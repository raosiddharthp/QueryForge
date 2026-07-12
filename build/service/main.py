"""
QueryForge pipeline orchestrator. Mirrors the design doc's §2 pipeline exactly:
validate -> classify -> decompose (multi-hop only) -> parallel retrieve
(dense + sparse) -> hybrid combine -> HyDE fallback (if dense similarity is
low) -> rerank (multi-hop only) -> RRF fuse -> recommend + log.

The only network call that leaves this container is to the Gemini Developer
API (gemini_client.py). Everything else — embeddings, BM25, reranking,
Firestore — runs inside this process or against Google Cloud Always-Free
resources. See the design doc's Figure 2 for the annotated version of this
same flow.
"""
import asyncio
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from budget_guard import BudgetExhaustedError
from config import HYDE_SIMILARITY_FLOOR, resolve_route
from embeddings import rerank
from fusion import rrf_fuse, tag_strategy
from gemini_client import classify, decompose, generate_hyde_document
from models import OptimizeRequest, OptimizeResponse
from recommender import build_config_recommendation, log_query, timer_ms
from retrieval import dense_retrieve, hybrid_combine, sparse_retrieve

app = FastAPI(title="QueryForge", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://queryforge-prod.web.app",
        "https://queryforge-prod.firebaseapp.com",
        "http://localhost:8000",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/v1/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest) -> OptimizeResponse:
    t0 = time.time()

    # ---- Classify ---------------------------------------------------------
    try:
        raw_classification = classify(req.query)
    except BudgetExhaustedError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Classifier call failed: {e}")

    query_type = raw_classification["type"]
    confidence = float(raw_classification["confidence"])
    route = resolve_route(query_type, confidence)

    # ---- Decompose (multi-hop only) ---------------------------------------
    sub_queries: list[str] = []
    if route["decompose"]:
        sub_queries = decompose(req.query)
    queries_to_retrieve = sub_queries if sub_queries else [req.query]

    # ---- Parallel dense + sparse retrieval, per sub-query -----------------
    async def retrieve_for(q: str) -> tuple[list[dict], list[dict]]:
        dense, sparse = await asyncio.gather(
            dense_retrieve(q, req.corpus_id, req.top_k),
            sparse_retrieve(q, req.corpus_id, req.top_k, date_boost=route["bm25_date_boost"]),
        )
        return dense, sparse

    retrieval_pairs = await asyncio.gather(*(retrieve_for(q) for q in queries_to_retrieve))

    # ---- HyDE fallback if the best dense hit is weak -----------------------
    hyde_triggered = False
    all_dense = [d for dense, _ in retrieval_pairs for d in dense]
    best_similarity = max((d["similarity"] for d in all_dense), default=0.0)
    if best_similarity < HYDE_SIMILARITY_FLOOR:
        hyde_triggered = True
        hyde_doc = generate_hyde_document(req.query)
        hyde_dense = await dense_retrieve(hyde_doc, req.corpus_id, req.top_k)
        retrieval_pairs.append((hyde_dense, []))

    # ---- Hybrid combine per sub-query, then merge into strategy-tagged lists
    hybrid_lists, dense_lists, sparse_lists = [], [], []
    for dense, sparse in retrieval_pairs:
        dense_lists.extend(dense)
        sparse_lists.extend(sparse)
        if dense or sparse:
            hybrid_lists.extend(hybrid_combine(dense, sparse, route["alpha"]))

    candidate_sets = [
        tag_strategy(sorted(dense_lists, key=lambda d: d["similarity"], reverse=True), "dense"),
        tag_strategy(sorted(sparse_lists, key=lambda d: d["bm25_score"], reverse=True), "sparse"),
        tag_strategy(hybrid_lists, "hybrid"),
    ]

    # ---- Rerank (multi-hop only, ~600ms tax paid only when it's earned) ---
    if route["rerank"]:
        top_candidates = candidate_sets[2][: req.top_k * 2]
        reranked = rerank(req.query, top_candidates)
        candidate_sets.append(tag_strategy(reranked, "reranked"))

    # ---- RRF fuse -----------------------------------------------------------
    fused = rrf_fuse(*[c for c in candidate_sets if c])[: req.top_k]

    # ---- Recommend + log ------------------------------------------------
    config_recommendation = build_config_recommendation(query_type, route)
    latency_ms = timer_ms(t0)
    try:
        log_query(req.query, raw_classification | {"fallback_applied": route["fallback_applied"]},
                   sub_queries, route, latency_ms)
    except Exception:
        pass  # logging failure should never fail the request

    return OptimizeResponse(
        results=fused,
        config_recommendation=config_recommendation,
        classifier_explanation={**raw_classification, "fallback_applied": route["fallback_applied"]},
        sub_queries=sub_queries,
        hyde_triggered=hyde_triggered,
        latency_ms=latency_ms,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
