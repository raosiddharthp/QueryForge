"""
Three retrieval strategies, run concurrently by main.py via asyncio.gather().
Dense uses Firestore's native vector search (ADR-002) so the index survives
Cloud Run's scale-to-zero cold starts. Sparse is a self-hosted, in-process
BM25 index — no external service, no per-query cost either way.
"""
import asyncio
from datetime import datetime, timezone

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from rank_bm25 import BM25Okapi

from embeddings import embed

_db: firestore.Client | None = None
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}  # corpus_id -> (index, docs)


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


async def dense_retrieve(query: str, corpus_id: str, k: int) -> list[dict]:
    """Cosine similarity over Firestore Vector Search. Runs in a thread since
    the firestore client here is synchronous."""
    def _run():
        query_vector = embed(query)
        docs = (
            _get_db()
            .collection("corpus_chunks")
            .where("corpus_id", "==", corpus_id)
            .find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=k,
                distance_result_field="vector_distance",
            )
            .get()
        )
        results = []
        for d in docs:
            data = d.to_dict()
            results.append({
                "doc_id": data.get("doc_id", d.id),
                "text": data.get("text", ""),
                "similarity": 1 - data.get("vector_distance", 1.0),
            })
        return results
    return await asyncio.to_thread(_run)


def _load_bm25_corpus(corpus_id: str) -> tuple[BM25Okapi, list[dict]]:
    """BM25 needs the full tokenized corpus in memory. Cached per corpus_id
    for the lifetime of the container — rebuilt on cold start, which is fine
    at demo-corpus scale (see MVP Scope, corpus capped at ~50MB)."""
    if corpus_id in _bm25_cache:
        return _bm25_cache[corpus_id]
    docs = list(_get_db().collection("corpus_chunks").where("corpus_id", "==", corpus_id).stream())
    parsed = [{"doc_id": d.to_dict().get("doc_id", d.id), "text": d.to_dict().get("text", ""),
               "metadata": d.to_dict().get("metadata", {})} for d in docs]
    tokenized = [p["text"].lower().split() for p in parsed]
    index = BM25Okapi(tokenized) if tokenized else None
    _bm25_cache[corpus_id] = (index, parsed)
    return index, parsed


async def sparse_retrieve(query: str, corpus_id: str, k: int, date_boost: float = 1.0) -> list[dict]:
    def _run():
        index, docs = _load_bm25_corpus(corpus_id)
        if index is None:
            return []
        scores = index.get_scores(query.lower().split())
        if date_boost != 1.0:
            now = datetime.now(timezone.utc)
            for i, d in enumerate(docs):
                effective_date = d.get("metadata", {}).get("effective_date")
                if effective_date:
                    age_days = (now - datetime.fromisoformat(effective_date).replace(tzinfo=timezone.utc)).days
                    scores[i] *= date_boost if age_days < 365 else 1.0
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"doc_id": docs[i]["doc_id"], "text": docs[i]["text"], "bm25_score": float(scores[i])}
                for i in ranked_idx if scores[i] > 0]
    return await asyncio.to_thread(_run)


def hybrid_combine(dense: list[dict], sparse: list[dict], alpha: float) -> list[dict]:
    """alpha * dense_score + (1-alpha) * bm25_score, normalized to [0,1] within
    each list first so the two scales are comparable before blending."""
    def _normalize(items: list[dict], key: str) -> dict[str, float]:
        if not items:
            return {}
        values = [i[key] for i in items]
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        return {i["doc_id"]: (i[key] - lo) / span for i in items}

    dense_norm = _normalize(dense, "similarity")
    sparse_norm = _normalize(sparse, "bm25_score")
    text_by_id = {d["doc_id"]: d["text"] for d in dense + sparse}

    all_ids = set(dense_norm) | set(sparse_norm)
    blended = []
    for doc_id in all_ids:
        score = alpha * dense_norm.get(doc_id, 0.0) + (1 - alpha) * sparse_norm.get(doc_id, 0.0)
        blended.append({"doc_id": doc_id, "text": text_by_id.get(doc_id, ""), "hybrid_score": score})
    return sorted(blended, key=lambda x: x["hybrid_score"], reverse=True)
