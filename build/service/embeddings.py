"""
ADR-003: dense embeddings and reranking run on open weights bundled into this
container's own image, not a metered Vertex AI endpoint. Cost is Cloud Run
vCPU-seconds (Always-Free up to 180K/month), not a per-token API charge.
Models are loaded once per container instance and reused across requests.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from config import EMBED_MODEL, RERANK_MODEL


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def embed(text: str) -> list[float]:
    vec = _embedder().encode(text, normalize_embeddings=True)
    return vec.tolist()


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """candidates: list of {"doc_id", "text", ...}. Returns the same list,
    re-sorted by cross-encoder score, with a `rerank_score` field added."""
    if not candidates:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = _reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
