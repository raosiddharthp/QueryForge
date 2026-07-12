"""Reciprocal Rank Fusion. Rank-based, not score-based — this is deliberate:
dense similarity, BM25 score, and cross-encoder logits live on incompatible
scales, and RRF sidesteps having to reconcile them by only ever looking at
where a document sits in each list, not how it scored."""
from config import RRF_K


def rrf_fuse(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """Each ranked_lists entry is a list of dicts with a 'doc_id' key, already
    sorted best-first. Returns a single fused, sorted list with 'rrf_score'
    and 'source_strategies' (which input lists contributed to this doc)."""
    scores: dict[str, float] = {}
    text_by_id: dict[str, str] = {}
    sources: dict[str, set[str]] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        strategy_name = ranked[0].get("_strategy", f"strategy_{list_idx}") if ranked else f"strategy_{list_idx}"
        for rank, doc in enumerate(ranked):
            doc_id = doc["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            text_by_id.setdefault(doc_id, doc.get("text", ""))
            sources.setdefault(doc_id, set()).add(strategy_name)

    fused = [
        {"doc_id": doc_id, "text": text_by_id[doc_id], "rrf_score": round(score, 6),
         "source_strategies": sorted(sources[doc_id])}
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda d: d["rrf_score"], reverse=True)


def tag_strategy(ranked: list[dict], strategy_name: str) -> list[dict]:
    for d in ranked:
        d["_strategy"] = strategy_name
    return ranked
