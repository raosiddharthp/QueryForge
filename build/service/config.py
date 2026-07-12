"""
Routing table for QueryForge's classifier output. This is the single source of
truth for the adaptive-alpha defaults documented in the design doc (§3.2) —
the grid-search results (Luan et al.) are encoded here, not re-derived per request.
"""

QUERY_TYPES = ["single-hop", "multi-hop-entity", "comparative", "temporal", "entity-heavy"]

# alpha: dense/sparse blend weight. decompose/rerank: whether those stages run.
# bm25_date_boost: only used for temporal queries.
ROUTING_TABLE = {
    "single-hop":        {"alpha": 0.70, "decompose": False, "rerank": False, "bm25_date_boost": 1.0},
    "multi-hop-entity":  {"alpha": 0.40, "decompose": True,  "rerank": True,  "bm25_date_boost": 1.0},
    "comparative":       {"alpha": 0.55, "decompose": False, "rerank": False, "bm25_date_boost": 1.0},
    "temporal":          {"alpha": 0.50, "decompose": False, "rerank": False, "bm25_date_boost": 2.0},
    "entity-heavy":      {"alpha": 0.40, "decompose": False, "rerank": False, "bm25_date_boost": 1.0},
}

# ADR-003: confidence below this floor always falls back to the safest general
# strategy (hybrid + decompose) regardless of what the classifier predicted.
CONFIDENCE_FLOOR = 0.75
FALLBACK_ROUTE = {"alpha": 0.55, "decompose": True, "rerank": True, "bm25_date_boost": 1.0}

# HyDE fires when dense similarity on the first retrieval pass is below this.
HYDE_SIMILARITY_FLOOR = 0.65

RRF_K = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"


def resolve_route(query_type: str, confidence: float) -> dict:
    """ADR-003's confidence-floor fallback, applied once, in one place."""
    if confidence < CONFIDENCE_FLOOR:
        return {**FALLBACK_ROUTE, "fallback_applied": True}
    return {**ROUTING_TABLE.get(query_type, FALLBACK_ROUTE), "fallback_applied": False}
