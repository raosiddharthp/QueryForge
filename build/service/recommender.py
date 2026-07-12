"""Every response includes a ready-to-paste config_recommendation, and every
request is logged to Firestore's query_logs collection — the same collection
that closes the MLOps loop back to §6.3's experiment-grid re-runs."""
import time
import uuid

from google.cloud import firestore

from config import RERANK_MODEL

_db: firestore.Client | None = None


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def build_config_recommendation(query_type: str, route: dict) -> dict:
    strategy = "decompose+hybrid+rrf" if route["decompose"] else "hybrid+rrf"
    return {
        "query_type": query_type,
        "recommended_strategy": strategy,
        "alpha": route["alpha"],
        "top_k_per_subquery": 5,
        "reranking": route["rerank"],
        "reranker": RERANK_MODEL if route["rerank"] else None,
    }


def log_query(query: str, classifier_explanation: dict, sub_queries: list[str],
              route: dict, latency_ms: int) -> None:
    """Fire-and-forget-ish write to Firestore. Kept synchronous and cheap —
    this is one write per request, well inside the 20K writes/day Always-Free
    quota at demo volume (see design doc §8.2)."""
    doc_id = f"qlog_{uuid.uuid4().hex[:8]}"
    _get_db().collection("query_logs").document(doc_id).set({
        "query_id": doc_id,
        "query_text": query,
        "classifier_type": classifier_explanation["type"],
        "confidence": classifier_explanation["confidence"],
        "sub_queries": sub_queries,
        "alpha": route["alpha"],
        "reranked": route["rerank"],
        "fallback_applied": route["fallback_applied"],
        "latency_ms": latency_ms,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })


def timer_ms(start: float) -> int:
    return round((time.time() - start) * 1000)
