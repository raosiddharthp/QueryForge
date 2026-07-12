"""
budget_guard.py — ADR-006 revised.

Google's own billing data lags by "at least 24 hours" per their published
documentation. That makes a reactive Cloud Billing Budget alert (Pub/Sub
notification -> Cloud Function -> disable) structurally unable to enforce a
$0.01 cap in real time: by the time any billing-data-driven trigger fires,
the overspend it's reacting to already happened.

This module self-enforces the cap instead. Every Gemini call is checked
against a running total in Firestore BEFORE the call is made, using
published per-token pricing to estimate cost — zero reliance on GCP's
billing pipeline, zero lag. The Cloud Billing Budget provisioned in
infra/main.tf remains in place as an independent secondary tripwire:
defense in depth in case this code itself has a bug, not the primary
enforcement mechanism.
"""
import time
from datetime import datetime, timezone

from google.cloud import firestore

_db: firestore.Client | None = None
DOC_PATH = ("queryforge_budget", "gemini_spend")
CAP_USD = 0.01

# USD per 1,000,000 tokens. Source: ai.google.dev/gemini-api/docs/pricing,
# confirmed current as of this build's design doc (§8.2) — re-verify before
# relying on these long-term, Google revises pricing without much notice.
PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
}


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def _current_month_start() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _get_state() -> dict:
    doc_ref = _get_db().collection(DOC_PATH[0]).document(DOC_PATH[1])
    doc = doc_ref.get()
    month = _current_month_start()

    if not doc.exists or doc.to_dict().get("period") != month:
        fresh = {"total_usd": 0.0, "period": month, "call_count": 0}
        doc_ref.set(fresh)
        return fresh
    return doc.to_dict()


class BudgetExhaustedError(Exception):
    def __init__(self, spent: float):
        self.spent = spent
        super().__init__(f"Gemini budget cap (${CAP_USD}) reached for this period — spent ${spent:.6f}")


def check_budget_available() -> dict:
    state = _get_state()
    remaining = max(0.0, CAP_USD - state["total_usd"])
    return {"ok": state["total_usd"] < CAP_USD, "spent": state["total_usd"], "remaining": remaining}


def record_usage(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0  # unpriced model — spend not tracked for this call, logged by caller

    cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

    doc_ref = _get_db().collection(DOC_PATH[0]).document(DOC_PATH[1])

    @firestore.transactional
    def _update(transaction):
        snapshot = doc_ref.get(transaction=transaction)
        month = _current_month_start()
        current = snapshot.to_dict() if snapshot.exists and snapshot.to_dict().get("period") == month \
            else {"total_usd": 0.0, "period": month, "call_count": 0}
        transaction.set(doc_ref, {
            "total_usd": current["total_usd"] + cost,
            "period": current["period"],
            "call_count": current["call_count"] + 1,
            "last_updated": time.time(),
        })

    _update(_get_db().transaction())
    return cost


def require_budget() -> None:
    """Call before any Gemini request. Raises BudgetExhaustedError if the
    self-tracked spend for this period has already reached the cap."""
    state = check_budget_available()
    if not state["ok"]:
        raise BudgetExhaustedError(state["spent"])
