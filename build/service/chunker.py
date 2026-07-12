"""Chunking strategy router — §4 of the design doc. Uniform token splitting
is deliberately not used; each content type gets the chunking strategy that
preserves the structure that actually matters for that document class."""
import re

CHUNK_CONFIG = {
    "policy_docs":  {"strategy": "section-aware", "max_tokens": 1024},
    "runbooks":     {"strategy": "step-aware",     "max_tokens": 512},
    "faq":          {"strategy": "qa-pair",        "max_tokens": 256},
    "email_slack":  {"strategy": "message-boundary", "max_tokens": 256},
    "spreadsheet":  {"strategy": "row-group",      "max_tokens": 512},
}

CHUNK_VERSION = "v1"


def _section_aware(text: str) -> list[str]:
    """Split on §-numbered sections or markdown-style headers."""
    parts = re.split(r"(?=^§\d|^#{1,3}\s)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _step_aware(text: str) -> list[str]:
    """Keep numbered steps intact — never split mid-step."""
    parts = re.split(r"(?=^\d+\.\s)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _qa_pair(text: str) -> list[str]:
    """Keep a Q: ... A: ... pair together as one chunk."""
    parts = re.split(r"(?=^Q:)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _message_boundary(text: str) -> list[str]:
    """One chunk per message, preserving [author] prefixes for thread context."""
    parts = re.split(r"(?=^\[[\w\s]+\]:)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _row_group(rows: list[str], header: str, group_size: int = 20) -> list[str]:
    """Include the header row in every chunk so a fragment is never
    unlabelled data."""
    return [f"{header}\n" + "\n".join(rows[i:i + group_size]) for i in range(0, len(rows), group_size)]


_STRATEGIES = {
    "section-aware": _section_aware,
    "step-aware": _step_aware,
    "qa-pair": _qa_pair,
    "message-boundary": _message_boundary,
}


def chunk_document(text: str, content_type: str) -> list[dict]:
    """Returns a list of {"text", "chunk_strategy", "chunk_version"} dicts."""
    config = CHUNK_CONFIG.get(content_type, CHUNK_CONFIG["policy_docs"])
    strategy_name = config["strategy"]

    if strategy_name == "row-group":
        lines = [l for l in text.splitlines() if l.strip()]
        header, rows = (lines[0], lines[1:]) if lines else ("", [])
        pieces = _row_group(rows, header)
    else:
        pieces = _STRATEGIES.get(strategy_name, _section_aware)(text)

    return [{"text": p, "chunk_strategy": strategy_name, "chunk_version": CHUNK_VERSION} for p in pieces]
