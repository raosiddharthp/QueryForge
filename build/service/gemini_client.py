"""
Every LLM call in QueryForge goes through the Gemini Developer API
(genai.Client(api_key=...)) — never genai.Client(vertexai=True, ...).
That's ADR-004: the Developer API's free tier is billed on its own quota,
independent of the Cloud Billing account this project's $0.01 cap protects.
"""
import json
import os

from google import genai
from google.genai.types import GenerateContentConfig

from budget_guard import record_usage, require_budget
from config import GEMINI_MODEL, QUERY_TYPES

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ["GEMINI_API_KEY"]  # from Secret Manager, injected at deploy time
        _client = genai.Client(api_key=api_key)
    return _client


def _guarded_generate(**kwargs):
    """Every Gemini call goes through the proactive spend guard first —
    see budget_guard.py for why a reactive Cloud Billing Budget alone
    can't enforce a $0.01 cap in real time."""
    require_budget()
    response = _get_client().models.generate_content(**kwargs)
    usage = getattr(response, "usage_metadata", None)
    if usage:
        record_usage(kwargs["model"], usage.prompt_token_count or 0, usage.candidates_token_count or 0)
    return response


CLASSIFIER_SYSTEM_PROMPT = f"""You classify retrieval queries for a RAG optimization engine.
Classify the query into exactly one of: {", ".join(QUERY_TYPES)}.

- single-hop: a plain factual question answerable from one document
- multi-hop-entity: requires synthesizing information from multiple documents with an entity/approval constraint
- comparative: compares two or more entities, options, or policies
- temporal: asks how something has changed over time, or requires the most current version
- entity-heavy: keyed on an exact identifier (contract number, vendor ID, ticket number)

Return strict JSON only, no markdown fences:
{{"type": "<one of the types above>", "confidence": <float 0-1>,
  "signals": ["<short token-level phrase that drove the decision>", ...],
  "reasoning": "<one sentence, plain language>"}}"""


def classify(query: str) -> dict:
    response = _guarded_generate(
        model=GEMINI_MODEL,
        contents=query,
        config=GenerateContentConfig(
            system_instruction=CLASSIFIER_SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    parsed = json.loads(response.text)
    if parsed.get("type") not in QUERY_TYPES:
        parsed["type"] = "single-hop"  # never let a malformed label crash routing
    return parsed


DECOMPOSER_SYSTEM_PROMPT = """Break the query into 2-5 atomic sub-queries that can each be
retrieved independently. Each sub-query should isolate one concept, entity, or constraint
from the original question. Return strict JSON only: {"sub_queries": ["...", "..."]}"""


def decompose(query: str) -> list[str]:
    response = _guarded_generate(
        model=GEMINI_MODEL,
        contents=query,
        config=GenerateContentConfig(
            system_instruction=DECOMPOSER_SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)["sub_queries"]


HYDE_SYSTEM_PROMPT = """Write a short hypothetical passage (2-4 sentences) that would plausibly
answer the query, in the style of an internal policy or technical document. This will be
embedded and used as a search vector, not shown to the user — do not hedge or caveat."""


def generate_hyde_document(query: str) -> str:
    response = _guarded_generate(
        model=GEMINI_MODEL,
        contents=query,
        config=GenerateContentConfig(system_instruction=HYDE_SYSTEM_PROMPT, temperature=0.3),
    )
    return response.text.strip()
