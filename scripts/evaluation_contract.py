"""AIE2026 Lecture 02: provenance-aware glossary evaluation.

This module is standard-library only. In a source-backed workflow, an answer is
anchored to the *supplied* glossary, not the model's unsupported guess about
whether a real-world term exists.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

REVIEW_MESSAGE = "この用語は用語集に見つかりませんでした。確認が必要です。"


def validate_output(data: Any) -> bool:
    """Enforce an exact two-field schema; do not silently coerce strings to bool."""
    return (type(data) is dict and set(data) == {"answer", "needs_review"}
            and type(data["answer"]) is str
            and type(data["needs_review"]) is bool)


def parse_model_json(raw: str) -> dict | None:
    """Reject malformed or nonconforming outputs; never silently repair model claims."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if validate_output(data) else None


def normalize_alias(value: str) -> str:
    """Normalize width/case and surrounding whitespace, but do not infer meaning."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def lookup_exact_quoted(question: str, glossary: list[dict]) -> dict | None:
    """Return exactly one source document matching the Japanese quoted term.

    No quote / no hit / alias collision => abstain. Short-substring matching is
    intentionally avoided (e.g. API vs API key).
    """
    match = re.search(r"「([^」]+)」", question)
    if not match:
        return None
    key = normalize_alias(match.group(1))
    hits = [entry for entry in glossary if any(
        normalize_alias(alias) == key for alias in
        [entry["term"], *entry.get("aliases", [])]
    )]
    return hits[0] if len(hits) == 1 else None


def source_grounded_answer(question: str, glossary: list[dict]) -> tuple[dict, str | None]:
    """Deterministic source-only benchmark, **not** an LLM/RAG generation score."""
    match = lookup_exact_quoted(question, glossary)
    if match is None:
        return {"answer": REVIEW_MESSAGE, "needs_review": True}, None
    return {"answer": match["definition"], "needs_review": False}, match["id"]


def accept_grounded_generation(raw: str, source_answer: dict) -> tuple[dict, bool]:
    """Safe system-level fallback: serve only verbatim documented definitions.

    Returns (served, fallback_used). A valid *model* JSON response is not proof
    of factuality, so it is retained only when its contents match the source.
    For a conversational app, preserving semantic paraphrases requires a
    separate validated entailment/evidence-checking mechanism.
    """
    data = parse_model_json(raw)
    if data is not None and data == source_answer:
        return data, False
    return source_answer, True
