"""Pure-Python evaluation contract for the Lecture 3 agent comparison.

These metrics deliberately avoid claiming that a token/keyword check proves semantic
correctness. No SDK, API key, or network access is needed to import this module.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

CASES: tuple[dict[str, Any], ...] = (
    {"id": "course_levels", "category": "course_grounding", "question": "AIE2026の講座の宿題にある『3レベル』とは何ですか？", "required_tools": ["search_glossary"], "check_terms": ["基礎", "実装", "発展"]},
    {"id": "lora_qlora", "category": "multi_term", "question": "LoRAとQLoRAの違いを説明してください。", "required_tools": ["search_glossary"], "check_terms": ["LoRA", "QLoRA"]},
    {"id": "quiz02", "category": "tool_selection", "question": "第2回の内容から確認問題を1問出してください。", "required_tools": ["make_quiz"], "check_terms": []},
    {"id": "lecture03", "category": "lecture_summary", "question": "第3回では何を学びますか？", "required_tools": ["get_lecture_summary"], "check_terms": ["エージェント"]},
    {"id": "rag", "category": "glossary", "question": "RAGとは何ですか？", "required_tools": ["search_glossary"], "check_terms": []},
    {"id": "missing_glossary", "category": "uncertainty", "question": "AIE2026の用語集に『完全架空概念XYZ991』は載っていますか？", "required_tools": ["search_glossary"], "check_terms": []},
    {"id": "summary02", "category": "lecture_summary", "question": "第2回の授業では何を学びましたか？", "required_tools": ["get_lecture_summary"], "check_terms": []},
    {"id": "quiz99", "category": "edge_case", "question": "第99回から確認問題を1問出してください。", "required_tools": ["make_quiz"], "check_terms": []},
)

QUICK_IDS = ("course_levels", "lora_qlora", "quiz02", "lecture03")


def select_cases(mode: str) -> list[dict[str, Any]]:
    if mode == "quick":
        return [dict(c) for c in CASES if c["id"] in QUICK_IDS]
    if mode == "full":
        return [dict(c) for c in CASES]
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_response(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Score observable tool use and lexical evidence; never label semantic accuracy."""
    if not result.get("ok"):
        return {"tool_coverage": None, "phrase_check": None, "needs_human_review": True}
    tools_seen = {call.get("tool") for call in result.get("tool_calls", [])}
    required = set(case["required_tools"])
    answer = str(result.get("answer", ""))
    phrases = case.get("check_terms", [])
    return {
        "tool_coverage": bool(required.issubset(tools_seen)),
        "phrase_check": all(term.casefold() in answer.casefold() for term in phrases) if phrases else None,
        "needs_human_review": True,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate with explicit denominators; no silent exclusion of failed calls."""
    by_framework: dict[str, dict[str, Any]] = {}
    for key in ("pydantic_ai", "google_adk"):
        rs = [pair[key] for pair in records]
        successes = [r for r in rs if r.get("ok")]
        covered = [r.get("evaluation", {}).get("tool_coverage") for r in successes]
        phrases = [r.get("evaluation", {}).get("phrase_check") for r in successes]
        phrases = [p for p in phrases if p is not None]
        tool_counts = Counter(call.get("tool") for r in successes for call in r.get("tool_calls", []))
        by_framework[key] = {
            "attempts": len(rs),
            "succeeded": len(successes),
            "failed": len(rs) - len(successes),
            "required_tool_ok": sum(x is True for x in covered),
            "required_tool_denominator": len(successes),
            "lexical_checks_ok": sum(x is True for x in phrases),
            "lexical_checks_denominator": len(phrases),
            "tool_call_counts": dict(sorted(tool_counts.items())),
            "latencies_s": [r.get("seconds") for r in successes],
            "note": "Lexical checks only; not semantic correctness or hallucination rate.",
        }
    return {
        "paired_attempts": len(records),
        "paired_both_succeeded": sum(p["pydantic_ai"].get("ok") and p["google_adk"].get("ok") for p in records),
        "frameworks": by_framework,
    }


def safe_jsonable(value: Any) -> Any:
    """Normalize SDK objects to stable JSON without relying on private object internals."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return safe_jsonable(value.model_dump(mode="json", exclude_none=True))
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        return f"<bytes length={len(value)}>"
    return str(value)[:300]


def validate_record_file(obj: dict[str, Any]) -> None:
    """Reject incomplete or contradictory evidence before publishing a PR."""
    assert isinstance(obj.get("records"), list), "records missing"
    assert isinstance(obj.get("plan"), dict), "plan missing"
    observed_keys: set[tuple[str, int]] = set()
    for row in obj["records"]:
        key = (row["case_id"], row["repeat"])
        assert key not in observed_keys, f"duplicate measurement: {key}"
        observed_keys.add(key)
        assert isinstance(row.get("question"), str) and row["question"]
        for framework in ("pydantic_ai", "google_adk"):
            result = row[framework]
            assert type(result.get("ok")) is bool, f"{framework} no status"
            assert isinstance(result.get("seconds"), (float, int))
            assert not result["ok"] or isinstance(result.get("answer"), str)


if __name__ == "__main__":
    print(json.dumps({"quick_cases": QUICK_IDS, "full_cases": len(CASES)}, ensure_ascii=False))
