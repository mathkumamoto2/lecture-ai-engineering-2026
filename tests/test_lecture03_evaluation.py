"""Pure offline tests: no API key or ADK dependency required."""

from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

from scripts.lec03_eval import CASES, evaluate_response, safe_jsonable, select_cases, summarize, validate_record_file
from scripts.lec03_compare import extract_adk_events, extract_pydantic, make_report, mask_exception


def test_testplan_has_multiple_categories_and_unique_ids():
    assert len(CASES) >= 8
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert len({case["category"] for case in CASES}) >= 5
    assert len(select_cases("quick")) < len(select_cases("full"))


def test_tool_coverage_never_claims_semantic_accuracy():
    case = CASES[0]
    ok = evaluate_response(case, {"ok": True, "answer": "基礎 実装 発展", "tool_calls": [{"tool": "search_glossary", "args": {"query": "3レベル"}}]})
    assert ok["tool_coverage"] and ok["phrase_check"]
    assert ok["needs_human_review"] is True
    skipped = evaluate_response(case, {"ok": False})
    assert skipped["tool_coverage"] is None
    wrong = evaluate_response(case, {"ok": True, "answer": "基礎", "tool_calls": []})
    assert wrong["tool_coverage"] is False and wrong["phrase_check"] is False


def test_summary_explicit_denominators_and_errors():
    case = CASES[0]
    a = {"ok": True, "seconds": 1.2, "answer": "基礎 実装 発展", "tool_calls": [{"tool": "search_glossary"}]}
    a["evaluation"] = evaluate_response(case, a)
    b = {"ok": False, "seconds": 0.1, "error_type": "RateLimit"}
    s = summarize([{"pydantic_ai": a, "google_adk": b}])
    assert s["paired_both_succeeded"] == 0
    assert s["frameworks"]["pydantic_ai"]["required_tool_ok"] == 1
    assert s["frameworks"]["google_adk"]["attempts"] == 1
    assert s["frameworks"]["google_adk"]["failed"] == 1


def test_duplicate_record_validation_is_rejected():
    record = {"case_id": "x", "repeat": 1, "question": "a", "pydantic_ai": {"ok": False, "seconds": 1}, "google_adk": {"ok": False, "seconds": 2}}
    try:
        validate_record_file({"plan": {}, "records": [record, record]})
    except AssertionError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate not rejected")


def test_safe_jsonable_handles_bare_and_sdk_shapes():
    obj = SimpleNamespace(model_dump=lambda **kwargs: {"part": [1, "a"]})
    assert safe_jsonable(obj) == {"part": [1, "a"]}
    assert safe_jsonable({"v": (1, 2)}) == {"v": [1, 2]}


@dataclass
class Call:
    name: str
    args: dict


@dataclass
class Response:
    name: str
    response: dict


class Event:
    def __init__(self, calls=(), responses=(), final=False, text=None, usage=False):
        self.calls, self.responses = calls, responses
        self.final = final
        self.content = SimpleNamespace(parts=[SimpleNamespace(text=text)] if text else [])
        self.usage_metadata = SimpleNamespace(prompt_token_count=8, candidates_token_count=4, total_token_count=12) if usage else None

    def get_function_calls(self):
        return self.calls

    def get_function_responses(self):
        return self.responses

    def is_final_response(self):
        return self.final


def test_adk_event_extractor_preserves_order_and_usage():
    events = [Event(calls=[Call("search_glossary", {"query": "LoRA"})], usage=True),
              Event(responses=[Response("search_glossary", {"result": "value"})]),
              Event(final=True, text="LoRAの定義", usage=True)]
    data = extract_adk_events(events)
    assert len(data["tool_calls"]) == len(data["tool_returns"]) == 1
    assert data["trace"][0]["kind"] == "tool_call"
    assert data["trace"][1]["kind"] == "tool_return"
    assert data["usage_events"] == 2
    assert data["answer"] == "LoRAの定義"
    assert data["token_totals_event_sum"]["total_tokens"] == 24


def test_api_key_is_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-abc123")
    masked = mask_exception(RuntimeError("failure secret-abc123"))
    assert "secret-abc123" not in masked["error_message"]
    assert "[REDACTED_KEY]" in masked["error_message"]


def test_report_does_not_claim_human_accuracy():
    case = CASES[0]
    a = {"ok": True, "answer": "基礎 実装 発展", "seconds": 0.5, "tool_calls": [{"tool": "search_glossary"}], "trace": [], "llm_requests": 2}
    b = {"ok": True, "answer": "基礎 実装 発展", "seconds": 0.7, "tool_calls": [{"tool": "search_glossary"}], "trace": [], "usage_events": 2}
    a["evaluation"] = evaluate_response(case, a)
    b["evaluation"] = evaluate_response(case, b)
    data = {"records": [{"case_id": case["id"], "repeat": 1, "question": case["question"], "pydantic_ai": a, "google_adk": b}],
            "model": "gemini-test", "updated_utc": "2026-11-09", "python_version": "3.12", "versions": {},
            "glossary_sha256": "placeholder", "instructions_sha256": "placeholder", "plan": {"mode": "quick", "repeat": 1}}
    report = make_report(data)
    assert "1/1" in report and "人手採点を実施したとは主張しない" in report
    assert "usage_events" in report


def test_pydantic_extractor_observes_calls(monkeypatch):
    module = types.ModuleType("pydantic_ai.messages")
    class ToolCallPart:
        def __init__(self, name, args):
            self.tool_name, self.args = name, args
    class ToolReturnPart:
        def __init__(self, name, content):
            self.tool_name, self.content = name, content
    module.ToolCallPart, module.ToolReturnPart = ToolCallPart, ToolReturnPart
    monkeypatch.setitem(sys.modules, "pydantic_ai", types.ModuleType("pydantic_ai"))
    monkeypatch.setitem(sys.modules, "pydantic_ai.messages", module)
    messages = [SimpleNamespace(parts=[ToolCallPart("make_quiz", {"lecture": 2}), ToolReturnPart("make_quiz", "done")])]
    fake = SimpleNamespace(all_messages=lambda: messages, usage=SimpleNamespace(requests=2), output="question")
    result = extract_pydantic(fake)
    assert result["llm_requests"] == 2
    assert result["tool_calls"][0]["tool"] == "make_quiz"
    assert result["tool_returns"][0]["preview"] == "done"


def test_resume_preserves_review_only_for_identical_answer(tmp_path, monkeypatch):
    import csv
    from scripts import lec03_compare as cmp
    monkeypatch.setattr(cmp, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cmp, "RESULT_PATH", tmp_path / "comparison_quick.json")
    monkeypatch.setattr(cmp, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(cmp, "REVIEW_PATH", tmp_path / "review.csv")
    monkeypatch.setattr(cmp, "ANSWER_PATH", tmp_path / "nonexistent.md")
    case = CASES[0]
    a = {"ok": True, "answer": "initial", "seconds": 0.1, "tool_calls": [{"tool": "search_glossary"}], "trace": []}
    b = {"ok": True, "answer": "other", "seconds": 0.1, "tool_calls": [{"tool": "search_glossary"}], "trace": []}
    for response in (a, b):
        response["evaluation"] = evaluate_response(case, response)
    data = {"schema_version": 2, "created_utc": "2026-11-09", "updated_utc": "2026-11-09", "plan": {"mode": "quick", "repeat": 1},
            "model": "fake", "glossary_sha256": "sha", "instructions_sha256": "sha", "python_version": "3.12", "versions": {},
            "records": [{"case_id": case["id"], "repeat": 1, "question": case["question"], "pydantic_ai": a, "google_adk": b}]}
    cmp.write_evidence(data)
    with cmp.REVIEW_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["human_grounded_0_or_1"] = "1"
    rows[0]["review_notes"] = "human checked"
    with cmp.REVIEW_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    cmp.write_evidence(data)
    with cmp.REVIEW_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["human_grounded_0_or_1"] == "1"
    assert rows[0]["review_notes"] == "human checked"
    a["answer"] = "changed on retry"
    cmp.write_evidence(data)
    with cmp.REVIEW_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["answer"] == "changed on retry"
    assert rows[0]["human_grounded_0_or_1"] == ""
    assert rows[0]["review_notes"] == ""
