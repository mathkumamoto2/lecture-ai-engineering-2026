"""Lecture 3 Advanced (a): controlled Pydantic AI vs Google ADK comparison.

Run from the repository root, on the user's Codespace:
    uv run --with 'google-adk>=2,<3' python scripts/lec03_compare.py --mode quick
    uv run --with 'google-adk>=2,<3' python scripts/lec03_compare.py --mode full --repeat 2 --pause 12 --publish

Only actual API observations are stored as measurements. Automatic heuristic scores
are NOT semantic accuracy. Failed pairs are retryable with --resume.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lec03_eval import evaluate_response, safe_jsonable, select_cases, summarize, validate_record_file

OUT_DIR = ROOT / "lectures" / "03" / "evidence"
RESULT_PATH = OUT_DIR / "comparison.json"
REVIEW_PATH = OUT_DIR / "human_review.csv"
REPORT_PATH = OUT_DIR / "comparison_report.md"
ANSWER_PATH = ROOT / "lectures" / "03" / "answer.md"
PR_BODY_PATH = ROOT / "lectures" / "03" / "PR_DESCRIPTION.md"
BEGIN = "<!-- LEC03_EVIDENCE_BEGIN -->"
END = "<!-- LEC03_EVIDENCE_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_exception(exc: Exception) -> dict[str, Any]:
    """Preserve diagnostic status but redact any accidental API key disclosure."""
    raw = str(exc)[:750]
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(name)
        if val:
            raw = raw.replace(val, "[REDACTED_KEY]")
    raw = re.sub(r"AIza[0-9A-Za-z_-]{12,}", "[REDACTED_KEY]", raw)
    return {
        "error_type": type(exc).__name__,
        "error_message": raw,
        "http_status": safe_jsonable(getattr(exc, "status_code", getattr(exc, "code", None))),
    }


def is_rate_limit(result: dict[str, Any]) -> bool:
    if result.get("ok"):
        return False
    return (str(result.get("http_status")) == "429" or "429" in result.get("error_message", "")
            or "ResourceExhausted" in result.get("error_type", ""))


def extract_pydantic(result: Any) -> dict[str, Any]:
    """Read actual Pydantic AI messages and account for model requests."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    calls, returns, ordered = [], [], []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                item = {"tool": part.tool_name, "args": safe_jsonable(part.args)}
                calls.append(item)
                ordered.append({"kind": "tool_call", **item})
            elif isinstance(part, ToolReturnPart):
                item = {"tool": part.tool_name, "preview": str(part.content)[:350]}
                returns.append(item)
                ordered.append({"kind": "tool_return", **item})
    usage = result.usage
    return {"answer": str(result.output), "tool_calls": calls, "tool_returns": returns,
            "trace": ordered, "llm_requests": safe_jsonable(usage.requests)}


def extract_adk_events(events: list[Any]) -> dict[str, Any]:
    """Extract per-event ADK calls, returns, final text and usage fields."""
    calls, returns, ordered, final_texts = [], [], [], []
    usage_events = 0
    tokens = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
    for event in events:
        for call in event.get_function_calls():
            item = {"tool": call.name, "args": safe_jsonable(call.args)}
            calls.append(item)
            ordered.append({"kind": "tool_call", **item})
        for response in event.get_function_responses():
            item = {"tool": response.name, "preview": str(response.response)[:350]}
            returns.append(item)
            ordered.append({"kind": "tool_return", **item})
        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            usage_events += 1
            for key, attr in (("prompt_tokens", "prompt_token_count"),
                              ("candidate_tokens", "candidates_token_count"),
                              ("total_tokens", "total_token_count")):
                value = getattr(usage, attr, 0) or 0
                if isinstance(value, (float, int)):
                    tokens[key] += int(value)
        if event.is_final_response():
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) or []
            text = "".join(part.text for part in parts if getattr(part, "text", None))
            if text:
                final_texts.append(text)
    return {"answer": "\n".join(final_texts), "tool_calls": calls,
            "tool_returns": returns, "trace": ordered,
            "usage_events": usage_events, "token_totals_event_sum": tokens,
            "usage_note": "ADK usage_events is not proven equivalent to Pydantic AI requests."}


async def pydantic_run(question: str) -> dict[str, Any]:
    from app.agent import agent, build_model
    started = time.perf_counter()
    try:
        obj = extract_pydantic(await agent.run(question, model=build_model()))
        return {"ok": bool(obj["answer"].strip()), "seconds": round(time.perf_counter() - started, 3), **obj}
    except Exception as exc:
        return {"ok": False, "seconds": round(time.perf_counter() - started, 3), **mask_exception(exc)}


async def adk_run(question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from adk_agent.glossary_agent.agent import root_agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        service = InMemorySessionService()
        session = await service.create_session(app_name="aie2026_lec03", user_id="comparison")
        runner = Runner(agent=root_agent, app_name="aie2026_lec03", session_service=service)
        events = []
        # Release runner-managed resources before the event loop closes.
        async with runner:
            async for event in runner.run_async(
                user_id="comparison", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
            ):
                events.append(event)
        obj = extract_adk_events(events)
        return {"ok": bool(obj["answer"].strip()), "seconds": round(time.perf_counter() - started, 3), **obj}
    except Exception as exc:
        return {"ok": False, "seconds": round(time.perf_counter() - started, 3), **mask_exception(exc)}


def new_evidence(plan: dict[str, Any], model: str, instructions: str,
                 tools: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 2, "created_utc": utc_now(), "updated_utc": utc_now(),
        "plan": plan, "model": model,
        "same_instructions": True, "same_python_tools": True, "registered_tools": tools,
        "instructions_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        "glossary_sha256": sha256_file(ROOT / "data/glossary.json"),
        "versions": {p: version_of(p) for p in ("google-adk", "pydantic-ai-slim", "pydantic-ai", "google-genai")},
        "python_version": sys.version.split()[0], "records": [],
        "limitations": ["Model sampling and backend scheduling may differ.",
                        "ADK usage_events is not a verified request-count equivalent.",
                        "Automatic tool/keyword metrics are not semantic correctness.",
                        "Agent framework versions may change; record package versions."],
    }


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_md(value: Any, limit: int = 280) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\r", " ").replace("\n", " ").replace("`", "'")[:limit]


def quote_md(text: Any, limit: int = 2200) -> str:
    value = str(text or "（出力なし）")[:limit]
    return "\n".join("> " + line for line in value.splitlines()) or "> （出力なし）"


def make_report(data: dict[str, Any]) -> str:
    records = data["records"]
    summary = summarize(records)
    lines = [
        "# 第3回 Advanced (a) — Pydantic AI / Google ADK 比較実験",
        "", f"- 計測日時（UTC）：{data['updated_utc']}",
        f"- Geminiモデル：`{data['model']}`（双方共通）",
        f"- Python / ADK / Pydantic AI：`{data.get('python_version')}` / `{data.get('versions',{}).get('google-adk')}` / `{data.get('versions',{}).get('pydantic-ai-slim')}`",
        f"- 用語集 SHA-256：`{data['glossary_sha256']}`",
        f"- 共通指示SHA-256：`{data['instructions_sha256']}`",
        f"- 条件：`{data['plan']['mode']}`、各質問{data['plan']['repeat']}回、独立セッション、実行順序を交互に変更",
        "", "## 成功・ツール使用・語句検査（実測から集計）", "",
        "| 評価項目 | Pydantic AI | Google ADK |", "|---|---:|---:|",
    ]
    for label, key in (("実行成功", "succeeded"), ("実行失敗", "failed"),
                       ("指定ツールの呼び出し", "required_tool_ok"),
                       ("指定語句の文字列検査", "lexical_checks_ok")):
        l = summary["frameworks"]["pydantic_ai"]
        r = summary["frameworks"]["google_adk"]
        if key == "succeeded":
            fmt = lambda m: f"{m['succeeded']}/{m['attempts']}"
        elif key == "required_tool_ok":
            fmt = lambda m: f"{m[key]}/{m['required_tool_denominator']}"
        elif key == "lexical_checks_ok":
            fmt = lambda m: f"{m[key]}/{m['lexical_checks_denominator']}"
        else:
            fmt = lambda m: str(m[key])
        lines.append(f"| {label} | {fmt(l)} | {fmt(r)} |")
    lines += ["", f"- 両方式とも成功したペア：**{summary['paired_both_succeeded']}/{summary['paired_attempts']}**", ""]
    for label, key in (("Pydantic AI", "pydantic_ai"), ("Google ADK", "google_adk")):
        vals = summary["frameworks"][key]["latencies_s"]
        if vals:
            lines.append(f"- {label}：成功例のレイテンシ中央値 {statistics.median(vals):.2f}秒（{len(vals)}件）。APIの混雑・リトライを含むため単独の速度優位性を保証しない。")
    lines += [
        "", "> **採点上の注意**：指定ツールの呼び出し率は観測可能な行動指標です。語句の有無は内容正解率ではありません。意味の正確性、根拠との整合性、説明の妥当性は `human_review.csv` で確認してください。",
        "", "> **使用量の注意**：Pydantic AI の `llm_requests` と、ADK の `usage_events` は同じ定義だと確認できていません。したがって直接差し引きません。",
        "", "## 質問・呼び出し順序・LLM出力", "",
    ]
    for row in records:
        lines += [f"### {row['case_id']}（反復{row['repeat']}）", "", f"質問：{row['question']}", ""]
        for name, key in (("Pydantic AI", "pydantic_ai"), ("Google ADK", "google_adk")):
            result = row[key]
            lines += [f"**{name}** — {'成功' if result.get('ok') else '失敗'} / {result.get('seconds')}秒", ""]
            if result.get("ok"):
                trace = [f"{x.get('kind')}: {x.get('tool')}({x.get('args', x.get('preview',''))})" for x in result.get("trace", [])]
                lines += ["- 呼び出し：" + safe_md(" → ".join(trace) if trace else "ツール呼び出しなし", 1300)]
                label = "requests" if key == "pydantic_ai" else "ADK usage events（近似指標）"
                count = result.get("llm_requests" if key == "pydantic_ai" else "usage_events", "不明")
                lines += [f"- {label}：{count}", "", quote_md(result.get("answer")), ""]
            else:
                lines += [f"- エラー：{safe_md(result.get('error_type'))} / HTTP {safe_md(result.get('http_status'))} / {safe_md(result.get('error_message'))}", ""]
    lines += [
        "## 課題の要件に関する考察", "",
        "- **実装の比較**：Pydantic AI は `Agent(instructions=...)` に対し `agent.tool_plain(function)` で登録。ADK は `Agent(instruction=..., tools=[...])` で登録し、`Runner` のイベント列を解析した。",
        "- **実験上の統制**：同じモデルID、同じ `INSTRUCTIONS` 文字列、同じPythonツール、同じ質問、独立セッション。実行順を交互にした。ただしSDK既定の生成パラメータや内部処理差は残る。",
        "- **根拠の評価**：ツールを呼んだことは正しい回答の必要条件になり得るが十分条件ではない。回答原文を用語集と照合することが必要。",
        "- **限界**：標本数・APIの負荷・レート制限・各フレームワークの非同一なメトリクス定義から、観測差をフレームワーク固有の性能差と断定しない。",
        "", "## 第1回の素のLLMとの比較", "",
        "第1回の回答原文と当時のモデル条件を今回の計測に取り込んでいないため、過去回答との直接的な正答比較は**未検証**。ツールを使わず単発生成する構成と、検索を経由する構成の差を説明するにとどめる。",
        "", "## 出典", "",
        "- AIE2026 第3回配布資料・`lectures/03/problem.md`",
        "- [Google ADK公式ドキュメント](https://google.github.io/adk-docs/)",
        "- [Pydantic AI公式ドキュメント](https://ai.pydantic.dev/)",
        "", "※ 本報告は保存したAPI結果のみから作成。人手採点を実施したとは主張しない。", "",
    ]
    return "\n".join(lines)


def write_evidence(data: dict[str, Any]) -> None:
    data["updated_utc"] = utc_now()
    validate_record_file(data)
    data["summary"] = summarize(data["records"])
    atomic_json(RESULT_PATH, data)
    REPORT_PATH.write_text(make_report(data), encoding="utf-8")
    # Rebuild machine-generated columns after every resume. Preserve reviewer annotations
    # only if the exact answer remains unchanged; otherwise an old label is invalid.
    annotations: dict[tuple[str, int, str], dict[str, str]] = {}
    if REVIEW_PATH.exists():
        with REVIEW_PATH.open(encoding="utf-8", newline="") as f:
            for prev in csv.DictReader(f):
                annotations[(prev["case_id"], int(prev["repeat"]), prev["framework"])] = prev
    tmp_csv = REVIEW_PATH.with_suffix(".csv.tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "repeat", "framework", "question", "answer", "required_tool_ok", "human_grounded_0_or_1", "human_adequate_0_or_1", "review_notes"])
        for pair in data["records"]:
            for framework in ("pydantic_ai", "google_adk"):
                row_key = (pair["case_id"], pair["repeat"], framework)
                r = pair[framework]
                answer = r.get("answer", "")
                prev = annotations.get(row_key, {})
                unchanged = prev.get("answer") == answer
                writer.writerow([*row_key, pair["question"], answer,
                                 r.get("evaluation", {}).get("tool_coverage"),
                                 prev.get("human_grounded_0_or_1", "") if unchanged else "",
                                 prev.get("human_adequate_0_or_1", "") if unchanged else "",
                                 prev.get("review_notes", "") if unchanged else ""])
    tmp_csv.replace(REVIEW_PATH)
    if ANSWER_PATH.exists() and data["plan"]["mode"] == "full":
        answer = ANSWER_PATH.read_text(encoding="utf-8")
        if answer.count(BEGIN) == 1 and answer.count(END) == 1:
            short = [
                "- 比較条件：" + f"`{data['model']}`、同一の指示・ツール、独立したセッション。",
                f"- ペア実験：{data['summary']['paired_both_succeeded']}/{data['summary']['paired_attempts']}件で両方式が成功。",
                "- 結果の詳細：`lectures/03/evidence/comparison_report.md` と `comparison.json`。",
                "- 判定上の注意：ツール利用の成否と、回答の意味的正確性を区別した。後者の独立した人手確認は未実施。",
            ]
            start = answer.index(BEGIN) + len(BEGIN)
            end = answer.index(END)
            answer = answer[:start] + "\n" + "\n".join(short) + "\n" + answer[end:]
            ANSWER_PATH.write_text(answer, encoding="utf-8")


def publish() -> None:
    """Publish only owned artifacts and only from the correct fork's lec03 branch."""
    def git(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr[-600:]}")
        return p.stdout.strip()

    branch = git("branch", "--show-current")
    remote = git("remote", "get-url", "origin")
    if branch != "lec03" or not (remote.endswith("mathkumamoto2/lecture-ai-engineering-2026.git") or
                                    remote.endswith("mathkumamoto2/lecture-ai-engineering-2026")):
        raise RuntimeError("PRの誤提出を防ぐため停止: origin と lec03 ブランチを確認してください")
    stage = ["scripts/lec03_compare.py", "lectures/03/answer.md", "lectures/03/evidence/comparison.json",
             "lectures/03/evidence/comparison_report.md", "lectures/03/evidence/human_review.csv"]
    stage += [str(p.relative_to(ROOT)) for p in
              (OUT_DIR / "comparison_quick.json", OUT_DIR / "comparison_quick_report.md",
               OUT_DIR / "human_review_quick.csv") if p.is_file()]
    git("add", "--", *stage)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "Lecture 3: record paired Pydantic AI and ADK measurements")
    git("push", "-u", "origin", "lec03")
    print("[OK] 実測データを mathkumamoto2 の lec03 にPushしました。")
    print("[PR] https://github.com/mathkumamoto2/lecture-ai-engineering-2026/compare/main...lec03?expand=1")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--repeat", type=int, default=1, help="1〜5回、同じ質問を独立セッションで反復")
    parser.add_argument("--pause", type=float, default=12.0, help="フレームワーク間のAPI間隔（秒）")
    parser.add_argument("--resume", action="store_true", help="同じ計画を再開。失敗ペアは再測定")
    parser.add_argument("--retries", type=int, default=2, help="HTTP 503の追加再試行回数（0〜4）")
    parser.add_argument("--retry-base", type=float, default=8.0, help="503再試行の初期間隔（秒、2倍ずつ増加）")
    parser.add_argument("--publish", action="store_true", help="全ペア成功時のみ測定結果をGitへPush")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5:
        parser.error("--repeat must be 1..5")
    if not 0 <= args.pause <= 120:
        parser.error("--pause must be 0..120")
    if not 0 <= args.retries <= 4:
        parser.error("--retries must be 0..4")
    if not 0 <= args.retry_base <= 120:
        parser.error("--retry-base must be 0..120")
    if not os.environ.get("GEMINI_API_KEY"):
        parser.error("Codespaces secret GEMINI_API_KEY が見つかりません。内容は表示しないでください")
    if (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1"}
            or os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in {"true", "1"}):
        parser.error("Google Cloud/VertexAI設定を無効にして、両者で同じAI Studioキーを使ってください")
    if os.environ.get("GOOGLE_API_KEY") and os.environ["GOOGLE_API_KEY"] != os.environ["GEMINI_API_KEY"]:
        parser.error("GOOGLE_API_KEYとGEMINI_API_KEYが異なります。キーを表示せず設定を統一してください")
    # google-genai prefers GOOGLE_API_KEY over GEMINI_API_KEY. Keep only the
    # configured course key so both frameworks authenticate the same way.
    os.environ.pop("GOOGLE_API_KEY", None)
    global RESULT_PATH, REPORT_PATH, REVIEW_PATH
    if args.mode == "quick":
        RESULT_PATH = OUT_DIR / "comparison_quick.json"
        REPORT_PATH = OUT_DIR / "comparison_quick_report.md"
        REVIEW_PATH = OUT_DIR / "human_review_quick.csv"
    else:
        RESULT_PATH = OUT_DIR / "comparison.json"
        REPORT_PATH = OUT_DIR / "comparison_report.md"
        REVIEW_PATH = OUT_DIR / "human_review.csv"
    if args.publish and args.mode != "full":
        parser.error("--publish は --mode full の完了後だけ許可します")
    from app.agent import INSTRUCTIONS
    from app.config import MODEL
    from adk_agent.glossary_agent.agent import root_agent
    from app import tools

    expected_tools = ("search_glossary", "get_lecture_summary", "make_quiz")
    if root_agent.instruction != INSTRUCTIONS or root_agent.model != MODEL:
        parser.error("ADKとPydantic AIでモデルIDまたは指示文が一致しません")
    shared = [getattr(tools, name, None) for name in expected_tools]
    if not all(callable(func) for func in shared):
        parser.error("演習で指定されたツールが存在しません")
    if len(root_agent.tools) != len(shared) or any(a is not b for a, b in zip(root_agent.tools, shared)):
        parser.error("ADK側の登録ツールと講座の共通Python関数が一致しません")
    cases = select_cases(args.mode)
    plan = {"mode": args.mode, "repeat": args.repeat, "case_ids": [c["id"] for c in cases],
            "alternating_order": True, "pause_seconds": args.pause,
            "http_503_max_extra_retries": args.retries, "http_503_retry_base_seconds": args.retry_base}
    if RESULT_PATH.exists():
        if not args.resume:
            parser.error("既存の比較結果があります。上書き防止のため --resume を指定してください")
        data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if {k: data["plan"].get(k) for k in ("mode", "repeat", "case_ids")} != {k: plan[k] for k in ("mode", "repeat", "case_ids")}:
            parser.error("既存の実験計画と異なります。既存JSONを退避してから新しい実験を開始してください")
        if data["model"] != MODEL or data["glossary_sha256"] != sha256_file(ROOT / "data/glossary.json") or data["instructions_sha256"] != hashlib.sha256(INSTRUCTIONS.encode()).hexdigest():
            parser.error("モデル・用語集・指示が前回と異なります。実験条件を分けて保存してください")
        validate_record_file(data)
    else:
        data = new_evidence(plan, MODEL, INSTRUCTIONS, list(expected_tools))
    indexes = {(r["case_id"], r["repeat"]): idx for idx, r in enumerate(data["records"])}
    total = len(cases) * args.repeat
    print(f"[計画] {len(cases)}問 × {args.repeat}反復 × 2方式 / {total}ペア、モデル={MODEL}")
    interrupted = False
    for repeat in range(1, args.repeat + 1):
        for case in cases:
            key = (case["id"], repeat)
            prev_index = indexes.get(key)
            if prev_index is not None and all(
                data["records"][prev_index][fw].get("ok")
                for fw in ("pydantic_ai", "google_adk")
            ):
                continue
            # Alternating which framework goes first reduces order bias.
            order = ("pydantic_ai", "google_adk") if (repeat + cases.index(case)) % 2 else ("google_adk", "pydantic_ai")
            if prev_index is None:
                record: dict[str, Any] = {
                    "case_id": case["id"], "category": case["category"],
                    "repeat": repeat, "question": case["question"],
                    "required_tools": case["required_tools"],
                    "check_terms": case["check_terms"], "order": list(order),
                    # 'pending' is not an attempted API call. An unfinished
                    # record remains recognizable in the saved evidence.
                    "pydantic_ai": {"ok": False, "seconds": 0.0, "error_type": "pending"},
                    "google_adk": {"ok": False, "seconds": 0.0, "error_type": "pending"},
                }
                data["records"].append(record)
                indexes[key] = len(data["records"]) - 1
            else:
                record = data["records"][prev_index]
            print(f"[{indexes[key]+1}/{total}] {case['id']} (repeat={repeat}, first={order[0]})", flush=True)
            attempted_this_pair = 0
            for framework in order:
                if record[framework].get("ok"):
                    print(f"  {framework}: SKIP (既存の成功結果を維持)", flush=True)
                    continue
                if attempted_this_pair and args.pause:
                    await asyncio.sleep(args.pause)
                # 503 is a transient server overload, not a quality metric.
                # A limited backoff avoids interpreting a transient failure as an
                # agent failure. Every attempted API call is durably recorded.
                retry_errors = []
                for retry_number in range(args.retries + 1):
                    answer = (await pydantic_run(case["question"]) if framework == "pydantic_ai"
                              else await adk_run(case["question"]))
                    answer["evaluation"] = evaluate_response(case, answer)
                    answer["retry_count"] = retry_number
                    answer["retry_errors"] = list(retry_errors)
                    # `seconds` measures only the *final* attempt, not the
                    # cumulative time across failed attempts and backoff.
                    # Runs with retries should not be naively used for
                    # framework latency comparisons without qualification.
                    record[framework] = answer
                    write_evidence(data)
                    if answer["ok"] or str(answer.get("http_status")) != "503" or retry_number >= args.retries:
                        break
                    retry_errors.append({
                        "error_type": answer.get("error_type"),
                        "http_status": answer.get("http_status"),
                        "error_message": answer.get("error_message"),
                        "seconds": answer.get("seconds"),
                    })
                    delay = min(120.0, args.retry_base * (2 ** retry_number))
                    print(f"  {framework}: HTTP 503 一時的混雑。{delay:g}秒後に再試行 "
                          f"({retry_number + 1}/{args.retries})", flush=True)
                    await asyncio.sleep(delay)
                attempted_this_pair += 1
                print(f"  {framework}: {'OK' if answer['ok'] else 'ERROR'}; "
                      f"tools={len(answer.get('tool_calls', []))}; 503_retries={answer['retry_count']}", flush=True)
                if not answer["ok"]:
                    reason = answer.get("error_message", "")
                    print(f"  [原因] {answer.get('error_type', 'unknown')} HTTP={answer.get('http_status')} {reason[:320]}", flush=True)
                    print("[停止] APIエラーを記録しました。原因を解消した後、--resume で再開してください。", flush=True)
                    interrupted = True
                    break
            if interrupted:
                break
            if args.pause:
                await asyncio.sleep(args.pause)
        if interrupted:
            break
    complete = len(data["records"]) == total and all(
        all(r[framework].get("ok") for framework in ("pydantic_ai", "google_adk"))
        for r in data["records"])
    print(f"[保存] {RESULT_PATH}")
    print(f"[比較レポート] {REPORT_PATH}")
    print(f"[人手評価] {REVIEW_PATH}（空欄は未評価）")
    if not complete:
        print("[未完了] 全ペア成功していません。原因を確認し、--resume を付けて再実行してください")
        return 1
    print("[OK] 予定した比較ペアがすべて成功しました")
    if args.publish:
        publish()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"[停止] {exc}", file=sys.stderr)
        raise SystemExit(1)
