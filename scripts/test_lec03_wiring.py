"""ADKが未導入でも実行できる構造検証テスト。

実際のモデル呼び出し成功や Gemini API の挙動は保証しない。
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent.parent


def test_adk_agent_uses_shared_instruction_and_tools():
    path = ROOT / "adk_agent/glossary_agent/agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.ImportFrom) and node.module == "app.agent" and
               any(alias.name == "INSTRUCTIONS" for alias in node.names) for node in tree.body)
    values = [node.value for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "root_agent" for target in node.targets)]
    assert len(values) == 1
    call = values[0]
    assert isinstance(call, ast.Call)
    kw = {item.arg: item.value for item in call.keywords}
    assert isinstance(kw.get("instruction"), ast.Name) and kw["instruction"].id == "INSTRUCTIONS"
    assert isinstance(kw.get("tools"), ast.List)
    names = {node.attr for node in kw["tools"].elts if isinstance(node, ast.Attribute)}
    assert names == {"search_glossary", "get_lecture_summary", "make_quiz"}
