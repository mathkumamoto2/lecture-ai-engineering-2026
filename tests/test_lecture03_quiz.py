"""第3回で追加するmake_quizの決定的な単体テスト。"""
import json

from app import tools


def test_quiz_uses_lecture_2_glossary_definition():
    payload = json.loads(tools.make_quiz(2))
    assert set(payload) == {"question", "answer"}
    assert any(
        term["definition"] == payload["answer"] and term["term"] in payload["question"]
        for term in tools.load_glossary()
        if term["lecture"] == 2
    )


def test_quiz_is_repeatable_for_framework_comparison():
    assert tools.make_quiz(2) == tools.make_quiz(2)


def test_quiz_nonexistent_lecture_is_not_an_exception():
    assert "該当する用語がありません" in tools.make_quiz(0)
