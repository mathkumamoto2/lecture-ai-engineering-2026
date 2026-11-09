"""講座の Pydantic AI エージェントと同じ指示・ツールを ADK で構成する。

このファイルではデータ検索ロジックやプロンプトを再実装しない。
フレームワーク以外の条件を可能な限り揃えて比較するため。
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    name="glossary_agent",
    model=MODEL,
    instruction=INSTRUCTIONS,
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
    ],
)
