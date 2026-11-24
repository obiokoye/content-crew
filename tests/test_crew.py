"""End-to-end crew runs with the scripted LLM: real CrewAI, no network."""

import pytest
from crewai import BaseLLM

from content_crew.crew import Brief, run_crew
from content_crew.demo import ScriptedLLM
from content_crew.guardrails import check_article


def run(library, topic, words=300):
    tools, tasks, rejects = [], [], []
    result = run_crew(
        ScriptedLLM(), library, Brief(topic, "Laravel developers", words),
        on_tool=lambda name, args, out: tools.append((name, args, out)),
        on_task=lambda agent, out: tasks.append(agent.strip()),
        on_reject=lambda attempt, problems: rejects.append(problems),
    )
    return result, tools, tasks, rejects


def test_pipeline_runs_researcher_writer_editor_in_order(library):
    result, tools, tasks, _ = run(library, "How MCP lets AI agents use tools")
    assert tasks == ["Researcher", "Writer", "Editor"]
    assert [t[0] for t in tools] == ["search_sources", "search_sources"]
    assert "[mcp]" in tools[0][2]  # the real tool returned real passages
    assert result.research.count("[mcp]") >= 3  # notes are built from those passages


def test_guardrail_rejects_first_edit_then_accepts_the_fix(library):
    result, _, _, rejects = run(library, "How MCP lets AI agents use tools")
    assert rejects == [["Sources section is missing: prompt-injection."]]
    article = result.article
    assert check_article(article, set(library.ids()), 300) == []
    assert "game-changer" not in article.body
    assert set(article.citations()) == {"mcp", "tool-calling", "prompt-injection"}


@pytest.mark.parametrize("topic", ["When to use a crew of multiple AI agents", "RAG explained"])
def test_other_demo_topics_produce_valid_articles(library, topic):
    result, *_ = run(library, topic)
    assert check_article(result.article, set(library.ids()), 300) == []


def test_editor_trims_to_the_word_limit(library):
    result, *_ = run(library, "RAG explained", words=120)
    assert result.article.word_count <= 120 * 1.15
    assert any("word limit" in e for e in result.article.edits)


def test_unscripted_topic_builds_article_from_research(library):
    result, *_ = run(library, "prompt injection defences")
    assert "[prompt-injection]" in result.article.body
    assert check_article(result.article, set(library.ids()), 300) == []


class AlwaysBadEditor(ScriptedLLM):
    def call(self, messages, *args, from_agent=None, **kwargs):
        if getattr(from_agent, "role", "") == "Editor":
            return "Thought: done\nFinal Answer: This is a revolutionary article with no sources."
        return super().call(messages, *args, from_agent=from_agent, **kwargs)


def test_crew_fails_loudly_when_editor_never_passes(library):
    assert issubclass(AlwaysBadEditor, BaseLLM)
    with pytest.raises(Exception, match="guardrail"):
        run_crew(AlwaysBadEditor(), library, Brief("MCP"))
