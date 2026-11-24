from content_crew.guardrails import check_article, parse_editor_output

KNOWN = {"mcp", "rag"}

GOOD = """# MCP in short

MCP connects AI apps to tools [mcp]. RAG adds your documents to the prompt [rag].

## Sources

- [mcp] Model Context Protocol (MCP)
- [rag] Retrieval-Augmented Generation (RAG)

EDITS:
- Cut the intro.
"""


def test_parses_title_body_sources_and_edits():
    article = parse_editor_output(GOOD)
    assert article.title == "MCP in short"
    assert article.citations() == ["mcp", "rag"]
    assert len(article.sources) == 2
    assert article.edits == ["Cut the intro."]
    assert "EDITS" not in article.to_markdown()


def test_good_article_passes():
    assert check_article(parse_editor_output(GOOD), KNOWN, max_words=100) == []


def test_rejects_unknown_citation_and_missing_source():
    text = GOOD.replace("[rag].", "[wikipedia].").replace("- [rag] Retrieval-Augmented Generation (RAG)\n", "")
    problems = check_article(parse_editor_output(text), KNOWN, 100)
    assert any("Unknown source ids cited: wikipedia" in p for p in problems)

    text = GOOD.replace("- [rag] Retrieval-Augmented Generation (RAG)\n", "")
    assert check_article(parse_editor_output(text), KNOWN, 100) == ["Sources section is missing: rag."]


def test_rejects_hype_words_length_and_missing_parts():
    hype = GOOD.replace("MCP connects", "MCP is a game-changer that connects")
    assert any("game-changer" in p for p in check_article(parse_editor_output(hype), KNOWN, 100))

    assert any(p.startswith("Too long") for p in check_article(parse_editor_output(GOOD), KNOWN, max_words=5))

    problems = check_article(parse_editor_output("Just some text with no citations."), KNOWN, 100)
    assert len(problems) >= 4  # no title, no citation, no sources, no edits
