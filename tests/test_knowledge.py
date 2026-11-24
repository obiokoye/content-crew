from content_crew.knowledge import SourceLibrary, format_hits, tokenize


def test_loads_every_markdown_file_with_titles(library):
    assert set(library.ids()) == {"agent-evals", "mcp", "multi-agent", "prompt-injection", "rag", "tool-calling"}
    assert library.get("mcp").title == "Model Context Protocol (MCP)"
    assert library.get("RAG.md") is library.get("rag")


def test_search_ranks_the_matching_source_first(library):
    hits = library.search("MCP server tools")
    assert hits and hits[0].source_id == "mcp"
    assert len(hits) <= 5


def test_search_ignores_stopwords_and_handles_no_match(library):
    assert tokenize("What is the RAG?") == ["rag"]
    assert library.search("the and of") == []
    assert format_hits([]).startswith("No matching passages")


def test_format_hits_prefixes_source_ids():
    lib = SourceLibrary()
    lib.add("laravel", "# Laravel\n\nQueues run jobs in the background.\n\nHorizon watches queues.")
    text = format_hits(lib.search("queues"))
    assert text.splitlines()[0].startswith("[laravel] ")
