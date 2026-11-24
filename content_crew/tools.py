"""CrewAI tools for the Researcher agent.

Both tools are read-only and only see the local ``sources/`` library, so the
researcher can't browse the web or touch the file system. Every call is
reported to an optional ``on_call`` hook, which the terminal UI uses to show
tool calls live and the tests use to assert on them.
"""

from __future__ import annotations

from typing import Callable

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from .knowledge import SourceLibrary, format_hits

ToolHook = Callable[[str, dict, str], None]


class SearchArgs(BaseModel):
    query: str = Field(..., description="Keywords to look for, e.g. 'MCP servers tools'")


class ReadArgs(BaseModel):
    source_id: str = Field(..., description="A source id from search results, e.g. 'mcp'")


class _LibraryTool(BaseTool):
    _library: SourceLibrary = PrivateAttr()
    _on_call: ToolHook | None = PrivateAttr(default=None)

    def __init__(self, library: SourceLibrary, on_call: ToolHook | None = None, **data):
        super().__init__(**data)
        self._library = library
        self._on_call = on_call

    def _report(self, args: dict, result: str) -> str:
        if self._on_call:
            self._on_call(self.name, args, result)
        return result


class SearchSourcesTool(_LibraryTool):
    name: str = "search_sources"
    description: str = (
        "Search the trusted source library. Returns matching passages, each "
        "prefixed with its source id in square brackets, like [mcp]."
    )
    args_schema: type[BaseModel] = SearchArgs

    def _run(self, query: str) -> str:
        return self._report({"query": query}, format_hits(self._library.search(query)))


class ReadSourceTool(_LibraryTool):
    name: str = "read_source"
    description: str = "Read one whole source by its id, e.g. 'rag'."
    args_schema: type[BaseModel] = ReadArgs

    def _run(self, source_id: str) -> str:
        source = self._library.get(source_id)
        if source is None:
            result = f"Unknown source '{source_id}'. Known ids: {', '.join(self._library.ids())}"
        else:
            result = f"[{source.id}] {source.title}\n\n{source.text}"
        return self._report({"source_id": source_id}, result)
