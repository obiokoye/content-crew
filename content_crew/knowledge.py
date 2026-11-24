"""A tiny local source library the Researcher agent searches.

Each Markdown file in ``sources/`` is one source. Its id is the file name
without ``.md`` (``sources/mcp.md`` -> ``mcp``), and the writer cites it as
``[mcp]``. Search is plain keyword scoring over paragraphs: no embeddings,
no network, fully deterministic, so it is easy to test and to explain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "why", "with", "you", "your", "about", "should", "explain", "into",
}


def tokenize(text: str) -> list[str]:
    """Lower-case words, minus stopwords and very short tokens."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass(frozen=True)
class Hit:
    source_id: str
    title: str
    paragraph: str
    score: float


@dataclass
class SourceLibrary:
    sources: dict[str, Source] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, directory: Path | str = DEFAULT_SOURCES_DIR) -> "SourceLibrary":
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Sources folder not found: {directory}")
        library = cls()
        for path in sorted(directory.glob("*.md")):
            library.add(path.stem, path.read_text(encoding="utf-8"))
        return library

    def add(self, source_id: str, markdown: str) -> Source:
        lines = markdown.strip().splitlines()
        title = source_id
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            lines = lines[1:]
        paragraphs = tuple(p.strip() for p in "\n".join(lines).split("\n\n") if p.strip())
        source = Source(source_id, title, paragraphs)
        self.sources[source_id] = source
        return source

    def ids(self) -> list[str]:
        return list(self.sources)

    def get(self, source_id: str) -> Source | None:
        return self.sources.get(source_id.strip().lower().removesuffix(".md"))

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        """Score each paragraph by query-word overlap; title matches count double."""
        terms = set(tokenize(query))
        if not terms:
            return []
        hits: list[Hit] = []
        for source in self.sources.values():
            title_terms = set(tokenize(source.title + " " + source.id.replace("-", " ")))
            title_bonus = 2.0 * len(terms & title_terms)
            for paragraph in source.paragraphs:
                words = tokenize(paragraph)
                overlap = sum(1 for w in words if w in terms)
                if overlap or title_bonus:
                    score = overlap + title_bonus
                    hits.append(Hit(source.id, source.title, paragraph, score))
        hits.sort(key=lambda h: (-h.score, h.source_id))
        return hits[:limit]


def format_hits(hits: list[Hit]) -> str:
    """Render hits as text the model reads as a tool observation."""
    if not hits:
        return "No matching passages. Try different keywords."
    return "\n".join(f"[{h.source_id}] {h.paragraph}" for h in hits)
