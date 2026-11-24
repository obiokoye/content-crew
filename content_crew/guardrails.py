"""Code checks on the Editor's output.

The editor is an LLM, so its output is checked in plain Python before the
crew accepts it. If a check fails, CrewAI sends the list of problems back to
the editor and asks it to try again (up to ``guardrail_max_retries`` times).
This is the same idea as a Laravel Form Request: never trust the input,
even when the "user" is a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
EDITS_MARKER = re.compile(r"^\s*EDITS:\s*$", re.MULTILINE)
SOURCES_HEADING = re.compile(r"^##\s+Sources\s*$", re.MULTILINE | re.IGNORECASE)

# Words the house style bans. Plain, specific writing beats hype.
BANNED_WORDS = (
    "revolutionary", "game-changer", "game-changing", "groundbreaking",
    "cutting-edge", "seamless", "seamlessly", "unleash", "supercharge", "delve",
)


@dataclass
class Article:
    title: str
    body: str
    sources: list[str]
    edits: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\b[\w'-]+\b", CITATION.sub("", self.body)))

    def citations(self) -> list[str]:
        """Source ids cited in the body, in first-seen order."""
        seen: dict[str, None] = {}
        for match in CITATION.findall(self.body):
            seen.setdefault(match, None)
        return list(seen)

    def to_markdown(self) -> str:
        sources = "\n".join(f"- {line}" for line in self.sources)
        return f"# {self.title}\n\n{self.body.strip()}\n\n## Sources\n\n{sources}\n"


def parse_editor_output(text: str) -> Article:
    """Split '# Title / body / ## Sources / EDITS:' into an Article.

    Missing parts come back empty, and ``check_article`` reports them.
    """
    text = text.strip().strip("`").strip()
    edits: list[str] = []
    marker = EDITS_MARKER.search(text)
    if marker:
        edits = _bullets(text[marker.end():])
        text = text[: marker.start()].rstrip()

    sources: list[str] = []
    heading = SOURCES_HEADING.search(text)
    if heading:
        sources = _bullets(text[heading.end():])
        text = text[: heading.start()].rstrip()

    title = ""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
    return Article(title=title, body="\n".join(lines).strip(), sources=sources, edits=edits)


def check_article(article: Article, known_sources: set[str], max_words: int) -> list[str]:
    """Return a list of problems. An empty list means the article passes."""
    problems: list[str] = []
    if not article.title:
        problems.append("Start with a '# Title' line.")
    if not article.body:
        problems.append("The article body is empty.")

    limit = int(max_words * 1.15)
    if article.word_count > limit:
        problems.append(f"Too long: {article.word_count} words, limit is {limit}. Cut it down.")

    cited = article.citations()
    if not cited:
        problems.append("Cite at least one source inline, like [mcp].")
    unknown = [c for c in cited if c not in known_sources]
    if unknown:
        problems.append(f"Unknown source ids cited: {', '.join(unknown)}. Only cite: {', '.join(sorted(known_sources))}.")

    if not article.sources:
        problems.append("Add a '## Sources' section listing every cited source.")
    else:
        listed = {m for line in article.sources for m in CITATION.findall(line)}
        missing = [c for c in cited if c in known_sources and c not in listed]
        if missing:
            problems.append(f"Sources section is missing: {', '.join(missing)}.")

    lowered = f"{article.title} {article.body}".lower()
    hype = [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]
    if hype:
        problems.append(f"Remove banned hype words: {', '.join(hype)}.")

    if not article.edits:
        problems.append("End with an 'EDITS:' line followed by a bullet list of the changes you made.")
    return problems


def _bullets(block: str) -> list[str]:
    items = []
    for line in block.strip().splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            items.append(line[2:].strip())
    return items
