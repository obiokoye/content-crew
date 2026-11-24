"""Terminal UI for the content crew.

    python main.py "How MCP lets AI agents use tools"
    python main.py "RAG explained" --audience "Laravel developers" --words 250
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from content_crew.config import load_settings, make_llm  # noqa: E402  (load .env first)
from content_crew.crew import Brief, run_crew  # noqa: E402
from content_crew.knowledge import DEFAULT_SOURCES_DIR, SourceLibrary  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

warnings.filterwarnings("ignore", message="A function callback cannot be saved in a checkpoint")
logging.getLogger("crewai").setLevel(logging.ERROR)

console = Console(highlight=False)

AGENTS = [
    ("Researcher", "🔎", "cyan", "searches the source library"),
    ("Writer", "📝", "magenta", "drafts the article"),
    ("Editor", "🧐", "yellow", "fact-checks, trims, formats"),
]
COLORS = {name: color for name, _, color, _ in AGENTS}
ICONS = {name: icon for name, icon, _, _ in AGENTS}
DEMO_TOPICS = [
    "How MCP lets AI agents use tools",
    "When to use a crew of multiple AI agents",
    "RAG explained for Laravel developers",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "article"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Researcher, Writer and Editor agents write a sourced article.")
    parser.add_argument("topic", nargs="*", help="what the article is about")
    parser.add_argument("--audience", default="software developers", help="who the article is for")
    parser.add_argument("--words", type=int, default=300, help="word limit for the final article")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_DIR), help="folder of .md source notes")
    parser.add_argument("--out", default="output", help="where to save the article")
    return parser.parse_args(argv)


def header(settings, library: SourceLibrary) -> None:
    pipeline = Text()
    for i, (name, icon, color, job) in enumerate(AGENTS):
        if i:
            pipeline.append("  →  ", style="dim")
        pipeline.append(f"{icon} {name}", style=f"bold {color}")
    console.print(Panel(pipeline, title="[bold]📰 Content Crew[/]", subtitle=f"[dim]{settings.label}[/]", expand=False))
    if settings.demo:
        console.print(f"[bold yellow]Demo mode: scripted AI replies[/] [dim]({settings.reason}; add a key to .env for real output)[/]")
    console.print(f"[dim]Source library: {len(library.sources)} notes ({', '.join(library.ids())})[/]\n")


def pick_topic(args: argparse.Namespace) -> str:
    if args.topic:
        return " ".join(args.topic)
    console.print("[bold]Try one of these, or type your own:[/]")
    for i, topic in enumerate(DEMO_TOPICS, 1):
        console.print(f"  [cyan]{i}[/] {topic}")
    answer = console.input("\n[bold green]topic ›[/] ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(DEMO_TOPICS):
        return DEMO_TOPICS[int(answer) - 1]
    return answer or DEMO_TOPICS[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = load_settings()
    library = SourceLibrary.from_dir(args.sources)
    header(settings, library)
    brief = Brief(pick_topic(args), args.audience, args.words)
    console.print(f"[bold]Brief:[/] {brief.topic}  [dim]· for {brief.audience} · ≤ {brief.max_words} words[/]\n")

    order = [name for name, *_ in AGENTS]
    status = console.status(f"[cyan]{ICONS['Researcher']} Researcher is searching…[/]", spinner="dots")

    def on_tool(name: str, tool_args: dict, result: str) -> None:
        arg = next(iter(tool_args.values()), "")
        found = result.count("\n[") + 1 if result.startswith("[") else 0
        console.print(f"  [cyan]🔧 {name}[/]([white]\"{arg}\"[/]) [dim]→ {found} passage(s)[/]")

    def on_task(agent: str, output: str) -> None:
        agent = agent.strip()
        color = COLORS.get(agent, "white")
        if agent == "Researcher":
            body = Text(output.strip(), no_wrap=True, overflow="ellipsis")
        elif agent == "Writer":
            lines = output.strip().splitlines()
            preview = "\n".join(lines[:6]) + (f"\n… ({len(output.split())} words)" if len(lines) > 6 else "")
            body = Text(preview)
        else:
            body = Text("Approved by the guardrail ✓", style="green")
        console.print(Panel(body, title=f"[bold {color}]{ICONS.get(agent, '')} {agent}[/]", border_style=color, title_align="left"))
        nxt = order.index(agent) + 1 if agent in order else len(order)
        if nxt < len(order):
            name = order[nxt]
            status.update(f"[{COLORS[name]}]{ICONS[name]} {name} is working…[/]")

    def on_reject(attempt: int, problems: list[str]) -> None:
        console.print(f"  [bold red]🛑 Guardrail rejected the editor's attempt {attempt}:[/]")
        for problem in problems:
            console.print(f"     [red]• {problem}[/]")
        console.print("     [dim]→ sent back to the Editor with this feedback[/]")

    started = time.perf_counter()
    try:
        with status:
            result = run_crew(make_llm(settings), library, brief, on_tool, on_task, on_reject)
    except Exception as exc:  # network errors, bad keys, guardrail giving up
        console.print(f"\n[bold red]The crew stopped:[/] {exc}")
        return 1
    elapsed = time.perf_counter() - started

    article = result.article
    retries = len(result.guardrail_rejections)
    console.print(Rule("[bold]Final article[/]"))
    console.print(Panel(Markdown(article.to_markdown()), border_style="green", padding=(1, 2)))

    edits = Table.grid(padding=(0, 1))
    for edit in article.edits:
        edits.add_row("[yellow]✎[/]", edit)
    console.print(Panel(edits, title="[bold yellow]Editor's changes[/]", border_style="yellow", title_align="left"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(brief.topic)}.md"
    path.write_text(article.to_markdown(), encoding="utf-8")
    console.print(
        f"[green]✓[/] {article.word_count} words · {len(article.citations())} sources cited · "
        f"{retries} guardrail {'retry' if retries == 1 else 'retries'} · {elapsed:.1f}s · saved to [bold]{path.as_posix()}[/]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
