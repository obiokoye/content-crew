"""The crew: three agents with narrow roles, run as a sequential pipeline.

    Researcher --notes--> Writer --draft--> Editor --(code guardrail)--> article

Each Task names one agent, says what to do, and says what a good result
looks like (``expected_output``). ``context=[...]`` hands earlier task
outputs to later agents, like passing data between chained queue jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task
from crewai.tasks.task_output import TaskOutput

from .guardrails import BANNED_WORDS, Article, check_article, parse_editor_output
from .knowledge import SourceLibrary
from .tools import ReadSourceTool, SearchSourcesTool, ToolHook

TaskHook = Callable[[str, str], None]
GuardrailHook = Callable[[int, list[str]], None]


@dataclass
class Brief:
    topic: str
    audience: str = "software developers"
    max_words: int = 300


@dataclass
class CrewResult:
    brief: Brief
    research: str
    draft: str
    article: Article
    guardrail_rejections: list[list[str]] = field(default_factory=list)


def build_agents(llm: Any, library: SourceLibrary, on_tool: ToolHook | None = None) -> dict[str, Agent]:
    common = dict(llm=llm, allow_delegation=False, verbose=False, max_iter=6)
    return {
        "researcher": Agent(
            role="Researcher",
            goal="Find accurate facts about {topic} in the source library, each tagged with its source id.",
            backstory=(
                "You are a careful technical researcher. You only trust the source library, "
                "you never invent facts, and you note where every fact came from."
            ),
            tools=[SearchSourcesTool(library, on_tool), ReadSourceTool(library, on_tool)],
            **common,
        ),
        "writer": Agent(
            role="Writer",
            goal="Turn research notes into a clear, friendly article for {audience}.",
            backstory=(
                "You write plain, specific explanations with short paragraphs and concrete examples. "
                "You cite sources inline exactly as the notes do, like [mcp]."
            ),
            **common,
        ),
        "editor": Agent(
            role="Editor",
            goal="Make the draft accurate, tight and on-style, and deliver it in the house format.",
            backstory=(
                "You are a strict editor. You check every claim against the research notes, "
                "cut hype and filler, and keep to the word limit."
            ),
            **common,
        ),
    }


def build_tasks(agents: dict[str, Agent], library: SourceLibrary, brief: Brief, guardrail: Callable) -> list[Task]:
    sources = "\n".join(f"- [{s.id}] {s.title}" for s in library.sources.values())
    research = Task(
        description=(
            "Research the topic '{topic}' for an article aimed at {audience}. "
            "Use search_sources (and read_source for a promising source). "
            "Collect 5 to 10 key facts. Use only facts returned by the tools."
        ),
        expected_output="A bullet list of 5-10 facts, each ending with its source id, e.g. '- MCP servers expose tools. [mcp]'",
        agent=agents["researcher"],
    )
    write = Task(
        description=(
            "Write an article draft about '{topic}' for {audience}, about {max_words} words. "
            "Use only the research notes. Keep the inline source ids, like [mcp], after each claim. "
            "Start with a '# Title' line."
        ),
        expected_output="A Markdown article draft with a '# Title' line and inline [source-id] citations.",
        agent=agents["writer"],
        context=[research],
    )
    edit = Task(
        description=(
            "Edit the draft about '{topic}'. Check each claim against the research notes and remove anything unsupported. "
            "Keep it at most {max_words} words. Remove hype words such as: " + ", ".join(BANNED_WORDS[:6]) + ". "
            "Only these source ids exist:\n" + sources + "\n\n"
            "Return exactly this format:\n"
            "# Title\n\n<article body with inline [source-id] citations>\n\n## Sources\n\n- [id] Source title\n\n"
            "EDITS:\n- <one line per change you made>"
        ),
        expected_output="The final article in the exact format above, ending with an EDITS: list.",
        agent=agents["editor"],
        context=[research, write],
        guardrail=guardrail,
        guardrail_max_retries=2,
    )
    return [research, write, edit]


def make_guardrail(library: SourceLibrary, brief: Brief, rejections: list[list[str]], on_reject: GuardrailHook | None):
    known = set(library.ids())

    def guardrail(output: TaskOutput):  # returns (passed, result_or_feedback); CrewAI rejects string annotations here
        problems = check_article(parse_editor_output(output.raw), known, brief.max_words)
        if problems:
            rejections.append(problems)
            if on_reject:
                on_reject(len(rejections), problems)
            return False, "The article failed review. Fix these problems and return the full article again:\n- " + "\n- ".join(problems)
        return True, output.raw

    return guardrail


def run_crew(
    llm: Any,
    library: SourceLibrary,
    brief: Brief,
    on_tool: ToolHook | None = None,
    on_task: TaskHook | None = None,
    on_reject: GuardrailHook | None = None,
) -> CrewResult:
    rejections: list[list[str]] = []
    agents = build_agents(llm, library, on_tool)
    tasks = build_tasks(agents, library, brief, make_guardrail(library, brief, rejections, on_reject))

    def task_callback(output: TaskOutput) -> None:
        if on_task:
            on_task(output.agent, output.raw)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        task_callback=task_callback,
        verbose=False,
    )
    result = crew.kickoff(inputs={"topic": brief.topic, "audience": brief.audience, "max_words": brief.max_words})
    research, draft, final = (t.raw for t in result.tasks_output)
    return CrewResult(brief, research, draft, parse_editor_output(final), rejections)
