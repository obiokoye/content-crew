"""Demo mode: a scripted LLM that plays all three agents with no API key.

``ScriptedLLM`` is a real CrewAI ``BaseLLM``, so CrewAI runs exactly as it
would with Gemini: the same prompts, the same ReAct tool loop, the same
task hand-offs and the same guardrail retry. Only the model's replies are
canned. The tests use this same class.

What is scripted and what is real:
- Researcher: the tool calls are scripted, but the notes are built from the
  passages the real ``search_sources`` tool returned.
- Writer: a pre-written draft for three demo topics (with one hype word left
  in on purpose), or a draft assembled from the notes for any other topic.
- Editor: the first attempt forgets one source in the Sources list, so you can
  watch the code guardrail reject it; the retry fixes it, removes the hype
  word and trims to the word limit.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import BaseLLM
from pydantic import PrivateAttr

from .guardrails import BANNED_WORDS, CITATION

NOTE_LINE = re.compile(r"^- ([^\[\n].*\. \[[a-z0-9-]+\])$", re.MULTILINE)
HIT_LINE = re.compile(r"^(?:Observation:\s*)?\[([a-z0-9-]+)\] (.+)$", re.MULTILINE)

DEMO_TOPICS: dict[str, dict[str, Any]] = {
    "mcp": {
        "match": ("mcp", "model context"),
        "queries": ["MCP protocol server client tools", "untrusted content tool arguments injection risk"],
        "draft": """# MCP: one plug for every AI tool

If you have ever written the same integration twice, once for each AI vendor, the Model Context Protocol is for you. MCP is an open protocol from Anthropic that standardises how AI applications connect to tools and data [mcp].

## What a server exposes

An MCP server offers three things: tools the model can call, read-only resources such as files or database rows, and reusable prompts [mcp]. A client inside the host app, such as an IDE or a chat app, connects to one server and passes those capabilities to the model [mcp].

## Why it is a game-changer

Write a server once and every MCP-compatible client can use it [mcp]. Under the hood it is still tool calling: the model returns a tool name and JSON arguments, and your code runs the function [tool-calling]. The model never runs code itself, so your app stays in charge [tool-calling].

## Keep it safe

A server can run code and read data, so ask the user before running sensitive tools and expose only the actions you need [mcp]. Treat tool arguments as untrusted input [tool-calling], and remember that an agent that reads untrusted content and can act on it is the riskiest combination [prompt-injection].

Start small: one read-only tool, one client, and grow from there.""",
    },
    "multi-agent": {
        "match": ("multi", "crew", "several agents", "team of agents"),
        "queries": ["multi agent researcher writer editor roles", "fake model client test guardrails offline evals"],
        "draft": """# When one AI agent is not enough

Most tasks only need one well-written prompt. Some jobs, though, are really three jobs in a trench coat. That is where a multi-agent system helps: it splits the work across agents that each have a narrow role, their own instructions and only the tools they need [multi-agent].

## The pipeline pattern

The classic setup is a pipeline. A researcher gathers facts, a writer drafts, and an editor checks the draft, and each agent gets the previous agent's output as context [multi-agent]. CrewAI describes each agent with a role, a goal and a backstory, and each piece of work as a task with an expected output [multi-agent].

## Why narrow roles help

Short, single-purpose prompts are easier to reason about and to test, because each agent has one job and one expected output [multi-agent]. With a fake model client you can test the control flow and guardrails offline for free [agent-evals].

## The revolutionary catch

More agents means more tokens and more latency, so only split the work when the steps need different skills [multi-agent]. And check every hand-off in code, for example that the editor cites real sources, so one agent's mistake does not flow to the next [multi-agent].

Then measure it: a fixed eval set shows whether a change made the crew better or worse [agent-evals].""",
    },
    "rag": {
        "match": ("rag", "retrieval"),
        "queries": ["RAG retrieval chunks citations", "indirect prompt injection retrieved content"],
        "draft": """# RAG in plain words

Retrieval-Augmented Generation answers questions from your own documents. Instead of hoping the model already knows your data, you fetch the relevant passages first and add them to the prompt [rag].

## The pipeline

Split documents into chunks, turn each chunk into a vector, and store the vectors. At question time, embed the question and fetch the closest chunks [rag]. If you build with Laravel, it is a lot like a search index that feeds a view, except the view is a prompt.

## What actually matters

Retrieval quality matters more than model size: bad chunks in, confident wrong answers out [rag]. Return citations with file and page so readers can check every claim [rag]. The usual failures are chunks that are too big, missing metadata, and questions that need several documents at once [rag].

## Why teams like it

Your private data stays out of model training, and you update knowledge by re-indexing documents instead of fine-tuning [rag]. That makes it a seamless fit for internal docs and support content.

## One warning

Retrieved text can carry hidden instructions, which is called indirect prompt injection [prompt-injection]. Treat everything you retrieve as data, never as commands [prompt-injection].""",
    },
}


# How the scripted editor fixes the hype words the scripted writer left in.
REWRITES = {
    "Why it is a game-changer": "Why it matters",
    "The revolutionary catch": "The catch",
    "a seamless fit": "a good fit",
}


def pick_script(topic: str) -> dict[str, Any] | None:
    lowered = topic.lower()
    for script in DEMO_TOPICS.values():
        if any(word in lowered for word in script["match"]):
            return script
    return None


class ScriptedLLM(BaseLLM):
    """Plays Researcher, Writer and Editor using CrewAI's text (ReAct) format."""

    _last_draft: str = PrivateAttr(default="")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="demo-scripted", **kwargs)

    def supports_function_calling(self) -> bool:
        return False  # use CrewAI's "Action: / Action Input:" text protocol

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        text = messages if isinstance(messages, str) else " ".join(str(m.get("content", "")) for m in messages)
        return len(text) // 4 + 1  # rough estimate; a scripted model has no tokenizer

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs) -> str:
        transcript = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages
        )
        role = getattr(from_agent, "role", "") or _guess_role(transcript)
        topic = _between(transcript, "topic '", "'") or _between(transcript, "about '", "'") or "AI agents"
        if role == "Researcher":
            return self._research(transcript, topic)
        if role == "Writer":
            self._last_draft = _draft_for(topic, transcript)
            return _final(self._last_draft)
        if role == "Editor":
            # A real editor reads the draft from its context; the scripted one remembers it.
            return _final(_edit(transcript, self._last_draft or _draft_for(topic, transcript)))
        return _final("Done.")

    # --- Researcher -----------------------------------------------------------------
    def _research(self, transcript: str, topic: str) -> str:
        script = pick_script(topic)
        queries = script["queries"] if script else [topic, f"{topic} risks"]
        searches_done = transcript.count("Action: search_sources")
        if searches_done < len(queries):
            query = queries[searches_done]
            thought = "I should search the source library." if searches_done == 0 else "I have the basics. Let me check the risks too."
            return f'Thought: {thought}\nAction: search_sources\nAction Input: {json.dumps({"query": query})}'
        return _final(_notes_from_hits(transcript))


def _notes_from_hits(transcript: str) -> str:
    facts: list[str] = []
    seen: set[str] = set()
    for source_id, paragraph in HIT_LINE.findall(transcript):
        sentence = re.split(r"(?<=\.)\s", paragraph.strip())[0].rstrip(".")
        if sentence not in seen:
            seen.add(sentence)
            facts.append(f"- {sentence}. [{source_id}]")
    return "\n".join(facts[:10]) or "- No facts found in the source library."


def _draft_for(topic: str, transcript: str) -> str:
    script = pick_script(topic)
    if script:
        return script["draft"]
    notes = list(dict.fromkeys(NOTE_LINE.findall(transcript)))
    body = "\n\n".join(" ".join(notes[i:i + 2]) for i in range(0, len(notes), 2))
    return f"# {topic.strip().capitalize()}: the short version\n\n{body or 'Nothing in the source library covers this yet.'}"


def _edit(transcript: str, draft: str) -> str:
    max_words = int(_between(transcript, "at most ", " words") or 300)
    retry = "failed review" in transcript

    edits: list[str] = []
    for phrase, better in REWRITES.items():
        if phrase in draft:
            draft = draft.replace(phrase, better)
            edits.append(f"Replaced '{phrase}' with '{better}' (hype word).")
    for word in BANNED_WORDS:  # anything left over is simply removed
        pattern = re.compile(rf"\b{re.escape(word)}\b ?", re.IGNORECASE)
        if pattern.search(draft):
            draft = pattern.sub("", draft)
            edits.append(f"Removed the hype word '{word}'.")

    title, _, body = draft.partition("\n")
    paragraphs = [p for p in body.strip().split("\n\n") if p.strip()]
    trimmed = 0
    while len(paragraphs) > 2 and _words("\n\n".join(paragraphs)) > max_words:
        paragraphs.pop(-2)  # keep the intro and the closing line, cut from the middle-end
        paragraphs = _drop_empty_headings(paragraphs)
        trimmed += 1
    if trimmed:
        edits.append(f"Cut {trimmed} paragraph(s) to fit the {max_words}-word limit.")
    edits.append("Checked every claim against the research notes.")
    body = "\n\n".join(paragraphs)

    cited = list(dict.fromkeys(CITATION.findall(body)))
    titles = dict(re.findall(r"^- \[([a-z0-9-]+)\] (.+)$", transcript, re.MULTILINE))
    listed = cited if retry else cited[:-1] or cited  # first attempt "forgets" the last source
    if retry:
        edits.append("Added the missing source to the Sources list.")
    sources = "\n".join(f"- [{c}] {titles.get(c, c)}" for c in listed)
    return f"{title}\n\n{body}\n\n## Sources\n\n{sources}\n\nEDITS:\n" + "\n".join(f"- {e}" for e in edits)


def _drop_empty_headings(paragraphs: list[str]) -> list[str]:
    kept = []
    for i, p in enumerate(paragraphs):
        nxt = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
        if p.startswith("## ") and (nxt is None or nxt.startswith("## ")):
            continue
        kept.append(p)
    return kept


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", CITATION.sub("", text)))


def _final(text: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


def _between(text: str, start: str, end: str) -> str | None:
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i + len(start))
    return text[i + len(start): j] if j >= 0 else None


def _guess_role(transcript: str) -> str:
    for role in ("Researcher", "Writer", "Editor"):
        if f"You are {role}" in transcript:
            return role
    return ""
