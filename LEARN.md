# LEARN: Day 15 for a Laravel developer

Day 1 had one agent in a loop. Today there are three agents, each with one job, handing work to the next. This guide maps the pieces to Laravel and explains the one idea to take into an interview: **split the job into narrow roles, and check every hand-off in code.**

## 1. Project setup

| Python / CrewAI                         | Laravel equivalent                         |
|-----------------------------------------|--------------------------------------------|
| `pip install -r requirements.txt`       | `composer install`                         |
| `.env` + `python-dotenv`                | `.env` + `env()`                           |
| `content_crew/` package                 | the `App\` namespace                       |
| `pytest`                                | `php artisan test`                         |
| `crewai.Agent`                          | a service class with one responsibility    |
| `crewai.Task`                           | a queued job: what to do + what "done" means |
| `Crew(process=Process.sequential)`      | `Bus::chain([...])`                        |
| `context=[research, write]`             | passing one job's result into the next     |
| Task `guardrail`                        | a Form Request that validates the result   |
| `ScriptedLLM` in tests                  | `Http::fake()`                             |

## 2. The crew, step by step (`content_crew/crew.py`)

```python
Agent(role="Researcher", goal="Find accurate facts about {topic}...", backstory="...", tools=[...])
```
An **Agent** is a prompt with a job title. `role`, `goal` and `backstory` become the system prompt. Only the Researcher gets tools, so the Writer and Editor can't search or call anything: least privilege, like giving each service only the dependencies it needs.

```python
Task(description="Research '{topic}'...", expected_output="A bullet list of 5-10 facts...", agent=researcher)
```
A **Task** says what to do and what a good answer looks like. `{topic}` placeholders are filled from `crew.kickoff(inputs={...})`, much like route parameters.

```python
Task(..., agent=writer, context=[research])
Task(..., agent=editor, context=[research, write], guardrail=guardrail, guardrail_max_retries=2)
```
`context` is the hand-off. The Writer sees the notes; the Editor sees the notes **and** the draft, so it can check claims against the facts.

The flow is `Bus::chain([new Research, new Write, new Edit])`, except each job is an LLM call.

## 3. Tools (`content_crew/tools.py`)

`SearchSourcesTool` is a `BaseTool` with a Pydantic `args_schema`. CrewAI turns that schema into the tool description the model reads. Pydantic models are the Python cousin of Form Request rules: they declare and validate the shape of input.

In demo mode the Researcher uses CrewAI's text (ReAct) format. With a real provider CrewAI can use the model's native function calling instead, but the loop is the same:

```text
Thought: I should search the source library.
Action: search_sources
Action Input: {"query": "MCP protocol server client tools"}
Observation: [mcp] An MCP client lives inside the host application...
```

CrewAI parses `Action`, runs the tool, appends the `Observation`, and asks the model again until it writes `Final Answer:`. It's the Day 1 loop, managed by the framework.

## 4. The key concept: guardrails on agent output (`content_crew/guardrails.py`)

An LLM editor will sometimes cite a source that doesn't exist, forget the Sources list, or run long. Asking nicely in the prompt helps, but it isn't a guarantee. So the Editor's output goes through plain Python:

```python
problems = check_article(parse_editor_output(output.raw), known_sources, max_words)
if problems:
    return False, "The article failed review. Fix these problems...\n- " + "\n- ".join(problems)
return True, output.raw
```

Returning `False` makes CrewAI send the feedback back to the Editor and try again (max 2 retries); after that the crew raises an error instead of saving a bad article. Think of it as `$request->validate()` with a retry: the model is the user, and you never trust user input.

Run demo mode and you'll see it happen: the scripted Editor forgets `[prompt-injection]` in the Sources list, the guardrail rejects it, and the retry fixes it.

## 5. Security note

- The Researcher's tools are **read-only** and only see `sources/`. There's no web, shell or file-write access, so a bad prompt can't make an agent do damage.
- Source files are **data, not instructions**. If you add a web-search tool later, a page could contain text like "ignore your instructions". Keep tools least-privilege and keep the code guardrail: it checks citations against a fixed list of known ids, whatever the model says.
- API keys live in `.env`, which is git-ignored. Demo mode needs no key at all.

## 6. Testing without an API key

`tests/test_crew.py` runs the real CrewAI pipeline with `ScriptedLLM`, a `BaseLLM` subclass that returns canned ReAct replies. The tests check the agent order, the tool calls, the guardrail rejection and retry, the word-limit trim, and that a crew whose Editor never passes fails loudly. It's `Http::fake()` for an LLM: test your orchestration, not someone else's API.

## Try this

1. Add a rule to `check_article()` that the title must be under 60 characters, then write a test for it in `tests/test_guardrails.py`.
2. Add a fourth agent, "SEO", that runs after the Editor and writes a 155-character meta description. Give its task `context=[edit]`.
3. Put a few of your own Laravel notes in `sources/` (for example `queues.md`) and run `python main.py "Laravel queues for beginners"` with a real Gemini key. Read the article and check every `[id]` against your notes.
