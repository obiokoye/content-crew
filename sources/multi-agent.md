# Multi-agent systems

A multi-agent system splits one job across several agents, each with a narrow role, its own instructions and only the tools it needs.

A common pattern is a pipeline: a researcher gathers facts, a writer drafts, and an editor checks the draft. Each agent receives the previous agent's output as context.

Narrow roles make prompts shorter and behaviour easier to test, because each agent has one job and one expected output.

Frameworks such as CrewAI describe agents with a role, a goal and a backstory, and describe work as tasks with an expected output.

Multi-agent setups cost more tokens and add latency, so they pay off when the steps really are different skills, not when one prompt would do.

Validating each hand-off, for example checking that the editor's output cites real sources, stops one agent's mistake from flowing to the next.
