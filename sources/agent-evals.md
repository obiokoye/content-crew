# Evaluating agents

Agents are non-deterministic, so they need evaluation suites, not just unit tests.

A typical eval runs a fixed set of inputs through the agent and scores each output, either with code checks (exact match, JSON validity, required fields) or with an LLM acting as a judge against a rubric.

Code checks are cheap and reliable; LLM judges handle open-ended quality such as tone or completeness.

Unit tests still matter: with a fake model client, the agent's control flow, tool dispatch and guardrails can be tested offline for free.

Tracking evals over time shows whether a prompt change or a model upgrade made things better or worse.
