# Prompt injection

Prompt injection is when text the model reads, such as a web page, an email or a document, contains instructions that try to override the developer's instructions.

Direct injection comes from the user. Indirect injection hides inside content the agent retrieves, which makes it harder to spot.

No prompt wording fully prevents injection. Defences are layered: keep tools least-privilege, treat retrieved text as data, require approval for risky actions, and validate outputs in code.

An agent that can both read untrusted content and take actions, such as sending messages, is the highest-risk combination.
