# Tool calling

Tool calling (also called function calling) lets a language model request that the application run a function, described to the model as a JSON Schema.

The model never runs code itself. It returns a tool name and JSON arguments; the application validates them, runs the function and sends the result back.

This loop of model call, tool run and model call again is the core of every AI agent. A step limit stops runaway loops.

Tool arguments are untrusted input. They should be validated exactly like form input from a user.

Read-only tools are low risk. Tools that send email, spend money or delete data should require human approval.
