# Model Context Protocol (MCP)

MCP is an open protocol, introduced by Anthropic in November 2024, that standardises how AI applications connect to external tools and data.

An MCP server exposes three kinds of capability: tools (functions the model can call), resources (read-only data such as files or database rows) and prompts (reusable templates).

An MCP client lives inside the host application, such as an IDE, a desktop chat app or a custom agent. One client connects to one server.

Servers talk to clients over JSON-RPC 2.0. Local servers usually run over stdio; remote servers use Streamable HTTP.

The main benefit is reuse: a server written once works with every MCP-compatible client, instead of writing a separate integration for each model vendor.

Because a server can run code and read data, clients should ask the user before running sensitive tools, and servers should expose the smallest set of actions they need.
