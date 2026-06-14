# flashmind-cli TODO

## MCP support

- Add `flashmind mcp add <server>` / `flashmind mcp remove` / `flashmind mcp list` subcommands
- Register MCP tools dynamically via `ToolBuilder::mcp()`
- OAuth flow: print auth URL to terminal, start local callback server, wait for redirect, exchange code for token
- Persist MCP server configs so they auto-reconnect on next launch
- Surface `mcp_add`, `mcp_remove`, `mcp_list`, `mcp_auth` as agent tools during chat

## Memory support

- Wire up `flashmind-memory` vector store (SQLite + sqlite-vec + FTS5)
- Configure embedding provider from config.toml (OpenAI/Ollama/OpenRouter)
- Inject memory search results into system prompt as developer messages
- Register `memory_store`, `memory_search`, `memory_forget` as agent tools
- Include `MEMORY_INSTRUCTIONS` prompt fragment in system prompt
- Store memories in `~/.flashmind/memory.db`
