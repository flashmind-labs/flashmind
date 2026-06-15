# flashmind-cli TODO

## MCP support

- [x] Add `flashmind mcp add <server>` / `flashmind mcp remove` / `flashmind mcp list` subcommands
- [x] Register MCP tools dynamically via `ToolBuilder::mcp()`
- [x] OAuth flow: handled by `mcp_auth` tool via `McpAuthHandler` interrupt flow
- [x] Persist MCP server configs so they auto-reconnect on next launch
- [x] Surface `mcp_add`, `mcp_remove`, `mcp_list`, `mcp_auth` as agent tools during chat

## Memory support

- Wire up `flashmind-memory` vector store (SQLite + sqlite-vec + FTS5)
- Configure embedding provider from config.toml (OpenAI/Ollama/OpenRouter)
- Inject memory search results into system prompt as developer messages
- Register `memory_store`, `memory_search`, `memory_forget` as agent tools
- Include `MEMORY_INSTRUCTIONS` prompt fragment in system prompt
- Store memories in `~/.flashmind/memory.db`
