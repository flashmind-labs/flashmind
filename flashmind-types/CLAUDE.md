# flashmind-types

Shared traits, wire types, and events for the Flashmind framework. This is the leaf crate that all other crates depend on — it has no internal workspace dependencies.

## Key Traits

- `LlmProvider` (`llm.rs`) — streaming completions interface. Only `complete()` is required.
- `Tool` (`tool.rs`) — tool definition + execution. Implement `name()`, `description()`, `parameters()`, `execute()`, `humanize()`.
- `MemoryProvider` (`memory.rs`) — vector memory store/search/forget interface.

## Key Types

- `AgentEvent` (`event.rs`) — streaming events: `TextDelta`, `ToolStart`, `ToolResult`, `Usage`, `Done`, `Error`, `SubagentEvent`, etc.
- `AgentInput` (`event.rs`) — `User { content, context, parts }` or `Resume`
- `Message` / `ContentPart` (`message.rs`) — wire format for LLM messages
- `Model` / `Provider` / `SamplingParams` (`model.rs`) — model identification and config
- `CompletionRequest` / `StreamEvent` (`llm.rs`) — LLM request/response types
- `ToolCall` / `ToolRegistry` / `ToolContext` (`tool.rs`) — tool calling infrastructure
- `InjectQueue` / `InjectEvent` (`event.rs`) — thread-safe async queue for injecting messages into a running agent (push/drain/notified/cancel). Used for parent↔child agent communication and TUI input.
- `TurnStatus` (`event.rs`) — result of a single agent turn: `Continue`, `Done`, `ToolCalls`, `Interrupted`, `CompactionNeeded`
- `Outcome<O>` / `AgentStream` (`stream.rs`) — stream wrapper with terminal `Done` variant

## Conventions

- Types here must stay provider-agnostic — no provider-specific logic
- `AgentEvent` is `#[derive(Serialize, Deserialize)]` except `Started` which is `#[serde(skip)]`
- `TokenUsage` is in `llm.rs`, `TurnUsage` is in `event.rs` — they convert via `From`
