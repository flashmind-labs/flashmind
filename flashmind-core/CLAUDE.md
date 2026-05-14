# flashmind-core

Agent runtime: the main loop that drives LLM calls, tool execution, and conversation management.

Depends only on `flashmind-types` (provider-agnostic).

## Key Types

- `Agent` / `AgentBuilder` (`agent.rs`) — the agent runtime. `Agent::start()` returns `impl Stream<Item = AgentEvent>`. Built via `Agent::builder(provider).scope(...).tools(...).build()`.
- `Conversation` / `ConversationEntry` / `EntryKind` (`conversation.rs`) — conversation as an IR richer than raw messages. Entry kinds (`SystemPrompt`, `Reminder`, `Memory`, `Summary`, etc.) control how entries map to wire format via `to_messages()`.
- Compaction (`compaction.rs`) — multi-stage context window management: truncate → summarize → prune → strip → last-exchange fallback.
- Streaming (`streaming.rs`) — bridges `CompletionStream` from the provider into `AgentEvent`s.
- `AgentManager` / `AgentHandle` / `AgentStatus` (`subagent/`) — spawns child agents as tokio tasks, tracks status, routes messages via `InjectQueue`.
- `AwaitMessageTool` (`subagent/await_tool.rs`) — auto-registered tool that lets spawned agents block until a message arrives, enabling back-and-forth conversations. Sets `AgentStatus::Awaiting` and notifies the parent.

## Agent Loop Flow

1. `stream_llm_response()` — sends conversation to provider, streams back `AgentEvent`s
2. `execute_tool_calls()` — runs any tool calls from the response
3. `try_compact()` — compacts conversation if context pressure detected
4. Loop back to 1 until `TurnStatus::Done` or max iterations reached

## Subagent Architecture (`subagent/`)

- `manager.rs` — `AgentManager` orchestrates child agents: spawn (with concurrency/depth limits), resolve by name/UUID, send messages, wait for completion. Each child runs in a tokio task via `run_agent()` which uses `tokio::select!` over the agent stream and cancellation.
- `handle.rs` — `AgentHandle` wraps a single child: status, cancel token, inject queue, join handle. `AgentStatus` variants: `Running`, `Awaiting`, `Completed`, `Failed`, `Cancelled`.
- `builder.rs` — `SpawnBuilder` fluent API for configuring child agents (model, tools, system prompt, iteration limits).
- `await_tool.rs` — `AwaitMessageTool` is auto-registered for every spawned agent. When a child calls it, the tool sets `AgentStatus::Awaiting`, notifies the parent via `AgentProgress`, then blocks on `InjectQueue::notified()` until a message arrives (with timeout and cancellation). This prevents agents from finishing prematurely during conversations.

### Message flow (parent → child)

1. Parent calls `communicate("child_name", "message")`
2. `AgentHandle::send_message()` pushes `InjectEvent::UserMessage` to child's `InjectQueue`
3. `run_agent()` forwards from `child_queue` to the agent's internal inject queue
4. Agent loop drains inject queue at top of next iteration → message appears in conversation

### Message flow (child → parent)

Child progress/errors are pushed to the parent's `InjectQueue` as `AgentProgress`/`AgentError` events, which the parent's agent loop drains into its conversation as developer entries.

## Important

- `agent.start()` borrows `&mut self` and `&mut Conversation` — the returned stream holds both borrows. This means it cannot be returned from closures; callers must use a state-machine pattern (see `flashmind-tui` REPL for example).
