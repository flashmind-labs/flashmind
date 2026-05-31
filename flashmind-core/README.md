# flashmind-core

Agent runtime for the [Flashmind](https://github.com/flashmind-labs/flashmind) AI framework.

Drives the LLM call → tool execution → compaction loop. Depends only on `flashmind-types` (provider-agnostic).

## Key Types

- **`Agent`** / **`AgentBuilder`** — build and run agents
- **`Conversation`** — rich conversation IR with typed entries
- **`AgentManager`** — spawn and coordinate child agents

## Usage

```rust
use flashmind_core::{Agent, Conversation, CancellationToken};
use flashmind_types::{AgentEvent, AgentInput};
use futures::StreamExt;

let mut agent = Agent::builder(provider).tools(tools).build_sync();
let mut conversation = Conversation::new();

let stream = agent.start(&mut conversation, CancellationToken::new(), AgentInput::user("Hi"), None);
tokio::pin!(stream);
while let Some(event) = stream.next().await {
    // handle events
}
```

## Features

- `subagent` — enables child agent spawning via `AgentManager`

## License

MPL-2.0
