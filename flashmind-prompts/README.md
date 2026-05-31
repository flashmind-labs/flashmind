# flashmind-prompts

Reusable system prompt fragments for [Flashmind](https://github.com/flashmind-labs/flashmind) agents.

## Contents

- `CODING_AGENT` — system prompt for code-generation agents
- `CONVERSATIONAL` — general-purpose assistant prompt
- `TOOL_USE_INSTRUCTIONS` — instructions for tool-calling behavior
- `SAFETY_GUARDRAILS` — safety and content policy rules
- `MEMORY_INSTRUCTIONS` — how to use long-term memory
- `CODE_REVIEW` — code review persona
- `SUMMARIZER` — conversation summarization prompt

Plus `build_project_instructions()` for workspace-aware context injection.

## Usage

```rust
use flashmind_prompts::{CODING_AGENT, TOOL_USE_INSTRUCTIONS};

let system = format!("{CODING_AGENT}\n\n{TOOL_USE_INSTRUCTIONS}");
```

Zero dependencies.

## License

MPL-2.0
