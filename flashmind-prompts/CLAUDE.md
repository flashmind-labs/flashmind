# flashmind-prompts

System prompt constants and a workspace-aware prompt builder. Zero dependencies.

## Contents

All in `lib.rs`:

- **Constants**: `CODING_AGENT`, `CONVERSATIONAL`, `TOOL_USE_INSTRUCTIONS`, `CONCISE_OUTPUT`, `SAFETY_GUARDRAILS`, `MEMORY_INSTRUCTIONS`, `CODE_REVIEW`, `SUMMARIZER`
- **Function**: `build_project_instructions(workspace: &Path) -> String` — scans a workspace directory and generates context-aware system instructions
