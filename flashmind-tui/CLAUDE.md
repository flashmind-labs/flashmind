# flashmind-tui

TUI primitives for building interactive agent CLIs. Uses ratatui for styling primitives and crossterm for terminal I/O, but renders in append mode (no full-screen viewport) to preserve scrollback.

## Key Types

- `Tui` (`term.rs`) — append-mode terminal wrapper. `println`, `draw_lines`, `erase`, `raw_mode`, `draw_widget`, `width`/`size`.
- `RawModeGuard` (`widgets/repl.rs`) — RAII guard: enables raw mode on creation, restores on drop.
- `Repl` / `ReplConfig` / `ReplEvent` (`widgets/repl.rs`) — state-machine REPL: `read_input()` → `ReplEvent`, then `stream_response(stream)`. Configurable prompt, placeholder, max input height, token usage display. `set_usage(prompt, completion)` for manual token display. Activity indicator (`set_activity(name)`/`clear_activity()`) shows spinner + label in input bar title during agent turns. Reverse-i-search (Ctrl+R) searches history case-insensitively with match cycling.
- `EventRenderer` (`event_render.rs`) — renders `AgentEvent` → `Vec<RenderAction>`. Buffers text deltas, flushes on newlines or `Done`. `RenderAction::Append` for new lines, `RenderAction::ReplaceTool` for in-place tool result updates.

## Widgets (`widgets/`)

All widgets are standalone state machines: own state, handle keys via `handle_key() -> Option<Action>`, render via `lines() -> Vec<Line>`.

| Widget | Module | Purpose |
|--------|--------|---------|
| `PlanPicker` | `plan_picker.rs` | Plan approval with checkboxes, inline editing, add/delete steps |
| `ChoicePicker` | `choice_picker.rs` | Single-select choice with optional text input |
| `Dropdown` | `dropdown.rs` | Scrollable dropdown with selection highlighting |
| `Tree` / `TreeItem` | `tree.rs` | Hierarchical tree view with box-drawing connectors |
| `StatusBar` | `status_bar.rs` | Composable status line with spinner, sections, auto-expiring toasts |
| `Spinner` | `spinner.rs` | Braille spinner animation |
| `TextArea` | `textarea.rs` | Multi-line input with soft-wrapping, Emacs keybindings, word navigation |

## Architecture

```
Tui                               ← append-mode terminal (println, draw_lines, erase)
  └─ term::render_widget_to_stdout()  ← rasterizes ratatui Widget → crossterm escape sequences
       ↑
Repl::draw_input()                ← renders TextArea widget with anchor-based redraw
Repl::stream_response()          ← renders AgentEvents via EventRenderer + print_line()
```

Widget rendering pattern (used by all interactive widgets):
```rust
let _raw = tui.raw_mode()?;
let mut drawn = tui.draw_lines(&widget.lines())?;
loop {
    let key = read_key()?;
    if let Some(action) = widget.handle_key(key) { break action; }
    tui.erase(drawn)?;
    drawn = tui.draw_lines(&widget.lines())?;
}
```

## Important

- `agent.start()` borrows `&mut Agent` + `&mut Conversation`, so the stream can't be returned from closures. The REPL uses a state-machine pattern instead of a callback API.
- Raw mode stays on during `read_input()` — enabled once, disabled on submit/quit via `RawModeGuard`.
- `render_widget_to_stdout` prints `\r\n` between rows but NOT after the last row.

## Features

- `markdown` — enables rich markdown rendering (bold, headings, code blocks with syntax highlighting, tables, lists). Adds `nom`, `nu-ansi-term`, `regex`, `comfy-table`, `ansi-to-tui` deps. When enabled, `EventRenderer` renders text through `markdown.rs` (parser) → `markdown_render.rs` (ANSI renderer) → `ansi-to-tui` (ratatui Lines). Block types: `Paragraph`, `Heading`, `CodeBlock`, `Table`, `BulletList`, `OrderedList`, `Blockquote`, `HorizontalRule`. Incremental flush (`render_text_incremental`) renders complete blocks during streaming while holding incomplete ones (unclosed code fences, partial tables). Paragraphs join single `\n` into spaces (markdown soft breaks). Lone ordered items (e.g. `1. heading`) stay as paragraphs with prefix preserved — NOT stripped, to avoid losing numbers during streaming when list items arrive one at a time.

## Examples

```bash
cargo run -p flashmind --example widgets           # interactive widget demo
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl   # full REPL
```
