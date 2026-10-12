# flashmind-tui

TUI primitives for building interactive agent CLIs. All interactive views use the alternate screen through Ratatui. Chat keeps the input fixed below a scrollable transcript.

## Key Types

- `Tui` (`term.rs`): styled output helpers and `fullscreen()` for a scoped Ratatui menu. The menu restores the main screen on drop.
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

## Rendering

Menus use `Tui::fullscreen()` and `screen.draw()` to render a Ratatui frame.
Use the frame area for layout and handle resize events by drawing again.
`draw_list()` keeps the selected row visible between fixed headers and footers.

The REPL shares a `Tui` with CLI status messages, shell output, and menus.
Completed lines enter the transcript; cached wrap offsets let it render only
the visible history. Input, autocomplete, pets, and live progress use frames.
Keep running tool lines in the live transcript until the result arrives.

`clear_input_bar()` suspends the chat screen before an independent menu.
Mouse wheel, PgUp/PgDn, and Ctrl+Home/End scroll the transcript. Scrolling up
pauses auto-follow until the user returns to the bottom. Ctrl+F searches the
transcript. Mouse dragging selects and copies text, with OSC 52 for remote
clipboards. All interactive rendering uses the alternate screen.

## Features

- `markdown` — enables rich markdown rendering (bold, headings, code blocks with syntax highlighting, tables, lists). Adds `nom`, `nu-ansi-term`, `regex`, `comfy-table`, `ansi-to-tui` deps. When enabled, `EventRenderer` renders text through `markdown.rs` (parser) → `markdown_render.rs` (ANSI renderer) → `ansi-to-tui` (ratatui Lines). Block types: `Paragraph`, `Heading`, `CodeBlock`, `Table`, `BulletList`, `OrderedList`, `Blockquote`, `HorizontalRule`. Incremental flush (`render_text_incremental`) renders complete blocks during streaming while holding incomplete ones (unclosed code fences, partial tables). Paragraphs join single `\n` into spaces (markdown soft breaks). Lone ordered items (e.g. `1. heading`) stay as paragraphs with prefix preserved — NOT stripped, to avoid losing numbers during streaming when list items arrive one at a time.

## Examples

```bash
cargo run -p flashmind --example widgets           # interactive widget demo
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl   # full REPL
```
