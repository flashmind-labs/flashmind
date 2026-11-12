# flashmind-tui

Terminal UI primitives for agent CLIs, built on Ratatui.

- All interactive views use the alternate screen.
- Widgets are state machines: they own their state, handle keys, and return
  lines. Keep terminal IO out of them.
- Store transcript lines unwrapped and lay them out when drawing, so they
  reflow on resize.
- Redraw at most once per frame, inside a synchronized update.
- Code must build without the optional `markdown` feature.

```bash
cargo run -p flashmind --example widgets
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl
```
