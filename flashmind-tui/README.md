# flashmind-tui

Terminal UI primitives for building interactive agent CLIs with [Flashmind](https://github.com/flashmind-labs/flashmind).

Append-mode rendering (preserves scrollback) using ratatui for styling and crossterm for I/O.

## Key Components

- **`Repl`**  -  state-machine REPL: read input, stream agent response
- **`EventRenderer`**  -  renders `AgentEvent` stream into styled terminal lines
- **`TextArea`**  -  multi-line input with Emacs keybindings and word navigation
- **Widgets**  -  `PlanPicker`, `ChoicePicker`, `Dropdown`, `Tree`, `StatusBar`, `Spinner`

## Usage

```rust
use flashmind_tui::{Repl, ReplConfig, Tui};

let tui = Tui::new()?;
let mut repl = Repl::new(ReplConfig::default(), tui);

loop {
    let input = repl.read_input().await?;
    // ... start agent stream ...
    repl.stream_response(stream).await?;
}
```

## Features

- `markdown`  -  rich markdown rendering (headings, code blocks, tables, lists)

## License

MPL-2.0
