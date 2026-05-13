//! Slash command parser and handler.

/// Parsed slash command.
pub enum Command {
    Help,
    Clear,
    Compact,
    Context,
    Info,
    Model(Option<String>),
    Temperature(Option<String>),
    Thinking(Option<String>),
    Title(Option<String>),
    Sessions,
    Save(Option<String>),
    Setup,
    Soul,
    Sub(Option<String>),
    Subagents,
    Quit,
    Unknown(String),
}

/// Parse a slash command from user input. Returns `None` if the input
/// is not a command (doesn't start with `/`).
pub fn parse(input: &str) -> Option<Command> {
    let input = input.trim();
    if !input.starts_with('/') {
        return None;
    }

    let mut parts = input[1..].splitn(2, ' ');
    let name = parts.next().unwrap_or("");
    let arg = parts.next().map(|s| s.trim().to_string());

    Some(match name {
        "help" | "h" => Command::Help,
        "clear" => Command::Clear,
        "compact" => Command::Compact,
        "context" | "ctx" => Command::Context,
        "info" => Command::Info,
        "model" | "m" => Command::Model(arg),
        "temperature" | "temp" => Command::Temperature(arg),
        "thinking" | "think" => Command::Thinking(arg),
        "title" => Command::Title(arg),
        "sessions" => Command::Sessions,
        "save" => Command::Save(arg),
        "setup" => Command::Setup,
        "soul" => Command::Soul,
        "sub" => Command::Sub(arg),
        "subagents" | "subs" => Command::Subagents,
        "quit" | "q" | "exit" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

pub fn help_text() -> &'static str {
    "\
Commands:
  /help, /h              Show this help
  /clear                 Clear conversation history
  /compact               Force context compaction
  /context               Show token usage
  /info                  Session overview
  /model [name]          Show or switch model
  /temperature [0.0-2.0] Show or set temperature
  /thinking [on|off]     Show or toggle reasoning
  /title [text]          Show or set session title
  /sessions              List saved sessions
  /save <path>           Save conversation to file
  /setup                 Interactive config setup wizard
  /soul                  Edit SOUL.md (system prompt) in $EDITOR
  /sub <task>            Spawn a background subagent
  /subagents             List active subagents
  /quit, /q              Exit"
}

/// All command names for tab completion.
pub fn command_names() -> Vec<String> {
    vec![
        "help".into(),
        "clear".into(),
        "compact".into(),
        "context".into(),
        "info".into(),
        "model".into(),
        "temperature".into(),
        "thinking".into(),
        "title".into(),
        "sessions".into(),
        "save".into(),
        "setup".into(),
        "soul".into(),
        "sub".into(),
        "subagents".into(),
        "quit".into(),
    ]
}
