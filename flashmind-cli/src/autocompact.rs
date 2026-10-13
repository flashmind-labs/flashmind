//! Argument parsing for the `/autocompact` command.

use flashmind_core::Agent;

/// What `/autocompact` was asked to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutoCompactArg {
    /// Show the current setting.
    Show,
    /// Compact automatically.
    On,
    /// Never compact before the context window overflows.
    Off,
    /// Drop the token limit and use the model's default trigger.
    Default,
    /// Compact once the prompt passes this many tokens.
    Tokens(u32),
}

pub(crate) const USAGE: &str = "  Usage: /autocompact [on|off|auto|<tokens>]  (e.g. 80000 or 80k)";

impl AutoCompactArg {
    pub(crate) fn parse(input: &str) -> Option<Self> {
        let input = input.trim().to_ascii_lowercase();
        match input.as_str() {
            "" => Some(Self::Show),
            "on" => Some(Self::On),
            "off" => Some(Self::Off),
            "default" | "reset" | "auto" => Some(Self::Default),
            _ => parse_tokens(&input).map(Self::Tokens),
        }
    }

    /// Apply the change to `agent`. `Show` changes nothing.
    pub(crate) fn apply(self, agent: &mut Agent) {
        match self {
            Self::Show => {}
            Self::On => agent.set_auto_compact(true),
            Self::Off => agent.set_auto_compact(false),
            Self::Default => {
                agent.set_auto_compact(true);
                agent.set_compact_at_tokens(None);
            }
            Self::Tokens(tokens) => {
                agent.set_auto_compact(true);
                agent.set_compact_at_tokens(Some(tokens));
            }
        }
    }
}

/// Describe the current setting in one line.
pub(crate) fn describe(agent: &Agent) -> String {
    if !agent.auto_compact() {
        return "  Auto-compact: off".into();
    }
    let trigger = agent.compaction_trigger_tokens();
    let window = agent.context_window();
    let source = match agent.compact_at_tokens() {
        Some(limit) if limit > trigger => {
            format!(" (limit {limit} is past the model default, so the default applies)")
        }
        Some(_) => String::new(),
        None => " (default)".into(),
    };
    format!("  Auto-compact: at {trigger} of {window} tokens{source}")
}

fn parse_tokens(input: &str) -> Option<u32> {
    let (digits, scale) = match input.strip_suffix('k') {
        Some(digits) => (digits, 1_000),
        None => (input, 1),
    };
    let digits = digits.replace(['_', ','], "");
    let tokens = digits.parse::<u32>().ok()?.checked_mul(scale)?;
    (tokens > 0).then_some(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_words_and_token_counts() {
        assert_eq!(AutoCompactArg::parse(""), Some(AutoCompactArg::Show));
        assert_eq!(AutoCompactArg::parse(" OFF "), Some(AutoCompactArg::Off));
        assert_eq!(
            AutoCompactArg::parse("reset"),
            Some(AutoCompactArg::Default)
        );
        assert_eq!(
            AutoCompactArg::parse("80k"),
            Some(AutoCompactArg::Tokens(80_000))
        );
        assert_eq!(
            AutoCompactArg::parse("120_000"),
            Some(AutoCompactArg::Tokens(120_000))
        );
        assert_eq!(AutoCompactArg::parse("0"), None);
        assert_eq!(AutoCompactArg::parse("lots"), None);
        assert_eq!(AutoCompactArg::parse("9999999k"), None);
    }
}
