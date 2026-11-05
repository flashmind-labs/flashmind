//! A caller-supplied compaction rule that outranks every other setting.

use std::str::FromStr;

use thiserror::Error;

/// A compaction rule that beats [`Agent::set_auto_compact`](super::Agent::set_auto_compact)
/// and [`Agent::set_compact_at_tokens`](super::Agent::set_compact_at_tokens),
/// so a host can fix the behavior of one process whatever its config says.
///
/// Parses from `off`, `<n>%` (1 to 100) or `<n>` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionOverride {
    /// Never compact automatically.
    Off,
    /// Compact once the prompt passes this percent of the context window.
    Percent(u8),
    /// Compact once the prompt passes this many tokens.
    Tokens(u32),
}

/// The text is not `off`, `<n>%` or `<n>`.
#[derive(Debug, Error)]
#[error("expected `off`, `<n>%` (1 to 100) or a token count, got {0:?}")]
pub struct ParseCompactionOverrideError(String);

impl FromStr for CompactionOverride {
    type Err = ParseCompactionOverrideError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || ParseCompactionOverrideError(text.to_string());
        let value = text.trim();
        if value.eq_ignore_ascii_case("off") {
            return Ok(Self::Off);
        }
        if let Some(digits) = value.strip_suffix('%') {
            let percent = digits.parse::<u8>().map_err(|_| invalid())?;
            return (1..=100)
                .contains(&percent)
                .then_some(Self::Percent(percent))
                .ok_or_else(invalid);
        }
        match value.parse::<u32>() {
            Ok(tokens) if tokens > 0 => Ok(Self::Tokens(tokens)),
            _ => Err(invalid()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> CompactionOverride {
        text.parse().unwrap()
    }

    #[test]
    fn parses_off_percent_and_tokens() {
        assert_eq!(parse("off"), CompactionOverride::Off);
        assert_eq!(parse(" OFF "), CompactionOverride::Off);
        assert_eq!(parse("70%"), CompactionOverride::Percent(70));
        assert_eq!(parse("100%"), CompactionOverride::Percent(100));
        assert_eq!(parse("300000"), CompactionOverride::Tokens(300_000));
    }

    #[test]
    fn rejects_everything_else() {
        for text in [
            "",
            "on",
            "0",
            "0%",
            "101%",
            "-5%",
            "7.5%",
            "%",
            "70 %",
            "300k",
            "-1",
            "99999999999",
        ] {
            assert!(text.parse::<CompactionOverride>().is_err(), "{text:?}");
        }
    }
}
