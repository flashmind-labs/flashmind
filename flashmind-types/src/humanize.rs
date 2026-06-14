//! Human-readable formatting for numbers, token counts, and pricing.

use std::fmt;

/// Format a token count as a compact human-readable string.
///
/// Uses SI-style suffixes: `1M`, `1.5M`, `128k`, `512`.
pub fn format_tokens(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        let whole = tokens / 1_000_000;
        let rem = tokens % 1_000_000;
        if rem == 0 {
            format!("{whole}M")
        } else {
            format!("{:.1}M", tokens as f64 / 1_000_000.0)
        }
    } else if tokens >= 1_000 {
        let whole = tokens / 1_000;
        let rem = tokens % 1_000;
        if rem == 0 {
            format!("{whole}k")
        } else {
            format!("{:.1}k", tokens as f64 / 1_000.0)
        }
    } else {
        format!("{tokens}")
    }
}

/// Format a large number with comma separators: `1,234,567`.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

/// Wrapper for displaying a token count in human-readable form via [`fmt::Display`].
///
/// ```
/// use flashmind_types::humanize::Tokens;
/// assert_eq!(Tokens(200_000).to_string(), "200k");
/// assert_eq!(Tokens(1_000_000).to_string(), "1M");
/// ```
pub struct Tokens(pub u32);

impl fmt::Display for Tokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tokens(self.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(512), "512");
        assert_eq!(format_tokens(1_000), "1k");
        assert_eq!(format_tokens(4_096), "4.1k");
        assert_eq!(format_tokens(128_000), "128k");
        assert_eq!(format_tokens(200_000), "200k");
        assert_eq!(format_tokens(1_000_000), "1M");
        assert_eq!(format_tokens(1_500_000), "1.5M");
        assert_eq!(format_tokens(2_000_000), "2M");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn test_tokens_display() {
        assert_eq!(format!("{}", Tokens(128_000)), "128k");
        assert_eq!(format!("{}", Tokens(1_000_000)), "1M");
    }
}
