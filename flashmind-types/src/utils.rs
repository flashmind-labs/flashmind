//! Shared utility functions for safe string manipulation.
//!
//! All functions are UTF-8-safe — they never split multi-byte characters, and the
//! newline variant stops at line boundaries for log-preview use cases.

/// Truncate `s` to at most `max` bytes, landing on a valid UTF-8 character boundary.
///
/// Unlike naive `&s[..max]` this avoids splitting multi-byte characters (emoji, CJK).
pub fn truncate_utf8(s: &str, max: usize) -> &str {
    &s[..s.floor_char_boundary(max)]
}

/// Like [`truncate_utf8`] but also stops at the first newline if it appears before `max`.
/// Useful for log previews where long multi-line output should show only the first line.
pub fn truncate_utf8_line(s: &str, max: usize) -> &str {
    let max = s
        .bytes()
        .position(|b| b == b'\r' || b == b'\n')
        .map_or(max, |nl| nl.min(max));

    truncate_utf8(s, max)
}
