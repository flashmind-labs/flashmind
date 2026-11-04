//! Keeps file data a tool printed as text, such as base64, out of the model's context.

use std::borrow::Cow;

/// A base64 run this long is file data, not text worth reading.
const MIN_BLOB_CHARS: usize = 2048;

fn is_base64(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=' | b'-' | b'_')
}

/// Encoded data mixes cases and digits; a run of one letter is not data.
fn looks_encoded(run: &[u8]) -> bool {
    run.iter().any(u8::is_ascii_uppercase)
        && run.iter().any(u8::is_ascii_lowercase)
        && run.iter().any(u8::is_ascii_digit)
}

/// `text` with every long base64 run replaced by a note of its size.
pub(crate) fn without_blobs(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut kept: Option<String> = None;
    let mut copied = 0;
    let mut start = 0;
    while start < bytes.len() {
        if !is_base64(bytes[start]) {
            start += 1;
            continue;
        }
        let end = bytes[start..]
            .iter()
            .position(|byte| !is_base64(*byte))
            .map_or(bytes.len(), |length| start + length);
        if end - start >= MIN_BLOB_CHARS && looks_encoded(&bytes[start..end]) {
            let out = kept.get_or_insert_with(String::new);
            // Both ends sit next to ASCII bytes, so they are char boundaries.
            out.push_str(&text[copied..start]);
            let size = ((end - start) / 4 * 3).div_ceil(1024);
            out.push_str(&format!("[{size} KB of binary data left out]"));
            copied = end;
        }
        start = end;
    }
    match kept {
        Some(mut out) => {
            out.push_str(&text[copied..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_base64_blob_in_debug_text_is_left_out() {
        let blob = "JVBERi0xLjcK".repeat(100_000);
        let text = format!(
            "Resource(EmbeddedResource {{ resource: BlobResourceContents {{ uri: \"x\", \
             mime_type: Some(\"application/pdf\"), blob: \"{blob}\" }} }})"
        );
        let kept = without_blobs(&text);
        assert_eq!(
            kept,
            "Resource(EmbeddedResource { resource: BlobResourceContents { uri: \"x\", \
             mime_type: Some(\"application/pdf\"), blob: \"[879 KB of binary data left out]\" } })"
        );
    }

    #[test]
    fn ordinary_text_is_borrowed_unchanged() {
        let text = "id: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 ✓";
        assert!(matches!(without_blobs(text), Cow::Borrowed(t) if t == text));
    }
}
