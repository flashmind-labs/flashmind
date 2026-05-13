//! File attachment parsing for `@file` syntax in user input.
//!
//! Extracts `@path` markers from input text, reads the files, and converts
//! media files (images, video, audio, PDFs) into [`ContentPart`] for
//! multimodal LLM input.

use std::path::{Path, PathBuf};

use flashmind_types::ContentPart;

const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"];
const VIDEO_EXTS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm"];
const AUDIO_EXTS: &[&str] = &["mp3", "m4a", "wav", "flac", "aac", "ogg", "oga", "opus"];
const DOC_EXTS: &[&str] = &["pdf"];

pub struct Attachments {
    pub text: String,
    pub parts: Vec<ContentPart>,
    pub errors: Vec<String>,
}

/// Extract `@file` markers from input, load the files, and return cleaned
/// text + content parts.
pub fn collect(input: &str, cwd: &Path) -> Attachments {
    let mut text = String::new();
    let mut parts = Vec::new();
    let mut errors = Vec::new();

    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '@' {
            let mut path_str = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_whitespace() {
                    break;
                }
                path_str.push(nc);
                chars.next();
            }

            if path_str.is_empty() {
                text.push('@');
                continue;
            }

            let resolved = resolve_path(&path_str, cwd);
            if !resolved.exists() {
                // Not a file path — keep the original text
                text.push('@');
                text.push_str(&path_str);
                continue;
            }

            match load_file(&resolved) {
                Ok(part) => parts.push(part),
                Err(e) => errors.push(format!("{}: {e}", resolved.display())),
            }
        } else {
            text.push(c);
        }
    }

    Attachments {
        text: text.trim().to_string(),
        parts,
        errors,
    }
}

fn resolve_path(raw: &str, cwd: &Path) -> PathBuf {
    let expanded = if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            home.join(stripped)
        } else {
            PathBuf::from(raw)
        }
    } else {
        PathBuf::from(raw)
    };

    if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    }
}

fn load_file(path: &Path) -> anyhow::Result<ContentPart> {
    let meta = std::fs::metadata(path)?;
    if meta.len() > MAX_FILE_SIZE {
        anyhow::bail!(
            "file too large ({:.1} MB, max {:.0} MB)",
            meta.len() as f64 / 1024.0 / 1024.0,
            MAX_FILE_SIZE as f64 / 1024.0 / 1024.0
        );
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    if IMAGE_EXTS.contains(&ext.as_str()) {
        let data = std::fs::read(path)?;
        let media_type = mime_for_ext(&ext);
        Ok(ContentPart::Image {
            media_type,
            data: base64_encode(&data),
        })
    } else if VIDEO_EXTS.contains(&ext.as_str()) {
        let data = std::fs::read(path)?;
        let media_type = mime_for_ext(&ext);
        Ok(ContentPart::Video {
            media_type,
            filename,
            data: base64_encode(&data),
        })
    } else if AUDIO_EXTS.contains(&ext.as_str()) {
        let data = std::fs::read(path)?;
        let media_type = mime_for_ext(&ext);
        Ok(ContentPart::Audio {
            media_type,
            filename,
            data: base64_encode(&data),
        })
    } else if DOC_EXTS.contains(&ext.as_str()) {
        let data = std::fs::read(path)?;
        let media_type = mime_for_ext(&ext);
        Ok(ContentPart::Document {
            media_type,
            filename,
            data: base64_encode(&data),
        })
    } else {
        // Text file — inline as text part
        let content = std::fs::read_to_string(path)?;
        Ok(ContentPart::Text {
            text: format!("```{}\n{content}\n```", filename),
        })
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn mime_for_ext(ext: &str) -> String {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "avi" => "video/x-msvideo",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
    .to_string()
}
