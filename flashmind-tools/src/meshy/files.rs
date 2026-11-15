//! Inlines local files as data URIs, since Meshy accepts URLs or data URIs only.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value};
use tokio::io::AsyncReadExt;

const MAX_MODEL_BYTES: u64 = 100 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

/// What a local file argument must contain.
#[derive(Clone, Copy)]
pub(super) enum FileKind {
    Model,
    Image,
}

/// Replace local paths in the named string or string-array arguments with data URIs.
/// URLs and existing data URIs pass through unchanged.
pub(super) async fn inline_local_files(
    args: &mut Map<String, Value>,
    fields: &[(&str, FileKind)],
    working_dir: Option<&PathBuf>,
) -> anyhow::Result<()> {
    for (field, kind) in fields {
        match args.get_mut(*field) {
            Some(Value::String(value)) => {
                *value = inline_one(value, *kind, working_dir).await?;
            }
            Some(Value::Array(values)) => {
                for value in values {
                    if let Value::String(text) = value {
                        *text = inline_one(text, *kind, working_dir).await?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

async fn inline_one(
    value: &str,
    kind: FileKind,
    working_dir: Option<&PathBuf>,
) -> anyhow::Result<String> {
    let lower = value.get(..8).unwrap_or(value).to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("data:") {
        return Ok(value.to_owned());
    }
    let path = working_dir.map_or_else(|| PathBuf::from(value), |dir| dir.join(value));
    let (mime, limit) = match kind {
        FileKind::Model => (model_mime(&path)?, MAX_MODEL_BYTES),
        FileKind::Image => (image_mime(&path)?, MAX_IMAGE_BYTES),
    };
    let file = tokio::fs::File::open(&path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    if u64::try_from(bytes.len())? > limit {
        bail!(
            "{} exceeds the {} MB limit for local files; use a URL instead",
            path.display(),
            limit / (1024 * 1024)
        );
    }
    if bytes.is_empty() {
        bail!("{} is empty", path.display());
    }
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn model_mime(path: &Path) -> anyhow::Result<&'static str> {
    match extension(path).as_str() {
        "glb" | "gltf" | "obj" | "fbx" | "stl" => Ok("application/octet-stream"),
        _ => bail!(
            "{} must be a GLB, glTF, OBJ, FBX or STL file",
            path.display()
        ),
    }
}

fn image_mime(path: &Path) -> anyhow::Result<&'static str> {
    match extension(path).as_str() {
        "png" => Ok("image/png"),
        "jpg" | "jpeg" => Ok("image/jpeg"),
        "webp" => Ok("image/webp"),
        _ => bail!("{} must be a PNG, JPEG or WebP image", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn local_files_become_data_uris_and_urls_pass_through() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.glb"), b"glTF").unwrap();
        std::fs::write(dir.path().join("b.PNG"), b"png").unwrap();
        let working_dir = dir.path().to_path_buf();
        let mut args = json!({
            "model_url": "a.glb",
            "images": ["b.PNG", "https://example.com/c.png", "data:image/png;base64,AA=="],
            "other": "left.txt",
        })
        .as_object()
        .unwrap()
        .clone();
        inline_local_files(
            &mut args,
            &[("model_url", FileKind::Model), ("images", FileKind::Image)],
            Some(&working_dir),
        )
        .await
        .unwrap();
        assert_eq!(
            args["model_url"],
            "data:application/octet-stream;base64,Z2xURg=="
        );
        assert_eq!(
            args["images"],
            json!([
                "data:image/png;base64,cG5n",
                "https://example.com/c.png",
                "data:image/png;base64,AA=="
            ])
        );
        assert_eq!(args["other"], "left.txt");
    }

    #[tokio::test]
    async fn rejects_wrong_types_missing_and_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        std::fs::write(dir.path().join("empty.glb"), b"").unwrap();
        let working_dir = dir.path().to_path_buf();
        for (name, kind) in [
            ("a.txt", FileKind::Model),
            ("a.txt", FileKind::Image),
            ("empty.glb", FileKind::Model),
            ("missing.glb", FileKind::Model),
        ] {
            let mut args = json!({ "f": name }).as_object().unwrap().clone();
            assert!(
                inline_local_files(&mut args, &[("f", kind)], Some(&working_dir))
                    .await
                    .is_err(),
                "{name}"
            );
        }
    }
}
