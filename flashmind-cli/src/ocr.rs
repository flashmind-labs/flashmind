use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use clap::Args;
use flashmind_llm::MistralProvider;
use flashmind_llm::mistral::{OcrDocument, OcrRequest};
use flashmind_types::Provider;
use tokio::io::AsyncReadExt;
use url::Url;

use crate::config::Config;

const MAX_DOCUMENT_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Args, Clone)]
pub(crate) struct OcrArgs {
    /// PDF or image file, or a publicly accessible URL
    input: String,
    /// Treat a URL as an image when its extension does not identify it
    #[arg(long)]
    image: bool,
    /// Print the complete provider response as JSON
    #[arg(long)]
    json: bool,
}

pub(crate) async fn run(args: &OcrArgs, model: Option<&str>, config: &Config) -> Result<()> {
    let model = model.unwrap_or("mistral:mistral-ocr-latest");
    let model: flashmind_types::Model = model.parse()?;
    ensure!(
        model.provider == Provider::Mistral,
        "OCR currently supports Mistral models"
    );
    let key = config
        .api_key_for(&Provider::Mistral)
        .context("set MISTRAL_API_KEY or providers.mistral.api_key")?;
    let mut provider = MistralProvider::new(key.to_owned());
    if let Some(url) = &config.endpoint_override.url {
        provider = provider.with_base_url(url.as_str())?;
    }
    let document = document(&args.input, args.image).await?;
    let result = provider
        .ocr(&OcrRequest {
            model: model.name().to_owned(),
            document,
            options: config.endpoint_override.extra_args.clone(),
        })
        .await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        let pages = result
            .get("pages")
            .and_then(|value| value.as_array())
            .context("Mistral OCR response has no pages")?;
        for (index, page) in pages.iter().enumerate() {
            let markdown = page
                .get("markdown")
                .and_then(|value| value.as_str())
                .context("Mistral OCR page has no markdown")?;
            if index > 0 {
                println!();
            }
            println!("{markdown}");
        }
    }
    Ok(())
}

async fn document(input: &str, image: bool) -> Result<OcrDocument> {
    let (value, is_image) = if let Ok(url) = Url::parse(input) {
        ensure!(
            matches!(url.scheme(), "https" | "http"),
            "document URL must use HTTP or HTTPS"
        );
        let is_image = image
            || matches!(
                extension(url.path()).as_deref(),
                Some("png" | "jpg" | "jpeg" | "webp")
            );
        (url.to_string(), is_image)
    } else {
        let mime = match extension(input).as_deref() {
            Some("pdf") => "application/pdf",
            Some("png") => "image/png",
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => bail!("document must be a PDF, PNG, JPEG or WebP file"),
        };
        let file = tokio::fs::File::open(input)
            .await
            .context("cannot open OCR document")?;
        let mut bytes = Vec::new();
        file.take(MAX_DOCUMENT_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .context("cannot read OCR document")?;
        ensure!(
            bytes.len() as u64 <= MAX_DOCUMENT_BYTES,
            "OCR document exceeds 20 MiB"
        );
        (
            format!("data:{mime};base64,{}", STANDARD.encode(bytes)),
            mime.starts_with("image/"),
        )
    };
    Ok(if is_image {
        OcrDocument::ImageUrl { image_url: value }
    } else {
        OcrDocument::DocumentUrl {
            document_url: value,
        }
    })
}

fn extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()?
        .to_str()
        .map(str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builds_image_and_pdf_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("document.pdf");
        tokio::fs::write(&path, b"pdf").await.unwrap();
        assert!(
            matches!(document(path.to_str().unwrap(), false).await.unwrap(), OcrDocument::DocumentUrl { document_url } if document_url == "data:application/pdf;base64,cGRm")
        );
        assert!(matches!(
            document("https://example.com/image.png", false)
                .await
                .unwrap(),
            OcrDocument::ImageUrl { .. }
        ));
        assert!(matches!(
            document("https://example.com/file", true).await.unwrap(),
            OcrDocument::ImageUrl { .. }
        ));
        assert!(document("file:///example.pdf", false).await.is_err());
    }
}
