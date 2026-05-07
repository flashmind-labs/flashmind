//! Generate images via a dedicated image API (e.g. DALL-E).
//!
//! ```sh
//! cargo run -p flashmind --example image_gen -- \
//!   --api-key YOUR_KEY \
//!   --prompt "A cat wearing a top hat"
//!
//! # With OpenAI directly:
//! cargo run -p flashmind --example image_gen -- \
//!   --api-key YOUR_KEY \
//!   --prompt "A cat wearing a top hat" \
//!   --model openai:dall-e-3 \
//!   --quality hd
//! ```

use std::sync::Arc;

use clap::Parser;
use futures::StreamExt;

use flashmind::llm::OpenRouterProvider;
use flashmind::types::{ImageGenRequest, LlmProvider, Model, StreamEvent};

#[derive(Parser)]
#[command(about = "Generate an image using a dedicated image generation API")]
struct Args {
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    #[arg(
        long,
        default_value = "openrouter:openai/dall-e-3",
        help = "Image model [e.g. openrouter:openai/dall-e-3, openai:dall-e-2, openai:gpt-image-1]"
    )]
    model: Model,

    #[arg(long)]
    prompt: String,

    #[arg(
        long,
        help = "Image dimensions [dall-e-3: 1024x1024, 1792x1024, 1024x1792] [dall-e-2: 256x256, 512x512, 1024x1024] [gpt-image-1: 1024x1024, 1536x1024, 1024x1536, auto]"
    )]
    size: Option<String>,

    #[arg(
        long,
        help = "Image quality [dall-e-3: standard, hd] [gpt-image-1: low, medium, high]"
    )]
    quality: Option<String>,

    #[arg(long, help = "Image style [dall-e-3 only: natural, vivid]")]
    style: Option<String>,

    #[arg(
        long,
        help = "Number of images to generate [dall-e-2: 1-10, dall-e-3/gpt-image-1: 1]"
    )]
    n: Option<u32>,

    #[arg(long, default_value = "output.png")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let provider = Arc::new(OpenRouterProvider::new(args.api_key));

    let request = ImageGenRequest {
        model: args.model.name().to_string(),
        prompt: args.prompt,
        size: args.size,
        aspect_ratio: None,
        quality: args.quality,
        style: args.style,
        n: args.n,
    };

    eprintln!("Generating image via dedicated API...");
    let mut stream = provider.generate_image(request);
    let mut count = 0u32;

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::FileAttachment {
                data, media_type, ..
            }) => {
                count += 1;
                let filename = if count == 1 {
                    args.output.clone()
                } else {
                    let stem = args
                        .output
                        .rsplit_once('.')
                        .map(|(s, _)| s)
                        .unwrap_or(&args.output);
                    let ext = args
                        .output
                        .rsplit_once('.')
                        .map(|(_, e)| e)
                        .unwrap_or("png");
                    format!("{stem}_{count}.{ext}")
                };
                std::fs::write(&filename, &data)?;
                eprintln!("Saved {filename} ({} KB, {media_type})", data.len() / 1024);
            }
            Ok(StreamEvent::Finished(reason)) => {
                eprintln!("Finished: {reason}");
                break;
            }
            Err(e) => return Err(e),
            _ => {}
        }
    }

    if count == 0 {
        eprintln!("No images were generated.");
    }

    Ok(())
}
