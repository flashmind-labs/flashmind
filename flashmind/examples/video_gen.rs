//! Generate a video via a dedicated video generation API (e.g. Veo).
//!
//! ```sh
//! cargo run -p flashmind --example video_gen -- \
//!   --api-key YOUR_KEY \
//!   --prompt "A timelapse of a sunset over the ocean"
//!
//! # With custom settings:
//! cargo run -p flashmind --example video_gen -- \
//!   --api-key YOUR_KEY \
//!   --prompt "A cat playing piano" \
//!   --model openrouter:google/veo-2.0-generate-001 \
//!   --aspect-ratio 16:9 \
//!   --duration 8
//! ```

use std::sync::Arc;

use clap::Parser;
use futures::StreamExt;

use flashmind::llm::{OpenRouterProvider, http::create_rate_limiter};
use flashmind::types::{LlmProvider, Model, StreamEvent, VideoGenRequest};

#[derive(Parser)]
#[command(about = "Generate a video from a text prompt")]
struct Args {
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    #[arg(long, default_value = "openrouter:google/veo-2.0-generate-001",
          help = "Video model [e.g. openrouter:google/veo-2.0-generate-001, openrouter:minimax/video-01-live]")]
    model: Model,

    #[arg(long)]
    prompt: String,

    #[arg(long, help = "Video resolution [e.g. 720p, 1080p]")]
    resolution: Option<String>,

    #[arg(long, help = "Aspect ratio [e.g. 16:9, 9:16, 1:1]")]
    aspect_ratio: Option<String>,

    #[arg(long, help = "Duration in seconds [veo-2: 5-8]")]
    duration: Option<u32>,

    #[arg(long, default_value = "output.mp4")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let provider = Arc::new(OpenRouterProvider::new(
        args.api_key,
        create_rate_limiter(60),
    ));

    let request = VideoGenRequest {
        model: args.model.name().to_string(),
        description: args.prompt,
        resolution: args.resolution,
        aspect_ratio: args.aspect_ratio,
        duration: args.duration,
        generate_audio: Some(true),
        frame_images: vec![],
        input_references: vec![],
    };

    eprintln!("Generating video (this may take several minutes)...");
    let mut stream = provider.generate_video(request);
    let mut saved = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::ContentDelta(text)) => {
                eprint!("{text}");
            }
            Ok(StreamEvent::FileAttachment {
                data, media_type, ..
            }) => {
                std::fs::write(&args.output, &data)?;
                eprintln!(
                    "Saved {} ({} MB, {media_type})",
                    args.output,
                    data.len() / (1024 * 1024)
                );
                saved = true;
            }
            Ok(StreamEvent::Finished(reason)) => {
                eprintln!("Finished: {reason}");
                break;
            }
            Err(e) => return Err(e),
            _ => {}
        }
    }

    if !saved {
        eprintln!("No video was generated.");
    }

    Ok(())
}
