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
//!
//! # With a reference video (samples frames as style guidance):
//! cargo run -p flashmind --example video_gen -- \
//!   --api-key YOUR_KEY \
//!   --prompt "Same scene but at night" \
//!   --reference clip.mp4 \
//!   --reference-frames 4
//! ```

use std::path::PathBuf;
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

    #[arg(
        long,
        default_value = "openrouter:google/veo-2.0-generate-001",
        help = "Video model [e.g. openrouter:google/veo-2.0-generate-001, openrouter:minimax/video-01-live]"
    )]
    model: Model,

    #[arg(long)]
    prompt: String,

    #[arg(
        long,
        help = "Path to a reference video (frames are extracted as style guidance)"
    )]
    reference: Option<PathBuf>,

    #[arg(
        long,
        default_value = "4",
        help = "Number of frames to sample from the reference video"
    )]
    reference_frames: u32,

    #[arg(long, help = "Video resolution [e.g. 720p, 1080p]")]
    resolution: Option<String>,

    #[arg(long, help = "Aspect ratio [e.g. 16:9, 9:16, 1:1]")]
    aspect_ratio: Option<String>,

    #[arg(long, help = "Duration in seconds [veo-2: 5-8]")]
    duration: Option<u32>,

    #[arg(long, default_value = "output.mp4")]
    output: String,
}

fn extract_frames(video: &std::path::Path, count: u32) -> anyhow::Result<Vec<String>> {
    let tmp = tempfile::tempdir()?;

    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .output()?;
    let total_frames: u32 = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .parse()
        .unwrap_or(100);

    let select = (0..count)
        .map(|i| {
            let frame = (i as u64 * total_frames as u64) / count as u64;
            format!("eq(n\\,{frame})")
        })
        .collect::<Vec<_>>()
        .join("+");

    let vf = format!("select='{select}',scale='min(768,iw):-2'");
    let pattern_jpg = tmp.path().join("frame_%03d.jpg");

    let status = std::process::Command::new("ffmpeg")
        .args(["-i"])
        .arg(video)
        .args(["-vf", &vf, "-vsync", "vfr", "-q:v", "5"])
        .arg(&pattern_jpg)
        .args(["-hide_banner", "-loglevel", "error"])
        .status()?;
    anyhow::ensure!(status.success(), "ffmpeg failed");

    let mut uris = Vec::new();
    for i in 1..=count {
        let path = tmp.path().join(format!("frame_{i:03}.jpg"));
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
            uris.push(format!("data:image/jpeg;base64,{b64}"));
        }
    }
    Ok(uris)
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

    let input_references = match &args.reference {
        Some(video_path) => {
            eprintln!(
                "Extracting {} frames from {}...",
                args.reference_frames,
                video_path.display()
            );
            extract_frames(video_path, args.reference_frames)?
        }
        None => vec![],
    };

    let request = VideoGenRequest {
        model: args.model.name().to_string(),
        description: args.prompt,
        resolution: args.resolution,
        aspect_ratio: args.aspect_ratio,
        duration: args.duration,
        generate_audio: Some(true),
        frame_images: vec![],
        input_references,
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
