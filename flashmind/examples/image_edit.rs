//! Edit or generate images via a multimodal chat model (e.g. GPT-image, Gemini).
//!
//! ```sh
//! cargo run -p flashmind --example image_edit -- \
//!   --api-key YOUR_KEY \
//!   --prompt "A cat wearing a top hat" \
//!   --model openrouter:openai/gpt-5-image
//!
//! # With a reference image:
//! cargo run -p flashmind --example image_edit -- \
//!   --api-key YOUR_KEY \
//!   --prompt "Same style but with a dog" \
//!   --reference photo.jpg
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use futures::StreamExt;

use flashmind::llm::OpenRouterProvider;
use flashmind::types::{
    CompletionRequest, ImageGenConfig, LlmProvider, Modality, StreamEvent, model::ReasoningLevel,
};

#[derive(Parser)]
#[command(about = "Generate an image from a text prompt")]
struct Args {
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    #[arg(
        long,
        default_value = "openrouter:openai/gpt-5-image",
        help = "Chat model with image output [e.g. openrouter:openai/gpt-5-image, openrouter:google/gemini-2.0-flash-exp:free]"
    )]
    model: flashmind::types::Model,

    #[arg(long)]
    prompt: String,

    #[arg(long, help = "Path to a reference image for style guidance")]
    reference: Option<PathBuf>,

    #[arg(long, default_value = "output.png", help = "Output filename")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let provider = Arc::new(OpenRouterProvider::new(args.api_key));

    let message = match &args.reference {
        Some(path) => {
            use flashmind::types::message::ContentPart;
            let data_uri = ImageGenConfig::data_uri_from_path(path)?;
            flashmind::types::Message::user_with_parts(
                &args.prompt,
                vec![ContentPart::ImageUrl { url: data_uri }],
            )
        }
        None => flashmind::types::Message::user(&args.prompt),
    };

    let request = CompletionRequest {
        model: args.model,
        messages: vec![message],
        tools: vec![],
        max_tokens: Some(4096),
        reasoning: ReasoningLevel::Off,
        sampling: Default::default(),
        modalities: vec![Modality::Image, Modality::Text],
        audio_config: None,
        image_config: Some(ImageGenConfig {
            aspect_ratio: None,
            size: None,
        }),
        user: None,
        provider_preferences: None,
    };

    eprintln!("Generating image...");
    let mut stream = provider.complete(request);

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::FileAttachment {
                data, media_type, ..
            }) => {
                std::fs::write(&args.output, &data)?;
                eprintln!(
                    "Saved {} ({} KB, {})",
                    args.output,
                    data.len() / 1024,
                    media_type
                );
            }
            Ok(StreamEvent::ContentDelta(text)) => {
                print!("{text}");
            }
            Ok(StreamEvent::Finished(reason)) => {
                eprintln!("Finished: {reason}");
                break;
            }
            Ok(other) => {
                eprintln!("Event: {other:?}");
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
}
