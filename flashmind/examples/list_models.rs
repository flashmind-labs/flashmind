//! List available models from OpenRouter with optional filtering.
//!
//! ```sh
//! cargo run -p flashmind --example list_models -- --api-key YOUR_KEY
//!
//! # Filter by category:
//! cargo run -p flashmind --example list_models -- --api-key YOUR_KEY --category vision
//!
//! # Search by name:
//! cargo run -p flashmind --example list_models -- --api-key YOUR_KEY --query gemini
//!
//! # List dedicated image generation models:
//! cargo run -p flashmind --example list_models -- --api-key YOUR_KEY --image
//!
//! # List dedicated video generation models:
//! cargo run -p flashmind --example list_models -- --api-key YOUR_KEY --video
//! ```

use std::str::FromStr;
use std::sync::Arc;

use clap::Parser;

use flashmind::llm::{OpenRouterProvider, http::create_rate_limiter};
use flashmind::types::{LlmProvider, ModelCategory};

#[derive(Parser)]
#[command(about = "List available OpenRouter models")]
struct Args {
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    #[arg(
        long,
        help = "Filter by category [chat, reasoning, vision, image_generation, tts, stt, video_generation, video_input]"
    )]
    category: Option<String>,

    #[arg(long, help = "Search models by name or ID (case-insensitive)")]
    query: Option<String>,

    #[arg(long, help = "List dedicated image generation models")]
    image: bool,

    #[arg(long, help = "List dedicated video generation models")]
    video: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let provider = Arc::new(OpenRouterProvider::new(
        args.api_key,
        create_rate_limiter(60),
    ));

    let query_lower = args.query.as_deref().map(|q| q.to_lowercase());

    if args.image || args.video {
        if args.image {
            let models = provider.list_image_models().await?;
            print_simple_models("image", &models, &query_lower);
        }
        if args.video {
            let models = provider.list_video_models().await?;
            print_simple_models("video", &models, &query_lower);
        }
        return Ok(());
    }

    let category_filter = args
        .category
        .as_deref()
        .map(ModelCategory::from_str)
        .transpose()
        .map_err(|_| {
            anyhow::anyhow!(
                "Unknown category. Valid: chat, reasoning, vision, image_generation, tts, stt, video_generation, video_input"
            )
        })?;

    let models = provider
        .list_models()
        .await
        .ok_or_else(|| anyhow::anyhow!("Failed to fetch models"))?;

    let mut count = 0;
    for model in &models {
        if let Some(cat) = category_filter {
            if !model.categories.contains(&cat) {
                continue;
            }
        }
        if let Some(ref q) = query_lower {
            let id = model.id.to_lowercase();
            let name = model.name.as_deref().unwrap_or("").to_lowercase();
            if !id.contains(q.as_str()) && !name.contains(q.as_str()) {
                continue;
            }
        }

        let cats: Vec<_> = model.categories.iter().map(|c| c.to_string()).collect();
        let name = model.name.as_deref().unwrap_or("");

        let mut meta = Vec::new();
        if let Some(ctx) = model.context_length {
            meta.push(format!("{}k ctx", ctx / 1000));
        }
        if let (Some(inp), Some(out)) = (model.pricing.prompt, model.pricing.completion) {
            if inp > 0.0 || out > 0.0 {
                meta.push(format!(
                    "${:.2}/M in, ${:.2}/M out",
                    inp * 1_000_000.0,
                    out * 1_000_000.0
                ));
            }
        }

        let meta_str = if meta.is_empty() {
            String::new()
        } else {
            format!("  ({})", meta.join(" | "))
        };

        println!(
            "openrouter:{}  {}  [{}]{}",
            model.id,
            name,
            cats.join(", "),
            meta_str
        );
        count += 1;
    }

    eprintln!("\n{count} models");
    Ok(())
}

struct SimpleModel<'a> {
    id: &'a str,
    name: Option<&'a str>,
}

fn print_simple_models(label: &str, models: &[impl AsSimpleModel], query: &Option<String>) {
    let mut count = 0;
    for model in models {
        let m = model.as_simple();
        if let Some(q) = query {
            let id = m.id.to_lowercase();
            let name = m.name.unwrap_or("").to_lowercase();
            if !id.contains(q.as_str()) && !name.contains(q.as_str()) {
                continue;
            }
        }
        let name = m.name.unwrap_or("");
        println!("openrouter:{}  {}", m.id, name);
        count += 1;
    }
    eprintln!("\n{count} {label} models");
}

trait AsSimpleModel {
    fn as_simple(&self) -> SimpleModel<'_>;
}

impl AsSimpleModel for flashmind::llm::openrouter::ImageModelInfo {
    fn as_simple(&self) -> SimpleModel<'_> {
        SimpleModel {
            id: &self.id,
            name: self.name.as_deref(),
        }
    }
}

impl AsSimpleModel for flashmind::llm::openrouter::VideoModelInfo {
    fn as_simple(&self) -> SimpleModel<'_> {
        SimpleModel {
            id: &self.id,
            name: self.name.as_deref(),
        }
    }
}
