//! List available models tool.
//!
//! Allows the agent to query available models from configured providers.
//! Supports filtering by provider and category, with pagination and rich metadata output.

use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::llm::LlmProvider;
use flashmind_types::model::Provider;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

pub type ProviderRegistry = Arc<std::collections::HashMap<Provider, Arc<dyn LlmProvider>>>;

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 100;

#[derive(Deserialize)]
struct Args {
    provider: Option<String>,
    category: Option<String>,
    query: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}

pub struct ListModelsTool {
    pub providers: ProviderRegistry,
}

fn format_cost(cost_per_token: f64) -> String {
    let per_million = cost_per_token * 1_000_000.0;
    if per_million < 0.01 {
        format!("${:.4}/M", per_million)
    } else if per_million < 1.0 {
        format!("${:.3}/M", per_million)
    } else {
        format!("${:.2}/M", per_million)
    }
}

fn format_context(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        let m = tokens as f64 / 1_000_000.0;
        if m == m.floor() {
            format!("{}M", m as u32)
        } else {
            format!("{:.1}M", m)
        }
    } else {
        format!("{}k", tokens / 1000)
    }
}

#[async_trait]
impl Tool for ListModelsTool {
    fn name(&self) -> &str {
        "list_models"
    }

    fn description(&self) -> &str {
        "List available models from configured LLM providers. Filter by provider, category, or search query. Returns model IDs, categories, pricing, and context window info. Supports pagination via offset/limit.\n\nCategories: chat, reasoning, vision, image_generation, tts, stt, video_generation, video_input"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Filter by provider (e.g. openrouter, ollama, anthropic, openai). Lists all providers if omitted."
                },
                "category": {
                    "type": "string",
                    "description": "Filter by model category: chat, reasoning, vision, image_generation, tts, stt, video_generation, video_input"
                },
                "query": {
                    "type": "string",
                    "description": "Search models by name or ID (case-insensitive substring match)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many matching models (for pagination). Default: 0"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of models to return (1-100). Default: 50"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: Args = ctx.parse_args(self.name())?;
        let offset = args.offset.unwrap_or(0);
        let limit = args.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

        let providers_to_query: Vec<(&Provider, &Arc<dyn LlmProvider>)> =
            if let Some(ref name) = args.provider {
                let provider = match Provider::from_str(name) {
                    Ok(p) => p,
                    Err(_) => {
                        let available: Vec<String> =
                            self.providers.keys().map(|p| p.to_string()).collect();
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!(
                                "Unknown provider '{}'. Available: {}",
                                name,
                                available.join(", ")
                            ),
                        ));
                    }
                };

                match self.providers.get_key_value(&provider) {
                    Some(pair) => vec![pair],
                    None => {
                        let available: Vec<String> =
                            self.providers.keys().map(|p| p.to_string()).collect();
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!(
                                "Provider '{}' is not configured. Available: {}",
                                name,
                                available.join(", ")
                            ),
                        ));
                    }
                }
            } else {
                self.providers.iter().collect()
            };

        let category_filter = if let Some(ref cat) = args.category {
            use flashmind_types::llm::ModelCategory;
            match ModelCategory::from_str(cat) {
                Ok(c) => Some(c),
                Err(_) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!(
                            "Unknown category '{}'. Available: chat, reasoning, vision, image_generation, tts, stt, video_generation, video_input",
                            cat
                        ),
                    ));
                }
            }
        } else {
            None
        };

        let query_lower = args.query.as_deref().map(|q| q.to_lowercase());

        let mut lines = Vec::new();
        let mut matched = 0usize;
        let mut returned = 0usize;
        let mut total_available = 0usize;

        for (provider, llm) in &providers_to_query {
            match llm.list_models().await {
                Some(models) => {
                    debug!(provider = %provider, count = models.len(), "listed models");

                    for model in &models {
                        if let Some(cat) = category_filter
                            && !model.categories.contains(&cat)
                        {
                            continue;
                        }

                        if let Some(ref q) = query_lower {
                            let id_lower = model.id.to_lowercase();
                            let name_lower = model
                                .name
                                .as_deref()
                                .map(|n| n.to_lowercase())
                                .unwrap_or_default();
                            if !id_lower.contains(q.as_str()) && !name_lower.contains(q.as_str()) {
                                continue;
                            }
                        }

                        total_available += 1;

                        if matched < offset {
                            matched += 1;
                            continue;
                        }

                        if returned >= limit {
                            matched += 1;
                            continue;
                        }

                        matched += 1;

                        let cats: Vec<String> =
                            model.categories.iter().map(|c| c.to_string()).collect();

                        let mut parts = vec![format!("{}:{}", provider, model.id)];

                        if let Some(ref name) = model.name {
                            parts.push(format!("  {}", name));
                        }

                        let mut meta = Vec::new();

                        if let Some(ctx_len) = model.context_length {
                            meta.push(format!("{} ctx", format_context(ctx_len)));
                        }
                        if let Some(max_out) = model.max_completion_tokens {
                            meta.push(format!("{} max out", format_context(max_out)));
                        }

                        let p = &model.pricing;
                        if let (Some(inp), Some(out)) = (p.prompt, p.completion)
                            && (inp > 0.0 || out > 0.0)
                        {
                            meta.push(format!(
                                "in:{} out:{}",
                                format_cost(inp),
                                format_cost(out)
                            ));
                        }
                        if let Some(img_cost) = p.image && img_cost > 0.0 {
                            let per_image = img_cost * 1_000_000.0;
                            meta.push(format!("${:.2}/img", per_image));
                        }

                        if !meta.is_empty() {
                            parts.push(format!("  {}", meta.join(" | ")));
                        }

                        parts.push(format!("  [{}]", cats.join(", ")));

                        lines.push(parts.join("\n"));
                        returned += 1;
                    }
                }
                None => {
                    lines.push(format!("{}: model listing not supported", provider));
                }
            }
        }

        if lines.is_empty() {
            let msg = if category_filter.is_some() || query_lower.is_some() {
                "No models found matching the given filters."
            } else {
                "No models found."
            };
            return Ok(ToolResult::success(ctx.tool_call_id, msg.to_string()));
        }

        let remaining = total_available.saturating_sub(offset + returned);
        if remaining > 0 {
            lines.push(format!(
                "--- Showing {}-{} of {} matches ({} more available, use offset: {})",
                offset + 1,
                offset + returned,
                total_available,
                remaining,
                offset + returned
            ));
        } else {
            lines.push(format!(
                "--- Showing {}-{} of {} matches",
                offset + 1,
                offset + returned,
                total_available
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, lines.join("\n\n")))
    }

    fn humanize(&self, args: &Value) -> String {
        let mut parts = Vec::new();
        if let Some(p) = args.get("provider").and_then(|v| v.as_str()) {
            parts.push(format!("from {}", p));
        }
        if let Some(c) = args.get("category").and_then(|v| v.as_str()) {
            parts.push(format!("category={}", c));
        }
        if let Some(q) = args.get("query").and_then(|v| v.as_str()) {
            parts.push(format!("matching '{}'", q));
        }
        if parts.is_empty() {
            "Listing models from all providers".to_string()
        } else {
            format!("Listing models {}", parts.join(", "))
        }
    }
}
