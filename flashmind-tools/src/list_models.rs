//! List available models tool.
//!
//! Allows the agent to query available models from configured providers.
//! Optionally filters by provider. Returns up to 50 results.

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

const MAX_RESULTS: usize = 50;

#[derive(Deserialize)]
struct Args {
    provider: Option<String>,
}

pub struct ListModelsTool {
    pub providers: ProviderRegistry,
}

#[async_trait]
impl Tool for ListModelsTool {
    fn name(&self) -> &str {
        "list_models"
    }

    fn description(&self) -> &str {
        "List available models from configured LLM providers. Optionally filter by provider name. Returns up to 50 models."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Filter by provider (e.g. openrouter, ollama, anthropic, openai). Lists all providers if omitted."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: Args = ctx.parse_args(self.name())?;

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

        let mut lines = Vec::new();
        let mut total = 0usize;

        for (provider, llm) in &providers_to_query {
            match llm.list_models().await {
                Some(models) => {
                    debug!(provider = %provider, count = models.len(), "listed models");

                    for model in &models {
                        if total >= MAX_RESULTS {
                            break;
                        }

                        let mut caps = Vec::new();
                        if model.capabilities.tool_calling {
                            caps.push("tools");
                        }
                        if model.capabilities.images {
                            caps.push("vision");
                        }
                        if model.capabilities.reasoning {
                            caps.push("reasoning");
                        }
                        if model.capabilities.image_generation {
                            caps.push("image_gen");
                        }
                        if model.capabilities.video_generation {
                            caps.push("video_gen");
                        }
                        if model.capabilities.audio_output {
                            caps.push("audio_out");
                        }

                        let ctx_str = model
                            .context_length
                            .map(|c| format!(" ({}k ctx)", c / 1000))
                            .unwrap_or_default();

                        let caps_str = if caps.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", caps.join(", "))
                        };

                        lines.push(format!("{}:{}{}{}", provider, model.id, ctx_str, caps_str));
                        total += 1;
                    }
                }
                None => {
                    lines.push(format!("{}: model listing not supported", provider));
                }
            }

            if total >= MAX_RESULTS {
                lines.push(format!("... truncated at {} models", MAX_RESULTS));
                break;
            }
        }

        if lines.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No models found.".to_string(),
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, lines.join("\n")))
    }

    fn humanize(&self, args: &Value) -> String {
        match args.get("provider").and_then(|v| v.as_str()) {
            Some(p) => format!("Listing models from {}", p),
            None => "Listing models from all providers".to_string(),
        }
    }
}
