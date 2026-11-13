//! List available models tool.
//!
//! Allows the agent to query available models from configured providers.
//! Supports filtering by provider and category, with pagination and rich metadata output.

use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::llm::{ModelCategory, ModelInfo};
use flashmind_types::model::Provider;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

pub use flashmind_types::llm::ProviderRegistry;

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

/// Available model listing tool.
pub struct ListModelsTool {
    pub providers: ProviderRegistry,
}

/// Query configured catalogs live and return IDs in `provider:model` form.
///
/// `None` selects all categories or providers. The query matches names and fully
/// qualified IDs case-insensitively. Providers without listing support are skipped.
/// Meshy and Tripo use known catalogs because they have no model-listing endpoint.
/// Returns an error if the selected provider is not configured.
pub async fn list_models(
    providers: &ProviderRegistry,
    category: Option<ModelCategory>,
    provider: Option<&Provider>,
    query: Option<&str>,
) -> anyhow::Result<Vec<ModelInfo>> {
    if let Some(provider) = provider {
        anyhow::ensure!(
            providers.contains_key(provider),
            "Provider '{provider}' is not configured"
        );
    }
    let query = query.map(str::to_lowercase);
    let mut selected: Vec<_> = providers
        .iter()
        .filter(|(id, _)| provider.is_none_or(|provider| provider == *id))
        .collect();
    selected.sort_by_key(|(id, _)| id.to_string());
    let mut result = Vec::new();
    for (id, client) in selected {
        let Some(models) = client.list_models().await else {
            continue;
        };
        for mut model in models {
            if category.is_some_and(|category| !model.categories.contains(&category)) {
                continue;
            }
            model.id = format!("{id}:{}", model.id);
            if query.as_ref().is_some_and(|query| {
                !model.id.to_lowercase().contains(query)
                    && !model
                        .name
                        .as_deref()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(query)
            }) {
                continue;
            }
            result.push(model);
        }
    }
    Ok(result)
}

fn format_cost(cost_per_token: rust_decimal::Decimal) -> String {
    let per_million = cost_per_token * rust_decimal::Decimal::from(1_000_000);
    if per_million < rust_decimal::Decimal::new(1, 2) {
        format!("${:.4}/M", per_million)
    } else if per_million < rust_decimal::Decimal::ONE {
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
        "Search the current provider's model catalog. Filter by category or name/ID; use offset/limit for pagination. Returns provider:model IDs for media tools and delegate's model override, capabilities, pricing, and context windows. Endpoint availability still depends on provider routing settings. An explicit provider can select another configured catalog.\n\nCategories: chat, reasoning, vision, image_generation, tts, stt, video_generation, image_to_video, video_input, image_editing, 3d"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Provider to query. Defaults to the current agent's provider, or all configured providers outside an agent."
                },
                "category": {
                    "type": "string",
                    "description": "Filter by model category: chat, reasoning, vision, image_generation, tts, stt, video_generation, image_to_video, video_input, image_editing, 3d"
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
        let mut providers = self.providers.as_ref().clone();
        if let Some(current) = &ctx.provider {
            if args.provider.is_none() {
                providers.clear();
            }
            providers.insert(current.provider(), current.clone());
        }

        let provider_filter = args
            .provider
            .as_deref()
            .map(Provider::from_str)
            .transpose()?;

        let category_filter = if let Some(ref cat) = args.category {
            use flashmind_types::llm::ModelCategory;
            match ModelCategory::from_str(cat) {
                Ok(c) => Some(c),
                Err(_) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!(
                            "Unknown category '{}'. Available: chat, reasoning, vision, image_generation, tts, stt, video_generation, image_to_video, video_input, image_editing, 3d",
                            cat
                        ),
                    ));
                }
            }
        } else {
            None
        };

        let models = match list_models(
            &Arc::new(providers),
            category_filter,
            provider_filter.as_ref(),
            args.query.as_deref(),
        )
        .await
        {
            Ok(models) => models,
            Err(error) => return Ok(ToolResult::failure(ctx.tool_call_id, error.to_string())),
        };
        let total_available = models.len();
        let mut lines = Vec::new();
        let mut returned = 0;
        for model in models.iter().skip(offset).take(limit) {
            let cats: Vec<String> = model.categories.iter().map(|c| c.to_string()).collect();

            let mut parts = vec![model.id.clone()];

            if let Some(ref name) = model.name {
                parts.push(format!("  {}", name));
            }

            let mut meta = Vec::new();
            meta.push(format!("tools: {}", model.capabilities.tool_calling));
            if let Some(created) = model.created_at {
                meta.push(format!("released {}", created.format("%Y-%m-%d")));
            }

            if let Some(ctx_len) = model.context_length {
                meta.push(format!("{} ctx", format_context(ctx_len)));
            }
            if let Some(max_out) = model.max_completion_tokens {
                meta.push(format!("{} max out", format_context(max_out)));
            }

            let p = &model.pricing;
            if let (Some(inp), Some(out)) = (p.prompt, p.completion) {
                meta.push(format!("in:{} out:{}", format_cost(inp), format_cost(out)));
            }
            if let Some(img_cost) = p.image
                && !img_cost.is_zero()
            {
                let per_image = img_cost * rust_decimal::Decimal::from(1_000_000);
                meta.push(format!("${:.2}/img", per_image));
            }

            if !meta.is_empty() {
                parts.push(format!("  {}", meta.join(" | ")));
            }

            parts.push(format!("  [{}]", cats.join(", ")));

            lines.push(parts.join("\n"));
            returned += 1;
        }

        if lines.is_empty() {
            let msg = if category_filter.is_some() || args.query.is_some() {
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
            "Listing models".to_string()
        } else {
            format!("Listing models {}", parts.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use flashmind_types::LlmProvider;
    use std::collections::HashMap;

    use flashmind_types::{
        CompletionRequest, CompletionStream, ModelCapabilities, ModelCategory, ModelInfo,
        ModelPricing,
    };
    use rust_decimal::Decimal;
    use tokio_util::sync::CancellationToken;

    use super::*;

    struct CatalogProvider {
        provider: Provider,
        models: Vec<ModelInfo>,
    }

    #[async_trait]
    impl LlmProvider for CatalogProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            self.provider.clone()
        }
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::empty())
        }
        async fn list_models(&self) -> Option<Vec<ModelInfo>> {
            Some(self.models.clone())
        }
    }

    fn catalog(provider: Provider, ids: &[&str]) -> Arc<dyn LlmProvider> {
        Arc::new(CatalogProvider {
            provider,
            models: ids
                .iter()
                .map(|id| ModelInfo {
                    image: Default::default(),
                    id: (*id).to_string(),
                    name: Some(format!("Example {id}")),
                    created_at: None,
                    context_length: Some(128_000),
                    max_completion_tokens: Some(4096),
                    capabilities: ModelCapabilities {
                        tool_calling: true,
                        ..Default::default()
                    },
                    categories: vec![ModelCategory::Chat],
                    pricing: ModelPricing {
                        prompt: Some(Decimal::ZERO),
                        completion: Some(Decimal::ZERO),
                        ..Default::default()
                    },
                })
                .collect(),
        })
    }

    #[tokio::test]
    async fn helper_filters_live_catalogs_and_qualifies_ids() {
        let mut image = catalog(Provider::OpenAi, &["example-image"])
            .list_models()
            .await
            .unwrap()
            .remove(0);
        image.categories = vec![ModelCategory::ImageGeneration];
        let providers: ProviderRegistry = Arc::new(HashMap::from([
            (
                Provider::OpenAi,
                Arc::new(CatalogProvider {
                    provider: Provider::OpenAi,
                    models: vec![image],
                }) as Arc<dyn LlmProvider>,
            ),
            (
                Provider::OpenRouter,
                catalog(Provider::OpenRouter, &["chat-one", "chat-two"]),
            ),
        ]));
        for (category, provider, query, ids) in [
            (
                Some(ModelCategory::ImageGeneration),
                None,
                None,
                vec!["openai:example-image"],
            ),
            (
                Some(ModelCategory::Chat),
                None,
                Some("TWO"),
                vec!["openrouter:chat-two"],
            ),
            (
                None,
                Some(&Provider::OpenAi),
                Some("EXAMPLE"),
                vec!["openai:example-image"],
            ),
            (None, None, Some("OPENAI:"), vec!["openai:example-image"]),
            (Some(ModelCategory::ThreeD), None, None, vec![]),
            (None, None, Some("missing"), vec![]),
        ] {
            let models = list_models(&providers, category, provider, query)
                .await
                .unwrap();
            assert_eq!(
                models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
                ids
            );
            for model in models {
                let parsed: flashmind_types::Model = model.id.parse().unwrap();
                assert_eq!(parsed.to_string(), model.id);
            }
        }
        assert!(
            list_models(&providers, None, Some(&Provider::Mistral), None)
                .await
                .is_err()
        );

        struct LiveCatalog(std::sync::atomic::AtomicUsize);
        #[async_trait]
        impl LlmProvider for LiveCatalog {
            fn name(&self) -> &str {
                "live"
            }
            fn provider(&self) -> Provider {
                Provider::OpenAi
            }
            fn complete(&self, _: CompletionRequest) -> CompletionStream {
                Box::pin(futures_util::stream::empty())
            }
            async fn list_models(&self) -> Option<Vec<ModelInfo>> {
                let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                catalog(Provider::OpenAi, &[&format!("call-{call}")])
                    .list_models()
                    .await
            }
        }
        let providers: ProviderRegistry = Arc::new(HashMap::from([(
            Provider::OpenAi,
            Arc::new(LiveCatalog(Default::default())) as Arc<dyn LlmProvider>,
        )]));
        for call in 0..2 {
            assert_eq!(
                list_models(&providers, None, None, None).await.unwrap()[0].id,
                format!("openai:call-{call}")
            );
        }
    }

    #[tokio::test]
    async fn listing_follows_current_provider_instead_of_stale_registry() {
        let tool = ListModelsTool {
            providers: Arc::new(HashMap::from([(
                Provider::OpenRouter,
                catalog(Provider::OpenRouter, &["stale"]),
            )])),
        };
        let cancel = CancellationToken::new();
        for provider in [Provider::OpenRouter, Provider::Ollama] {
            for args in [json!({}), json!({"provider": provider.to_string()})] {
                let result = tool
                    .execute(
                        ToolContext::new("list", args, None, &cancel)
                            .with_provider(catalog(provider.clone(), &["current"])),
                    )
                    .await
                    .unwrap();
                assert!(result.is_success());
                assert!(result.output().contains(&format!("{provider}:current")));
                assert!(!result.output().contains("stale"));
            }
        }
    }

    #[tokio::test]
    async fn current_catalog_supports_search_pagination_and_selection_metadata() {
        let tool = ListModelsTool {
            providers: Arc::new(HashMap::new()),
        };
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                ToolContext::new(
                    "list",
                    json!({
                        "query": "WORKER", "category": "chat", "offset": 1, "limit": 1
                    }),
                    None,
                    &cancel,
                )
                .with_provider(catalog(
                    Provider::OpenRouter,
                    &["worker-one", "unrelated", "worker-two", "worker-three"],
                )),
            )
            .await
            .unwrap();
        let output = result.output();
        assert!(result.is_success());
        assert!(output.contains("openrouter:worker-two"));
        assert!(!output.contains("worker-one"));
        assert!(!output.contains("worker-three"));
        assert!(output.contains("Showing 2-2 of 3 matches"));
        assert!(output.contains("tools: true"));
        assert!(output.contains("128k ctx"));
        assert!(output.contains("in:$0.0000/M out:$0.0000/M"));
    }
}
