use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

use flashmind_memory::{EmbeddingProviderConfig, MemoryStore, create_embedding_provider};
use flashmind_tools::{Tool, ToolContext, ToolResult};
use flashmind_types::memory::{MemoryMetadata, MemoryProvider};

use crate::config::{Config, config_dir};

// ---------------------------------------------------------------------------
// Memory tools
// ---------------------------------------------------------------------------

pub struct MemoryStoreTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryStoreTool {
    fn name(&self) -> &str {
        "memory_store"
    }

    fn description(&self) -> &str {
        "Store a fact in long-term memory for future conversations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or information to remember."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags for categorization (e.g. \"preference\", \"project\")."
                },
                "context": {
                    "type": "string",
                    "description": "Brief context about why this is being stored."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let content: String = ctx
            .args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if content.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "content is required"));
        }

        let tags: Vec<String> = ctx
            .args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let context = ctx
            .args
            .get("context")
            .and_then(|v| v.as_str())
            .map(String::from);

        let meta = MemoryMetadata {
            context,
            tags,
            expires_at: None,
        };

        let id = MemoryProvider::store(self.store.as_ref(), &content, meta).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Stored memory {id}"),
        ))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("…");
        let preview: String = content.chars().take(60).collect();
        format!("remember: {preview}")
    }
}

struct MemoryRecallTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }

    fn description(&self) -> &str {
        "Search long-term memory for relevant facts from previous conversations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let query = ctx.args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "query is required"));
        }

        let limit = ctx.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let results = MemoryProvider::search(self.store.as_ref(), query, limit).await?;

        if results.is_empty() {
            return Ok(ToolResult::success(ctx.tool_call_id, "No memories found."));
        }

        let mut out = String::new();
        for entry in &results {
            out.push_str(&format!(
                "[{}] (score: {:.2}) {}\n",
                entry.id, entry.score, entry.content
            ));
            if !entry.metadata.tags.is_empty() {
                out.push_str(&format!("  tags: {}\n", entry.metadata.tags.join(", ")));
            }
        }
        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("…");
        format!("recall: {query}")
    }
}

struct MemoryForgetTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryForgetTool {
    fn name(&self) -> &str {
        "memory_forget"
    }

    fn description(&self) -> &str {
        "Remove a specific memory by ID."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The memory ID to forget (from memory_recall results)."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let id = ctx.args.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if id.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "id is required"));
        }

        MemoryProvider::forget(self.store.as_ref(), id).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Forgot memory {id}"),
        ))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("…");
        format!("forget: {id}")
    }
}

// ---------------------------------------------------------------------------
// Memory store setup
// ---------------------------------------------------------------------------

fn build_embedding_config(config: &Config) -> Option<EmbeddingProviderConfig> {
    let provider = config.memory_provider.as_deref()?;
    match provider {
        "openrouter" => {
            let model = config
                .memory_model
                .clone()
                .unwrap_or_else(|| "openai/text-embedding-3-small".into());
            Some(EmbeddingProviderConfig::OpenRouter {
                api_key: config.openrouter_api_key.clone(),
                model,
            })
        }
        "openai" => {
            let model = config
                .memory_model
                .clone()
                .unwrap_or_else(|| "text-embedding-3-small".into());
            Some(EmbeddingProviderConfig::OpenAI {
                api_key: config.openai_api_key.clone(),
                model,
                base_url: config.openai_base_url.clone(),
            })
        }
        _ => None,
    }
}

pub async fn open_memory_store(config: &Config) -> Result<Option<Arc<MemoryStore>>> {
    let Some(embed_config) = build_embedding_config(config) else {
        return Ok(None);
    };

    let embedder = create_embedding_provider(&embed_config, None)
        .context("failed to create embedding provider for memory")?;

    let db_path = config_dir()?.join("memory.db");
    let store = MemoryStore::connect(&db_path, embedder)
        .await
        .context("failed to open memory store")?;

    Ok(Some(Arc::new(store)))
}

pub fn register_memory_tools(
    registry: &mut flashmind_types::ToolRegistry,
    store: &Arc<MemoryStore>,
) {
    registry.register(Arc::new(MemoryStoreTool {
        store: Arc::clone(store),
    }));
    registry.register(Arc::new(MemoryRecallTool {
        store: Arc::clone(store),
    }));
    registry.register(Arc::new(MemoryForgetTool {
        store: Arc::clone(store),
    }));
}
