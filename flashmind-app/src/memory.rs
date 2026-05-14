//! Memory tools for autonomous agent-driven long-term memory.
//!
//! The agent calls these tools on its own to store and recall information
//! across sessions. Uses SQLite + sqlite-vec + FTS5 for hybrid search.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_memory::embeddings::EmbeddingProvider;
use flashmind_memory::{DbStore, Scope, Source, Tag};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

fn parse_date_to_epoch(s: &str) -> Option<i64> {
    use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc};
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&dt).timestamp());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M") {
        return Some(Utc.from_utc_datetime(&dt).timestamp());
    }
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    Some(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0)?).timestamp())
}

fn parse_before_to_epoch(s: &str) -> Option<i64> {
    use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc};
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&dt).timestamp());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M") {
        return Some(Utc.from_utc_datetime(&dt).timestamp());
    }
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    let next = d.succ_opt()?;
    Some(
        Utc.from_utc_datetime(&next.and_hms_opt(0, 0, 0)?)
            .timestamp(),
    )
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s
            .char_indices()
            .nth(max_len)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        format!("{}...", &s[..end])
    }
}

// ---------------------------------------------------------------------------
// Shared deps
// ---------------------------------------------------------------------------

type EmbedderRef = Arc<dyn EmbeddingProvider>;

/// Register all memory tools into a tool registry.
pub fn register_tools(
    registry: &mut flashmind_types::tool::ToolRegistry,
    db: DbStore,
    embedder: EmbedderRef,
) {
    registry.register(Arc::new(MemoryStoreTool {
        db: db.clone(),
        embedder: embedder.clone(),
    }));
    registry.register(Arc::new(MemoryRecallTool {
        db: db.clone(),
        embedder: embedder.clone(),
    }));
    registry.register(Arc::new(MemoryForgetTool { db: db.clone() }));
    registry.register(Arc::new(MemoryEditTool {
        db: db.clone(),
        embedder,
    }));
    registry.register(Arc::new(MemoryListTool { db }));
}

// ============================================================================
// MemoryStoreTool
// ============================================================================

#[derive(Deserialize)]
struct StoreArgs {
    content: String,
    scope: Option<Scope>,
    ttl_hours: Option<u64>,
    #[serde(default)]
    tags: Vec<Tag>,
    tool_name: Option<String>,
}

struct MemoryStoreTool {
    db: DbStore,
    embedder: EmbedderRef,
}

#[async_trait]
impl Tool for MemoryStoreTool {
    fn name(&self) -> &str {
        "memory_store"
    }

    fn description(&self) -> &str {
        "Store information in long-term memory. Use this after meaningful exchanges \
         to save facts, preferences, decisions, or context worth recalling later. \
         Always tag memories: 'fact' for concrete info, 'preference' for how the user \
         likes things done, 'tool' + tool_name for tool configs, 'project' for project \
         context, 'episode' for time-bound events, 'user-profile' for user summaries."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be specific — include context, not just the raw fact."
                },
                "scope": {
                    "type": "string",
                    "enum": ["local", "global"],
                    "description": "'local' (default): scoped to current session context. 'global': available across all sessions."
                },
                "ttl_hours": {
                    "type": "integer",
                    "description": "Time-to-live in hours. Memory auto-expires after this."
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["fact", "preference", "tool", "project", "episode", "user-profile"]
                    },
                    "description": "Categorization tags. Include at least one."
                },
                "tool_name": {
                    "type": "string",
                    "description": "When tags includes 'tool', specify the tool name."
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StoreArgs = ctx.parse_args(self.name())?;
        let scope = args.scope.unwrap_or(Scope::Local);
        let chat_key: Option<String> = match scope {
            Scope::Global => None,
            Scope::Local => Some("cli".to_string()),
        };

        let expires_at = args
            .ttl_hours
            .map(|h| (chrono::Utc::now() + chrono::Duration::hours(h as i64)).timestamp());

        let embedding = self
            .embedder
            .embed(&args.content)
            .await
            .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;

        // Dedup check
        let similar = self
            .db
            .find_similar(embedding.clone(), chat_key.as_deref(), Some(scope), 0.92, 1)
            .await
            .unwrap_or_default();
        if let Some((existing_id, score)) = similar.first() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Already known (id: {}, similarity: {:.0}%)",
                    existing_id.get(..8).unwrap_or(existing_id),
                    score * 100.0
                ),
            ));
        }

        match self
            .db
            .store(
                &args.content,
                embedding,
                Source::Manual,
                chat_key.as_deref(),
                None,
                &args.tags,
                expires_at,
                args.tool_name.as_deref(),
            )
            .await
        {
            Ok(id) => {
                let short = id.get(..8).unwrap_or(&id);
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("Stored memory (id: {short})"),
                ))
            }
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to store: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
        format!("Remembering \"{}\"", truncate(content, 40))
    }
}

// ============================================================================
// MemoryRecallTool
// ============================================================================

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
    limit: Option<u64>,
    tag: Option<Tag>,
    scope: Option<Scope>,
    after: Option<String>,
    before: Option<String>,
}

struct MemoryRecallTool {
    db: DbStore,
    embedder: EmbedderRef,
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }

    fn description(&self) -> &str {
        "Search long-term memory. Use at the START of conversations to check for \
         relevant context, and when the user references prior conversations."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — memories matching this text will be returned"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 10)"
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "local"],
                    "description": "Filter by scope. Omit to search both."
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by tag (e.g., 'tool', 'preference', 'fact')"
                },
                "after": {
                    "type": "string",
                    "description": "Only memories created on or after this date (YYYY-MM-DD)"
                },
                "before": {
                    "type": "string",
                    "description": "Only memories created on or before this date (YYYY-MM-DD)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: RecallArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(10) as usize;

        let query_embedding = self
            .embedder
            .embed(&args.query)
            .await
            .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;

        let chat_key = Some("cli".to_string());

        let results = if args.tag == Some(Tag::Tool) {
            self.db
                .search_for_tool(
                    query_embedding,
                    &args.query,
                    None,
                    limit,
                    chat_key.as_deref(),
                )
                .await
        } else {
            match args.scope {
                Some(Scope::Global) => {
                    self.db
                        .search_hybrid(
                            query_embedding,
                            &args.query,
                            limit,
                            None,
                            Some(Scope::Global),
                        )
                        .await
                }
                Some(Scope::Local) => {
                    self.db
                        .search_hybrid(
                            query_embedding,
                            &args.query,
                            limit,
                            chat_key.as_deref(),
                            Some(Scope::Local),
                        )
                        .await
                }
                None => {
                    self.db
                        .search_multi_context_hybrid(
                            query_embedding,
                            &args.query,
                            limit,
                            chat_key.as_deref(),
                        )
                        .await
                }
            }
        };

        let after_ts = args.after.as_deref().and_then(parse_date_to_epoch);
        let before_ts = args.before.as_deref().and_then(parse_before_to_epoch);

        match results {
            Ok(results) if !results.is_empty() => {
                let results: Vec<_> = results
                    .into_iter()
                    .filter(|r| {
                        if let Some(a) = after_ts
                            && r.created_at < a
                        {
                            return false;
                        }
                        if let Some(b) = before_ts
                            && r.created_at >= b
                        {
                            return false;
                        }
                        true
                    })
                    .collect();

                if results.is_empty() {
                    return Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!("No memories found matching '{}' in date range", args.query),
                    ));
                }

                let mut output = format!("Found {} memories:\n\n", results.len());
                for (i, r) in results.iter().enumerate() {
                    let date = chrono::DateTime::from_timestamp(r.created_at, 0)
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let scope_tag = if r.chat_key.is_none() {
                        " [global]"
                    } else {
                        ""
                    };
                    let id_tag =
                        r.id.as_deref()
                            .map(|id| format!(" ({})", id.get(..8).unwrap_or(id)))
                            .unwrap_or_default();
                    output.push_str(&format!(
                        "{}. [{}]{}{}\n   {}\n\n",
                        i + 1,
                        date,
                        id_tag,
                        scope_tag,
                        r.content
                    ));
                }
                Ok(ToolResult::success(ctx.tool_call_id, output))
            }
            _ => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("No memories found matching '{}'", args.query),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        format!("Recalling '{query}'")
    }
}

// ============================================================================
// MemoryForgetTool
// ============================================================================

#[derive(Deserialize)]
struct ForgetArgs {
    id: String,
}

struct MemoryForgetTool {
    db: DbStore,
}

#[async_trait]
impl Tool for MemoryForgetTool {
    fn name(&self) -> &str {
        "memory_forget"
    }

    fn description(&self) -> &str {
        "Remove a memory by ID. Use to delete outdated or incorrect information."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID (first 8 chars or full UUID)"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ForgetArgs = ctx.parse_args(self.name())?;
        match self.db.delete(&args.id).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Removed memory {}", args.id),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to delete: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Forgetting memory {id}")
    }
}

// ============================================================================
// MemoryEditTool
// ============================================================================

#[derive(Deserialize)]
struct EditArgs {
    id: String,
    content: Option<String>,
    tags: Option<Vec<String>>,
    scope: Option<Scope>,
    expires_at: Option<i64>,
}

struct MemoryEditTool {
    db: DbStore,
    embedder: EmbedderRef,
}

#[async_trait]
impl Tool for MemoryEditTool {
    fn name(&self) -> &str {
        "memory_edit"
    }

    fn description(&self) -> &str {
        "Edit an existing memory's content, tags, scope, or expiry. Preserves ID and timestamps."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to edit"
                },
                "content": {
                    "type": "string",
                    "description": "New content (optional)"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "New tags (replaces existing)"
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "local"],
                    "description": "Change scope"
                },
                "expires_at": {
                    "type": "integer",
                    "description": "Unix timestamp for expiry. 0 to remove expiry."
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: EditArgs = ctx.parse_args(self.name())?;

        let embedding = if let Some(content) = args.content.as_ref() {
            Some(
                self.embedder
                    .embed(content)
                    .await
                    .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?,
            )
        } else {
            None
        };

        let tags: Option<Vec<Tag>> = args
            .tags
            .as_ref()
            .map(|strs| strs.iter().filter_map(|s| s.parse().ok()).collect());

        let chat_key_update: Option<Option<String>> = match args.scope {
            Some(Scope::Global) => Some(None),
            Some(Scope::Local) => Some(Some("cli".to_string())),
            None => None,
        };

        match self
            .db
            .update(
                &args.id,
                args.content.as_deref(),
                embedding,
                tags.as_deref(),
                chat_key_update,
                args.expires_at.map(|v| if v == 0 { None } else { Some(v) }),
            )
            .await
        {
            Ok(full_id) => {
                let preview = args
                    .content
                    .as_ref()
                    .map(|c| truncate(c, 50))
                    .unwrap_or_else(|| "[unchanged]".to_string());
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!(
                        "Updated memory ({}): \"{}\"",
                        full_id.get(..8).unwrap_or(&full_id),
                        preview
                    ),
                ))
            }
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to update: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Editing memory {id}")
    }
}

// ============================================================================
// MemoryListTool
// ============================================================================

#[derive(Deserialize)]
struct ListArgs {
    limit: Option<u64>,
    cursor: Option<String>,
    scope: Option<Scope>,
    after: Option<String>,
    before: Option<String>,
}

struct MemoryListTool {
    db: DbStore,
}

#[async_trait]
impl Tool for MemoryListTool {
    fn name(&self) -> &str {
        "memory_list"
    }

    fn description(&self) -> &str {
        "List stored memories, newest first. Use cursor from last result for pagination."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 20)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from previous page"
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "local"],
                    "description": "Filter by scope"
                },
                "after": {
                    "type": "string",
                    "description": "Only memories after this date (YYYY-MM-DD)"
                },
                "before": {
                    "type": "string",
                    "description": "Only memories before this date (YYYY-MM-DD)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(20) as usize;

        let scope_filter: Option<String> = match args.scope {
            Some(Scope::Global) => Some("chat_key IS NULL".to_string()),
            Some(Scope::Local) => Some("chat_key = 'cli'".to_string()),
            None => None,
        };

        let cursor_i64: Option<i64> = args.cursor.as_deref().and_then(|c| c.parse().ok());
        let after_ts = args.after.as_deref().and_then(parse_date_to_epoch);
        let before_ts = args.before.as_deref().and_then(parse_before_to_epoch);

        let mut memories = self
            .db
            .list_all_dated(
                limit + 1,
                cursor_i64,
                scope_filter.as_deref(),
                after_ts,
                before_ts,
            )
            .await?;
        let has_more = memories.len() > limit;
        memories.truncate(limit);

        if memories.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No memories found.".to_string(),
            ));
        }

        let mut output = format!("Memories ({} results):\n\n", memories.len());
        for (i, mem) in memories.iter().enumerate() {
            let id_preview = mem.id.get(..8).unwrap_or(&mem.id);
            let date = chrono::DateTime::from_timestamp(mem.created_at, 0)
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let scope_tag = if mem.chat_key.is_none() {
                " [global]"
            } else {
                ""
            };

            let mut line = format!(
                "{}. [{}] {}{}\n   {}",
                i + 1,
                date,
                id_preview,
                scope_tag,
                mem.content
            );
            if !mem.tags.is_empty() {
                let tags_str: Vec<&str> = mem.tags.iter().map(|t| t.into()).collect();
                line.push_str(&format!(" [{}]", tags_str.join(",")));
            }
            output.push_str(&line);
            output.push_str("\n\n");
        }

        if has_more {
            let next = memories
                .last()
                .map(|m| m.created_at.to_string())
                .unwrap_or_default();
            output.push_str(&format!("More available. Next page: cursor=\"{next}\"\n"));
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let limit = args.get("limit").and_then(|v| v.as_i64());
        match limit {
            Some(n) => format!("Listing {n} memories"),
            None => "Listing memories".to_string(),
        }
    }
}
