//! Redis query and info tools.
//!
//! Provides two tools for interacting with Redis servers:
//! - `redis_query` — execute arbitrary Redis commands (with optional read-only gating)
//! - `redis_info` — retrieve server information via the `INFO` command

use async_trait::async_trait;
use redis::Value as RedisValue;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Redis server.
pub struct RedisConfig {
    /// Redis connection URL, e.g. `redis://host:port` or `redis://:password@host:port/db`.
    pub url: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Commands allowed in read-only mode.
const READONLY_COMMANDS: &[&str] = &[
    "GET",
    "MGET",
    "KEYS",
    "SCAN",
    "EXISTS",
    "TYPE",
    "TTL",
    "PTTL",
    "STRLEN",
    "LLEN",
    "LRANGE",
    "LINDEX",
    "SCARD",
    "SMEMBERS",
    "SISMEMBER",
    "HGET",
    "HGETALL",
    "HKEYS",
    "HVALS",
    "HLEN",
    "HEXISTS",
    "ZCARD",
    "ZRANGE",
    "ZRANGEBYSCORE",
    "ZRANK",
    "ZSCORE",
    "DBSIZE",
    "INFO",
    "PING",
    "ECHO",
    "SELECT",
    "XLEN",
    "XRANGE",
    "XINFO",
    "OBJECT",
    "RANDOMKEY",
    "DUMP",
];

/// Returns `true` when `cmd` is in the read-only allowlist (case-insensitive).
fn is_readonly_command(cmd: &str) -> bool {
    let upper = cmd.to_ascii_uppercase();
    READONLY_COMMANDS.iter().any(|&c| c == upper)
}

/// Parse a command string into tokens, respecting double-quoted and
/// single-quoted substrings.
fn parse_command(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars = input.chars().peekable();
    let mut in_quote: Option<char> = None;

    for ch in chars {
        match in_quote {
            Some(q) if ch == q => {
                in_quote = None;
            }
            Some(_) => {
                current.push(ch);
            }
            None if ch == '"' || ch == '\'' => {
                in_quote = Some(ch);
            }
            None if ch.is_ascii_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            None => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Format a `redis::Value` into human-readable text.
fn format_redis_value(value: &RedisValue) -> String {
    match value {
        RedisValue::Nil => "(nil)".to_string(),
        RedisValue::Int(n) => format!("(integer) {n}"),
        RedisValue::BulkString(bytes) => match std::str::from_utf8(bytes) {
            Ok(s) => format!("\"{s}\""),
            Err(_) => format!("<binary {}B>", bytes.len()),
        },
        RedisValue::Array(items) => {
            if items.is_empty() {
                return "(empty array)".to_string();
            }
            let mut out = String::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&format!("{}) {}", i + 1, format_redis_value(item)));
            }
            out
        }
        RedisValue::SimpleString(s) => s.clone(),
        RedisValue::Okay => "OK".to_string(),
        RedisValue::Map(pairs) => {
            if pairs.is_empty() {
                return "(empty map)".to_string();
            }
            let mut out = String::new();
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&format!(
                    "{}) {} -> {}",
                    i + 1,
                    format_redis_value(k),
                    format_redis_value(v)
                ));
            }
            out
        }
        RedisValue::Double(f) => format!("(double) {f}"),
        RedisValue::Boolean(b) => format!("(boolean) {b}"),
        RedisValue::VerbatimString { format: _, text } => text.clone(),
        RedisValue::BigNumber(n) => format!("(big number) {n}"),
        RedisValue::Set(items) => {
            if items.is_empty() {
                return "(empty set)".to_string();
            }
            let mut out = String::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&format!("{}) {}", i + 1, format_redis_value(item)));
            }
            out
        }
        RedisValue::Attribute { data, attributes } => {
            let mut out = format_redis_value(data);
            for (k, v) in attributes.iter() {
                out.push_str(&format!(
                    "\n  attr {} = {}",
                    format_redis_value(k),
                    format_redis_value(v)
                ));
            }
            out
        }
        RedisValue::Push { kind, data } => {
            let mut out = format!("(push {kind})");
            for (i, item) in data.iter().enumerate() {
                out.push_str(&format!("\n{}) {}", i + 1, format_redis_value(item)));
            }
            out
        }
        RedisValue::ServerError(e) => format!("(error) {}", e.details().unwrap_or("")),
    }
}

// ---------------------------------------------------------------------------
// RedisQueryTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RedisQueryArgs {
    command: String,
}

/// Execute an arbitrary Redis command against a server.
///
/// When `readonly` is `true`, only read-side commands (GET, KEYS, INFO, …) are
/// permitted. Mutating commands like SET, DEL, FLUSHDB are rejected.
pub struct RedisQueryTool {
    /// Redis connection URL.
    pub url: String,
    /// When `true`, only read-only commands are allowed.
    pub readonly: bool,
}

#[async_trait]
impl Tool for RedisQueryTool {
    fn name(&self) -> &str {
        "redis_query"
    }

    fn description(&self) -> &str {
        "Execute a Redis command against a connected server. \
         Pass the full command string, e.g. \"GET mykey\" or \"HGETALL users:42\". \
         In read-only mode only query commands are allowed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Full Redis command, e.g. \"GET key\", \"SET key value\", \"HGETALL hash\"."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Command cancelled"));
        }

        let args: RedisQueryArgs = ctx.parse_args(self.name())?;
        let tokens = parse_command(&args.command);

        if tokens.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Empty command"));
        }

        let cmd_name = &tokens[0];

        if self.readonly && !is_readonly_command(cmd_name) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Command '{}' is not allowed in read-only mode. \
                     Only read commands (GET, KEYS, INFO, …) are permitted.",
                    cmd_name.to_ascii_uppercase()
                ),
            ));
        }

        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| anyhow::anyhow!("Failed to create Redis client: {e}"))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to Redis: {e}"))?;

        let mut cmd = redis::cmd(&cmd_name.to_ascii_uppercase());
        for arg in &tokens[1..] {
            cmd.arg(arg.as_str());
        }

        let result: RedisValue = cmd
            .query_async(&mut conn)
            .await
            .map_err(|e| anyhow::anyhow!("Redis error: {e}"))?;

        let output = format_redis_value(&result);
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        format!("Running Redis command: {command}")
    }
}

// ---------------------------------------------------------------------------
// RedisInfoTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RedisInfoArgs {
    section: Option<String>,
}

/// Retrieve Redis server information via the `INFO` command.
///
/// Always read-only. Optionally accepts a `section` parameter to narrow
/// the output (e.g. `"memory"`, `"cpu"`, `"replication"`).
pub struct RedisInfoTool {
    /// Redis connection URL.
    pub url: String,
}

#[async_trait]
impl Tool for RedisInfoTool {
    fn name(&self) -> &str {
        "redis_info"
    }

    fn description(&self) -> &str {
        "Retrieve Redis server information via the INFO command. \
         Optionally specify a section (memory, cpu, replication, etc.) \
         to limit the output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Optional INFO section to retrieve (e.g. \"memory\", \"cpu\", \"replication\"). Omit for full output."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Command cancelled"));
        }

        let args: RedisInfoArgs = ctx.parse_args(self.name())?;

        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| anyhow::anyhow!("Failed to create Redis client: {e}"))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to Redis: {e}"))?;

        let mut cmd = redis::cmd("INFO");
        if let Some(ref section) = args.section {
            cmd.arg(section.as_str());
        }

        let result: RedisValue = cmd
            .query_async(&mut conn)
            .await
            .map_err(|e| anyhow::anyhow!("Redis error: {e}"))?;

        let output = format_redis_value(&result);
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        match args.get("section").and_then(|v| v.as_str()) {
            Some(section) => format!("Fetching Redis INFO ({section})"),
            None => "Fetching Redis INFO".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_command -------------------------------------------------------

    #[test]
    fn test_parse_simple_command() {
        assert_eq!(parse_command("GET mykey"), vec!["GET", "mykey"]);
    }

    #[test]
    fn test_parse_multiple_args() {
        assert_eq!(parse_command("SET foo bar"), vec!["SET", "foo", "bar"]);
    }

    #[test]
    fn test_parse_quoted_string() {
        assert_eq!(
            parse_command(r#"SET greeting "hello world""#),
            vec!["SET", "greeting", "hello world"]
        );
    }

    #[test]
    fn test_parse_single_quoted() {
        assert_eq!(
            parse_command("SET key 'value with spaces'"),
            vec!["SET", "key", "value with spaces"]
        );
    }

    #[test]
    fn test_parse_empty() {
        let result: Vec<String> = parse_command("   ");
        assert!(result.is_empty());
    }

    // -- is_readonly_command -------------------------------------------------

    #[test]
    fn test_readonly_allows_get() {
        assert!(is_readonly_command("GET"));
        assert!(is_readonly_command("get"));
        assert!(is_readonly_command("Get"));
    }

    #[test]
    fn test_readonly_allows_info() {
        assert!(is_readonly_command("INFO"));
    }

    #[test]
    fn test_readonly_blocks_set() {
        assert!(!is_readonly_command("SET"));
    }

    #[test]
    fn test_readonly_blocks_del() {
        assert!(!is_readonly_command("DEL"));
    }

    #[test]
    fn test_readonly_blocks_flushdb() {
        assert!(!is_readonly_command("FLUSHDB"));
    }

    // -- format_redis_value --------------------------------------------------

    #[test]
    fn test_format_nil() {
        assert_eq!(format_redis_value(&RedisValue::Nil), "(nil)");
    }

    #[test]
    fn test_format_int() {
        assert_eq!(format_redis_value(&RedisValue::Int(42)), "(integer) 42");
    }

    #[test]
    fn test_format_bulk_string() {
        let val = RedisValue::BulkString(b"hello".to_vec());
        assert_eq!(format_redis_value(&val), "\"hello\"");
    }

    #[test]
    fn test_format_ok() {
        assert_eq!(format_redis_value(&RedisValue::Okay), "OK");
    }

    #[test]
    fn test_format_simple_string() {
        let val = RedisValue::SimpleString("PONG".to_string());
        assert_eq!(format_redis_value(&val), "PONG");
    }

    #[test]
    fn test_format_empty_array() {
        let val = RedisValue::Array(vec![]);
        assert_eq!(format_redis_value(&val), "(empty array)");
    }

    #[test]
    fn test_format_array() {
        let val = RedisValue::Array(vec![
            RedisValue::BulkString(b"a".to_vec()),
            RedisValue::BulkString(b"b".to_vec()),
        ]);
        let out = format_redis_value(&val);
        assert!(out.contains("1) \"a\""));
        assert!(out.contains("2) \"b\""));
    }

    #[test]
    fn test_format_boolean() {
        assert_eq!(
            format_redis_value(&RedisValue::Boolean(true)),
            "(boolean) true"
        );
    }

    #[test]
    fn test_format_double() {
        let val = RedisValue::Double(3.14);
        assert!(format_redis_value(&val).contains("3.14"));
    }

    // -- tool metadata -------------------------------------------------------

    #[test]
    fn test_query_tool_name() {
        let tool = RedisQueryTool {
            url: "redis://localhost".to_string(),
            readonly: false,
        };
        assert_eq!(tool.name(), "redis_query");
    }

    #[test]
    fn test_info_tool_name() {
        let tool = RedisInfoTool {
            url: "redis://localhost".to_string(),
        };
        assert_eq!(tool.name(), "redis_info");
    }

    #[test]
    fn test_humanize_query() {
        let tool = RedisQueryTool {
            url: "redis://localhost".to_string(),
            readonly: false,
        };
        let desc = tool.humanize(&json!({"command": "GET foo"}));
        assert_eq!(desc, "Running Redis command: GET foo");
    }

    #[test]
    fn test_humanize_info_with_section() {
        let tool = RedisInfoTool {
            url: "redis://localhost".to_string(),
        };
        let desc = tool.humanize(&json!({"section": "memory"}));
        assert_eq!(desc, "Fetching Redis INFO (memory)");
    }

    #[test]
    fn test_humanize_info_no_section() {
        let tool = RedisInfoTool {
            url: "redis://localhost".to_string(),
        };
        let desc = tool.humanize(&json!({}));
        assert_eq!(desc, "Fetching Redis INFO");
    }

    // -- readonly gating (async) ---------------------------------------------

    #[tokio::test]
    async fn test_readonly_rejects_set_command() {
        let tool = RedisQueryTool {
            url: "redis://localhost:6379".to_string(),
            readonly: true,
        };
        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("id", json!({"command": "SET foo bar"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("not allowed in read-only mode"));
    }

    #[tokio::test]
    async fn test_empty_command_rejected() {
        let tool = RedisQueryTool {
            url: "redis://localhost:6379".to_string(),
            readonly: false,
        };
        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("id", json!({"command": "   "}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Empty command"));
    }

    #[tokio::test]
    async fn test_cancelled_returns_failure() {
        let tool = RedisQueryTool {
            url: "redis://localhost:6379".to_string(),
            readonly: false,
        };
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new("id", json!({"command": "GET foo"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("cancelled"));
    }
}
