//! Tools that read Meshy catalogs and account data, with no task to wait for.

use std::fmt::Write as _;

use async_trait::async_trait;
use flashmind_llm::three_d::ThreeDClient;
use serde_json::{Map, Value};

use crate::{Tool, ToolContext, ToolResult};

/// Most library entries listed in one reply.
const MAX_LIBRARY_LINES: usize = 300;

/// Which Meshy lookup a tool performs.
pub(super) enum Query {
    AnimationLibrary,
    Account,
}

/// Static description of one lookup tool.
pub(super) struct QuerySpec {
    pub name: &'static str,
    pub description: &'static str,
    pub query: Query,
    pub schema: fn() -> Value,
}

/// A Meshy lookup that returns immediately.
pub struct MeshyQueryTool {
    client: ThreeDClient,
    spec: &'static QuerySpec,
    schema: Value,
}

impl MeshyQueryTool {
    pub(super) fn new(client: ThreeDClient, spec: &'static QuerySpec) -> Self {
        Self {
            client,
            spec,
            schema: (spec.schema)(),
        }
    }

    async fn animation_library(&self, args: &Map<String, Value>) -> anyhow::Result<String> {
        let query = query_pairs(args, &["search", "category", "sub_category", "action_ids"]);
        let library = self
            .client
            .meshy_get("openapi/v1/animations/library", &query)
            .await?;
        let entries = library.as_array().cloned().unwrap_or_default();
        if entries.is_empty() {
            return Ok("No animations match.".into());
        }
        let mut text = String::new();
        for entry in entries.iter().take(MAX_LIBRARY_LINES) {
            let _ = writeln!(
                text,
                "{} {} [{}/{}]",
                entry["action_id"],
                entry["name"].as_str().unwrap_or_default(),
                entry["category"].as_str().unwrap_or_default(),
                entry["sub_category"].as_str().unwrap_or_default(),
            );
        }
        if entries.len() > MAX_LIBRARY_LINES {
            let _ = writeln!(
                text,
                "{} more not shown; narrow with search or category.",
                entries.len() - MAX_LIBRARY_LINES
            );
        }
        Ok(text.trim_end().to_owned())
    }

    async fn account(&self, args: &Map<String, Value>) -> anyhow::Result<String> {
        match args.get("kind").and_then(Value::as_str) {
            Some("balance") => {
                let balance = self.client.meshy_get("openapi/v1/balance", &[]).await?;
                Ok(format!("Meshy credit balance: {}", balance["balance"]))
            }
            Some("usage") => {
                let query = query_pairs(
                    args,
                    &[
                        "page_num",
                        "page_size",
                        "sort_by",
                        "start_time",
                        "end_time",
                        "endpoints",
                        "status",
                    ],
                );
                let usage = self
                    .client
                    .meshy_get("openapi/v1/usage/tasks", &query)
                    .await?;
                let records = usage.as_array().cloned().unwrap_or_default();
                if records.is_empty() {
                    return Ok("No tasks match.".into());
                }
                let mut text = String::from("task_id endpoint status credits created_at_ms\n");
                for record in &records {
                    let _ = writeln!(
                        text,
                        "{} {} {} {} {}",
                        record["task_id"].as_str().unwrap_or_default(),
                        record["endpoint"].as_str().unwrap_or_default(),
                        record["status"].as_str().unwrap_or_default(),
                        record["consumed_credits"],
                        record["created_at"],
                    );
                }
                Ok(text.trim_end().to_owned())
            }
            _ => anyhow::bail!("kind must be balance or usage"),
        }
    }
}

#[async_trait]
impl Tool for MeshyQueryTool {
    fn name(&self) -> &str {
        self.spec.name
    }

    fn description(&self) -> &str {
        self.spec.description
    }

    fn parameters(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let Value::Object(args) = &ctx.args else {
            anyhow::bail!("Tool '{}': arguments must be an object", self.name());
        };
        for key in args.keys() {
            if !self.schema["properties"]
                .as_object()
                .is_some_and(|properties| properties.contains_key(key))
            {
                anyhow::bail!("Tool '{}': unknown argument '{key}'", self.name());
            }
        }
        let text = tokio::select! {
            biased;
            _ = ctx.cancel_token().cancelled() => anyhow::bail!("{} cancelled", self.name()),
            text = async {
                match self.spec.query {
                    Query::AnimationLibrary => self.animation_library(args).await,
                    Query::Account => self.account(args).await,
                }
            } => text?,
        };
        Ok(ToolResult::success(ctx.tool_call_id, text))
    }
}

/// Query string pairs for the named arguments; arrays become comma lists.
fn query_pairs(args: &Map<String, Value>, keys: &[&'static str]) -> Vec<(&'static str, String)> {
    let mut pairs = Vec::new();
    for key in keys {
        let text = match args.get(*key) {
            None | Some(Value::Null) => continue,
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| match item {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .collect::<Vec<_>>()
                .join(","),
            Some(other) => other.to_string(),
        };
        pairs.push((*key, text));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn query_pairs_join_arrays_and_skip_missing_values() {
        let args = json!({
            "search": "walk",
            "action_ids": [1, 2, 3],
            "page_size": 50,
            "category": null,
            "ignored": "x",
        });
        let pairs = query_pairs(
            args.as_object().unwrap(),
            &["search", "category", "action_ids", "page_size"],
        );
        assert_eq!(
            pairs,
            [
                ("search", "walk".to_owned()),
                ("action_ids", "1,2,3".to_owned()),
                ("page_size", "50".to_owned()),
            ]
        );
    }

    #[tokio::test]
    async fn library_lists_entries_and_account_reports_balance() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio_util::sync::CancellationToken;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = url::Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let client = ThreeDClient::new(flashmind_llm::three_d::ThreeDProvider::Meshy, "key")
            .unwrap()
            .with_base_url(base);
        let tool = |name: &str| {
            let spec = super::super::specs::QUERIES
                .iter()
                .find(|spec| spec.name == name)
                .unwrap();
            MeshyQueryTool::new(client.clone(), spec)
        };
        let server = async {
            for reply in [
                r#"[{"action_id":7,"name":"Walk","category":"WalkAndRun","sub_category":"Walking"}]"#,
                r#"{"balance":42}"#,
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = [0_u8; 4096];
                let _ = socket.read(&mut buffer).await.unwrap();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                            reply.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        };
        let cancel = CancellationToken::new();
        let run = async {
            let library = tool("animation_library_3d")
                .execute(ToolContext::new(
                    "call",
                    json!({"search": "walk"}),
                    None,
                    &cancel,
                ))
                .await
                .unwrap();
            let account = tool("meshy_account")
                .execute(ToolContext::new(
                    "call",
                    json!({"kind": "balance"}),
                    None,
                    &cancel,
                ))
                .await
                .unwrap();
            let unknown = tool("meshy_account")
                .execute(ToolContext::new("call", json!({"bogus": 1}), None, &cancel))
                .await;
            assert!(unknown.is_err());
            (library, account)
        };
        let ((library, account), ()) = tokio::join!(run, server);
        let text = |result: &ToolResult| match result {
            ToolResult::Success { content, .. } => format!("{content:?}"),
            other => panic!("{other:?}"),
        };
        assert!(text(&library).contains("7 Walk [WalkAndRun/Walking]"));
        assert!(text(&account).contains("Meshy credit balance: 42"));
    }
}
