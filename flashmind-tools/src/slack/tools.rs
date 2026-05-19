//! Slack `Tool` trait implementations.
//!
//! Six tools covering channels, messages, threads, search, posting, and reactions.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::SlackClient;
use super::types::*;

// ---------------------------------------------------------------------------
// slack_list_channels
// ---------------------------------------------------------------------------

/// List Slack channels visible to the authenticated user.
pub struct SlackListChannelsTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct ListChannelsArgs {
    types: Option<String>,
    limit: Option<u32>,
}

#[async_trait]
impl Tool for SlackListChannelsTool {
    fn name(&self) -> &str {
        "slack_list_channels"
    }

    fn description(&self) -> &str {
        "List Slack channels visible to the authenticated user. \
         Returns channel names, IDs, and member counts."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "types": {
                    "type": "string",
                    "description": "Comma-separated channel types to include (default: public_channel,private_channel)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of channels to return (default 100, max 1000)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListChannelsArgs = parse_args(self.name(), ctx.args)?;
        let types = args
            .types
            .as_deref()
            .unwrap_or("public_channel,private_channel");
        let limit = args.limit.unwrap_or(100).min(1000);
        let limit_str = limit.to_string();

        let data: ChannelListData = self
            .client
            .api_get(
                "conversations.list",
                &[("types", types), ("limit", &limit_str)],
            )
            .await?;

        let mut out = String::new();
        if data.channels.is_empty() {
            out.push_str("No channels found.");
        } else {
            for ch in &data.channels {
                let vis = if ch.is_private { "private" } else { "public" };
                let members = ch
                    .num_members
                    .map(|n| format!("{n} members"))
                    .unwrap_or_else(|| "—".into());
                let topic = ch.topic.as_ref().map(|t| t.value.as_str()).unwrap_or("");
                let _ = writeln!(
                    out,
                    "- #{} ({}) [{vis}, {members}]{}",
                    ch.name,
                    ch.id,
                    if topic.is_empty() {
                        String::new()
                    } else {
                        format!("\n  Topic: {topic}")
                    },
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Slack channels".into()
    }
}

// ---------------------------------------------------------------------------
// slack_read_channel
// ---------------------------------------------------------------------------

/// Read recent messages from a Slack channel.
pub struct SlackReadChannelTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct ReadChannelArgs {
    channel: String,
    limit: Option<u32>,
    oldest: Option<String>,
    latest: Option<String>,
}

#[async_trait]
impl Tool for SlackReadChannelTool {
    fn name(&self) -> &str {
        "slack_read_channel"
    }

    fn description(&self) -> &str {
        "Read recent messages from a Slack channel. Returns messages with \
         timestamps, users, and text content."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID (e.g. C1234567890)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of messages to return (default 20, max 1000)"
                },
                "oldest": {
                    "type": "string",
                    "description": "Only messages after this Unix timestamp (inclusive)"
                },
                "latest": {
                    "type": "string",
                    "description": "Only messages before this Unix timestamp (inclusive)"
                }
            },
            "required": ["channel"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ReadChannelArgs = parse_args(self.name(), ctx.args)?;
        let limit = args.limit.unwrap_or(20).min(1000);
        let limit_str = limit.to_string();

        let mut params: Vec<(&str, &str)> = vec![("channel", &args.channel), ("limit", &limit_str)];
        if let Some(ref oldest) = args.oldest {
            params.push(("oldest", oldest));
        }
        if let Some(ref latest) = args.latest {
            params.push(("latest", latest));
        }

        let data: HistoryData = self
            .client
            .api_get("conversations.history", &params)
            .await?;

        let mut out = String::new();
        if data.messages.is_empty() {
            out.push_str("No messages found.");
        } else {
            for msg in &data.messages {
                let user = msg.user.as_deref().unwrap_or("unknown");
                let thread_info = if let Some(count) = msg.reply_count {
                    format!(" [{count} replies]")
                } else {
                    String::new()
                };
                let reactions = format_reactions(&msg.reactions);
                let _ = writeln!(
                    out,
                    "[{}] {}: {}{}{}",
                    msg.ts, user, msg.text, thread_info, reactions,
                );
            }
            if data.has_more == Some(true) {
                let _ = writeln!(
                    out,
                    "\n(more messages available — use oldest/latest to paginate)"
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let channel = args.get("channel").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Reading Slack channel {channel}")
    }
}

// ---------------------------------------------------------------------------
// slack_read_thread
// ---------------------------------------------------------------------------

/// Read all messages in a Slack thread.
pub struct SlackReadThreadTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct ReadThreadArgs {
    channel: String,
    thread_ts: String,
}

#[async_trait]
impl Tool for SlackReadThreadTool {
    fn name(&self) -> &str {
        "slack_read_thread"
    }

    fn description(&self) -> &str {
        "Read all messages in a Slack thread, including the parent message and all replies."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID containing the thread"
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Timestamp of the parent message (thread root)"
                }
            },
            "required": ["channel", "thread_ts"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ReadThreadArgs = parse_args(self.name(), ctx.args)?;

        let data: RepliesData = self
            .client
            .api_get(
                "conversations.replies",
                &[("channel", &args.channel), ("ts", &args.thread_ts)],
            )
            .await?;

        let mut out = String::new();
        if data.messages.is_empty() {
            out.push_str("No messages found in thread.");
        } else {
            for msg in &data.messages {
                let user = msg.user.as_deref().unwrap_or("unknown");
                let reactions = format_reactions(&msg.reactions);
                let _ = writeln!(out, "[{}] {}: {}{}", msg.ts, user, msg.text, reactions);
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let ts = args
            .get("thread_ts")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        format!("Reading Slack thread {ts}")
    }
}

// ---------------------------------------------------------------------------
// slack_search_messages
// ---------------------------------------------------------------------------

/// Search Slack messages across all channels.
pub struct SlackSearchMessagesTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct SearchMessagesArgs {
    query: String,
    count: Option<u32>,
    sort: Option<String>,
}

#[async_trait]
impl Tool for SlackSearchMessagesTool {
    fn name(&self) -> &str {
        "slack_search_messages"
    }

    fn description(&self) -> &str {
        "Search Slack messages across all accessible channels. \
         Supports Slack search syntax (in:#channel, from:@user, etc.)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (supports Slack search syntax)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (default 20, max 100)"
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: score or timestamp (default: score)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchMessagesArgs = parse_args(self.name(), ctx.args)?;
        let count = args.count.unwrap_or(20).min(100);
        let sort = args.sort.as_deref().unwrap_or("score");
        let count_str = count.to_string();

        let data: SearchData = self
            .client
            .api_get(
                "search.messages",
                &[
                    ("query", args.query.as_str()),
                    ("count", &count_str),
                    ("sort", sort),
                ],
            )
            .await?;

        let mut out = format!("{} results found.\n\n", data.messages.total);
        if data.messages.matches.is_empty() {
            out.push_str("No matching messages.");
        } else {
            for m in &data.messages.matches {
                let user = m.username.as_deref().unwrap_or("unknown");
                let permalink = m.permalink.as_deref().unwrap_or("");
                let _ = writeln!(
                    out,
                    "- [{}] #{} ({}) — {}\n  {}",
                    m.ts, m.channel.name, user, m.text, permalink,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
        format!("Searching Slack messages: {query}")
    }
}

// ---------------------------------------------------------------------------
// slack_send_message
// ---------------------------------------------------------------------------

/// Send a message to a Slack channel or thread.
pub struct SlackSendMessageTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct SendMessageArgs {
    channel: String,
    text: String,
    thread_ts: Option<String>,
}

#[async_trait]
impl Tool for SlackSendMessageTool {
    fn name(&self) -> &str {
        "slack_send_message"
    }

    fn description(&self) -> &str {
        "Send a message to a Slack channel. Optionally reply to a thread \
         by providing the parent message timestamp."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID to post to"
                },
                "text": {
                    "type": "string",
                    "description": "Message text (supports Slack mrkdwn formatting)"
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Parent message timestamp to reply in a thread (optional)"
                }
            },
            "required": ["channel", "text"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SendMessageArgs = parse_args(self.name(), ctx.args)?;

        let mut payload = json!({
            "channel": args.channel,
            "text": args.text,
        });
        if let Some(thread_ts) = &args.thread_ts {
            payload["thread_ts"] = json!(thread_ts);
        }

        let resp: PostMessageResponse = self.client.api_post("chat.postMessage", &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Message sent to channel {} (timestamp: {})",
                resp.channel, resp.ts
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let channel = args.get("channel").and_then(|v| v.as_str()).unwrap_or("?");
        if args.get("thread_ts").is_some() {
            format!("Replying in Slack channel {channel}")
        } else {
            format!("Sending message to Slack channel {channel}")
        }
    }
}

// ---------------------------------------------------------------------------
// slack_add_reaction
// ---------------------------------------------------------------------------

/// Add an emoji reaction to a Slack message.
pub struct SlackAddReactionTool {
    /// Shared Slack API client.
    pub client: Arc<SlackClient>,
}

#[derive(Deserialize)]
struct AddReactionArgs {
    channel: String,
    timestamp: String,
    name: String,
}

/// Empty response data for `reactions.add` (only `ok` is meaningful).
#[derive(Deserialize)]
struct EmptyData {}

#[async_trait]
impl Tool for SlackAddReactionTool {
    fn name(&self) -> &str {
        "slack_add_reaction"
    }

    fn description(&self) -> &str {
        "Add an emoji reaction to a Slack message. Use the emoji name without \
         colons (e.g. 'thumbsup' not ':thumbsup:')."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID containing the message"
                },
                "timestamp": {
                    "type": "string",
                    "description": "Timestamp of the message to react to"
                },
                "name": {
                    "type": "string",
                    "description": "Emoji name without colons (e.g. 'thumbsup')"
                }
            },
            "required": ["channel", "timestamp", "name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: AddReactionArgs = parse_args(self.name(), ctx.args)?;

        let payload = json!({
            "channel": args.channel,
            "timestamp": args.timestamp,
            "name": args.name,
        });

        let _: EmptyData = self.client.api_post("reactions.add", &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Added :{}: reaction to message {}",
                args.name, args.timestamp
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Adding :{name}: reaction")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format reactions for display output.
fn format_reactions(reactions: &Option<Vec<Reaction>>) -> String {
    match reactions {
        Some(reactions) if !reactions.is_empty() => {
            let items: Vec<String> = reactions
                .iter()
                .map(|r| format!(":{}: {}", r.name, r.count))
                .collect();
            format!(" [{}]", items.join(", "))
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> Arc<SlackClient> {
        Arc::new(SlackClient::new_for_test())
    }

    // -- name() tests -------------------------------------------------------

    #[test]
    fn tool_names() {
        let c = make_client();
        assert_eq!(
            SlackListChannelsTool { client: c.clone() }.name(),
            "slack_list_channels"
        );
        assert_eq!(
            SlackReadChannelTool { client: c.clone() }.name(),
            "slack_read_channel"
        );
        assert_eq!(
            SlackReadThreadTool { client: c.clone() }.name(),
            "slack_read_thread"
        );
        assert_eq!(
            SlackSearchMessagesTool { client: c.clone() }.name(),
            "slack_search_messages"
        );
        assert_eq!(
            SlackSendMessageTool { client: c.clone() }.name(),
            "slack_send_message"
        );
        assert_eq!(
            SlackAddReactionTool { client: c.clone() }.name(),
            "slack_add_reaction"
        );
    }

    // -- humanize() tests ---------------------------------------------------

    #[test]
    fn humanize_list_channels() {
        let c = make_client();
        let tool = SlackListChannelsTool { client: c };
        assert_eq!(tool.humanize(&json!({})), "Listing Slack channels");
    }

    #[test]
    fn humanize_read_channel() {
        let c = make_client();
        let tool = SlackReadChannelTool { client: c };
        let h = tool.humanize(&json!({ "channel": "C123" }));
        assert!(h.contains("C123"));
    }

    #[test]
    fn humanize_read_thread() {
        let c = make_client();
        let tool = SlackReadThreadTool { client: c };
        let h = tool.humanize(&json!({ "thread_ts": "1234567890.123456" }));
        assert!(h.contains("1234567890.123456"));
    }

    #[test]
    fn humanize_search_messages() {
        let c = make_client();
        let tool = SlackSearchMessagesTool { client: c };
        let h = tool.humanize(&json!({ "query": "deploy status" }));
        assert!(h.contains("deploy status"));
    }

    #[test]
    fn humanize_send_message() {
        let c = make_client();
        let tool = SlackSendMessageTool { client: c.clone() };
        let h = tool.humanize(&json!({ "channel": "C123" }));
        assert!(h.contains("Sending message"));
        assert!(h.contains("C123"));

        let h_thread = tool.humanize(&json!({ "channel": "C123", "thread_ts": "123.456" }));
        assert!(h_thread.contains("Replying"));
    }

    #[test]
    fn humanize_add_reaction() {
        let c = make_client();
        let tool = SlackAddReactionTool { client: c };
        let h = tool.humanize(&json!({ "name": "thumbsup" }));
        assert!(h.contains("thumbsup"));
    }

    // -- parameters() schema tests ------------------------------------------

    #[test]
    fn parameters_are_objects() {
        let c = make_client();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(SlackListChannelsTool { client: c.clone() }),
            Box::new(SlackReadChannelTool { client: c.clone() }),
            Box::new(SlackReadThreadTool { client: c.clone() }),
            Box::new(SlackSearchMessagesTool { client: c.clone() }),
            Box::new(SlackSendMessageTool { client: c.clone() }),
            Box::new(SlackAddReactionTool { client: c.clone() }),
        ];

        for tool in &tools {
            let params = tool.parameters();
            assert_eq!(
                params.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "tool '{}' parameters should have type: object",
                tool.name()
            );
            assert!(
                params.get("properties").is_some(),
                "tool '{}' parameters should have properties",
                tool.name()
            );
        }
    }

    #[test]
    fn required_fields_present() {
        let c = make_client();

        let read_channel = SlackReadChannelTool { client: c.clone() };
        let required = read_channel.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"channel".to_string()));

        let read_thread = SlackReadThreadTool { client: c.clone() };
        let required = read_thread.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"channel".to_string()));
        assert!(required.contains(&"thread_ts".to_string()));

        let search = SlackSearchMessagesTool { client: c.clone() };
        let required = search.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"query".to_string()));

        let send = SlackSendMessageTool { client: c.clone() };
        let required = send.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"channel".to_string()));
        assert!(required.contains(&"text".to_string()));

        let reaction = SlackAddReactionTool { client: c.clone() };
        let required = reaction.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"channel".to_string()));
        assert!(required.contains(&"timestamp".to_string()));
        assert!(required.contains(&"name".to_string()));
    }
}
