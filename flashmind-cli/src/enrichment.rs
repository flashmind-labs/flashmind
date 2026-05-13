//! Session title enrichment — fire-and-forget.
//!
//! After a turn completes, spawns a background task that asks the LLM to
//! generate a short kebab-case title for the conversation, then persists it
//! via `crate::session::set_title`.

use std::sync::Arc;

use futures::StreamExt;
use rust_decimal::dec;
use tracing::{info, warn};

use flashmind_memory::tokio_rusqlite;
use flashmind_types::model::{Model, ReasoningLevel, SamplingParams};
use flashmind_types::{CompletionRequest, LlmProvider, Message, Role};

use crate::session;

const TITLE_PROMPT: &str = "\
Generate a short kebab-case title summarizing this conversation. \
Format: lowercase-words-separated-by-hyphens (e.g. fix-slack-streaming-bug). \
Max 6 words. Reply with ONLY the title — no quotes, no punctuation.";

/// Spawn a fire-and-forget enrichment task that generates a kebab-case title
/// from conversation messages and persists it on the `local_sessions` row.
pub fn spawn_title_enrichment(
    session_key: String,
    messages: Vec<Message>,
    model: Model,
    provider: Arc<dyn LlmProvider>,
    conn: tokio_rusqlite::Connection,
) {
    let messages: Vec<Message> = messages
        .into_iter()
        .filter(|m| m.role != Role::System)
        .collect();

    if messages.len() < 2 {
        return;
    }

    tokio::spawn(async move {
        let mut full_messages = Vec::with_capacity(messages.len() + 2);
        full_messages.push(Message::system(TITLE_PROMPT));
        full_messages.extend(messages);
        full_messages.push(Message::user("Generate the kebab-case title now."));

        let request = CompletionRequest {
            model: model.clone(),
            messages: full_messages,
            tools: vec![],
            max_tokens: Some(40),
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.3)),
                ..Default::default()
            },
            modalities: vec![],
            audio_config: None,
            image_config: None,
        };

        let mut stream = provider.complete(request);
        let mut raw = String::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(flashmind_types::StreamEvent::ContentDelta(text)) => {
                    raw.push_str(&text);
                }
                Ok(flashmind_types::StreamEvent::Finished(_)) => break,
                Err(e) => {
                    warn!(error = %e, model = %model, "enrichment LLM error");
                    return;
                }
                _ => {}
            }
        }

        let title = sanitize_title(&raw);
        if title.is_empty() {
            warn!(model = %model, raw = %raw, "enrichment produced empty title");
            return;
        }

        if let Err(e) = session::set_title(&conn, &session_key, Some(&title)).await {
            warn!(error = %e, session_key, "failed to persist enrichment title");
            return;
        }

        info!(session_key, title, "enrichment title set");
    });
}

/// Normalize a model-generated title into safe kebab-case.
fn sanitize_title(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
        .trim();

    let lower = trimmed.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_dash = false;

    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if (c == '-' || c.is_whitespace() || c == '_') && !out.is_empty() && !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }

    while out.ends_with('-') {
        out.pop();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_basic() {
        assert_eq!(
            sanitize_title("Fix Slack Streaming Bug"),
            "fix-slack-streaming-bug"
        );
    }

    #[test]
    fn sanitize_strips_quotes() {
        assert_eq!(sanitize_title("\"fix-bug.\""), "fix-bug");
    }

    #[test]
    fn sanitize_collapses_separators() {
        assert_eq!(sanitize_title("add   new  --  feature"), "add-new-feature");
    }

    #[test]
    fn sanitize_empty_on_garbage() {
        assert_eq!(sanitize_title("!!!"), "");
    }
}
