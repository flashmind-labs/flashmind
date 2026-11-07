//! Pre-standard `_session/steering`, as specified by codex-acp's steering
//! contract. Idle requests start a normal session-owned turn by default.
//! `_meta.steering.idleBehavior = "promptRequired"` leaves idle input with
//! the client instead. Active requests are answered only after insertion.
//! Contract: <https://github.com/agentclientprotocol/codex-acp/blob/main/docs/steering.md>.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{ContentBlock, SessionId};
use agent_client_protocol::{Error, JsonRpcRequest, JsonRpcResponse, Responder};
use flashmind_core::{CancellationToken, Conversation, ConversationEntry};
use flashmind_types::ContentPart;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::server::lock;

const MAX_PENDING: usize = 8;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/steering", response = SteeringResponse)]
#[serde(rename_all = "camelCase")]
pub(super) struct SteeringRequest {
    pub session_id: SessionId,
    pub prompt: Vec<ContentBlock>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Map<String, Value>>,
}

impl SteeringRequest {
    pub fn prompt_required(&self) -> Result<bool, Error> {
        let Some(steering) = self.meta.as_ref().and_then(|meta| meta.get("steering")) else {
            return Ok(false);
        };
        let steering = steering
            .as_object()
            .ok_or_else(|| Error::invalid_params().data("steering metadata must be an object"))?;
        match steering.get("idleBehavior") {
            None => Ok(false),
            Some(Value::String(value)) if value == "promptRequired" => Ok(true),
            _ => Err(Error::invalid_params().data("unsupported steering idleBehavior")),
        }
    }

    pub fn validate(&self, images: bool) -> Result<usize, Error> {
        if self.prompt.is_empty() {
            return Err(Error::invalid_params().data("steering prompt must not be empty"));
        }
        let mut bytes = 0usize;
        for block in &self.prompt {
            let size = match block {
                ContentBlock::Text(text) => text.text.len(),
                ContentBlock::ResourceLink(link) => link.uri.len(),
                ContentBlock::Image(image) if images => {
                    image.data.len().saturating_add(image.mime_type.len())
                }
                ContentBlock::Image(_) => {
                    return Err(
                        Error::invalid_request().data("the active model does not support images")
                    );
                }
                _ => return Err(Error::invalid_params().data("unsupported steering content block")),
            };
            bytes = bytes.saturating_add(size);
            if bytes > MAX_BYTES {
                return Err(Error::invalid_request().data("steering prompt is too large"));
            }
        }
        if bytes == 0 {
            return Err(Error::invalid_params().data("steering prompt must not be empty"));
        }
        // Include annotations and extension metadata in the queue budget too.
        let bytes = serde_json::to_vec(&self.prompt)
            .map_err(|_| Error::invalid_params().data("invalid steering content"))?
            .len();
        if bytes > MAX_BYTES {
            return Err(Error::invalid_request().data("steering prompt is too large"));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonRpcResponse)]
#[serde(tag = "outcome", rename_all = "camelCase")]
pub(super) enum SteeringResponse {
    Injected,
    StartedNewTurn,
    PromptRequired { reason: NoRunningTurn },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum NoRunningTurn {
    NoRunningTurn,
}

impl SteeringResponse {
    pub fn idle() -> Self {
        Self::PromptRequired {
            reason: NoRunningTurn::NoRunningTurn,
        }
    }
}

pub(super) struct Pending {
    pub prompt: Vec<ContentBlock>,
    pub responder: Responder<SteeringResponse>,
    pub prompt_required: bool,
    pub bytes: usize,
}

impl Pending {
    fn reject(self) {
        let result = if self.prompt_required {
            self.responder.respond(SteeringResponse::idle())
        } else {
            self.responder.respond_with_error(
                Error::invalid_request().data("the turn ended before steering was applied"),
            )
        };
        if let Err(error) = result {
            tracing::debug!(%error, "could not answer steering request");
        }
    }
}

#[derive(Default)]
pub(super) struct State {
    active: Option<Active>,
}

struct Active {
    cancel: CancellationToken,
    pending: VecDeque<Pending>,
    bytes: usize,
}

impl State {
    /// Returns idle input to the handler without consuming it. Holding this
    /// lock also serializes arrivals against the turn's final acceptance check.
    pub fn enqueue(&mut self, pending: Pending) -> Result<Option<Pending>, Error> {
        let Some(active) = &mut self.active else {
            return Ok(Some(pending));
        };
        if active.cancel.is_cancelled() {
            return Ok(Some(pending));
        }
        if active.pending.len() >= MAX_PENDING
            || active.bytes.saturating_add(pending.bytes) > MAX_BYTES
        {
            pending
                .responder
                .respond_with_error(Error::invalid_request().data("steering queue is full"))?;
        } else {
            active.bytes += pending.bytes;
            active.pending.push_back(pending);
        }
        Ok(None)
    }
}

/// Closes acceptance on every exit path, including transport errors and task
/// drops. No pending request can leak into the next prompt.
pub(super) struct Turn(Arc<Mutex<State>>);

impl Turn {
    pub fn begin(state: Arc<Mutex<State>>, cancel: CancellationToken) -> Self {
        lock(&state).active = Some(Active {
            cancel,
            pending: VecDeque::new(),
            bytes: 0,
        });
        Self(state)
    }

    /// Insert only before a model request or after all tools and compaction.
    /// At settling, close acceptance atomically if no input remains.
    pub fn deliver(&self, conversation: &mut Conversation, settling: bool) -> Result<bool, Error> {
        let pending = {
            let mut state = lock(&self.0);
            let Some(active) = &mut state.active else {
                return Ok(false);
            };
            if active.cancel.is_cancelled() || (settling && active.pending.is_empty()) {
                let active = state.active.take();
                drop(state);
                if let Some(active) = active {
                    for pending in active.pending {
                        pending.reject();
                    }
                }
                return Ok(false);
            }
            active.bytes = 0;
            std::mem::take(&mut active.pending)
        };
        let added = !pending.is_empty();
        for pending in pending {
            conversation.add(entry(&pending.prompt));
            pending.responder.respond(SteeringResponse::Injected)?;
        }
        Ok(added)
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let active = lock(&self.0).active.take();
        if let Some(active) = active {
            for pending in active.pending {
                pending.reject();
            }
        }
    }
}

/// Keep text and image blocks in their wire order. Providers prepend the text
/// field to parts, so mixed prompts put all their content in parts instead.
pub(super) fn entry(blocks: &[ContentBlock]) -> ConversationEntry {
    if !blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Image(_)))
    {
        return ConversationEntry::user(super::server::prompt_text(blocks));
    }
    let parts = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(ContentPart::Text {
                text: text.text.clone(),
            }),
            ContentBlock::ResourceLink(link) => Some(ContentPart::Text {
                text: link.uri.clone(),
            }),
            ContentBlock::Image(image) => Some(ContentPart::Image {
                media_type: image.mime_type.clone(),
                data: image.data.clone(),
            }),
            _ => None,
        })
        .collect();
    ConversationEntry::user_with_parts("", parts)
}
