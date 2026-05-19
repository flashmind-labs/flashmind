//! Response types for the Slack Web API.
//!
//! All types derive `Deserialize` and use `#[serde(default)]` for optional
//! fields so that missing keys in partial API responses don't cause errors.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// A Slack channel (public or private).
#[derive(Debug, Deserialize)]
pub struct Channel {
    /// Channel ID (e.g. `C1234567890`).
    pub id: String,
    /// Channel name (without the `#` prefix).
    pub name: String,
    /// Whether this is a public channel.
    #[serde(default)]
    pub is_channel: bool,
    /// Whether this is a private channel (group).
    #[serde(default)]
    pub is_private: bool,
    /// Number of members in the channel.
    #[serde(default)]
    pub num_members: Option<u64>,
    /// Channel topic.
    #[serde(default)]
    pub topic: Option<TopicOrPurpose>,
    /// Channel purpose.
    #[serde(default)]
    pub purpose: Option<TopicOrPurpose>,
}

/// Topic or purpose metadata for a channel.
#[derive(Debug, Deserialize)]
pub struct TopicOrPurpose {
    /// The text value of the topic or purpose.
    pub value: String,
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A message in a Slack channel or thread.
#[derive(Debug, Deserialize)]
pub struct SlackMessage {
    /// Message timestamp (unique ID within a channel).
    pub ts: String,
    /// User ID of the message author.
    #[serde(default)]
    pub user: Option<String>,
    /// Message text content.
    #[serde(default)]
    pub text: String,
    /// Thread parent timestamp (present if this message is part of a thread).
    #[serde(default)]
    pub thread_ts: Option<String>,
    /// Number of replies in the thread (only on parent messages).
    #[serde(default)]
    pub reply_count: Option<u64>,
    /// Reactions on this message.
    #[serde(default)]
    pub reactions: Option<Vec<Reaction>>,
}

/// A reaction on a message.
#[derive(Debug, Deserialize)]
pub struct Reaction {
    /// Emoji name (without colons).
    pub name: String,
    /// Number of users who added this reaction.
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// A search result match from `search.messages`.
#[derive(Debug, Deserialize)]
pub struct SearchMatch {
    /// Message timestamp.
    pub ts: String,
    /// Message text content.
    #[serde(default)]
    pub text: String,
    /// Channel the message was found in.
    pub channel: SearchChannel,
    /// Username of the message author.
    #[serde(default)]
    pub username: Option<String>,
    /// Permalink to the message.
    #[serde(default)]
    pub permalink: Option<String>,
}

/// Lightweight channel info within a search result.
#[derive(Debug, Deserialize)]
pub struct SearchChannel {
    /// Channel ID.
    pub id: String,
    /// Channel name.
    pub name: String,
}

// ---------------------------------------------------------------------------
// Post message response
// ---------------------------------------------------------------------------

/// Response data from `chat.postMessage`.
#[derive(Debug, Deserialize)]
pub struct PostMessageResponse {
    /// Timestamp of the posted message.
    pub ts: String,
    /// Channel the message was posted to.
    pub channel: String,
}

// ---------------------------------------------------------------------------
// List endpoint wrappers
// ---------------------------------------------------------------------------

/// Response data from `conversations.list`.
#[derive(Debug, Deserialize)]
pub struct ChannelListData {
    /// List of channels.
    #[serde(default)]
    pub channels: Vec<Channel>,
}

/// Response data from `conversations.history`.
#[derive(Debug, Deserialize)]
pub struct HistoryData {
    /// List of messages (newest first).
    #[serde(default)]
    pub messages: Vec<SlackMessage>,
    /// Whether there are more messages to paginate through.
    #[serde(default)]
    pub has_more: Option<bool>,
}

/// Response data from `conversations.replies`.
#[derive(Debug, Deserialize)]
pub struct RepliesData {
    /// Thread messages (parent + replies).
    #[serde(default)]
    pub messages: Vec<SlackMessage>,
}

/// Response data from `search.messages`.
#[derive(Debug, Deserialize)]
pub struct SearchData {
    /// Search results container.
    pub messages: SearchMessages,
}

/// Inner container for search message results.
#[derive(Debug, Deserialize)]
pub struct SearchMessages {
    /// Matched messages.
    #[serde(default)]
    pub matches: Vec<SearchMatch>,
    /// Total number of matches.
    #[serde(default)]
    pub total: u64,
}
