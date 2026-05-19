//! Response types for the GitHub REST API.
//!
//! All types derive `Deserialize` and use `#[serde(default)]` for optional
//! fields so that missing keys in partial API responses don't cause errors.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Repository
// ---------------------------------------------------------------------------

/// A GitHub repository.
#[derive(Debug, Deserialize)]
pub struct Repository {
    /// Unique repository ID.
    pub id: u64,
    /// Full name in `owner/repo` form.
    pub full_name: String,
    /// Repository description.
    #[serde(default)]
    pub description: Option<String>,
    /// URL to the repository on GitHub.
    pub html_url: String,
    /// Whether the repository is private.
    #[serde(default)]
    pub private: bool,
    /// Primary programming language.
    #[serde(default)]
    pub language: Option<String>,
    /// Number of stars.
    #[serde(default)]
    pub stargazers_count: u64,
    /// Last updated timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// User
// ---------------------------------------------------------------------------

/// A GitHub user (lightweight, login only).
#[derive(Debug, Deserialize)]
pub struct User {
    /// GitHub username.
    pub login: String,
}

// ---------------------------------------------------------------------------
// Label
// ---------------------------------------------------------------------------

/// A label attached to an issue or pull request.
#[derive(Debug, Deserialize)]
pub struct Label {
    /// Label display name.
    pub name: String,
    /// Hex color code (without `#` prefix).
    pub color: String,
}

// ---------------------------------------------------------------------------
// Issue
// ---------------------------------------------------------------------------

/// A GitHub issue.
#[derive(Debug, Deserialize)]
pub struct Issue {
    /// Issue number within the repository.
    pub number: u64,
    /// Issue title.
    pub title: String,
    /// Current state (`open` or `closed`).
    pub state: String,
    /// Issue body (Markdown).
    #[serde(default)]
    pub body: Option<String>,
    /// Author of the issue.
    pub user: User,
    /// Labels applied to the issue.
    #[serde(default)]
    pub labels: Vec<Label>,
    /// Users assigned to the issue.
    #[serde(default)]
    pub assignees: Vec<User>,
    /// URL to the issue on GitHub.
    pub html_url: String,
    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created_at: String,
    /// Last updated timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Pull Request
// ---------------------------------------------------------------------------

/// A GitHub pull request.
#[derive(Debug, Deserialize)]
pub struct PullRequest {
    /// PR number within the repository.
    pub number: u64,
    /// PR title.
    pub title: String,
    /// Current state (`open`, `closed`).
    pub state: String,
    /// PR body (Markdown).
    #[serde(default)]
    pub body: Option<String>,
    /// Author of the pull request.
    pub user: User,
    /// URL to the pull request on GitHub.
    pub html_url: String,
    /// Whether the PR has been merged.
    #[serde(default)]
    pub merged: Option<bool>,
    /// Whether the PR is mergeable (null if not yet computed).
    #[serde(default)]
    pub mergeable: Option<bool>,
    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created_at: String,
    /// Last updated timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: String,
    /// Head branch (source).
    pub head: Branch,
    /// Base branch (target).
    pub base: Branch,
}

/// A branch reference in a pull request.
#[derive(Debug, Deserialize)]
pub struct Branch {
    /// Branch name.
    #[serde(rename = "ref")]
    pub ref_name: String,
    /// Commit SHA at the tip of the branch.
    pub sha: String,
}

// ---------------------------------------------------------------------------
// Comment
// ---------------------------------------------------------------------------

/// A comment on an issue or pull request.
#[derive(Debug, Deserialize)]
pub struct Comment {
    /// Unique comment ID.
    pub id: u64,
    /// Comment body (Markdown).
    pub body: String,
    /// Author of the comment.
    pub user: User,
    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

/// A GitHub notification.
#[derive(Debug, Deserialize)]
pub struct Notification {
    /// Unique notification ID.
    pub id: String,
    /// Subject of the notification (issue, PR, etc.).
    pub subject: NotificationSubject,
    /// Reason the user received this notification.
    pub reason: String,
    /// Whether this notification is unread.
    #[serde(default)]
    pub unread: bool,
    /// Last updated timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: String,
    /// Repository this notification belongs to.
    pub repository: NotificationRepo,
}

/// Subject metadata within a notification.
#[derive(Debug, Deserialize)]
pub struct NotificationSubject {
    /// Subject title (e.g. issue title).
    pub title: String,
    /// Subject type (`Issue`, `PullRequest`, `Release`, etc.).
    #[serde(rename = "type")]
    pub subject_type: String,
}

/// Lightweight repository info within a notification.
#[derive(Debug, Deserialize)]
pub struct NotificationRepo {
    /// Full name in `owner/repo` form.
    pub full_name: String,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Paginated search result wrapper.
#[derive(Debug, Deserialize)]
pub struct SearchResult<T> {
    /// Total number of matching results.
    pub total_count: u64,
    /// Results for the current page.
    pub items: Vec<T>,
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Result of merging a pull request.
#[derive(Debug, Deserialize)]
pub struct MergeResult {
    /// Merge commit SHA (if successful).
    #[serde(default)]
    pub sha: Option<String>,
    /// Whether the merge was successful.
    #[serde(default)]
    pub merged: bool,
    /// Human-readable merge status message.
    #[serde(default)]
    pub message: String,
}
