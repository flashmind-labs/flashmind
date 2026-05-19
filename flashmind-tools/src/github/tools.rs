//! GitHub `Tool` trait implementations.
//!
//! Ten tools covering repositories, issues, pull requests, and notifications.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::GitHubClient;
use super::types::*;

// ---------------------------------------------------------------------------
// github_list_repos
// ---------------------------------------------------------------------------

/// List repositories for the authenticated user.
pub struct GitHubListReposTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct ListReposArgs {
    sort: Option<String>,
    per_page: Option<u32>,
    #[serde(rename = "type")]
    type_: Option<String>,
}

#[async_trait]
impl Tool for GitHubListReposTool {
    fn name(&self) -> &str {
        "github_list_repos"
    }

    fn description(&self) -> &str {
        "List repositories for the authenticated GitHub user. \
         Supports sorting and filtering by type."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "sort": {
                    "type": "string",
                    "description": "Sort field: created, updated, pushed, full_name (default: updated)"
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of repos to return (default 30, max 100)"
                },
                "type": {
                    "type": "string",
                    "description": "Filter by type: all, owner, member (default: all)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListReposArgs = parse_args(self.name(), ctx.args)?;
        let sort = args.sort.as_deref().unwrap_or("updated");
        let per_page = args.per_page.unwrap_or(30).min(100);
        let type_ = args.type_.as_deref().unwrap_or("all");

        let path = format!("user/repos?sort={sort}&per_page={per_page}&type={type_}");
        let repos: Vec<Repository> = self.client.get(&path).await?;

        let mut out = String::new();
        if repos.is_empty() {
            out.push_str("No repositories found.");
        } else {
            for repo in &repos {
                let desc = repo.description.as_deref().unwrap_or("");
                let lang = repo.language.as_deref().unwrap_or("—");
                let vis = if repo.private { "private" } else { "public" };
                let _ = writeln!(
                    out,
                    "- **{}** ({}, {}) ★{}\n  {}\n  {}",
                    repo.full_name, lang, vis, repo.stargazers_count, desc, repo.html_url,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing GitHub repositories".into()
    }
}

// ---------------------------------------------------------------------------
// github_search_issues
// ---------------------------------------------------------------------------

/// Search GitHub issues and pull requests.
pub struct GitHubSearchIssuesTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct SearchIssuesArgs {
    query: String,
    per_page: Option<u32>,
}

#[async_trait]
impl Tool for GitHubSearchIssuesTool {
    fn name(&self) -> &str {
        "github_search_issues"
    }

    fn description(&self) -> &str {
        "Search GitHub issues and pull requests using GitHub search syntax."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "GitHub search query (e.g. 'repo:owner/name is:open label:bug')"
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of results to return (default 30, max 100)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchIssuesArgs = parse_args(self.name(), ctx.args)?;
        let per_page = args.per_page.unwrap_or(30).min(100);
        let query = urlencoding::encode(&args.query);

        let path = format!("search/issues?q={query}&per_page={per_page}");
        let result: SearchResult<Issue> = self.client.get(&path).await?;

        let mut out = format!("{} results found.\n\n", result.total_count);
        for issue in &result.items {
            let _ = writeln!(
                out,
                "- #{} [{}] {}\n  {}",
                issue.number, issue.state, issue.title, issue.html_url,
            );
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
        format!("Searching GitHub issues: {query}")
    }
}

// ---------------------------------------------------------------------------
// github_get_issue
// ---------------------------------------------------------------------------

/// Get a specific GitHub issue with comments.
pub struct GitHubGetIssueTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct GetIssueArgs {
    owner: String,
    repo: String,
    number: u64,
}

#[async_trait]
impl Tool for GitHubGetIssueTool {
    fn name(&self) -> &str {
        "github_get_issue"
    }

    fn description(&self) -> &str {
        "Get a specific GitHub issue including its comments."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "number": {
                    "type": "integer",
                    "description": "Issue number"
                }
            },
            "required": ["owner", "repo", "number"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetIssueArgs = parse_args(self.name(), ctx.args)?;

        let issue_path = format!("repos/{}/{}/issues/{}", args.owner, args.repo, args.number);
        let issue: Issue = self.client.get(&issue_path).await?;

        let comments_path = format!(
            "repos/{}/{}/issues/{}/comments",
            args.owner, args.repo, args.number
        );
        let comments: Vec<Comment> = self.client.get(&comments_path).await?;

        let mut out = String::new();
        let _ = writeln!(out, "# #{} {}", issue.number, issue.title);
        let _ = writeln!(out, "State: {} | Author: {}", issue.state, issue.user.login);

        if !issue.labels.is_empty() {
            let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            let _ = writeln!(out, "Labels: {}", labels.join(", "));
        }
        if !issue.assignees.is_empty() {
            let assignees: Vec<&str> = issue.assignees.iter().map(|u| u.login.as_str()).collect();
            let _ = writeln!(out, "Assignees: {}", assignees.join(", "));
        }

        let _ = writeln!(
            out,
            "Created: {} | Updated: {}",
            issue.created_at, issue.updated_at
        );
        let _ = writeln!(out, "URL: {}\n", issue.html_url);

        if let Some(body) = &issue.body {
            let _ = writeln!(out, "{body}\n");
        }

        if !comments.is_empty() {
            let _ = writeln!(out, "---\n## Comments ({}):\n", comments.len());
            for comment in &comments {
                let _ = writeln!(
                    out,
                    "**{}** ({})\n{}\n",
                    comment.user.login, comment.created_at, comment.body
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let owner = args.get("owner").and_then(|v| v.as_str()).unwrap_or("?");
        let repo = args.get("repo").and_then(|v| v.as_str()).unwrap_or("?");
        let number = args.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        format!("Getting issue {owner}/{repo}#{number}")
    }
}

// ---------------------------------------------------------------------------
// github_create_issue
// ---------------------------------------------------------------------------

/// Create a new GitHub issue.
pub struct GitHubCreateIssueTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct CreateIssueArgs {
    owner: String,
    repo: String,
    title: String,
    body: Option<String>,
    labels: Option<Vec<String>>,
    assignees: Option<Vec<String>>,
}

#[async_trait]
impl Tool for GitHubCreateIssueTool {
    fn name(&self) -> &str {
        "github_create_issue"
    }

    fn description(&self) -> &str {
        "Create a new issue in a GitHub repository."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "title": {
                    "type": "string",
                    "description": "Issue title"
                },
                "body": {
                    "type": "string",
                    "description": "Issue body (Markdown)"
                },
                "labels": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Labels to apply"
                },
                "assignees": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "GitHub usernames to assign"
                }
            },
            "required": ["owner", "repo", "title"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateIssueArgs = parse_args(self.name(), ctx.args)?;

        let mut payload = json!({
            "title": args.title,
        });
        if let Some(body) = &args.body {
            payload["body"] = json!(body);
        }
        if let Some(labels) = &args.labels {
            payload["labels"] = json!(labels);
        }
        if let Some(assignees) = &args.assignees {
            payload["assignees"] = json!(assignees);
        }

        let path = format!("repos/{}/{}/issues", args.owner, args.repo);
        let issue: Issue = self.client.post(&path, &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Created issue #{} — {}\n{}",
                issue.number, issue.title, issue.html_url
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("...");
        format!("Creating issue: {title}")
    }
}

// ---------------------------------------------------------------------------
// github_comment_on_issue
// ---------------------------------------------------------------------------

/// Comment on a GitHub issue.
pub struct GitHubCommentOnIssueTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct CommentOnIssueArgs {
    owner: String,
    repo: String,
    number: u64,
    body: String,
}

#[async_trait]
impl Tool for GitHubCommentOnIssueTool {
    fn name(&self) -> &str {
        "github_comment_on_issue"
    }

    fn description(&self) -> &str {
        "Add a comment to a GitHub issue."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "number": {
                    "type": "integer",
                    "description": "Issue number"
                },
                "body": {
                    "type": "string",
                    "description": "Comment body (Markdown)"
                }
            },
            "required": ["owner", "repo", "number", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CommentOnIssueArgs = parse_args(self.name(), ctx.args)?;

        let path = format!(
            "repos/{}/{}/issues/{}/comments",
            args.owner, args.repo, args.number
        );
        let payload = json!({ "body": args.body });
        let comment: Comment = self.client.post(&path, &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Comment {} added to issue #{}.", comment.id, args.number),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let number = args.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        format!("Commenting on issue #{number}")
    }
}

// ---------------------------------------------------------------------------
// github_list_prs
// ---------------------------------------------------------------------------

/// List pull requests in a GitHub repository.
pub struct GitHubListPrsTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct ListPrsArgs {
    owner: String,
    repo: String,
    state: Option<String>,
    per_page: Option<u32>,
}

#[async_trait]
impl Tool for GitHubListPrsTool {
    fn name(&self) -> &str {
        "github_list_prs"
    }

    fn description(&self) -> &str {
        "List pull requests in a GitHub repository."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "state": {
                    "type": "string",
                    "description": "Filter by state: open, closed, all (default: open)"
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of PRs to return (default 30, max 100)"
                }
            },
            "required": ["owner", "repo"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListPrsArgs = parse_args(self.name(), ctx.args)?;
        let state = args.state.as_deref().unwrap_or("open");
        let per_page = args.per_page.unwrap_or(30).min(100);

        let path = format!(
            "repos/{}/{}/pulls?state={state}&per_page={per_page}",
            args.owner, args.repo
        );
        let prs: Vec<PullRequest> = self.client.get(&path).await?;

        let mut out = String::new();
        if prs.is_empty() {
            out.push_str("No pull requests found.");
        } else {
            for pr in &prs {
                let _ = writeln!(
                    out,
                    "- #{} [{}] {} (by {})\n  {}",
                    pr.number, pr.state, pr.title, pr.user.login, pr.html_url,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let owner = args.get("owner").and_then(|v| v.as_str()).unwrap_or("?");
        let repo = args.get("repo").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Listing PRs in {owner}/{repo}")
    }
}

// ---------------------------------------------------------------------------
// github_get_pr
// ---------------------------------------------------------------------------

/// Get a specific GitHub pull request.
pub struct GitHubGetPrTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct GetPrArgs {
    owner: String,
    repo: String,
    number: u64,
}

#[async_trait]
impl Tool for GitHubGetPrTool {
    fn name(&self) -> &str {
        "github_get_pr"
    }

    fn description(&self) -> &str {
        "Get details of a specific GitHub pull request."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "number": {
                    "type": "integer",
                    "description": "Pull request number"
                }
            },
            "required": ["owner", "repo", "number"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetPrArgs = parse_args(self.name(), ctx.args)?;

        let path = format!("repos/{}/{}/pulls/{}", args.owner, args.repo, args.number);
        let pr: PullRequest = self.client.get(&path).await?;

        let mut out = String::new();
        let _ = writeln!(out, "# PR #{} {}", pr.number, pr.title);
        let _ = writeln!(out, "State: {} | Author: {}", pr.state, pr.user.login);
        let _ = writeln!(out, "Branch: {} -> {}", pr.head.ref_name, pr.base.ref_name);
        let _ = writeln!(out, "Head SHA: {}", pr.head.sha);

        if let Some(merged) = pr.merged {
            let _ = writeln!(out, "Merged: {merged}");
        }
        if let Some(mergeable) = pr.mergeable {
            let _ = writeln!(out, "Mergeable: {mergeable}");
        }

        let _ = writeln!(
            out,
            "Created: {} | Updated: {}",
            pr.created_at, pr.updated_at
        );
        let _ = writeln!(out, "URL: {}\n", pr.html_url);

        if let Some(body) = &pr.body {
            let _ = writeln!(out, "{body}");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let owner = args.get("owner").and_then(|v| v.as_str()).unwrap_or("?");
        let repo = args.get("repo").and_then(|v| v.as_str()).unwrap_or("?");
        let number = args.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        format!("Getting PR {owner}/{repo}#{number}")
    }
}

// ---------------------------------------------------------------------------
// github_comment_on_pr
// ---------------------------------------------------------------------------

/// Comment on a GitHub pull request.
pub struct GitHubCommentOnPrTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct CommentOnPrArgs {
    owner: String,
    repo: String,
    number: u64,
    body: String,
}

#[async_trait]
impl Tool for GitHubCommentOnPrTool {
    fn name(&self) -> &str {
        "github_comment_on_pr"
    }

    fn description(&self) -> &str {
        "Add a comment to a GitHub pull request."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "number": {
                    "type": "integer",
                    "description": "Pull request number"
                },
                "body": {
                    "type": "string",
                    "description": "Comment body (Markdown)"
                }
            },
            "required": ["owner", "repo", "number", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CommentOnPrArgs = parse_args(self.name(), ctx.args)?;

        // PR comments use the issues endpoint
        let path = format!(
            "repos/{}/{}/issues/{}/comments",
            args.owner, args.repo, args.number
        );
        let payload = json!({ "body": args.body });
        let comment: Comment = self.client.post(&path, &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Comment {} added to PR #{}.", comment.id, args.number),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let number = args.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        format!("Commenting on PR #{number}")
    }
}

// ---------------------------------------------------------------------------
// github_merge_pr
// ---------------------------------------------------------------------------

/// Merge a GitHub pull request.
pub struct GitHubMergePrTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct MergePrArgs {
    owner: String,
    repo: String,
    number: u64,
    merge_method: Option<String>,
    commit_title: Option<String>,
}

#[async_trait]
impl Tool for GitHubMergePrTool {
    fn name(&self) -> &str {
        "github_merge_pr"
    }

    fn description(&self) -> &str {
        "Merge a GitHub pull request. Supports merge, squash, and rebase strategies."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "number": {
                    "type": "integer",
                    "description": "Pull request number"
                },
                "merge_method": {
                    "type": "string",
                    "description": "Merge strategy: merge, squash, or rebase (default: merge)"
                },
                "commit_title": {
                    "type": "string",
                    "description": "Custom title for the merge commit"
                }
            },
            "required": ["owner", "repo", "number"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MergePrArgs = parse_args(self.name(), ctx.args)?;

        let path = format!(
            "repos/{}/{}/pulls/{}/merge",
            args.owner, args.repo, args.number
        );
        let mut payload = json!({});
        if let Some(method) = &args.merge_method {
            payload["merge_method"] = json!(method);
        }
        if let Some(title) = &args.commit_title {
            payload["commit_title"] = json!(title);
        }

        let result: MergeResult = self.client.put(&path, &payload).await?;

        let sha_info = result
            .sha
            .as_deref()
            .map(|s| format!(" (SHA: {s})"))
            .unwrap_or_default();
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("PR #{}: {}{sha_info}", args.number, result.message),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let number = args.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        let method = args
            .get("merge_method")
            .and_then(|v| v.as_str())
            .unwrap_or("merge");
        format!("Merging PR #{number} ({method})")
    }
}

// ---------------------------------------------------------------------------
// github_list_notifications
// ---------------------------------------------------------------------------

/// List GitHub notifications for the authenticated user.
pub struct GitHubListNotificationsTool {
    /// Shared GitHub API client.
    pub client: Arc<GitHubClient>,
}

#[derive(Deserialize)]
struct ListNotificationsArgs {
    all: Option<bool>,
    participating: Option<bool>,
}

#[async_trait]
impl Tool for GitHubListNotificationsTool {
    fn name(&self) -> &str {
        "github_list_notifications"
    }

    fn description(&self) -> &str {
        "List GitHub notifications for the authenticated user."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "all": {
                    "type": "boolean",
                    "description": "Include read notifications (default: false)"
                },
                "participating": {
                    "type": "boolean",
                    "description": "Only show notifications where the user is participating (default: false)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListNotificationsArgs = parse_args(self.name(), ctx.args)?;
        let all = args.all.unwrap_or(false);
        let participating = args.participating.unwrap_or(false);

        let path = format!("notifications?all={all}&participating={participating}");
        let notifications: Vec<Notification> = self.client.get(&path).await?;

        let mut out = String::new();
        if notifications.is_empty() {
            out.push_str("No notifications.");
        } else {
            for notif in &notifications {
                let unread = if notif.unread { "UNREAD" } else { "read" };
                let _ = writeln!(
                    out,
                    "- [{}] [{}] {} ({}) — {}\n  Reason: {}",
                    unread,
                    notif.subject.subject_type,
                    notif.subject.title,
                    notif.repository.full_name,
                    notif.updated_at,
                    notif.reason,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing GitHub notifications".into()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> Arc<GitHubClient> {
        Arc::new(GitHubClient::new_for_test())
    }

    // -- name() tests -------------------------------------------------------

    #[test]
    fn tool_names() {
        let c = make_client();
        assert_eq!(
            GitHubListReposTool { client: c.clone() }.name(),
            "github_list_repos"
        );
        assert_eq!(
            GitHubSearchIssuesTool { client: c.clone() }.name(),
            "github_search_issues"
        );
        assert_eq!(
            GitHubGetIssueTool { client: c.clone() }.name(),
            "github_get_issue"
        );
        assert_eq!(
            GitHubCreateIssueTool { client: c.clone() }.name(),
            "github_create_issue"
        );
        assert_eq!(
            GitHubCommentOnIssueTool { client: c.clone() }.name(),
            "github_comment_on_issue"
        );
        assert_eq!(
            GitHubListPrsTool { client: c.clone() }.name(),
            "github_list_prs"
        );
        assert_eq!(
            GitHubGetPrTool { client: c.clone() }.name(),
            "github_get_pr"
        );
        assert_eq!(
            GitHubCommentOnPrTool { client: c.clone() }.name(),
            "github_comment_on_pr"
        );
        assert_eq!(
            GitHubMergePrTool { client: c.clone() }.name(),
            "github_merge_pr"
        );
        assert_eq!(
            GitHubListNotificationsTool { client: c.clone() }.name(),
            "github_list_notifications"
        );
    }

    // -- humanize() tests ---------------------------------------------------

    #[test]
    fn humanize_list_repos() {
        let c = make_client();
        let tool = GitHubListReposTool { client: c };
        assert_eq!(tool.humanize(&json!({})), "Listing GitHub repositories");
    }

    #[test]
    fn humanize_search_issues() {
        let c = make_client();
        let tool = GitHubSearchIssuesTool { client: c };
        let h = tool.humanize(&json!({ "query": "is:open bug" }));
        assert!(h.contains("is:open bug"));
    }

    #[test]
    fn humanize_get_issue() {
        let c = make_client();
        let tool = GitHubGetIssueTool { client: c };
        let h = tool.humanize(&json!({ "owner": "rust-lang", "repo": "rust", "number": 42 }));
        assert!(h.contains("rust-lang/rust#42"));
    }

    #[test]
    fn humanize_create_issue() {
        let c = make_client();
        let tool = GitHubCreateIssueTool { client: c };
        let h = tool.humanize(&json!({ "title": "Fix bug" }));
        assert!(h.contains("Fix bug"));
    }

    #[test]
    fn humanize_comment_on_issue() {
        let c = make_client();
        let tool = GitHubCommentOnIssueTool { client: c };
        let h = tool.humanize(&json!({ "number": 7 }));
        assert!(h.contains("#7"));
    }

    #[test]
    fn humanize_list_prs() {
        let c = make_client();
        let tool = GitHubListPrsTool { client: c };
        let h = tool.humanize(&json!({ "owner": "org", "repo": "app" }));
        assert!(h.contains("org/app"));
    }

    #[test]
    fn humanize_get_pr() {
        let c = make_client();
        let tool = GitHubGetPrTool { client: c };
        let h = tool.humanize(&json!({ "owner": "org", "repo": "app", "number": 10 }));
        assert!(h.contains("org/app#10"));
    }

    #[test]
    fn humanize_merge_pr() {
        let c = make_client();
        let tool = GitHubMergePrTool { client: c };
        let h = tool.humanize(&json!({ "number": 5, "merge_method": "squash" }));
        assert!(h.contains("#5"));
        assert!(h.contains("squash"));
    }

    #[test]
    fn humanize_list_notifications() {
        let c = make_client();
        let tool = GitHubListNotificationsTool { client: c };
        assert_eq!(tool.humanize(&json!({})), "Listing GitHub notifications");
    }

    // -- parameters() schema tests ------------------------------------------

    #[test]
    fn parameters_are_objects() {
        let c = make_client();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(GitHubListReposTool { client: c.clone() }),
            Box::new(GitHubSearchIssuesTool { client: c.clone() }),
            Box::new(GitHubGetIssueTool { client: c.clone() }),
            Box::new(GitHubCreateIssueTool { client: c.clone() }),
            Box::new(GitHubCommentOnIssueTool { client: c.clone() }),
            Box::new(GitHubListPrsTool { client: c.clone() }),
            Box::new(GitHubGetPrTool { client: c.clone() }),
            Box::new(GitHubCommentOnPrTool { client: c.clone() }),
            Box::new(GitHubMergePrTool { client: c.clone() }),
            Box::new(GitHubListNotificationsTool { client: c.clone() }),
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

        // Tools with required fields
        let search = GitHubSearchIssuesTool { client: c.clone() };
        let required = search.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"query".to_string()));

        let get_issue = GitHubGetIssueTool { client: c.clone() };
        let required = get_issue.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"owner".to_string()));
        assert!(required.contains(&"repo".to_string()));
        assert!(required.contains(&"number".to_string()));
    }
}
