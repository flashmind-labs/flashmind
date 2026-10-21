//! Git worktree discovery, creation and per-agent switching.

use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::Command;

use crate::bash::CommandApproval;
use flashmind_types::tool::{CommandAllowList, Tool, ToolContext, ToolResult};

/// Manage worktrees without changing another agent's directory.
#[derive(Default)]
pub struct WorktreeTool {
    pub(crate) allowlist: Option<Arc<dyn CommandAllowList>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    List,
    Status,
    Create,
    Switch,
}

#[derive(Deserialize)]
struct Args {
    action: Action,
    path: Option<PathBuf>,
    name: Option<String>,
    branch: Option<String>,
    base: Option<String>,
    #[serde(default)]
    switch: bool,
}

#[derive(Default, Serialize)]
struct Worktree {
    path: PathBuf,
    branch: Option<String>,
    head: Option<String>,
    bare: bool,
    current: bool,
    available: bool,
    dirty: bool,
    changes: Vec<String>,
}

async fn git(cwd: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .kill_on_drop(true)
        .output()
        .await
        .context("could not run git")?;
    if !output.status.success() {
        bail!(
            "git failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("git returned non-UTF-8 output")
}

fn parse_worktrees(output: &str) -> Result<Vec<Worktree>> {
    let mut trees = Vec::new();
    for record in output.split("\0\0").filter(|record| !record.is_empty()) {
        let mut tree = Worktree::default();
        for field in record.split('\0') {
            if let Some(path) = field.strip_prefix("worktree ") {
                tree.path = PathBuf::from(path);
            } else if let Some(branch) = field.strip_prefix("branch ") {
                tree.branch = Some(
                    branch
                        .strip_prefix("refs/heads/")
                        .unwrap_or(branch)
                        .to_owned(),
                );
            } else if let Some(head) = field.strip_prefix("HEAD ") {
                tree.head = Some(head.to_owned());
            } else if field == "bare" {
                tree.bare = true;
            }
        }
        if tree.path.as_os_str().is_empty() {
            bail!("git returned a worktree without a path");
        }
        trees.push(tree);
        if trees.len() > 256 {
            bail!("repository has more than 256 worktrees");
        }
    }
    if trees.is_empty() {
        bail!("not a Git repository with worktrees");
    }
    Ok(trees)
}

async fn inspect(tree: &mut Worktree, cwd: &Path) -> Result<()> {
    tree.current = tree.path == cwd;
    tree.available = match tokio::fs::metadata(&tree.path).await {
        Ok(metadata) => metadata.is_dir(),
        Err(error) if error.kind() == ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    if tree.available && !tree.bare {
        tree.changes = git(
            &tree.path,
            &["status", "--short", "--untracked-files=normal"],
        )
        .await?
        .lines()
        .map(str::to_owned)
        .collect();
    }
    tree.dirty = !tree.changes.is_empty();
    Ok(())
}

impl WorktreeTool {
    async fn run(&self, ctx: &ToolContext<'_>, args: Args) -> Result<ToolResult> {
        let cwd = ctx
            .working_dir
            .context("worktree requires an agent working directory")?;
        let cwd = tokio::fs::canonicalize(cwd).await?;
        let mut trees =
            parse_worktrees(&git(&cwd, &["worktree", "list", "--porcelain", "-z"]).await?)?;
        let root = trees[0].path.clone();
        let current = if trees[0].bare && cwd.starts_with(&root) {
            root.clone()
        } else {
            let top = git(&cwd, &["rev-parse", "--show-toplevel"]).await?;
            PathBuf::from(top.strip_suffix('\n').unwrap_or(&top))
        };
        if matches!(args.action, Action::Switch) && args.path.is_none() {
            bail!("switch requires path from list");
        }
        if matches!(args.action, Action::List) {
            for tree in &mut trees {
                inspect(tree, &current).await?;
            }
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                serde_json::to_string_pretty(&trees)?,
            ));
        }
        let target = match args.action {
            Action::Create => {
                let name = args.name.context("create requires name")?;
                let mut components = Path::new(&name).components();
                if !matches!(components.next(), Some(Component::Normal(_)))
                    || components.next().is_some()
                {
                    bail!("name must be a single directory name");
                }
                let branch = args.branch.unwrap_or_else(|| name.clone());
                git(&root, &["check-ref-format", "--branch", &branch]).await?;
                let base = format!("{}^{{commit}}", args.base.as_deref().unwrap_or("HEAD"));
                let revision =
                    git(&cwd, &["rev-parse", "--verify", "--end-of-options", &base]).await?;
                let directory = root.join(".worktrees");
                match tokio::fs::symlink_metadata(&directory).await {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        bail!(".worktrees must be a directory inside the repository")
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
                let path = directory.join(&name);
                let label = format!(
                    "git -C {root:?} worktree add -b {branch:?} -- {path:?} {:?}",
                    revision.trim()
                );
                if let Some(policy) = &self.allowlist
                    && !policy.is_allowed(&label)
                {
                    return Ok(ToolResult::interrupt(
                        ctx.tool_call_id,
                        Arc::new(CommandApproval::new(label, policy.clone())),
                    ));
                }
                tokio::fs::create_dir_all(&directory).await?;
                git(
                    &root,
                    &[
                        "worktree",
                        "add",
                        "-b",
                        &branch,
                        "--",
                        path.to_str().context("worktree path is not UTF-8")?,
                        revision.trim(),
                    ],
                )
                .await?;
                trees =
                    parse_worktrees(&git(&cwd, &["worktree", "list", "--porcelain", "-z"]).await?)?;
                tokio::fs::canonicalize(path).await?
            }
            Action::Status | Action::Switch => {
                let path = args.path.unwrap_or_else(|| current.clone());
                let path = if path.is_absolute() {
                    path
                } else {
                    cwd.join(path)
                };
                tokio::fs::canonicalize(path)
                    .await
                    .context("worktree path does not exist")?
            }
            Action::List => unreachable!("list returns before selecting a worktree"),
        };
        let tree = trees
            .iter_mut()
            .find(|tree| tree.path == target)
            .context("path is not a worktree of the current repository; use a path from list")?;
        if tree.bare && (matches!(args.action, Action::Switch) || args.switch) {
            bail!("cannot switch to a bare repository");
        }
        inspect(tree, &current).await?;
        let switching = matches!(args.action, Action::Switch)
            || (matches!(args.action, Action::Create) && args.switch);
        if switching {
            tree.current = true;
        }
        let result = ToolResult::success(ctx.tool_call_id, serde_json::to_string_pretty(tree)?);
        Ok(if switching {
            result.with_working_dir(target)
        } else {
            result
        })
    }
}

#[async_trait]
impl Tool for WorktreeTool {
    fn name(&self) -> &str {
        "worktree"
    }
    fn description(&self) -> &str {
        "List Git worktrees and their changes, inspect status, create a worktree under the main repository's .worktrees directory, or switch this agent to an existing worktree. Switching changes the session's working directory for subsequent commands and file tools. Other agents keep their own directories."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object", "properties":{
            "action":{"type":"string","enum":["list","status","create","switch"]},
            "path":{"type":"string","description":"Worktree root from list, absolute or relative to the current directory. Status defaults to the current worktree. Switch requires a path."},
            "name":{"type":"string","description":"Directory name under .worktrees for create."},
            "branch":{"type":"string","description":"New branch for create; defaults to name."},
            "base":{"type":"string","description":"Starting commit or ref for create; defaults to HEAD."},
            "switch":{"type":"boolean","description":"Also attach this session when creating a worktree; default false."}
        },"required":["action"]})
    }
    fn concurrent(&self) -> bool {
        false
    }
    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args = ctx.parse_args(self.name())?;
        match self.run(&ctx, args).await {
            Ok(result) => Ok(result),
            Err(error) => Ok(ToolResult::failure(ctx.tool_call_id, format!("{error:#}"))),
        }
    }
    fn humanize(&self, args: &Value) -> String {
        format!("Worktree {}", args["action"].as_str().unwrap_or("status"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    async fn repository() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q", "-b", "main"])
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("example.txt"), "main content\n")
            .await
            .unwrap();
        git(dir.path(), &["add", "example.txt"]).await.unwrap();
        git(
            dir.path(),
            &[
                "-c",
                "user.name=Example",
                "-c",
                "user.email=user@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "Initial example",
            ],
        )
        .await
        .unwrap();
        dir
    }

    async fn run(tool: &WorktreeTool, cwd: &Path, args: Value) -> ToolResult {
        let cwd = cwd.to_path_buf();
        let cancel = CancellationToken::new();
        tool.execute(ToolContext::new("test", args, Some(&cwd), &cancel))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn create_list_status_and_switch_keep_checkout_state_separate() {
        let repo = repository().await;
        let root = repo.path().canonicalize().unwrap();
        let tool = WorktreeTool::default();
        let result = run(
            &tool,
            &root,
            json!({"action":"create", "name":"task", "branch":"feature/task", "switch":true}),
        )
        .await;
        assert!(result.is_success(), "{}", result.output());
        let target = root.join(".worktrees/task").canonicalize().unwrap();
        assert_eq!(result.working_dir(), Some(target.as_path()));
        assert_eq!(
            git(&root, &["branch", "--show-current"])
                .await
                .unwrap()
                .trim(),
            "main"
        );
        assert_eq!(
            git(&target, &["branch", "--show-current"])
                .await
                .unwrap()
                .trim(),
            "feature/task"
        );
        tokio::fs::write(target.join("example.txt"), "worktree change\n")
            .await
            .unwrap();
        let result = run(&tool, &target, json!({"action":"status"})).await;
        let status: Value = serde_json::from_str(&result.output()).unwrap();
        assert_eq!(status["current"], true);
        assert_eq!(status["changes"][0], " M example.txt");
        let subdir = target.join("subdir");
        tokio::fs::create_dir(&subdir).await.unwrap();
        let result = run(&tool, &subdir, json!({"action":"list"})).await;
        let list: Value = serde_json::from_str(&result.output()).unwrap();
        assert_eq!(list.as_array().unwrap().len(), 2);
        assert_eq!(list[1]["current"], true);
        let switched = run(&tool, &target, json!({"action":"switch", "path":root})).await;
        assert_eq!(switched.working_dir(), Some(root.as_path()));
        assert_eq!(
            tokio::fs::read_to_string(root.join("example.txt"))
                .await
                .unwrap(),
            "main content\n"
        );
    }

    #[tokio::test]
    async fn invalid_targets_and_failed_creation_do_not_switch() {
        let repo = repository().await;
        let other = repository().await;
        let tool = WorktreeTool::default();
        for args in [
            json!({"action":"switch"}),
            json!({"action":"switch", "path":other.path()}),
            json!({"action":"create", "name":"../outside", "switch":true}),
            json!({"action":"create", "name":"task", "branch":"main", "switch":true}),
            json!({"action":"create", "name":"task", "base":"--invalid", "switch":true}),
        ] {
            let result = run(&tool, repo.path(), args).await;
            assert!(!result.is_success(), "{}", result.output());
            assert!(result.working_dir().is_none());
        }
        assert_eq!(
            parse_worktrees(
                &git(repo.path(), &["worktree", "list", "--porcelain", "-z"])
                    .await
                    .unwrap()
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[tokio::test]
    async fn creation_obeys_command_approval_before_mutating_the_repository() {
        struct Deny;
        impl CommandAllowList for Deny {
            fn is_allowed(&self, _: &str) -> bool {
                false
            }
            fn add_session_pattern(&self, _: &str) {}
        }
        let repo = repository().await;
        let tool = WorktreeTool {
            allowlist: Some(Arc::new(Deny)),
        };
        let result = run(
            &tool,
            repo.path(),
            json!({"action":"create", "name":"task", "switch":true}),
        )
        .await;
        assert!(result.is_interrupt());
        assert!(!repo.path().join(".worktrees").exists());
        assert!(
            git(repo.path(), &["show-ref", "--verify", "refs/heads/task"])
                .await
                .is_err()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn creation_rejects_an_escaping_worktree_directory_symlink() {
        use std::os::unix::fs::symlink;
        let repo = repository().await;
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), repo.path().join(".worktrees")).unwrap();
        let result = run(
            &WorktreeTool::default(),
            repo.path(),
            json!({"action":"create", "name":"task"}),
        )
        .await;
        assert!(!result.is_success());
        assert!(!outside.path().join("task").exists());
    }
}
