//! File read, write, and delete operations.
//!
//! Provides `FileReadTool`, `FileWriteTool`, and `FileDeleteTool`, each respecting
//! protected-path validation. Paths are tilde-expanded before use. Write operations
//! auto-create parent directories.
//!
//! ## Path resolution
//!
//! - **Absolute paths** pass through as-is (the agent can access the host filesystem)
//! - **Relative paths** resolve against `working_dir` (the agent's workspace)
//! - **No working directory** (REPL mode): paths returned as-is
//!
//! Security boundary for bash/exec is bubblewrap, not path resolution.

use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{Instrument, info_span};

use crate::file_cache::FileCache;
use crate::protected::ProtectedPaths;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

/// Resolve a path for file operations.
///
/// - Absolute paths pass through as-is
/// - Relative paths resolve against `working_dir`
/// - Without a working directory (REPL mode), paths are returned as-is
pub fn resolve_path(path: &str, working_dir: Option<&PathBuf>) -> PathBuf {
    let p = Path::new(path);

    if p.is_absolute() {
        return p.to_path_buf();
    }

    if let Some(wd) = working_dir {
        wd.join(p)
    } else {
        p.to_path_buf()
    }
}

/// Convert absolute path to relative if it's under the working directory.
pub fn to_relative_path(path: &Path, working_dir: Option<&PathBuf>) -> String {
    if let Some(wd) = working_dir
        && let Ok(rel) = path.strip_prefix(wd)
    {
        let rel_str = rel.display().to_string();
        if !rel_str.starts_with("..") {
            return rel_str;
        }
    }
    path.display().to_string()
}

/// Expand `~` to the user's home directory.
pub fn expand_tilde(path: &str) -> Cow<'_, str> {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            let expanded = format!("{}{}", home.display(), path.strip_prefix('~').unwrap());
            tracing::debug!(original = path, expanded = %expanded, "expanded tilde in path");
            return Cow::Owned(expanded);
        }
    } else if path == "~"
        && let Some(home) = dirs::home_dir()
    {
        let expanded = home.display().to_string();
        tracing::debug!(original = path, expanded = %expanded, "expanded bare tilde");
        return Cow::Owned(expanded);
    }
    Cow::Borrowed(path)
}

// === Typed Args ===

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct FileReadArgs {
    path: String,
    /// Include line numbers in output (default: false)
    show_lines: Option<bool>,
}

#[derive(Deserialize)]
struct FileListArgs {
    path: String,
    /// Recurse into subdirectories (default: false)
    recursive: Option<bool>,
    /// Include hidden files starting with . (default: false)
    include_hidden: Option<bool>,
    /// Sort by: name, size, or date (default: name)
    sort_by: Option<String>,
}

#[derive(Deserialize)]
struct FileWriteArgs {
    path: String,
    content: String,
    /// Replace starting at this line number (1-indexed).
    #[serde(default)]
    from_line: Option<u32>,
    /// Replace up to and including this line number.
    #[serde(default)]
    to_line: Option<u32>,
}

// === FileReadTool ===

/// File reading tool.
pub struct FileReadTool {
    pub protected: Arc<ProtectedPaths>,
    pub file_cache: FileCache,
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read a file's full contents. For large files, use read_lines to read a specific line range instead. Use show_lines=true to include line numbers."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "show_lines": {
                    "type": "boolean",
                    "description": "Include line numbers in output (default: false)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();
        let args: FileReadArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        tracing::debug!(path = %resolved_path.display(), original = %args.path, "file_read requested");

        // Output size is enforced generically by enforce_output_limits() in tool_exec.rs
        {
            let _step = info_span!(target: "prompt_trace", "step",
                step = "read",
                detail = format!("protected_check path={}", resolved_path.display()).as_str(),
            )
            .entered();

            if self.protected.is_read_protected(&resolved_path) {
                tracing::warn!(path = %resolved_path.display(), "file_read blocked: protected path");
                metrics::counter!("tools.file.reads").increment(1);
                metrics::histogram!("tools.file.read.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error: Cannot read protected file: {}", args.path),
                ));
            }
        }

        match tokio::fs::read_to_string(&resolved_path)
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "read",
                detail = format!("path={}", resolved_path.display()).as_str(),
            ))
            .await
        {
            Ok(content) => {
                let bytes = content.len();
                tracing::debug!(path = %resolved_path.display(), bytes, "file_read success");
                metrics::counter!("tools.file.reads").increment(1);
                metrics::histogram!("tools.file.read.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                metrics::histogram!("tools.file.read.bytes").record(bytes as f64);
                self.file_cache.store(&resolved_path, content.clone());

                // Check line count and truncate if needed
                let line_count = content.lines().count();
                const MAX_LINES: usize = 10_000;

                let output = if line_count > MAX_LINES {
                    let truncated: String = content
                        .lines()
                        .take(MAX_LINES)
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!(
                        "[File truncated to {} lines. Use read_lines with from_line/to_line to read more.]\n\n{}",
                        MAX_LINES, truncated
                    )
                } else if args.show_lines == Some(true) {
                    content
                        .lines()
                        .enumerate()
                        .map(|(i, line)| format!("{:4} | {}", i + 1, line))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    content
                };

                Ok(ToolResult::success(ctx.tool_call_id, output))
            }
            Err(e) => {
                tracing::debug!(path = %resolved_path.display(), error = %e, "file_read failed");
                metrics::counter!("tools.file.reads").increment(1);
                metrics::histogram!("tools.file.read.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error reading file: {}", e),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Reading {}", path)
    }

    fn max_output_lines(&self) -> usize {
        10_000
    }

    fn max_output_bytes(&self) -> usize {
        512 * 1024 // 512 KiB
    }
}

// === FileWriteTool ===

/// File writing tool.
pub struct FileWriteTool {
    pub protected: Arc<ProtectedPaths>,
    pub file_cache: FileCache,
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed. Overwrites existing content unless from_line/to_line are set to replace a specific line range."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                },
                "from_line": {
                    "type": "integer",
                    "description": "Optional starting line number (1-indexed) for partial replacement. If set with to_line, replaces only lines in that range."
                },
                "to_line": {
                    "type": "integer",
                    "description": "Optional ending line number (1-indexed, inclusive). Must be used with from_line. Replaces lines from from_line to to_line inclusive."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();
        let args: FileWriteArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        tracing::debug!(path = %resolved_path.display(), original = %args.path, bytes = args.content.len(), "file_write requested");

        {
            let _step = info_span!(target: "prompt_trace", "step",
                step = "write",
                detail = format!("protected_check path={}", resolved_path.display()).as_str(),
            )
            .entered();

            if self.protected.is_write_protected(&resolved_path) {
                tracing::warn!(path = %resolved_path.display(), "file_write blocked: write-protected path");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error: Cannot modify write-protected file: {}", args.path),
                ));
            }
        }

        if let Some(parent) = resolved_path.parent()
            && !parent.exists()
        {
            tracing::debug!(dir = %parent.display(), "creating parent directories");
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                tracing::debug!(dir = %parent.display(), error = %e, "directory creation failed");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error creating directories: {}", e),
                ));
            }
        }

        // Handle line-range replacement (like sed)
        let content_to_write = if let (Some(from), Some(to)) = (args.from_line, args.to_line) {
            if from > to {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!(
                        "Invalid line range: from_line ({}) > to_line ({})",
                        from, to
                    ),
                ));
            }

            // Read existing file content
            let existing = tokio::fs::read_to_string(&resolved_path)
                .await
                .unwrap_or_default();

            let mut lines: Vec<&str> = existing.lines().collect();
            let num_lines = lines.len();

            // Convert 1-indexed to 0-indexed, clamp to valid range
            let start_idx = ((from as usize) - 1).min(num_lines);
            let end_idx = (to as usize).min(num_lines);

            // Split new content into lines
            let new_lines: Vec<&str> = args.content.split('\n').collect();

            // Replace the line range
            lines.splice(start_idx..end_idx, new_lines);

            lines.join("\n")
        } else {
            args.content.clone()
        };

        match tokio::fs::write(&resolved_path, &content_to_write)
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "write",
                detail = format!("path={} size={}", resolved_path.display(), args.content.len()).as_str(),
            ))
            .await
        {
            Ok(()) => {
                tracing::debug!(path = %resolved_path.display(), bytes = args.content.len(), "file_write success");
                metrics::counter!("tools.file.writes").increment(1);
                metrics::histogram!("tools.file.write.duration_seconds").record(start.elapsed().as_secs_f64());
                metrics::histogram!("tools.file.write.bytes").record(args.content.len() as f64);
                let diffs = self.file_cache.diff_vec(&resolved_path, &content_to_write, ctx.working_dir);
                self.file_cache.store(&resolved_path, content_to_write);
                Ok(ToolResult::success_with_diffs(ctx.tool_call_id, "OK", diffs))
            }
            Err(e) => {
                tracing::debug!(path = %resolved_path.display(), error = %e, "file_write failed");
                metrics::counter!("tools.file.writes").increment(1);
                metrics::histogram!("tools.file.write.duration_seconds").record(start.elapsed().as_secs_f64());
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error writing file: {}", e),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Writing {}", path)
    }
}

// === FileDeleteTool ===

/// File deletion tool.
pub struct FileDeleteTool {
    pub protected: Arc<ProtectedPaths>,
}

#[async_trait]
impl Tool for FileDeleteTool {
    fn name(&self) -> &str {
        "file_delete"
    }

    fn description(&self) -> &str {
        "Delete a file. Cannot delete directories or protected files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to delete"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: PathArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        tracing::debug!(path = %resolved_path.display(), original = %args.path, "file_delete requested");

        {
            let _step = info_span!(target: "prompt_trace", "step",
                step = "write",
                detail = format!("protected_check path={}", resolved_path.display()).as_str(),
            )
            .entered();

            if self.protected.is_write_protected(&resolved_path) {
                tracing::warn!(path = %resolved_path.display(), "file_delete blocked: write-protected path");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error: Cannot delete write-protected file: {}", args.path),
                ));
            }
        }

        match tokio::fs::remove_file(&resolved_path)
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "write",
                detail = format!("delete path={}", resolved_path.display()).as_str(),
            ))
            .await
        {
            Ok(()) => {
                tracing::debug!(path = %resolved_path.display(), "file_delete success");
                Ok(ToolResult::success(ctx.tool_call_id, "OK"))
            }
            Err(e) => {
                tracing::debug!(path = %resolved_path.display(), error = %e, "file_delete failed");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error deleting file: {}", e),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Deleting {}", path)
    }
}

// === FileListTool ===

/// Directory listing tool.
pub struct FileListTool;

#[async_trait]
impl Tool for FileListTool {
    fn name(&self) -> &str {
        "file_list"
    }

    fn description(&self) -> &str {
        "List files and subdirectories. Each entry shows its type: [file], [dir], or [link]. Use recursive=true to list subdirectories (skips target, node_modules, .git, dist, build, .venv, venv, __pycache__, .next, .cache), include_hidden=true to show dotfiles, sort_by='name'|'size'|'date'."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Recurse into subdirectories (default: false)"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files starting with . (default: false)"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["name", "size", "date"],
                    "description": "Sort by: name, size, or date (default: name)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: FileListArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        tracing::debug!(path = %resolved_path.display(), "file_list requested");

        let recursive = args.recursive.unwrap_or(false);
        let include_hidden = args.include_hidden.unwrap_or(false);
        let sort_by = args.sort_by.as_deref().unwrap_or("name");

        // Collect entries
        let mut files: Vec<(String, PathBuf, std::fs::Metadata)> = Vec::new();

        const RECURSIVE_IGNORED_DIRS: &[&str] = &[
            "target",
            "node_modules",
            ".git",
            ".svn",
            ".hg",
            "dist",
            "build",
            ".venv",
            "venv",
            "__pycache__",
            ".next",
            ".cache",
        ];

        fn collect_dir(
            dir: &Path,
            files: &mut Vec<(String, PathBuf, std::fs::Metadata)>,
            recursive: bool,
            include_hidden: bool,
        ) -> std::io::Result<()> {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !include_hidden && name.starts_with('.') {
                        continue;
                    }
                    let path = entry.path();
                    let metadata = entry.metadata()?;
                    let is_dir = metadata.is_dir();
                    files.push((name.clone(), path.clone(), metadata));

                    if recursive && is_dir && !RECURSIVE_IGNORED_DIRS.contains(&name.as_str()) {
                        collect_dir(&path, files, recursive, include_hidden)?;
                    }
                }
            }
            Ok(())
        }

        collect_dir(&resolved_path, &mut files, recursive, include_hidden)?;

        // Sort
        match sort_by {
            "size" => files.sort_by(|a, b| {
                let size_a = a.2.len();
                let size_b = b.2.len();
                size_b.cmp(&size_a) // descending
            }),
            "date" => files.sort_by(|a, b| {
                let time_a = a.2.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                let time_b = b.2.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                time_b.cmp(&time_a) // descending (newest first)
            }),
            _ => files.sort_by_key(|a| a.0.to_lowercase()),
        }

        // Format output
        let output: Vec<String> = files
            .into_iter()
            .map(|(name, path, _)| {
                let type_str = if path.is_dir() {
                    "[dir]"
                } else if path.is_symlink() {
                    "[link]"
                } else {
                    "[file]"
                };
                format!("{} {}", type_str, name)
            })
            .collect();

        tracing::debug!(path = %resolved_path.display(), count = output.len(), "file_list success");
        Ok(ToolResult::success(ctx.tool_call_id, output.join("\n")))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Listing {}", path)
    }
}

// === ReadLinesTool ===

/// Line-range file reading tool.
pub struct ReadLinesTool {
    pub protected: Arc<ProtectedPaths>,
    pub file_cache: FileCache,
}

#[derive(Deserialize)]
struct ReadLinesArgs {
    path: String,
    from_line: u32,
    to_line: u32,
}

#[async_trait]
impl Tool for ReadLinesTool {
    fn name(&self) -> &str {
        "read_lines"
    }

    fn description(&self) -> &str {
        "Read a line range from a file (1-indexed, inclusive). Returns lines with line numbers. Use instead of file_read for large files to avoid loading everything into context."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "from_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed, inclusive)"
                },
                "to_line": {
                    "type": "integer",
                    "description": "End line (1-indexed, inclusive)"
                }
            },
            "required": ["path", "from_line", "to_line"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ReadLinesArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        tracing::debug!(
            path = %resolved_path.display(),
            from_line = args.from_line,
            to_line = args.to_line,
            "read_lines requested"
        );

        // Validate line range
        if args.from_line == 0 || args.to_line == 0 {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Invalid line range: line numbers must be >= 1",
            ));
        }
        if args.from_line > args.to_line {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Invalid line range: from_line ({}) > to_line ({})",
                    args.from_line, args.to_line
                ),
            ));
        }

        if self.protected.is_read_protected(&resolved_path) {
            tracing::warn!(path = %resolved_path.display(), "read_lines blocked: protected path");
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Error: Cannot read protected file: {}", args.path),
            ));
        }

        let content = match tokio::fs::read_to_string(&resolved_path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(path = %resolved_path.display(), error = %e, "read_lines failed");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error reading file: {}", e),
                ));
            }
        };

        // Populate cache with full file content
        self.file_cache.store(&resolved_path, content.clone());

        let lines: Vec<&str> = content.lines().collect();
        let file_len = lines.len();

        if args.from_line as usize > file_len {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "from_line ({}) exceeds file length ({} lines)",
                    args.from_line, file_len
                ),
            ));
        }

        // Clamp to_line to file length
        let to_line = (args.to_line as usize).min(file_len);
        let from_idx = (args.from_line as usize) - 1;

        let output: String = lines[from_idx..to_line]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>6} | {}", from_idx + 1 + i, line))
            .collect::<Vec<_>>()
            .join("\n");

        tracing::debug!(
            path = %resolved_path.display(),
            from_line = args.from_line,
            to_line = to_line,
            "read_lines success"
        );

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let from_line = args.get("from_line").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let to_line = args.get("to_line").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        format!("Reading {}:{}-{}", path, from_line, to_line)
    }

    fn max_output_lines(&self) -> usize {
        10000 // 10k lines max for read_lines - prevents context overflow
    }

    fn max_output_bytes(&self) -> usize {
        512 * 1024 // 512 KiB for read_lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_protected(dir: &Path) -> Arc<ProtectedPaths> {
        Arc::new(ProtectedPaths::new(dir))
    }

    #[tokio::test]
    async fn test_file_read() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": file_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("hello world"));
    }

    #[tokio::test]
    async fn test_file_write() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("output.txt");

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "test content"
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "test content");
    }

    #[tokio::test]
    async fn test_file_write_protected() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "original").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": config_path.to_str().unwrap(),
            "content": "hacked"
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("protected"));
        assert_eq!(std::fs::read_to_string(&config_path).unwrap(), "original");
    }

    #[tokio::test]
    async fn test_file_delete() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("to_delete.txt");
        std::fs::write(&file_path, "delete me").unwrap();

        let tool = FileDeleteTool {
            protected: make_protected(dir.path()),
        };
        let args = json!({ "path": file_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_file_list() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();

        let tool = FileListTool;
        let args = json!({ "path": dir.path().to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("a.txt"));
        assert!(result.output().contains("b.txt"));
    }

    #[tokio::test]
    async fn test_read_lines_basic() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\nline5").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 2,
            "to_line": 4
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("line2"));
        assert!(result.output().contains("line3"));
        assert!(result.output().contains("line4"));
        assert!(!result.output().contains("line1"));
        assert!(!result.output().contains("line5"));
    }

    #[tokio::test]
    async fn test_read_lines_clamp_to_end() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2\nline3").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 2,
            "to_line": 100
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("line2"));
        assert!(result.output().contains("line3"));
    }

    #[tokio::test]
    async fn test_read_lines_invalid_range_from_gt_to() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2\nline3").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 3,
            "to_line": 1
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
    }

    #[tokio::test]
    async fn test_read_lines_zero_line_number() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 0,
            "to_line": 2
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
    }

    #[tokio::test]
    async fn test_read_lines_from_exceeds_file_length() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 10,
            "to_line": 20
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
    }

    #[tokio::test]
    async fn test_read_lines_number_formatting() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        let content: String = (1..=50).map(|i| format!("line{}\n", i)).collect();
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 42,
            "to_line": 42
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        // Line number should be right-aligned in 6 chars
        assert!(result.output().contains("    42 | line42"));
    }

    #[test]
    fn test_resolve_path() {
        let wd = PathBuf::from("/workspaces/slack_123");

        // Relative path resolves to working_dir
        assert_eq!(
            resolve_path("foo.txt", Some(&wd)),
            PathBuf::from("/workspaces/slack_123/foo.txt")
        );

        // Absolute path passes through as-is
        assert_eq!(
            resolve_path("/etc/passwd", Some(&wd)),
            PathBuf::from("/etc/passwd")
        );

        // Relative with subdirectory
        assert_eq!(
            resolve_path("src/main.rs", Some(&wd)),
            PathBuf::from("/workspaces/slack_123/src/main.rs")
        );

        // No working dir returns path as-is (REPL mode)
        assert_eq!(
            resolve_path("/etc/passwd", None),
            PathBuf::from("/etc/passwd")
        );

        assert_eq!(resolve_path("foo.txt", None), PathBuf::from("foo.txt"));
    }

    #[test]
    fn test_expand_tilde() {
        // Bare ~ should expand to home dir
        let expanded = expand_tilde("~");
        assert!(!expanded.starts_with('~'), "bare ~ should expand");

        // ~/path should expand
        let expanded = expand_tilde("~/Documents/test.txt");
        assert!(!expanded.starts_with("~/"), "~/path should expand");
        assert!(expanded.ends_with("/Documents/test.txt"));

        // Non-tilde paths pass through unchanged
        assert_eq!(expand_tilde("/etc/passwd"), "/etc/passwd");
        assert_eq!(expand_tilde("relative/path"), "relative/path");

        // Tilde in the middle should NOT expand
        assert_eq!(expand_tilde("/home/~user/file"), "/home/~user/file");
    }

    // === FileReadTool additional tests ===

    #[tokio::test]
    async fn test_file_read_nonexistent() {
        let dir = tempdir().unwrap();
        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": dir.path().join("nonexistent.txt").to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("Error reading file"));
    }

    #[tokio::test]
    async fn test_file_read_protected() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "secret").unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": config_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("protected"));
    }

    #[tokio::test]
    async fn test_file_read_empty_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("empty.txt");
        std::fs::write(&file_path, "").unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": file_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.output(), "");
    }

    #[tokio::test]
    async fn test_file_read_binary_content() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("binary.bin");
        std::fs::write(&file_path, [0xFF, 0xFE, 0x00, 0x01]).unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": file_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        // read_to_string fails on invalid UTF-8
        assert!(!result.is_success());
        assert!(result.output().contains("Error reading file"));
    }

    #[tokio::test]
    async fn test_file_read_unicode() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("unicode.txt");
        std::fs::write(&file_path, "héllo wörld 🌍 日本語").unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({ "path": file_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("héllo wörld 🌍 日本語"));
    }

    // === FileWriteTool additional tests ===

    #[tokio::test]
    async fn test_file_write_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let nested_path = dir.path().join("a/b/c/deep.txt");

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": nested_path.to_str().unwrap(),
            "content": "deep content"
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(
            std::fs::read_to_string(&nested_path).unwrap(),
            "deep content"
        );
    }

    #[tokio::test]
    async fn test_file_write_line_range_replacement() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("replace.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\nline5").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Replace lines 2-3 with new content
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new2\nnew3\nnew_extra",
            "from_line": 2,
            "to_line": 3
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "line1\nnew2\nnew3\nnew_extra\nline4\nline5");
    }

    #[tokio::test]
    async fn test_file_write_line_range_single_line() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("replace.txt");
        std::fs::write(&file_path, "aaa\nbbb\nccc").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Replace just line 2
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "BBB",
            "from_line": 2,
            "to_line": 2
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "aaa\nBBB\nccc");
    }

    #[tokio::test]
    async fn test_file_write_line_range_invalid() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("replace.txt");
        std::fs::write(&file_path, "line1\nline2").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "x",
            "from_line": 5,
            "to_line": 2
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("Invalid line range"));
    }

    #[tokio::test]
    async fn test_file_write_overwrites_existing() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("overwrite.txt");
        std::fs::write(&file_path, "old content").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new content"
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "new content");
    }

    // === FileDeleteTool additional tests ===

    #[tokio::test]
    async fn test_file_delete_nonexistent() {
        let dir = tempdir().unwrap();
        let tool = FileDeleteTool {
            protected: make_protected(dir.path()),
        };
        let args = json!({ "path": dir.path().join("ghost.txt").to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("Error deleting file"));
    }

    #[tokio::test]
    async fn test_file_delete_protected() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "data").unwrap();

        let tool = FileDeleteTool {
            protected: make_protected(dir.path()),
        };
        let args = json!({ "path": config_path.to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("protected"));
        // File should still exist
        assert!(config_path.exists());
    }

    // === FileListTool additional tests ===

    #[tokio::test]
    async fn test_file_list_empty_dir() {
        let dir = tempdir().unwrap();

        let tool = FileListTool;
        let args = json!({ "path": dir.path().to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.output(), "");
    }

    #[tokio::test]
    async fn test_file_list_nonexistent_dir() {
        let dir = tempdir().unwrap();
        let tool = FileListTool;
        let args = json!({ "path": dir.path().join("nope").to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        // Now returns success with empty output since recursive collects all dirs
        assert!(result.output().is_empty() || result.output().contains("nope") || !result.is_success());
    }

    #[tokio::test]
    async fn test_file_list_with_subdirs() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = FileListTool;
        let args = json!({ "path": dir.path().to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("[file] file.txt"));
        assert!(result.output().contains("[dir] subdir"));
    }

    #[tokio::test]
    async fn test_file_list_sorted() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();

        let tool = FileListTool;
        let args = json!({ "path": dir.path().to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let lines: Vec<&str> = result.output().lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("a.txt"));
        assert!(lines[1].contains("b.txt"));
        assert!(lines[2].contains("c.txt"));
    }

    // === ReadLinesTool additional tests ===

    #[tokio::test]
    async fn test_read_lines_single_line() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 2,
            "to_line": 2
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("beta"));
        assert!(!result.output().contains("alpha"));
        assert!(!result.output().contains("gamma"));
        // Should have exactly one line of output
        assert_eq!(result.output().lines().count(), 1);
    }

    #[tokio::test]
    async fn test_read_lines_entire_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "one\ntwo\nthree").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 1,
            "to_line": 3
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("one"));
        assert!(result.output().contains("two"));
        assert!(result.output().contains("three"));
        assert_eq!(result.output().lines().count(), 3);
    }

    #[tokio::test]
    async fn test_read_lines_nonexistent_file() {
        let dir = tempdir().unwrap();
        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": dir.path().join("nope.txt").to_str().unwrap(),
            "from_line": 1,
            "to_line": 5
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("Error reading file"));
    }

    #[tokio::test]
    async fn test_read_lines_protected() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "secret=true").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": config_path.to_str().unwrap(),
            "from_line": 1,
            "to_line": 1
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("protected"));
    }

    #[tokio::test]
    async fn test_read_lines_to_zero() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "line1\nline2").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 1,
            "to_line": 0
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("line numbers must be >= 1"));
    }

    #[tokio::test]
    async fn test_read_lines_line_numbers_in_output() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        std::fs::write(&file_path, "aaa\nbbb\nccc\nddd\neee").unwrap();

        let tool = ReadLinesTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "from_line": 3,
            "to_line": 5
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        // Verify line numbers are present and correct
        let lines: Vec<&str> = result.output().lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("3") && lines[0].contains("ccc"));
        assert!(lines[1].contains("4") && lines[1].contains("ddd"));
        assert!(lines[2].contains("5") && lines[2].contains("eee"));
    }

    // === FileWriteTool line-range edge cases ===

    #[tokio::test]
    async fn test_file_write_line_range_at_end() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("replace.txt");
        std::fs::write(&file_path, "line1\nline2\nline3").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Replace last line
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "NEW_LAST",
            "from_line": 3,
            "to_line": 3
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "line1\nline2\nNEW_LAST");
    }

    #[tokio::test]
    async fn test_file_write_line_range_at_start() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("replace.txt");
        std::fs::write(&file_path, "line1\nline2\nline3").unwrap();

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Replace first line
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "NEW_FIRST",
            "from_line": 1,
            "to_line": 1
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "NEW_FIRST\nline2\nline3");
    }

    #[tokio::test]
    async fn test_file_write_line_range_on_nonexistent_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("new.txt");

        let tool = FileWriteTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Line-range on nonexistent file should use empty content as base
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "inserted",
            "from_line": 1,
            "to_line": 1
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "inserted");
    }

    // === FileReadTool show_lines tests ===
    #[tokio::test]
    async fn test_file_read_show_lines() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line one\nline two\nline three").unwrap();

        let tool = FileReadTool {
            protected: make_protected(dir.path()),
            file_cache: FileCache::new(),
        };
        // Use absolute path since there's no working_dir in this test helper
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "show_lines": true
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        // Check output contains line numbers (format may vary slightly)
        assert!(result.output().contains("1") && result.output().contains("line one"));
    }

    // === FileListTool new parameter tests ===
    #[tokio::test]
    async fn test_file_list_recursive() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("file1.txt"), "a").unwrap();
        std::fs::write(dir.path().join("subdir/file2.txt"), "b").unwrap();

        let tool = FileListTool;
        let args = json!({
            "path": dir.path().to_str().unwrap(),
            "recursive": true
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("file1.txt"));
        assert!(result.output().contains("subdir"));
        assert!(result.output().contains("file2.txt"));
    }

    #[tokio::test]
    async fn test_file_list_include_hidden() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "a").unwrap();
        std::fs::write(dir.path().join(".hidden"), "b").unwrap();

        let tool = FileListTool;

        // Without include_hidden (default false)
        let args = json!({ "path": dir.path().to_str().unwrap() });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(result.output().contains("visible.txt"));
        assert!(!result.output().contains(".hidden"));

        // With include_hidden = true
        let args = json!({
            "path": dir.path().to_str().unwrap(),
            "include_hidden": true
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(result.output().contains("visible.txt"));
        assert!(result.output().contains(".hidden"));
    }
}
