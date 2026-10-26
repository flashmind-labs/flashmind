use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use flashmind_core::CancellationToken;
use flashmind_tools::file_cache::FileCache;
use flashmind_tools::file_ops::{FileReadTool, FileWriteTool, ReadLinesTool, expand_tilde};
use flashmind_tools::glob::GlobTool;
use flashmind_tools::grep::GrepTool;
use flashmind_tools::lsp::LspManager;
use flashmind_tools::protected::ProtectedPaths;
use flashmind_tools::text_replace::StrReplaceTool;
use flashmind_types::tool::{Tool, ToolContext, ToolRegistry, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

pub fn register_file_tools(
    registry: &mut ToolRegistry,
    protected: &Arc<ProtectedPaths>,
    lsp: Option<Arc<LspManager>>,
) {
    for name in [
        "file_read",
        "file_write",
        "read_lines",
        "glob",
        "grep",
        "str_replace",
        "str_replace_regex",
        "str_diff",
        "file_delete",
        "file_list",
    ] {
        registry.remove(name);
    }
    let cache = FileCache::new();
    registry.register(Arc::new(ReadTool {
        full: FileReadTool {
            protected: protected.clone(),
            file_cache: cache.clone(),
        },
        lines: ReadLinesTool {
            protected: protected.clone(),
            file_cache: cache.clone(),
        },
    }));
    registry.register(Arc::new(WriteTool {
        tool: FileWriteTool {
            protected: protected.clone(),
            file_cache: cache.clone(),
        },
        lsp: lsp.clone(),
    }));
    registry.register(Arc::new(EditTool {
        tool: StrReplaceTool {
            protected: protected.clone(),
            file_cache: cache,
        },
        lsp,
    }));
    registry.register(Arc::new(FindTool {
        files: GlobTool {
            protected: protected.clone(),
        },
        contents: GrepTool,
    }));
    registry.alias("file_read", "read");
    registry.alias("read_lines", "read");
    registry.alias("file_write", "write");
    registry.alias("str_replace", "edit");
}

fn diagnostic_path(ctx: &ToolContext<'_>) -> Option<PathBuf> {
    let path = ctx.args.get("path")?.as_str()?;
    let path = PathBuf::from(expand_tilde(path).as_ref());
    Some(if path.is_absolute() {
        path
    } else {
        ctx.working_dir.map_or(path.clone(), |cwd| cwd.join(&path))
    })
}

async fn append_diagnostics(
    result: &mut ToolResult,
    lsp: Option<&LspManager>,
    path: Option<&Path>,
    cwd: Option<&Path>,
    cancel: &CancellationToken,
) {
    if !result.is_success() {
        return;
    }
    let (Some(lsp), Some(path)) = (lsp, path) else {
        return;
    };
    let message = match lsp.diagnostics_after_edit(path, cwd, cancel).await {
        Ok(Some(message)) => message,
        Ok(None) => return,
        Err(error) => format!("Diagnostics unavailable: {error:#}"),
    };
    if let ToolResult::Success { output, .. } = result {
        output.push_str(&format!("\n\n{message}"));
    }
}

struct ReadTool {
    full: FileReadTool,
    lines: ReadLinesTool,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a text file. Use optional from_line/to_line for an inclusive line range, with line numbers."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.full.parameters();
        let range = self.lines.parameters();
        schema["properties"]["from_line"] = range["properties"]["from_line"].clone();
        schema["properties"]["to_line"] = range["properties"]["to_line"].clone();
        schema
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.args.get("from_line").is_some() || ctx.args.get("to_line").is_some() {
            if ctx.args.get("from_line").is_none() {
                ctx.args["from_line"] = json!(1);
            }
            if ctx.args.get("to_line").is_none() {
                ctx.args["to_line"] = json!(u32::MAX);
            }
            self.lines.execute(ctx).await
        } else {
            self.full.execute(ctx).await
        }
    }

    fn humanize(&self, args: &Value) -> String {
        self.full.humanize(args)
    }
    fn max_output_bytes(&self) -> usize {
        self.full.max_output_bytes()
    }
    fn max_output_lines(&self) -> usize {
        self.full.max_output_lines()
    }
}

struct WriteTool {
    tool: FileWriteTool,
    lsp: Option<Arc<LspManager>>,
}

#[async_trait]
impl Tool for WriteTool {
    fn concurrent(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Create or replace a whole file. Creates parent directories. Use edit for targeted changes."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.tool.parameters();
        if let Some(properties) = schema["properties"].as_object_mut() {
            properties.remove("from_line");
            properties.remove("to_line");
        }
        schema
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.args.get("from_line").is_some() || ctx.args.get("to_line").is_some() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "write replaces a whole file. Use edit for targeted changes.",
            ));
        }
        let path = diagnostic_path(&ctx);
        let cwd = ctx.working_dir.cloned();
        let cancel = ctx.cancel_token().clone();
        let mut result = self.tool.execute(ctx).await?;
        append_diagnostics(
            &mut result,
            self.lsp.as_deref(),
            path.as_deref(),
            cwd.as_deref(),
            &cancel,
        )
        .await;
        Ok(result)
    }

    fn humanize(&self, args: &Value) -> String {
        self.tool.humanize(args)
    }
}

struct EditTool {
    tool: StrReplaceTool,
    lsp: Option<Arc<LspManager>>,
}

#[async_trait]
impl Tool for EditTool {
    fn concurrent(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. old_string must be nonempty and match exactly once; include enough context to make it unique."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.tool.parameters();
        schema["properties"]["old_string"]["description"] =
            json!("Exact text to replace. Must be nonempty and match exactly once.");
        schema["properties"]["old_string"]["minLength"] = json!(1);
        schema
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> Result<ToolResult> {
        if let Some(path) = ctx.args.get("path").and_then(Value::as_str) {
            let expanded = expand_tilde(path).into_owned();
            ctx.args["path"] = json!(expanded);
        }
        let path = diagnostic_path(&ctx);
        let cwd = ctx.working_dir.cloned();
        let cancel = ctx.cancel_token().clone();
        let mut result = self.tool.execute_unique(ctx).await?;
        append_diagnostics(
            &mut result,
            self.lsp.as_deref(),
            path.as_deref(),
            cwd.as_deref(),
            &cancel,
        )
        .await;
        Ok(result)
    }

    fn humanize(&self, args: &Value) -> String {
        self.tool.humanize(args)
    }
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FindMode {
    #[default]
    Files,
    Content,
}

#[derive(Deserialize)]
struct FindArgs {
    #[serde(default)]
    mode: FindMode,
    pattern: String,
    path: Option<String>,
}

struct FindTool {
    files: GlobTool,
    contents: GrepTool,
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by glob pattern (mode=files, default) or search their contents by regex (mode=content). path defaults to the working directory."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.contents.parameters();
        schema["properties"]["mode"] = json!({
            "type": "string", "enum": ["files", "content"], "default": "files",
            "description": "Search filenames with a glob, or file contents with a regex."
        });
        schema["properties"]["pattern"]["description"] =
            json!("Glob for files (e.g. **/*.rs), or regex for content.");
        schema["properties"]["path"]["description"] =
            json!("Search root or file path. Defaults to the working directory.");
        schema["properties"]["max_results"]["description"] = json!(
            "Maximum content matches (default: 1000). Filename searches preview up to 20 paths."
        );
        schema["required"] = json!(["pattern"]);
        schema
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: FindArgs = ctx.parse_args(self.name())?;
        let path = expand_tilde(args.path.as_deref().unwrap_or(".")).into_owned();
        match args.mode {
            FindMode::Files => {
                let pattern = Path::new(&path).join(&args.pattern);
                ctx.args = json!({ "pattern": pattern.to_string_lossy() });
                self.files.execute(ctx).await
            }
            FindMode::Content => {
                ctx.args["path"] = json!(path);
                self.contents.execute(ctx).await
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        self.contents.humanize(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use flashmind_tools::ToolBuilder;
    use flashmind_types::ToolCall;
    use tempfile::{TempDir, tempdir};
    use tokio_util::sync::CancellationToken;

    struct Fixture {
        dir: TempDir,
        registry: ToolRegistry,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempdir().unwrap();
            let protected = Arc::new(ProtectedPaths::new(dir.path()));
            let mut registry = ToolBuilder::new().file_ops(None, &protected).build();
            register_file_tools(&mut registry, &protected, None);
            Self { dir, registry }
        }

        async fn run(&self, name: &str, args: Value) -> ToolResult {
            let call = ToolCall {
                id: "test".into(),
                name: name.into(),
                arguments: args,
            };
            let cwd: PathBuf = self.dir.path().to_path_buf();
            self.registry
                .execute(&call, Some(&cwd), &CancellationToken::new())
                .await
        }
    }

    #[tokio::test]
    async fn writes_and_edits_return_language_diagnostics() {
        use flashmind_tools::lsp::LanguageServerConfig;
        use std::collections::BTreeMap;
        use std::process::Command;

        if !Command::new("python3")
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return;
        }
        let mut fixture = Fixture::new();
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("flashmind-tools/tests/fixtures/lsp_server.py");
        let lsp = Arc::new(LspManager::with_servers(BTreeMap::from([(
            "python".into(),
            LanguageServerConfig {
                command: vec![
                    "python3".into(),
                    "-u".into(),
                    script.to_string_lossy().into_owned(),
                ],
                extensions: vec!["py".into()],
                root_markers: vec!["pyproject.toml".into()],
                initialization: Value::Null,
                disabled: false,
            },
        )])));
        fs::write(fixture.dir.path().join("pyproject.toml"), "").unwrap();
        let protected = Arc::new(ProtectedPaths::new(fixture.dir.path()));
        register_file_tools(&mut fixture.registry, &protected, Some(lsp));
        let written = fixture
            .run("write", json!({"path":"app.py", "content":"broken = 1"}))
            .await;
        assert!(written.is_success());
        assert!(
            written
                .output()
                .contains("error: type error in python version 1"),
            "{}",
            written.output()
        );
        let edited = fixture
            .run(
                "edit",
                json!({"path":"app.py", "old_string":"broken", "new_string":"fixed"}),
            )
            .await;
        assert!(edited.is_success());
        assert!(
            edited.output().contains("No diagnostics reported."),
            "{}",
            edited.output()
        );
        assert_eq!(
            fs::read_to_string(fixture.dir.path().join("app.py")).unwrap(),
            "fixed = 1"
        );
    }

    #[test]
    fn exposes_consolidated_file_tools() {
        let fixture = Fixture::new();
        let names: Vec<_> = fixture
            .registry
            .definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(names, ["edit", "find", "image_read", "read", "write"]);
        for alias in ["file_read", "file_write", "read_lines", "str_replace"] {
            assert!(fixture.registry.contains(alias));
            assert!(!names.iter().any(|name| name == alias));
        }
        let write_schema = fixture.registry.get("write").unwrap().parameters();
        assert!(write_schema["properties"].get("from_line").is_none());
        assert!(write_schema["properties"].get("to_line").is_none());
        for name in ["read", "find"] {
            assert!(fixture.registry.get(name).unwrap().concurrent());
        }
        for name in ["write", "edit"] {
            assert!(!fixture.registry.get(name).unwrap().concurrent());
        }
    }

    #[tokio::test]
    async fn reads_full_files_and_optional_ranges() {
        let fixture = Fixture::new();
        fs::write(fixture.dir.path().join("sample.txt"), "one\ntwo\nthree\n").unwrap();
        fs::write(fixture.dir.path().join("empty.txt"), "").unwrap();
        let full = fixture.run("read", json!({"path": "sample.txt"})).await;
        assert!(full.is_success(), "{}", full.output());
        assert_eq!(full.output(), "one\ntwo\nthree\n");
        let range = fixture
            .run(
                "read",
                json!({"path": "sample.txt", "from_line": 2, "to_line": 2}),
            )
            .await;
        assert!(range.is_success());
        assert!(range.output().contains("2 | two"));
        assert!(!range.output().contains("one"));
        let tail = fixture
            .run("read", json!({"path": "sample.txt", "from_line": 3}))
            .await;
        assert!(tail.is_success());
        assert!(tail.output().contains("three"));
        let head = fixture
            .run("read", json!({"path": "sample.txt", "to_line": 1}))
            .await;
        assert!(head.is_success());
        assert!(head.output().contains("one"));
        let invalid = fixture
            .run("read", json!({"path": "sample.txt", "from_line": 0}))
            .await;
        assert!(!invalid.is_success());
        let empty = fixture.run("read", json!({"path": "empty.txt"})).await;
        assert!(empty.is_success());
        assert!(empty.output().is_empty());
    }

    #[tokio::test]
    async fn writes_and_edits_share_file_diffs() {
        let fixture = Fixture::new();
        let written = fixture
            .run(
                "write",
                json!({"path": "sub/sample.txt", "content": "hello world\n"}),
            )
            .await;
        assert!(written.is_success(), "{}", written.output());
        let read = fixture.run("read", json!({"path": "sub/sample.txt"})).await;
        assert!(read.is_success());
        let edit = fixture
            .run(
                "edit",
                json!({"path": "sub/sample.txt", "old_string": "world", "new_string": "rust"}),
            )
            .await;
        assert!(edit.is_success(), "{}", edit.output());
        assert!(!edit.diffs().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.dir.path().join("sub/sample.txt")).unwrap(),
            "hello rust\n"
        );
        let replaced = fixture
            .run(
                "write",
                json!({"path": "sub/sample.txt", "content": "replacement\n"}),
            )
            .await;
        assert!(replaced.is_success());
        assert!(!replaced.diffs().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.dir.path().join("sub/sample.txt")).unwrap(),
            "replacement\n"
        );
    }

    #[tokio::test]
    async fn edit_rejects_missing_empty_and_ambiguous_matches_without_writing() {
        let fixture = Fixture::new();
        let path = fixture.dir.path().join("sample.txt");
        fs::write(&path, "aaa hello hello").unwrap();
        for old in ["missing", "", "hello", "aa"] {
            let result = fixture
                .run(
                    "edit",
                    json!({"path": "sample.txt", "old_string": old, "new_string": "changed"}),
                )
                .await;
            assert!(!result.is_success(), "old_string={old:?}");
            assert_eq!(fs::read_to_string(&path).unwrap(), "aaa hello hello");
        }
        let unchanged = fixture
            .run(
                "edit",
                json!({"path": "sample.txt", "old_string": "aaa", "new_string": "aaa"}),
            )
            .await;
        assert!(unchanged.is_success());
    }

    #[tokio::test]
    async fn finds_files_and_contents_under_the_requested_path() {
        let fixture = Fixture::new();
        fs::create_dir(fixture.dir.path().join("src")).unwrap();
        fs::write(
            fixture.dir.path().join("src/sample.rs"),
            "fn example() {}\n",
        )
        .unwrap();
        fs::write(fixture.dir.path().join("other.rs"), "fn other() {}\n").unwrap();
        let files = fixture
            .run("find", json!({"pattern": "**/*.rs", "path": "src"}))
            .await;
        assert!(files.is_success(), "{}", files.output());
        assert!(files.output().contains("sample.rs"));
        assert!(!files.output().contains("other.rs"));
        let content = fixture
            .run(
                "find",
                json!({"mode": "content", "pattern": "example", "path": "src"}),
            )
            .await;
        assert!(content.is_success());
        assert!(content.output().contains("sample.rs:1:"));
        let default_path = fixture
            .run("find", json!({"mode": "content", "pattern": "other"}))
            .await;
        assert!(default_path.is_success());
        assert!(default_path.output().contains("other.rs:1:"));
        let invalid = fixture
            .run("find", json!({"mode": "invalid", "pattern": "*"}))
            .await;
        assert!(!invalid.is_success());
    }

    #[tokio::test]
    async fn preserves_protection_and_rejects_partial_writes() {
        let fixture = Fixture::new();
        fs::write(fixture.dir.path().join("config.toml"), "example").unwrap();
        fs::write(fixture.dir.path().join("sample.txt"), "unchanged").unwrap();
        let protected_write = fixture
            .run(
                "write",
                json!({"path": "config.toml", "content": "changed"}),
            )
            .await;
        assert!(!protected_write.is_success());
        let protected_edit = fixture
            .run(
                "edit",
                json!({"path": "config.toml", "old_string": "example", "new_string": "changed"}),
            )
            .await;
        assert!(!protected_edit.is_success());
        let partial = fixture
            .run(
                "write",
                json!({"path": "sample.txt", "content": "changed", "from_line": 1, "to_line": 1}),
            )
            .await;
        assert!(!partial.is_success());
        assert_eq!(
            fs::read_to_string(fixture.dir.path().join("sample.txt")).unwrap(),
            "unchanged"
        );
    }
}
