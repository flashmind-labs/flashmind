//! The `lsp` tool: semantic navigation for Rust code through rust-analyzer.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use url::Url;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::client::{LspClient, file_uri};
use super::manager::{LspManager, find_cargo_root, is_dependency_source};
use super::protocol::{
    DocumentSymbol, Location, Position, WorkspaceSymbol, hover_text, parse_locations,
    symbol_kind_name,
};

const READY_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_MAX_RESULTS: usize = 50;
const HOVER_MAX_CHARS: usize = 2000;
const SOURCE_LINE_MAX_CHARS: usize = 200;
/// How far from the given line to look for the symbol.
const LINE_SLACK: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
enum Operation {
    Definition,
    References,
    Implementations,
    Hover,
    Symbols,
    Outline,
}

#[derive(Deserialize)]
struct LspArgs {
    operation: Operation,
    symbol: Option<String>,
    file: Option<String>,
    line: Option<usize>,
    max_results: Option<usize>,
}

/// Semantic code navigation backed by rust-analyzer.
pub struct LspTool {
    manager: LspManager,
}

impl LspTool {
    /// Create the tool. Servers start on first use.
    pub fn new(manager: LspManager) -> Self {
        Self { manager }
    }
}

/// A point in a file the server is asked about.
struct Target {
    path: PathBuf,
    position: Position,
}

/// Everything one call needs, resolved from the arguments.
struct Call<'a> {
    client: &'a LspClient,
    cancel: &'a CancellationToken,
    working_dir: Option<&'a Path>,
    max_results: usize,
}

#[async_trait]
impl Tool for LspTool {
    fn name(&self) -> &str {
        "lsp"
    }

    fn description(&self) -> &str {
        "Semantic Rust navigation through rust-analyzer. Use INSTEAD of grep to find where a \
         type, function or trait is defined (definition), every use of it (references), what \
         implements a trait or type (implementations), its signature and docs (hover), symbols \
         by name across the workspace (symbols), or the items in one file (outline). Works \
         inside dependency crates too. Point at a symbol with `symbol` plus the `file` and \
         `line` where it appears; with only `symbol`, it is looked up by name. The first call \
         on a project waits for indexing."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["definition", "references", "implementations", "hover", "symbols", "outline"],
                    "description": "What to look up"
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name as written in the code, e.g. `Error` or `Client::new` (the last segment is located). For `symbols`, the search query. Not used by `outline`."
                },
                "file": {
                    "type": "string",
                    "description": "File where the symbol appears. Required for `outline`. Omit to look the symbol up by name."
                },
                "line": {
                    "type": "integer",
                    "description": "1-based line in `file` where the symbol appears. Nearby lines are also checked. Without it, the first occurrence in the file is used."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to list (default: 50)"
                }
            },
            "required": ["operation"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(READY_TIMEOUT.as_secs() + 90)
    }

    fn humanize(&self, args: &Value) -> String {
        let operation = args
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("lookup");
        let subject = args
            .get("symbol")
            .or_else(|| args.get("file"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        format!("LSP {operation} {subject}").trim_end().to_string()
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: LspArgs = ctx.parse_args(self.name())?;
        let id = ctx.tool_call_id;
        let working_dir = ctx.working_dir.map(PathBuf::as_path);

        let file = match &args.file {
            Some(file) => {
                if let Err(result) = ctx.check_absolute_path(file) {
                    return Ok(result);
                }
                let path = resolve_path(file, working_dir);
                if !path.is_file() {
                    return Ok(ToolResult::failure(id, format!("No such file: {file}")));
                }
                Some(path)
            }
            None => None,
        };

        let Some(root) = self.choose_root(file.as_deref(), working_dir).await else {
            return Ok(ToolResult::failure(
                id,
                "No Cargo project found. Pass a `file` inside a Rust project.",
            ));
        };
        let client = self.manager.client(&root).await?;

        if !client.wait_ready(Duration::ZERO, ctx.cancel_token()).await {
            ctx.progress(&format!(
                "Waiting for rust-analyzer to index {}",
                root.display()
            ));
            if !client.wait_ready(READY_TIMEOUT, ctx.cancel_token()).await {
                if !client.is_alive() {
                    return Ok(ToolResult::failure(
                        id,
                        "rust-analyzer exited while indexing.",
                    ));
                }
                return Ok(ToolResult::failure(
                    id,
                    format!(
                        "rust-analyzer is still indexing {} ({}s so far). Try again shortly, or use grep meanwhile.",
                        root.display(),
                        client.uptime().as_secs()
                    ),
                ));
            }
        }

        let call = Call {
            client: &client,
            cancel: ctx.cancel_token(),
            working_dir,
            max_results: args.max_results.unwrap_or(DEFAULT_MAX_RESULTS),
        };
        let output = match run(&call, &args, file.as_deref()).await {
            Ok(output) => output,
            Err(message) => return Ok(ToolResult::failure(id, message)),
        };
        client.touch();
        Ok(ToolResult::success(id, output))
    }
}

impl LspTool {
    /// Pick the workspace whose server should answer.
    ///
    /// A project file uses its own Cargo workspace. Dependency sources go to
    /// the server used last, since they belong to whichever project pulled them in.
    async fn choose_root(
        &self,
        file: Option<&Path>,
        working_dir: Option<&Path>,
    ) -> Option<PathBuf> {
        if let Some(file) = file {
            if is_dependency_source(file) {
                if let Some(root) = self.manager.most_recent_root().await {
                    return Some(root);
                }
            } else if let Some(root) = file.parent().and_then(find_cargo_root) {
                return Some(root);
            }
        }
        working_dir.and_then(find_cargo_root)
    }
}

/// Run one operation. `Err` holds a message for the model.
async fn run(call: &Call<'_>, args: &LspArgs, file: Option<&Path>) -> Result<String, String> {
    match args.operation {
        Operation::Outline => {
            let path = file.ok_or("`outline` needs `file`.")?;
            outline(call, path).await
        }
        Operation::Symbols => {
            let query = args
                .symbol
                .as_deref()
                .ok_or("`symbols` needs `symbol` as the query.")?;
            let symbols = workspace_symbols(call, query).await?;
            Ok(render_symbols(
                call,
                &format!("symbols matching `{query}`"),
                &symbols,
            ))
        }
        Operation::Definition
        | Operation::References
        | Operation::Implementations
        | Operation::Hover => {
            let symbol = args
                .symbol
                .as_deref()
                .ok_or("This operation needs `symbol`.")?;
            let target = match file {
                Some(path) => locate_in_file(path, symbol, args.line)?,
                None => match locate_by_name(call, symbol).await? {
                    ByName::Found(target) => target,
                    ByName::Definitions(output) if args.operation == Operation::Definition => {
                        return Ok(output);
                    }
                    ByName::Definitions(output) => {
                        return Err(format!(
                            "`{symbol}` matches several symbols. Pass `file` and `line` for one of them:\n{output}"
                        ));
                    }
                },
            };
            point_query(call, args.operation, symbol, &target).await
        }
    }
}

enum ByName {
    Found(Target),
    /// Zero or several exact matches, already rendered.
    Definitions(String),
}

/// Look a symbol up through `workspace/symbol` and keep exact name matches.
async fn locate_by_name(call: &Call<'_>, symbol: &str) -> Result<ByName, String> {
    let name = last_segment(symbol);
    let exact: Vec<_> = workspace_symbols(call, name)
        .await?
        .into_iter()
        .filter(|found| found.name == name)
        .collect();
    if exact.len() != 1 {
        if exact.is_empty() {
            return Err(format!(
                "No symbol named `{name}` in the workspace. Try `symbols` for a fuzzy search, or pass `file` and `line`."
            ));
        }
        return Ok(ByName::Definitions(render_symbols(
            call,
            &format!("definitions of `{name}`"),
            &exact,
        )));
    }
    let found = &exact[0];
    let path = uri_to_path(&found.location.uri).ok_or("Symbol has a non-file location.")?;
    let line = found
        .location
        .range
        .map(|range| range.start.line as usize + 1);
    Ok(ByName::Found(locate_in_file(&path, name, line)?))
}

async fn workspace_symbols(call: &Call<'_>, query: &str) -> Result<Vec<WorkspaceSymbol>, String> {
    let value = call
        .client
        .request("workspace/symbol", json!({"query": query}), call.cancel)
        .await
        .map_err(|e| e.to_string())?;
    Ok(Vec::<WorkspaceSymbol>::deserialize(&value).unwrap_or_default())
}

async fn point_query(
    call: &Call<'_>,
    operation: Operation,
    symbol: &str,
    target: &Target,
) -> Result<String, String> {
    call.client
        .sync_document(&target.path)
        .await
        .map_err(|e| e.to_string())?;
    let uri = file_uri(&target.path).map_err(|e| e.to_string())?;
    let mut params = json!({
        "textDocument": {"uri": uri.as_str()},
        "position": {"line": target.position.line, "character": target.position.character},
    });
    let method = match operation {
        Operation::Definition => "textDocument/definition",
        Operation::References => {
            params["context"] = json!({"includeDeclaration": true});
            "textDocument/references"
        }
        Operation::Implementations => "textDocument/implementation",
        Operation::Hover => "textDocument/hover",
        Operation::Symbols | Operation::Outline => {
            return Err(format!(
                "`{}` does not take a position.",
                <&str>::from(operation)
            ));
        }
    };
    let value = call
        .client
        .request(method, params, call.cancel)
        .await
        .map_err(|e| e.to_string())?;

    if operation == Operation::Hover {
        let text = hover_text(&value);
        if text.is_empty() {
            return Ok(format!("No hover information for `{symbol}`."));
        }
        return Ok(truncate_chars(&text, HOVER_MAX_CHARS));
    }

    let locations = parse_locations(value);
    let operation_name: &'static str = operation.into();
    let title = format!("{} for `{symbol}`", operation_name.replace('_', " "));
    Ok(render_locations(call, &title, &locations))
}

async fn outline(call: &Call<'_>, path: &Path) -> Result<String, String> {
    call.client
        .sync_document(path)
        .await
        .map_err(|e| e.to_string())?;
    let uri = file_uri(path).map_err(|e| e.to_string())?;
    let value = call
        .client
        .request(
            "textDocument/documentSymbol",
            json!({"textDocument": {"uri": uri.as_str()}}),
            call.cancel,
        )
        .await
        .map_err(|e| e.to_string())?;

    let display = display_path(path, call.working_dir);
    if let Ok(symbols) = Vec::<DocumentSymbol>::deserialize(&value) {
        let mut out = format!("Outline of {display}:\n");
        write_outline(&mut out, &symbols, 0);
        return Ok(out.trim_end().to_string());
    }
    let flat = Vec::<WorkspaceSymbol>::deserialize(&value).unwrap_or_default();
    Ok(render_symbols(
        call,
        &format!("outline of {display}"),
        &flat,
    ))
}

fn write_outline(out: &mut String, symbols: &[DocumentSymbol], depth: usize) {
    for symbol in symbols {
        let detail = symbol
            .detail
            .as_deref()
            .filter(|detail| !detail.is_empty())
            .map(|detail| format!(": {}", truncate_chars(detail, 120)))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "{:>5}  {}{} {}{}",
            symbol.selection_range.start.line + 1,
            "  ".repeat(depth),
            symbol_kind_name(symbol.kind),
            symbol.name,
            detail
        );
        write_outline(out, &symbol.children, depth + 1);
    }
}

fn render_locations(call: &Call<'_>, title: &str, locations: &[Location]) -> String {
    if locations.is_empty() {
        return format!("No {title}.");
    }
    let mut sources = SourceCache::default();
    let mut out = format!("{} {title}:\n", locations.len());
    for location in locations.iter().take(call.max_results) {
        let line = location.range.start.line as usize;
        let _ = writeln!(
            out,
            "{}",
            sources.render(&location.uri, line, call.working_dir)
        );
    }
    push_overflow(&mut out, locations.len(), call.max_results);
    out.trim_end().to_string()
}

fn render_symbols(call: &Call<'_>, title: &str, symbols: &[WorkspaceSymbol]) -> String {
    if symbols.is_empty() {
        return format!("No {title}.");
    }
    let mut sources = SourceCache::default();
    let mut out = format!("{} {title}:\n", symbols.len());
    for symbol in symbols.iter().take(call.max_results) {
        let container = symbol
            .container_name
            .as_deref()
            .filter(|name| !name.is_empty())
            .map(|name| format!(" in {name}"))
            .unwrap_or_default();
        let place = match symbol.location.range {
            Some(range) => sources.render(
                &symbol.location.uri,
                range.start.line as usize,
                call.working_dir,
            ),
            None => uri_to_path(&symbol.location.uri)
                .map(|path| display_path(&path, call.working_dir))
                .unwrap_or_else(|| symbol.location.uri.clone()),
        };
        let _ = writeln!(
            out,
            "{} {}{container}  {place}",
            symbol_kind_name(symbol.kind),
            symbol.name
        );
    }
    push_overflow(&mut out, symbols.len(), call.max_results);
    out.trim_end().to_string()
}

fn push_overflow(out: &mut String, total: usize, shown: usize) {
    if total > shown {
        let _ = writeln!(
            out,
            "... {} more (raise max_results to see them)",
            total - shown
        );
    }
}

/// Reads each file once per call to quote result lines.
#[derive(Default)]
struct SourceCache {
    files: HashMap<String, Option<Vec<String>>>,
}

impl SourceCache {
    /// `path:line: source`, with the path relative to the working dir when possible.
    fn render(&mut self, uri: &str, line: usize, working_dir: Option<&Path>) -> String {
        let Some(path) = uri_to_path(uri) else {
            return format!("{uri}:{}", line + 1);
        };
        let display = display_path(&path, working_dir);
        let lines = self.files.entry(uri.to_string()).or_insert_with(|| {
            std::fs::read_to_string(&path)
                .ok()
                .map(|text| text.lines().map(str::to_string).collect())
        });
        match lines.as_ref().and_then(|lines| lines.get(line)) {
            Some(source) => format!(
                "{display}:{}: {}",
                line + 1,
                truncate_chars(source.trim(), SOURCE_LINE_MAX_CHARS)
            ),
            None => format!("{display}:{}", line + 1),
        }
    }
}

/// Find `symbol` in `path`, at or near `line` (1-based), or its first occurrence.
fn locate_in_file(path: &Path, symbol: &str, line: Option<usize>) -> Result<Target, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    let position = find_position(&text, last_segment(symbol), line)?;
    Ok(Target {
        path: path.to_path_buf(),
        position,
    })
}

fn find_position(text: &str, name: &str, line: Option<usize>) -> Result<Position, String> {
    let lines: Vec<&str> = text.lines().collect();
    let candidates: Vec<usize> = match line {
        Some(line) => {
            let center = line.saturating_sub(1);
            let mut order = vec![center];
            for offset in 1..=LINE_SLACK {
                order.push(center.saturating_sub(offset));
                order.push(center + offset);
            }
            order.dedup();
            order
        }
        None => (0..lines.len()).collect(),
    };
    for index in candidates {
        let Some(source) = lines.get(index) else {
            continue;
        };
        if let Some(byte) = find_word(source, name) {
            return Ok(Position {
                line: index as u32,
                character: source[..byte].encode_utf16().count() as u32,
            });
        }
    }
    Err(match line {
        Some(line) => {
            let content = lines
                .get(line.saturating_sub(1))
                .map_or("<past end of file>", |l| l.trim());
            format!(
                "`{name}` is not within {LINE_SLACK} lines of line {line}. Line {line} is: {content}"
            )
        }
        None => format!("`{name}` does not appear in the file."),
    })
}

/// Byte offset of `name` as a whole identifier in `line`.
fn find_word(line: &str, name: &str) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    line.match_indices(name).map(|(at, _)| at).find(|&at| {
        let before = line[..at].chars().next_back();
        let after = line[at + name.len()..].chars().next();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

/// `crate::Client::new()` -> `new`.
fn last_segment(symbol: &str) -> &str {
    let symbol = symbol.trim().trim_end_matches("()");
    symbol
        .rsplit("::")
        .next()
        .unwrap_or(symbol)
        .trim_start_matches('&')
}

fn resolve_path(file: &str, working_dir: Option<&Path>) -> PathBuf {
    match working_dir {
        Some(dir) => dir.join(file),
        None => PathBuf::from(file),
    }
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    Url::parse(uri).ok()?.to_file_path().ok()
}

fn display_path(path: &Path, working_dir: Option<&Path>) -> String {
    working_dir
        .and_then(|dir| path.strip_prefix(dir).ok())
        .unwrap_or(path)
        .display()
        .to_string()
}

fn truncate_chars(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => format!("{}...", &text[..at]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_LIB: &str = "\
pub trait Shape {
    fn area(&self) -> f64;
}

/// A square with equal sides.
pub struct Square {
    pub side: f64,
}

impl Shape for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
}

pub fn total(shapes: &[Box<dyn Shape>]) -> f64 {
    shapes.iter().map(|shape| shape.area()).sum()
}

pub fn make() -> Square {
    Square { side: 2.0 }
}
";

    async fn call(tool: &LspTool, dir: &PathBuf, args: Value) -> ToolResult {
        let cancel = CancellationToken::new();
        let ctx = ToolContext::new("call", args, Some(dir), &cancel);
        tool.execute(ctx).await.expect("lsp tool call")
    }

    /// Runs every operation against a real rust-analyzer. Skips when it is not installed.
    #[tokio::test]
    async fn answers_queries_from_rust_analyzer() {
        let Some(command) = crate::lsp::find_rust_analyzer() else {
            eprintln!("rust-analyzer not on PATH, skipping");
            return;
        };
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        std::fs::create_dir_all(root.join("src")).expect("create src");
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("write manifest");
        std::fs::write(root.join("src/lib.rs"), FIXTURE_LIB).expect("write lib");
        let tool = LspTool::new(LspManager::new(command));

        let output = call(&tool, &root, json!({"operation": "definition", "symbol": "Square", "file": "src/lib.rs", "line": 21})).await;
        assert!(output.is_success(), "{}", output.output());
        assert!(
            output
                .output()
                .contains("src/lib.rs:6: pub struct Square {"),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "implementations", "symbol": "Shape"}),
        )
        .await;
        assert!(
            output
                .output()
                .contains("src/lib.rs:10: impl Shape for Square {"),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "references", "symbol": "area", "file": "src/lib.rs", "line": 2}),
        )
        .await;
        assert!(
            output.output().starts_with("3 references"),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "hover", "symbol": "Square", "file": "src/lib.rs", "line": 20}),
        )
        .await;
        assert!(
            output.output().contains("A square with equal sides."),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "symbols", "symbol": "tot"}),
        )
        .await;
        assert!(
            output.output().contains("function total"),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "outline", "file": "src/lib.rs"}),
        )
        .await;
        assert!(
            output.output().contains("struct Square"),
            "{}",
            output.output()
        );
        assert!(
            output.output().contains("    field side"),
            "{}",
            output.output()
        );

        let output = call(
            &tool,
            &root,
            json!({"operation": "definition", "symbol": "Box", "file": "src/lib.rs", "line": 16}),
        )
        .await;
        assert!(
            output.output().contains("pub struct Box"),
            "{}",
            output.output()
        );
    }

    #[test]
    fn last_segment_strips_paths_and_calls() {
        assert_eq!(last_segment("crate::Client::new()"), "new");
        assert_eq!(last_segment("Error"), "Error");
    }

    #[test]
    fn find_word_respects_identifier_boundaries() {
        assert_eq!(find_word("struct MyError; impl Error", "Error"), Some(21));
        assert_eq!(find_word("ErrorKind", "Error"), None);
    }

    #[test]
    fn find_position_checks_nearby_lines_and_counts_utf16() {
        let text = "fn a() {}\n// é Error\nlet x = 1;\n";
        let position = find_position(text, "Error", Some(3)).expect("within slack");
        assert_eq!(
            position,
            Position {
                line: 1,
                character: 5
            }
        );

        let error = find_position(text, "Missing", Some(1)).expect_err("not present");
        assert!(error.contains("Line 1 is: fn a() {}"), "{error}");

        let first = find_position(text, "x", None).expect("first occurrence");
        assert_eq!(first.line, 2);
    }

    #[test]
    fn display_path_is_relative_inside_working_dir() {
        let dir = Path::new("/srv/project");
        assert_eq!(
            display_path(Path::new("/srv/project/src/lib.rs"), Some(dir)),
            "src/lib.rs"
        );
        assert_eq!(
            display_path(Path::new("/other/lib.rs"), Some(dir)),
            "/other/lib.rs"
        );
    }

    #[test]
    fn truncate_chars_adds_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 3), "abc...");
        assert_eq!(truncate_chars("abc", 3), "abc");
    }
}
