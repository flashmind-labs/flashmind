//! Interactive MCP (Model Context Protocol) agent with TUI.
//!
//! Demonstrates using MCP management tools (`mcp_add`, `mcp_list`, `mcp_remove`)
//! to dynamically connect to MCP servers at runtime. The agent can discover and
//! call tools exposed by any connected MCP server.
//!
//! # Usage
//!
//! ```sh
//! # Basic — uses llama3.2 by default
//! cargo run -p flashmind --example mcp
//!
//! # With a specific model
//! MODEL=qwen3:8b cargo run -p flashmind --example mcp
//! ```
//!
//! # Example conversation
//!
//! ```text
//! > Add the filesystem MCP server for the current directory
//! ▶ mcp_add — Adding MCP server 'filesystem'
//! ✓ mcp_add (1.2s)
//!
//! > List my MCP servers
//! ▶ mcp_list — Listing MCP servers
//! ✓ mcp_list (0.1s)
//!
//! > Read the file at ./Cargo.toml
//! ▶ filesystem_read_file — [mcp:filesystem] read_file
//! ✓ filesystem_read_file (0.3s)
//! ```

use std::sync::Arc;

use flashmind::core::{Agent, Conversation, ConversationEntry};
use flashmind::llm::OllamaProvider;
use flashmind::tools::mcp::tools::{McpAddTool, McpListTool, McpRemoveTool};
use flashmind::tools::mcp::{McpDiskConfig, McpRegistry, McpToolOp, make_mcp_tool_wrappers};
use flashmind::types::{AgentInput, AgentLlmConfig, LlmProvider, ToolRegistry};
use flashmind_tui::{Repl, ReplConfig, ReplEvent};

const SYSTEM_PROMPT: &str = "\
You are an AI assistant with access to MCP (Model Context Protocol) servers. \
You can connect to external tool servers and use their capabilities.

Your MCP management tools:
- **mcp_add**: Connect to an MCP server (stdio via command+args, or HTTP via url)
- **mcp_list**: List connected servers and their tools
- **mcp_remove**: Disconnect and remove a server

When the user asks to add an MCP server, use mcp_add with the appropriate transport. Common examples:
- Filesystem: command=\"npx\", args=[\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/path\"]
- GitHub: command=\"npx\", args=[\"-y\", \"@modelcontextprotocol/server-github\"], env={\"GITHUB_PERSONAL_ACCESS_TOKEN\": \"...\"}
- SQLite: command=\"npx\", args=[\"-y\", \"@modelcontextprotocol/server-sqlite\", \"path/to/db.sqlite\"]

After connecting a server, its tools become available for you to call directly.";

/// Drain MCP pending operations and register/unregister tool wrappers on the agent.
fn apply_mcp_ops(agent: &mut Agent, mcp: &McpRegistry) {
    for op in mcp.drain_pending_ops() {
        match op {
            McpToolOp::Register {
                server_name,
                tool_defs,
            } => {
                let wrappers = make_mcp_tool_wrappers(mcp, &server_name, &tool_defs);
                for w in wrappers {
                    agent.tools_mut().register(w);
                }
            }
            McpToolOp::Unregister { server_name } => {
                agent
                    .tools_mut()
                    .strip_prefixes(&[&format!("{server_name}_")]);
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_str = std::env::var("MODEL").unwrap_or_else(|_| "llama3.2".into());
    let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(None, None));

    // MCP server configs persist across sessions
    let mcp_dir = home::home_dir()
        .expect("no home directory")
        .join(".flashmind")
        .join("mcp");
    let mcp_config = McpDiskConfig::new(mcp_dir);
    let mcp = McpRegistry::new(mcp_config, None);
    mcp.load_saved().await;

    // Register MCP management tools
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(McpAddTool { mcp: mcp.clone() }));
    tools.register(Arc::new(McpListTool { mcp: mcp.clone() }));
    tools.register(Arc::new(McpRemoveTool { mcp: mcp.clone() }));

    let model = format!("ollama:{model_str}").parse()?;
    let mut agent = Agent::builder(provider)
        .tools(tools)
        .llm(AgentLlmConfig::new(model))
        .build();

    // Register any tools from previously-saved MCP servers
    apply_mcp_ops(&mut agent, &mcp);

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system(SYSTEM_PROMPT));

    let config = ReplConfig {
        prompt: "mcp".to_string(),
        greeting: Some(format!(
            "flashmind MCP agent — model: {model_str} — Ctrl-D to quit\n\
             Try: \"Add the filesystem MCP server for the current directory\""
        )),
        ..Default::default()
    };

    let mut repl = Repl::new(config);
    repl.print_greeting()?;

    while let ReplEvent::UserInput(text) = repl.read_input()? {
        let stream = agent.start(&mut conversation, AgentInput::user(text), None);
        repl.stream_response(Box::pin(stream)).await?;

        // Register/unregister MCP tools that were added/removed during this turn
        apply_mcp_ops(&mut agent, &mcp);
    }

    mcp.shutdown_all().await;
    Ok(())
}
