//! Command approval via the interrupt mechanism.
//!
//! Demonstrates how the `CommandAllowList` + `BashTool` interrupt flow works:
//! commands not in the allow-list return `ToolResult::Interrupt` with a typed
//! `CommandApproval` payload. The caller handles the interrupt, decides whether
//! to approve, adds the tool result to the conversation, and resumes — the LLM
//! re-calls `exec` and this time the command passes.
//!
//! Uses a mock provider that always calls `exec` with `echo hello`, so you can
//! run this without any LLM backend.
//!
//! ```sh
//! cargo run -p flashmind --example command_approval
//! ```

use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;

use flashmind::core::{Agent, CancellationToken, Conversation, ConversationEntry};
use flashmind::tools::bash::CommandApproval;
use flashmind::types::tool::{CommandAllowList, ToolRegistry};
use flashmind::types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Provider, StreamEvent,
};

// ---------------------------------------------------------------------------
// Simple allow-list: only "echo *" is pre-approved
// ---------------------------------------------------------------------------

struct SimpleAllowList {
    approved: std::sync::Mutex<Vec<String>>,
}

impl SimpleAllowList {
    fn new() -> Self {
        Self {
            approved: std::sync::Mutex::new(vec!["echo *".to_string()]),
        }
    }
}

impl CommandAllowList for SimpleAllowList {
    fn is_allowed(&self, command: &str) -> bool {
        let approved = self.approved.lock().unwrap();
        approved.iter().any(|pat| {
            if let Some(prefix) = pat.strip_suffix(" *") {
                command.starts_with(prefix)
            } else {
                command == pat
            }
        })
    }

    fn add_session_pattern(&self, pattern: &str) {
        self.approved.lock().unwrap().push(pattern.to_string());
    }
}

// ---------------------------------------------------------------------------
// Mock provider that calls `exec` with a command
// ---------------------------------------------------------------------------

struct ExecCallingProvider;

#[async_trait]
impl LlmProvider for ExecCallingProvider {
    fn name(&self) -> &str {
        "mock-exec"
    }
    fn provider(&self) -> Provider {
        Provider::Ollama
    }
    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let has_tool_result = request
            .messages
            .iter()
            .any(|m| m.role == flashmind::types::Role::Tool);

        Box::pin(stream! {
            if has_tool_result {
                yield Ok(StreamEvent::ContentDelta("Done! The command executed successfully.".into()));
                yield Ok(StreamEvent::Finished(FinishReason::Stop));
            } else {
                yield Ok(StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "exec".into(),
                });
                yield Ok(StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: r#"{"command":"ls -la /tmp"}"#.into(),
                });
                yield Ok(StreamEvent::Finished(FinishReason::ToolCalls));
            }
        })
    }
}

#[tokio::main]
async fn main() {
    let provider: Arc<dyn LlmProvider> = Arc::new(ExecCallingProvider);
    let allowlist = Arc::new(SimpleAllowList::new());

    // Register the bash tool with our allow-list.
    // Only "echo *" is pre-approved, so "ls -la /tmp" will trigger an interrupt.
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(flashmind::tools::bash::BashTool {
        protected: Arc::new(flashmind::tools::protected::ProtectedPaths::new(
            &std::env::current_dir().unwrap(),
        )),
        secrets: Vec::new(),
        process_registry: flashmind::tools::process::ProcessRegistry::new(),
        forbidden_cmds: Vec::new(),
        allowlist: Some(allowlist.clone()),
    }));

    let mut agent = Agent::builder(provider).tools(tools).build().await;
    let mut conversation = Conversation::new();
    conversation.set_system("You have access to a bash tool. Use it when asked.");

    println!("=== Command Approval Example ===\n");
    println!("The allow-list only has 'echo *'. The LLM will try 'ls -la /tmp'.\n");

    // First turn: the LLM calls `exec` with a command not in the allow-list.
    // The stream yields an Interrupt event and stops.
    let mut pending_approval: Option<(String, String)> = None;
    {
        let cancel = CancellationToken::new();
        let s = agent.start(
            &mut conversation,
            cancel,
            AgentInput::user("List files in /tmp"),
            None,
        );
        tokio::pin!(s);

        while let Some(event) = s.next().await {
            match event {
                AgentEvent::TextDelta(t) => print!("{t}"),
                AgentEvent::ToolStart {
                    name, humanized, ..
                } => println!("[tool:start] {name} — {humanized}"),
                AgentEvent::ToolResult {
                    name,
                    output,
                    success,
                    ..
                } => println!("[tool:done]  {name} → {output} (ok={success})"),
                AgentEvent::Interrupted {
                    tool_call_id,
                    payload,
                    ..
                } => {
                    if let Some(approval) = payload
                        .as_ref()
                        .and_then(|p| p.as_any().downcast_ref::<CommandApproval>())
                    {
                        println!(
                            "[interrupt]  Command '{}' needs approval — auto-approving.",
                            approval.command
                        );
                        pending_approval = Some((tool_call_id.clone(), approval.command.clone()));
                    }
                }
                AgentEvent::Done(resp) => println!("\n\n[done] {resp}"),
                _ => {}
            }
        }
    }
    // Stream dropped — borrows released.

    // Handle the approval: add pattern to allow-list, inject tool result, resume.
    if let Some((tool_call_id, command)) = pending_approval {
        allowlist.add_session_pattern(&command);
        conversation.add(ConversationEntry::tool(
            &tool_call_id,
            "Approved. Execute the command again.",
        ));

        println!("\n--- Resuming after approval ---\n");

        let cancel = CancellationToken::new();
        let s = agent.start(&mut conversation, cancel, AgentInput::Resume, None);
        tokio::pin!(s);

        while let Some(event) = s.next().await {
            match event {
                AgentEvent::TextDelta(t) => print!("{t}"),
                AgentEvent::ToolStart {
                    name, humanized, ..
                } => println!("[tool:start] {name} — {humanized}"),
                AgentEvent::ToolResult {
                    name,
                    output,
                    success,
                    ..
                } => println!("[tool:done]  {name} → {output} (ok={success})"),
                AgentEvent::Done(resp) => println!("\n\n[done] {resp}"),
                _ => {}
            }
        }
    }
}
