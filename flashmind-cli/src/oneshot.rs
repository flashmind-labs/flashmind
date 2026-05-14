//! Non-interactive single-shot execution.
//!
//! Runs a single prompt through the agent and streams the response to stdout.
//! Supports stdin piping: `echo "code" | flashmind-cli -p "review this"`

use std::io::{self, Write};

use anyhow::Result;
use futures::StreamExt;

use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry};
use flashmind_types::AgentEvent;
use flashmind_types::AgentInput;
use flashmind_types::model::Model;

use crate::config::Config;

pub async fn run(model_override: Option<Model>, prompt: &str) -> Result<()> {
    let config = Config::load()?;
    Config::init()?;

    let llm_config = config.build_llm_config(model_override.as_ref())?;
    let provider = config.build_provider_for(&llm_config.model.provider)?;
    let tool_set = config.build_tools(provider.clone(), &llm_config).await?;

    let mut agent = Agent::builder(provider)
        .llm(llm_config)
        .tools(tool_set.tools)
        .build();

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system(config.system_prompt()));

    let cancel = CancellationToken::new();
    let stream = agent.start(
        &mut conversation,
        cancel,
        AgentInput::user(prompt.to_string()),
        None,
    );
    tokio::pin!(stream);

    let mut stdout = io::stdout().lock();
    while let Some(event) = stream.next().await {
        match &event {
            AgentEvent::TextDelta(content) => {
                write!(stdout, "{content}")?;
                stdout.flush()?;
            }
            AgentEvent::Done(_) => {
                writeln!(stdout)?;
            }
            AgentEvent::Error(message) => {
                eprintln!("Error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
