//! Tool batches shared by the agent loop and interactive callers.

use std::time::Instant;

use flashmind_types::tool::{ToolContext, ToolResult};
use flashmind_types::{AgentEvent, AgentStream, Outcome, ToolCall};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;

use crate::{Agent, CancellationToken, Conversation, ConversationEntry};

/// Calls that may execute together. Serial calls form individual batches.
pub struct ToolBatch<'a> {
    calls: Vec<&'a ToolCall>,
}

/// A completed call, before the caller handles approval or records its result.
pub struct CompletedToolCall {
    /// The requested call.
    pub call: ToolCall,
    /// The tool's result, including any interrupt payload.
    pub result: ToolResult,
    /// Time spent executing the call.
    pub elapsed_ms: u64,
}

impl CompletedToolCall {
    /// Record a normal result and return its display events. Interrupts leave
    /// the conversation untouched so the caller can handle their payload.
    pub fn finish(self, conversation: &mut Conversation) -> Vec<AgentEvent> {
        let mut events: Vec<_> = self
            .result
            .diffs()
            .iter()
            .map(|diff| AgentEvent::FileDiff {
                path: diff.path.clone(),
                diff: diff.diff.clone(),
            })
            .collect();
        let event = if self.result.is_interrupt() {
            AgentEvent::Interrupted {
                tool_call_id: self.call.id,
                tool_name: self.call.name,
                output: self.result.output(),
                payload: self.result.payload().cloned(),
            }
        } else {
            conversation.add(ConversationEntry::tool(&self.call.id, self.result.output()));
            AgentEvent::ToolResult {
                id: self.call.id,
                name: self.call.name,
                output: self.result.output(),
                success: self.result.is_success(),
                elapsed_ms: self.elapsed_ms,
                sources: self.result.sources().to_vec(),
            }
        };
        events.push(event);
        events
    }
}

impl Agent {
    /// Group consecutive concurrent calls. Serial calls separate batches so
    /// reads and writes keep the order requested by the model.
    pub fn tool_batches<'a>(&self, calls: &'a [ToolCall]) -> Vec<ToolBatch<'a>> {
        let mut batches = Vec::new();
        let mut parallel = Vec::new();
        for call in calls {
            if self.allow_parallel_tool_calls && self.tools().is_concurrent(call) {
                parallel.push(call);
            } else {
                if !parallel.is_empty() {
                    batches.push(ToolBatch {
                        calls: std::mem::take(&mut parallel),
                    });
                }
                batches.push(ToolBatch { calls: vec![call] });
            }
        }
        if !parallel.is_empty() {
            batches.push(ToolBatch { calls: parallel });
        }
        batches
    }

    /// Execute one batch, streaming starts and progress. Results are returned
    /// in call order after every call in the batch finishes.
    pub fn execute_tool_batch<'a>(
        &'a self,
        batch: &'a ToolBatch<'a>,
        cancel: &'a CancellationToken,
    ) -> AgentStream<'a, AgentEvent, Vec<CompletedToolCall>> {
        AgentStream::new(async_stream::stream! {
            for call in &batch.calls {
                yield Outcome::Item(AgentEvent::ToolStart {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    humanized: self.tools().humanize(call),
                });
            }
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
            let mut running: FuturesUnordered<_> = batch.calls.iter().enumerate().map(|(index, call)| {
                let forward = progress_tx.clone();
                async move {
                    let started = Instant::now();
                    let (tx, mut rx) = mpsc::unbounded_channel();
                    let ctx = ToolContext::new(&call.id, call.arguments.clone(), self.working_dir.as_ref(), cancel)
                        .with_llm(self.llm())
                        .with_provider(self.provider_arc().clone())
                        .with_progress(tx);
                    let execution = self.tools().execute_in_context(call, ctx);
                    let relay = async {
                        while let Some(line) = rx.recv().await {
                            let _ = forward.send((call.id.clone(), line));
                        }
                    };
                    let (result, ()) = tokio::join!(execution, relay);
                    (index, CompletedToolCall {
                        call: (*call).clone(),
                        result,
                        elapsed_ms: started.elapsed().as_millis() as u64,
                    })
                }
            }).collect();
            drop(progress_tx);
            let mut completed: Vec<Option<CompletedToolCall>> = batch.calls.iter().map(|_| None).collect();
            loop {
                tokio::select! {
                    Some(event) = self.next_background_event() => {
                        yield Outcome::Item(event);
                    }
                    Some((id, line)) = progress_rx.recv() => {
                        yield Outcome::Item(AgentEvent::ToolProgress { id, line });
                    }
                    next = running.next() => match next {
                        Some((index, result)) => completed[index] = Some(result),
                        None => break,
                    },
                }
            }
            while let Ok((id, line)) = progress_rx.try_recv() {
                yield Outcome::Item(AgentEvent::ToolProgress { id, line });
            }
            for event in self.drain_background_events() {
                yield Outcome::Item(event);
            }
            yield Outcome::Done(completed.into_iter().flatten().collect());
        })
    }
}
