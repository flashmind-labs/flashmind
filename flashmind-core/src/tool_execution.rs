//! Tool batches shared by the agent loop and interactive callers.

use std::time::Instant;

use flashmind_types::tool::{BackgroundRequest, ToolContext, ToolResult};
use flashmind_types::{AgentEvent, AgentStream, ContentPart, Outcome, ToolCall};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
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

/// Progress or an individual result from a running tool batch.
pub enum ToolBatchEvent {
    Event(AgentEvent),
    /// Results arrive in completion order. `index` is the call's position
    /// within the batch.
    Completed {
        index: usize,
        tool: CompletedToolCall,
    },
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
            let media: Vec<ContentPart> = self
                .result
                .content()
                .iter()
                .filter(|part| !matches!(part, ContentPart::Text { .. }))
                .cloned()
                .collect();
            let files = self.result.files().to_vec();
            if !media.is_empty() || !files.is_empty() {
                events.push(AgentEvent::ToolMedia {
                    id: self.call.id.clone(),
                    parts: media.clone(),
                    files,
                });
            }
            conversation.add(ConversationEntry::tool_with_parts(
                &self.call.id,
                self.result.output(),
                media,
            ));
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
        &'a mut self,
        batch: &'a ToolBatch<'a>,
        cancel: &'a CancellationToken,
    ) -> AgentStream<'a, AgentEvent, Vec<CompletedToolCall>> {
        AgentStream::new(async_stream::stream! {
            let mut execution = self.stream_tool_batch(batch, cancel);
            let mut completed = Vec::with_capacity(batch.calls.len());
            while let Some(event) = execution.next().await {
                match event {
                    ToolBatchEvent::Event(event) => yield Outcome::Item(event),
                    ToolBatchEvent::Completed { index, tool } => completed.push((index, tool)),
                }
            }
            completed.sort_unstable_by_key(|(index, _)| *index);
            yield Outcome::Done(completed.into_iter().map(|(_, tool)| tool).collect());
        })
    }

    /// Yield each result as soon as it finishes, allowing the caller to
    /// checkpoint it while other calls are still running.
    pub fn stream_tool_batch<'a>(
        &'a mut self,
        batch: &'a ToolBatch<'a>,
        cancel: &'a CancellationToken,
    ) -> AgentStream<'a, ToolBatchEvent, ()> {
        AgentStream::new(async_stream::stream! {
            let background_requests: Vec<_> = batch.calls.iter()
                .map(|call| self.tools().supports_background(call).then(BackgroundRequest::default))
                .collect();
            for (call, request) in batch.calls.iter().zip(&background_requests) {
                yield Outcome::Item(ToolBatchEvent::Event(AgentEvent::ToolStart {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    humanized: self.tools().humanize(call),
                }));
                if let Some(request) = request {
                    yield Outcome::Item(ToolBatchEvent::Event(AgentEvent::ToolBackgroundReady {
                        id: call.id.clone(),
                        request: Some(request.clone()),
                    }));
                }
            }
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
            let agent = &*self;
            let mut running: FuturesUnordered<_> = batch.calls.iter().zip(&background_requests).enumerate().map(|(index, (call, request))| {
                let forward = progress_tx.clone();
                async move {
                    let started = Instant::now();
                    let (tx, mut rx) = mpsc::unbounded_channel();
                    let ctx = ToolContext::new(&call.id, call.arguments.clone(), agent.working_dir.as_ref(), cancel)
                        .with_llm(agent.llm())
                        .with_provider(agent.provider_arc().clone())
                        .with_progress(tx);
                    let ctx = match request {
                        Some(request) => ctx.with_background_request(request),
                        None => ctx,
                    };
                    let execution = agent.tools().execute_in_context(call, ctx);
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
            let mut working_dir = None;
            loop {
                tokio::select! {
                    Some(event) = self.next_background_event() => {
                        yield Outcome::Item(ToolBatchEvent::Event(event));
                    }
                    Some((id, line)) = progress_rx.recv() => {
                        yield Outcome::Item(ToolBatchEvent::Event(AgentEvent::ToolProgress { id, line }));
                    }
                    next = running.next() => match next {
                        Some((index, tool)) => {
                            if let Some(path) = tool.result.working_dir()
                                && working_dir.as_ref().is_none_or(|(previous, _)| index > *previous)
                            {
                                working_dir = Some((index, path.to_path_buf()));
                            }
                            while let Ok((id, line)) = progress_rx.try_recv() {
                                yield Outcome::Item(ToolBatchEvent::Event(AgentEvent::ToolProgress { id, line }));
                            }
                            yield Outcome::Item(ToolBatchEvent::Completed { index, tool });
                        }
                        None => break,
                    },
                }
            }
            while let Ok((id, line)) = progress_rx.try_recv() {
                yield Outcome::Item(ToolBatchEvent::Event(AgentEvent::ToolProgress { id, line }));
            }
            for event in self.drain_background_events() {
                yield Outcome::Item(ToolBatchEvent::Event(event));
            }
            drop(running);
            if let Some((_, path)) = working_dir {
                self.working_dir = Some(path);
            }
            yield Outcome::Done(());
        })
    }
}
