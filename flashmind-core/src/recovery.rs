//! Error recovery shared by the agent loop and interactive callers.

use flashmind_types::{AgentEvent, AgentStream, LlmError, LlmErrorKind, Outcome};
use std::time::Duration;

use crate::{Agent, CancellationToken, Conversation, handle_llm_error};

const MAX_STREAM_RETRIES: u8 = 10;
const MAX_STREAM_RETRY_DELAY: Duration = Duration::from_secs(60);

/// Compaction attempts and stream retries since the last completed response.
#[derive(Default)]
pub struct RecoveryAttempts {
    compactions: u8,
    stream_retries: u8,
}

impl RecoveryAttempts {
    /// A completed response gives the next request fresh compaction and
    /// stream retry budgets, so a long tool loop can recover more than twice.
    pub fn record_success(&mut self) {
        if self.stream_retries > 0 {
            tracing::info!(retries = self.stream_retries, "LLM stream recovered");
        }
        self.compactions = 0;
        self.stream_retries = 0;
    }
}

impl Agent {
    /// Recover context errors, retry broken streams, or remove rejected binary
    /// attachments. The result says whether the caller should retry the turn.
    pub fn recover<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        cancel: &'a CancellationToken,
        error: &'a anyhow::Error,
        attempts: &'a mut RecoveryAttempts,
    ) -> AgentStream<'a, AgentEvent, bool> {
        AgentStream::new(async_stream::stream! {
            if cancel.is_cancelled() {
                yield Outcome::Done(false);
                return;
            }
            let llm_error = error.downcast_ref::<LlmError>();
            if llm_error.is_some_and(LlmError::is_recoverable) && attempts.compactions < 2 {
                attempts.compactions += 1;
                let mut recovery = handle_llm_error(
                    conversation,
                    self.provider(),
                    &self.llm().model,
                    cancel,
                    attempts.compactions,
                );
                let mut compacted = false;
                loop {
                    let event = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            yield Outcome::Done(false);
                            return;
                        }
                        event = recovery.next() => event,
                    };
                    match event {
                        Some(event) => {
                            compacted |= matches!(event, AgentEvent::Compacted(_));
                            yield Outcome::Item(event);
                        }
                        None => break,
                    }
                }
                yield Outcome::Done(compacted);
                return;
            }
            if llm_error.is_some_and(|error| error.kind == LlmErrorKind::StreamError) {
                if attempts.stream_retries >= MAX_STREAM_RETRIES {
                    tracing::warn!(model = %self.llm().model, attempts = attempts.stream_retries, "LLM stream retry limit reached");
                    yield Outcome::Item(AgentEvent::Status(format!(
                        "LLM stream retry limit reached after {MAX_STREAM_RETRIES} retries."
                    )));
                    yield Outcome::Done(false);
                    return;
                }
                attempts.stream_retries += 1;
                let delay = Duration::from_secs(1 << attempts.stream_retries)
                    .min(MAX_STREAM_RETRY_DELAY);
                tracing::warn!(
                    attempt = attempts.stream_retries,
                    max_retries = MAX_STREAM_RETRIES,
                    delay_secs = delay.as_secs(),
                    model = %self.llm().model,
                    error = %error,
                    "Retrying interrupted LLM stream"
                );
                yield Outcome::Item(AgentEvent::Status(format!(
                    "LLM stream interrupted, retrying in {}s ({}/{MAX_STREAM_RETRIES})...",
                    delay.as_secs(), attempts.stream_retries,
                )));
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        yield Outcome::Done(false);
                        return;
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
                yield Outcome::Done(true);
                return;
            }
            if conversation.strip_binary_parts() > 0 {
                yield Outcome::Item(AgentEvent::Status("Model rejected request, stripped images/documents and retrying...".into()));
                yield Outcome::Done(true);
                return;
            }
            yield Outcome::Done(false);
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use flashmind_types::{CompletionRequest, CompletionStream, LlmProvider, Provider};

    use super::*;
    use crate::ConversationEntry;

    #[test]
    fn a_completed_response_restores_the_compaction_budget() {
        let mut attempts = RecoveryAttempts {
            compactions: 2,
            stream_retries: 3,
        };
        attempts.record_success();
        assert_eq!(attempts.compactions, 0);
        assert_eq!(attempts.stream_retries, 0);
    }

    struct UnusedProvider;

    #[async_trait]
    impl LlmProvider for UnusedProvider {
        fn name(&self) -> &str {
            "unused"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        fn complete(&self, _request: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::empty())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stream_retries_stop_at_the_shared_limit() {
        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let error = LlmError::new(LlmErrorKind::StreamError, "Incomplete response").into();
        let mut attempts = RecoveryAttempts::default();
        for expected in std::iter::repeat_n(true, 10).chain([false]) {
            let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
            while recovery.next().await.is_some() {}
            assert_eq!(recovery.take_result(), Some(expected));
        }
    }

    #[tokio::test]
    async fn failed_compaction_stops_recovery_without_changing_context_or_model() {
        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let model = agent.llm().model.clone();
        let mut conversation = Conversation::with_system("Example system prompt.");
        for _ in 0..8 {
            conversation.add(ConversationEntry::user("Question"));
            conversation.add(ConversationEntry::assistant("Answer"));
        }
        let before = serde_json::to_value(&conversation).unwrap();
        let cancel = CancellationToken::new();
        let error = LlmError::new(LlmErrorKind::ContextLengthExceeded, "Example overflow").into();
        let mut attempts = RecoveryAttempts::default();
        let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
        while let Some(event) = recovery.next().await {
            assert!(!matches!(event, AgentEvent::Compacted(_)));
        }
        assert_eq!(recovery.take_result(), Some(false));
        drop(recovery);
        assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
        assert_eq!(agent.llm().model, model);
    }

    #[tokio::test(start_paused = true)]
    async fn stream_retries_back_off_and_cap_at_sixty_seconds() {
        use tokio::time::{Duration, Instant};

        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let error = LlmError::new(LlmErrorKind::StreamError, "Incomplete response").into();
        let mut attempts = RecoveryAttempts::default();
        for seconds in [2, 4, 8, 16, 32, 60, 60, 60, 60, 60] {
            let started = Instant::now();
            let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
            assert!(matches!(recovery.next().await, Some(AgentEvent::Status(_))));
            assert!(recovery.next().await.is_none());
            assert_eq!(recovery.take_result(), Some(true));
            assert_eq!(started.elapsed(), Duration::from_secs(seconds));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_interrupts_stream_retry_backoff() {
        use tokio::time::Instant;

        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let error = LlmError::new(LlmErrorKind::StreamError, "Incomplete response").into();
        let mut attempts = RecoveryAttempts::default();
        let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
        assert!(matches!(recovery.next().await, Some(AgentEvent::Status(_))));
        let started = Instant::now();
        cancel.cancel();
        assert!(recovery.next().await.is_none());
        assert_eq!(recovery.take_result(), Some(false));
        assert_eq!(started.elapsed(), std::time::Duration::ZERO);
    }

    #[tokio::test]
    async fn cancellation_does_not_consume_a_retry() {
        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = LlmError::new(LlmErrorKind::StreamError, "Incomplete response").into();
        let mut attempts = RecoveryAttempts::default();
        let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
        assert!(recovery.next().await.is_none());
        assert_eq!(recovery.take_result(), Some(false));
        drop(recovery);
        assert_eq!(attempts.stream_retries, 0);
    }
}
