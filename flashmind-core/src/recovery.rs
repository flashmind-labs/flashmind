//! Error recovery shared by the agent loop and interactive callers.

use flashmind_types::{AgentEvent, AgentStream, LlmError, LlmErrorKind, Outcome};

use crate::{Agent, CancellationToken, Conversation, handle_llm_error};

/// Recovery attempts consumed during the current user turn.
#[derive(Default)]
pub struct RecoveryAttempts {
    compactions: u8,
    stream_retries: u8,
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
                        Some(event) => yield Outcome::Item(event),
                        None => break,
                    }
                }
                yield Outcome::Done(true);
                return;
            }
            if llm_error.is_some_and(|error| error.kind == LlmErrorKind::StreamError)
                && attempts.stream_retries < 2
            {
                attempts.stream_retries += 1;
                yield Outcome::Item(AgentEvent::Status("Connection dropped, retrying...".into()));
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
            Box::pin(futures::stream::empty())
        }
    }

    #[tokio::test]
    async fn stream_retries_stop_at_the_shared_limit() {
        let agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let error = LlmError::new(LlmErrorKind::StreamError, "Incomplete response").into();
        let mut attempts = RecoveryAttempts::default();
        for expected in [true, true, false] {
            let mut recovery = agent.recover(&mut conversation, &cancel, &error, &mut attempts);
            while recovery.next().await.is_some() {}
            assert_eq!(recovery.take_result(), Some(expected));
        }
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
