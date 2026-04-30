//! [`AgentStream`] — a stream that yields events and terminates with a result.
//!
//! Replaces the `(impl Stream<Item = I>, oneshot::Receiver<O>)` pattern used
//! throughout the agent runtime. Producers yield [`Outcome::Item`] during
//! processing and [`Outcome::Done`] as the final value. Consumers iterate
//! the stream normally and call [`take_result`](AgentStream::take_result)
//! after exhaustion.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

/// A single element in an [`AgentStream`]: either an intermediate event or the
/// final result.
pub enum Outcome<I, O> {
    /// An intermediate item yielded during processing.
    Item(I),
    /// The final result — must be yielded exactly once, as the last element.
    Done(O),
}

/// A stream that yields events of type `I` and produces a final result `O`.
///
/// Implements [`Stream<Item = I>`] so it can be used with `.next().await` and
/// `StreamExt` combinators. When the inner stream yields [`Outcome::Done`],
/// the result is stored internally and the stream terminates. Retrieve it
/// with [`take_result`](Self::take_result).
///
/// # Thread safety
///
/// The inner stream must satisfy `Send + 'a`, which means `AgentStream` can be
/// safely moved across thread boundaries (e.g., spawned on a Tokio task).
///
/// # Example
///
/// ```ignore
/// fn work() -> AgentStream<'static, String, u32> {
///     AgentStream::new(async_stream::stream! {
///         yield Outcome::Item("hello".into());
///         yield Outcome::Item("world".into());
///         yield Outcome::Done(42);
///     })
/// }
///
/// let mut s = work();
/// while let Some(item) = s.next().await {
///     println!("{item}");
/// }
/// let answer = s.take_result().unwrap();
/// ```
pub struct AgentStream<'a, I, O> {
    inner: Pin<Box<dyn Stream<Item = Outcome<I, O>> + Send + 'a>>,
    result: Option<O>,
}

impl<'a, I, O> AgentStream<'a, I, O> {
    /// Wrap a stream of [`Outcome`] values into an `AgentStream`.
    pub fn new(inner: impl Stream<Item = Outcome<I, O>> + Send + 'a) -> Self {
        Self {
            inner: Box::pin(inner),
            result: None,
        }
    }

    /// Take the final result after the stream has been fully consumed.
    ///
    /// Returns `None` if the stream hasn't ended yet or if the producer
    /// never yielded [`Outcome::Done`].
    ///
    /// # Panics
    ///
    /// This method does not panic. It always returns an [`Option`], yielding
    /// `None` when no result is available.
    pub fn take_result(&mut self) -> Option<O> {
        self.result.take()
    }
}

impl<I, O: Unpin> AgentStream<'_, I, O> {
    /// Poll for the next item without requiring manual pinning or `StreamExt`.
    pub async fn next(&mut self) -> Option<I> {
        use futures::StreamExt;
        StreamExt::next(self).await
    }
}

impl<I, O: Unpin> Stream for AgentStream<'_, I, O> {
    type Item = I;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<I>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Outcome::Item(item))) => Poll::Ready(Some(item)),
            Poll::Ready(Some(Outcome::Done(result))) => {
                this.result = Some(result);
                Poll::Ready(None)
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn yields_items_then_result() {
        let mut s = AgentStream::new(async_stream::stream! {
            yield Outcome::Item("a");
            yield Outcome::Item("b");
            yield Outcome::Done(42);
        });

        let items: Vec<&str> = s.by_ref().collect().await;
        assert_eq!(items, vec!["a", "b"]);
        assert_eq!(s.take_result(), Some(42));
    }

    #[tokio::test]
    async fn empty_stream_no_result() {
        let mut s: AgentStream<&str, i32> = AgentStream::new(futures::stream::empty());

        let items: Vec<&str> = s.by_ref().collect().await;
        assert!(items.is_empty());
        assert_eq!(s.take_result(), None);
    }

    #[tokio::test]
    async fn result_only() {
        let mut s: AgentStream<&str, i32> = AgentStream::new(async_stream::stream! {
            yield Outcome::Done(99);
        });

        let items: Vec<&str> = s.by_ref().collect().await;
        assert!(items.is_empty());
        assert_eq!(s.take_result(), Some(99));
    }

    #[tokio::test]
    async fn next_convenience() {
        let mut s = AgentStream::new(async_stream::stream! {
            yield Outcome::Item("x");
            yield Outcome::Item("y");
            yield Outcome::Done(7);
        });

        assert_eq!(s.next().await, Some("x"));
        assert_eq!(s.next().await, Some("y"));
        assert_eq!(s.next().await, None);
        assert_eq!(s.take_result(), Some(7));
    }
}
