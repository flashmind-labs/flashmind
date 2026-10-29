//! Display pacing that catches up instead of blocking the event producer.

use std::collections::VecDeque;
use std::time::Duration;

use flashmind_types::AgentEvent;
use tokio::time::Instant;

// Skip pacing when either limit is reached. One event may exceed the byte limit.
const MAX_EVENTS: usize = 256;
const MAX_TEXT_BYTES: usize = 64 * 1024;
const NANOS_PER_CHAR: u128 = 1_000_000_000;

#[derive(Clone)]
pub(super) struct ToolDisplay {
    id: String,
    shown_at: Instant,
}

pub(super) struct StreamOutput {
    events: VecDeque<AgentEvent>,
    text_bytes: usize,
    offset: usize,
    rate: u32,
    credit: u128,
    tool_interval: Duration,
    last_tool: Option<ToolDisplay>,
    bypass: bool,
}

impl StreamOutput {
    pub(super) fn new(rate: u32) -> Self {
        Self {
            events: VecDeque::new(),
            text_bytes: 0,
            offset: 0,
            rate,
            credit: 0,
            tool_interval: Duration::ZERO,
            last_tool: None,
            bypass: false,
        }
    }

    pub(super) fn with_tool_display(
        mut self,
        interval: Duration,
        last_tool: Option<ToolDisplay>,
    ) -> Self {
        self.tool_interval = interval;
        self.last_tool = last_tool;
        self
    }

    pub(super) fn last_tool_display(&self) -> Option<&ToolDisplay> {
        self.last_tool.as_ref()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub(super) fn push(&mut self, event: AgentEvent) {
        if matches!(
            event,
            AgentEvent::Done(_) | AgentEvent::Cancelled | AgentEvent::Error(_)
        ) {
            self.skip_delay();
        }
        if let AgentEvent::TextDelta(text) | AgentEvent::ReasoningDelta(text) = &event {
            self.text_bytes += text.len();
        }
        self.events.push_back(event);
        if self.events.len() >= MAX_EVENTS || self.text_bytes >= MAX_TEXT_BYTES {
            self.skip_delay();
        }
    }

    pub(super) fn advance(&mut self, elapsed: Duration) {
        // Keep only fractional credit between frames. Idle time and missed
        // frames must not turn into a burst on the next redraw.
        self.credit %= NANOS_PER_CHAR;
        if !self.is_empty() {
            self.credit += elapsed.as_nanos() * u128::from(self.rate);
        }
    }

    pub(super) fn skip_delay(&mut self) {
        self.bypass = true;
    }

    pub(super) fn next_ready(&mut self) -> Option<AgentEvent> {
        self.next_ready_at(Instant::now())
    }

    fn next_ready_at(&mut self, now: Instant) -> Option<AgentEvent> {
        let event = self.events.front()?;
        let tool_id = match event {
            AgentEvent::ToolStart { id, .. } | AgentEvent::ToolResult { id, .. } => Some(id),
            AgentEvent::Interrupted { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        };
        let changing_tool =
            tool_id.is_some_and(|id| self.last_tool.as_ref().is_none_or(|tool| tool.id != *id));
        let response = matches!(event, AgentEvent::TextDelta(text) | AgentEvent::ReasoningDelta(text) if !text.is_empty())
            || matches!(event, AgentEvent::Compacted(_));
        if (changing_tool || response)
            && !self.bypass
            && self.last_tool.as_ref().is_some_and(|tool| {
                now.saturating_duration_since(tool.shown_at) < self.tool_interval
            })
        {
            return None;
        }
        if let Some(id) = tool_id.filter(|_| changing_tool) {
            self.last_tool = Some(ToolDisplay {
                id: id.clone(),
                shown_at: now,
            });
        }
        let (AgentEvent::TextDelta(text) | AgentEvent::ReasoningDelta(text)) = event else {
            return self.events.pop_front();
        };
        let remaining = &text[self.offset..];
        let mut end = self.offset;
        let mut count = 0;
        for ch in remaining.chars() {
            if self.rate != 0 && !self.bypass && count >= self.credit / NANOS_PER_CHAR {
                break;
            }
            end += ch.len_utf8();
            count += 1;
        }
        if end == self.offset && !remaining.is_empty() {
            return None;
        }
        if self.rate != 0 && !self.bypass {
            self.credit -= count * NANOS_PER_CHAR;
        }
        self.text_bytes -= end - self.offset;
        if self.offset == 0 && end == text.len() {
            return self.events.pop_front();
        }
        let chunk = text[self.offset..end].to_owned();
        let event = match event {
            AgentEvent::TextDelta(_) => AgentEvent::TextDelta(chunk),
            _ => AgentEvent::ReasoningDelta(chunk),
        };
        if end == text.len() {
            self.events.pop_front();
            self.offset = 0;
        } else {
            self.offset = end;
        }
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(event: AgentEvent) -> String {
        match event {
            AgentEvent::TextDelta(text) | AgentEvent::ReasoningDelta(text) => text,
            other => panic!("Expected a text delta, got {other:?}"),
        }
    }

    fn start(id: &str) -> AgentEvent {
        AgentEvent::ToolStart {
            id: id.into(),
            name: "exec".into(),
            humanized: "Example command".into(),
        }
    }

    fn result(id: &str) -> AgentEvent {
        AgentEvent::ToolResult {
            id: id.into(),
            name: "exec".into(),
            output: "Example output".into(),
            success: true,
            elapsed_ms: 20,
            sources: Vec::new(),
        }
    }

    #[test]
    fn fast_tools_finish_immediately_but_the_next_tool_waits_across_streams() {
        let now = Instant::now();
        let interval = Duration::from_millis(500);
        let mut output = StreamOutput::new(0).with_tool_display(interval, None);
        output.push(start("one"));
        assert!(matches!(
            output.next_ready_at(now),
            Some(AgentEvent::ToolStart { .. })
        ));
        output.push(result("one"));
        assert!(matches!(
            output.next_ready_at(now + Duration::from_millis(20)),
            Some(AgentEvent::ToolResult { .. })
        ));
        assert!(output.is_empty());

        let mut output =
            StreamOutput::new(0).with_tool_display(interval, output.last_tool_display().cloned());
        output.push(start("two"));
        assert!(
            output
                .next_ready_at(now + Duration::from_millis(499))
                .is_none()
        );
        assert!(
            matches!(output.next_ready_at(now + interval), Some(AgentEvent::ToolStart { id, .. }) if id == "two")
        );
    }

    #[test]
    fn parallel_results_remain_visible_before_the_next_result() {
        let now = Instant::now();
        let interval = Duration::from_millis(500);
        let mut output = StreamOutput::new(0).with_tool_display(interval, None);
        output.push(result("one"));
        output.push(result("two"));
        assert!(
            matches!(output.next_ready_at(now), Some(AgentEvent::ToolResult { id, .. }) if id == "one")
        );
        assert!(
            output
                .next_ready_at(now + Duration::from_millis(20))
                .is_none()
        );
        assert!(
            matches!(output.next_ready_at(now + interval), Some(AgentEvent::ToolResult { id, .. }) if id == "two")
        );
    }

    #[test]
    fn command_preview_gets_its_minimum_time_before_thinking_or_reply_text() {
        for event in [
            AgentEvent::ReasoningDelta("Considering the result.".into()),
            AgentEvent::TextDelta("Here is the result.".into()),
        ] {
            let now = Instant::now();
            let interval = Duration::from_millis(500);
            let mut output = StreamOutput::new(0).with_tool_display(interval, None);
            output.push(start("one"));
            assert!(output.next_ready_at(now).is_some());
            output.push(result("one"));
            assert!(
                output
                    .next_ready_at(now + Duration::from_millis(20))
                    .is_some()
            );
            output.push(event);
            assert!(
                output
                    .next_ready_at(now + Duration::from_millis(499))
                    .is_none()
            );
            assert!(output.next_ready_at(now + interval).is_some());
        }
    }

    #[test]
    fn cancellation_and_zero_interval_skip_tool_delays() {
        let now = Instant::now();
        for interval in [Duration::ZERO, Duration::from_millis(500)] {
            let mut output = StreamOutput::new(100).with_tool_display(interval, None);
            output.push(start("one"));
            output.push(start("two"));
            assert!(output.next_ready_at(now).is_some());
            if !interval.is_zero() {
                assert!(output.next_ready_at(now).is_none());
                output.skip_delay();
            }
            assert!(
                matches!(output.next_ready_at(now), Some(AgentEvent::ToolStart { id, .. }) if id == "two")
            );
        }
    }

    #[test]
    fn completion_releases_remaining_text_in_order_without_waiting() {
        let mut output = StreamOutput::new(100);
        output.push(AgentEvent::TextDelta("abcdef".into()));
        assert!(output.next_ready().is_none());
        output.advance(Duration::from_millis(20));
        assert_eq!(text(output.next_ready().unwrap()), "ab");
        output.push(AgentEvent::Done("abcdef".into()));
        assert_eq!(text(output.next_ready().unwrap()), "cdef");
        assert!(matches!(output.next_ready(), Some(AgentEvent::Done(_))));
        assert!(output.is_empty());
    }

    #[test]
    fn many_tokens_share_the_frame_budget_with_reasoning() {
        let mut output = StreamOutput::new(100);
        output.push(AgentEvent::ReasoningDelta("ab".into()));
        output.push(AgentEvent::TextDelta("c".into()));
        output.push(AgentEvent::TextDelta("d".into()));
        output.advance(Duration::from_millis(30));
        assert!(matches!(output.next_ready(), Some(AgentEvent::ReasoningDelta(s)) if s == "ab"));
        assert_eq!(text(output.next_ready().unwrap()), "c");
        assert!(output.next_ready().is_none());
        output.advance(Duration::from_millis(10));
        assert_eq!(text(output.next_ready().unwrap()), "d");
    }

    #[test]
    fn unicode_is_counted_in_characters_and_preserved() {
        let mut output = StreamOutput::new(2);
        output.push(AgentEvent::TextDelta("é日本🦀".into()));
        for expected in ["é", "日", "本", "🦀"] {
            output.advance(Duration::from_millis(500));
            assert_eq!(text(output.next_ready().unwrap()), expected);
        }
        assert!(output.is_empty());
        assert_eq!(output.text_bytes, 0);
    }

    #[test]
    fn idle_credit_is_discarded_but_fractional_credit_accumulates() {
        let mut output = StreamOutput::new(1);
        output.advance(Duration::from_secs(60));
        output.push(AgentEvent::TextDelta("abc".into()));
        for _ in 0..3 {
            output.advance(Duration::from_millis(250));
            assert!(output.next_ready().is_none());
        }
        output.advance(Duration::from_millis(250));
        assert_eq!(text(output.next_ready().unwrap()), "a");
        assert!(output.next_ready().is_none());
    }

    #[test]
    fn cancellation_releases_the_remaining_text_without_loss() {
        let mut output = StreamOutput::new(10);
        output.push(AgentEvent::TextDelta("abcdef".into()));
        output.advance(Duration::from_millis(200));
        assert_eq!(text(output.next_ready().unwrap()), "ab");
        output.push(AgentEvent::Cancelled);
        assert_eq!(text(output.next_ready().unwrap()), "cdef");
        assert!(matches!(output.next_ready(), Some(AgentEvent::Cancelled)));
        assert!(output.is_empty());
    }

    #[test]
    fn zero_rate_and_non_text_events_need_no_credit() {
        let mut output = StreamOutput::new(0);
        output.push(AgentEvent::TextDelta("abcdef".into()));
        assert_eq!(text(output.next_ready().unwrap()), "abcdef");
        let mut output = StreamOutput::new(100);
        output.push(AgentEvent::Status("Working".into()));
        output.push(AgentEvent::TextDelta(String::new()));
        assert!(matches!(output.next_ready(), Some(AgentEvent::Status(_))));
        assert_eq!(text(output.next_ready().unwrap()), "");
        assert!(output.is_empty());
    }

    #[test]
    fn event_backlog_catches_up_without_waiting_for_tool_timers() {
        let now = Instant::now();
        let mut output = StreamOutput::new(1).with_tool_display(Duration::from_secs(60), None);
        output.push(start("one"));
        assert!(output.next_ready_at(now).is_some());
        output.push(start("two"));
        assert!(output.next_ready_at(now).is_none());
        for _ in 1..MAX_EVENTS {
            output.push(AgentEvent::TextDelta("a".into()));
        }
        assert!(
            matches!(output.next_ready_at(now), Some(AgentEvent::ToolStart { id, .. }) if id == "two")
        );
        for _ in 1..MAX_EVENTS {
            assert_eq!(text(output.next_ready_at(now).unwrap()), "a");
        }
        assert!(output.is_empty());
    }

    #[test]
    fn text_backlog_catches_up_without_waiting_for_display_credit() {
        let mut output = StreamOutput::new(1);
        let reply = "é".repeat(MAX_TEXT_BYTES / 2);
        output.push(AgentEvent::TextDelta(reply.clone()));
        assert_eq!(text(output.next_ready().unwrap()), reply);
        assert!(output.is_empty());
    }

    #[test]
    fn terminal_events_release_tool_backlogs_without_waiting() {
        for event in [
            AgentEvent::Done(String::new()),
            AgentEvent::Cancelled,
            AgentEvent::Error("Example failure".into()),
        ] {
            let now = Instant::now();
            let mut output = StreamOutput::new(1).with_tool_display(Duration::from_secs(60), None);
            output.push(start("one"));
            assert!(output.next_ready_at(now).is_some());
            output.push(start("two"));
            assert!(output.next_ready_at(now).is_none());
            output.push(event);
            assert!(
                matches!(output.next_ready_at(now), Some(AgentEvent::ToolStart { id, .. }) if id == "two")
            );
            assert!(matches!(
                output.next_ready_at(now),
                Some(AgentEvent::Done(_) | AgentEvent::Cancelled | AgentEvent::Error(_))
            ));
            assert!(output.is_empty());
        }
    }
}
