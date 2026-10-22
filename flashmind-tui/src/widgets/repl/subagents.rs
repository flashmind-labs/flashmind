//! Selecting and talking to retained subagent conversations.

use std::io;
use std::sync::Arc;

use flashmind_types::AgentEvent;
use ratatui::crossterm::event::Event;
use uuid::Uuid;

use super::{KeyAction, PastedImage, Repl, ReplConfig};
use crate::term::Tui;
use crate::widgets::choice_picker::{ChoiceOption, ChoicePicker, ChoicePickerAction};

/// A selectable subagent conversation.
pub struct SubagentInfo {
    pub id: Uuid,
    pub title: String,
    pub label: String,
    pub status: String,
    pub running: bool,
}

/// Retained display events, independent of the agent runtime.
pub enum ConversationEvent {
    User(String),
    Agent(AgentEvent),
}

/// Connects the REPL's conversation browser to an application's subagent runtime.
pub trait SubagentSource: Send + Sync {
    fn list(&self) -> Vec<SubagentInfo>;
    fn events(&self, id: Uuid, offset: usize) -> Result<Vec<ConversationEvent>, String>;
    fn send(&self, id: Uuid, text: &str, images: Vec<PastedImage>) -> Result<(), String>;
    fn cancel(&self, id: Uuid);
}

pub(super) enum SubagentView {
    Picker {
        infos: Vec<SubagentInfo>,
        picker: Box<ChoicePicker>,
        tui: Tui,
    },
    Chat {
        id: Uuid,
        repl: Box<Repl<'static>>,
        offset: usize,
    },
}

impl Repl<'_> {
    pub fn set_subagent_source(&mut self, source: Arc<dyn SubagentSource>) {
        self.subagent_source = Some(source);
    }

    /// Open a conversation picker while the parent continues processing events.
    pub fn open_subagents(&mut self) -> io::Result<()> {
        let Some(source) = &self.subagent_source else {
            return Ok(());
        };
        let infos = source.list();
        if infos.is_empty() {
            self.toast = Some(("No active subagents".into(), self.tick));
            return Ok(());
        }
        let picker = ChoicePicker::new("Subagents".into(), infos.iter().map(choice).collect())
            .esc_hint("back");
        self.tui.suspend()?;
        self.subagent_view = Some(SubagentView::Picker {
            infos,
            picker: Box::new(picker),
            tui: Tui::new(),
        });
        Ok(())
    }

    /// Draw the selected view without drawing or replacing the parent's transcript.
    pub(super) fn draw_subagent_view(&mut self) -> io::Result<()> {
        let Some(source) = self.subagent_source.clone() else {
            return Ok(());
        };
        match self.subagent_view.as_mut() {
            Some(SubagentView::Picker { infos, picker, tui }) => {
                refresh_picker(infos, picker, source.list());
                tui.draw(|frame, _| picker.render(frame))
            }
            Some(SubagentView::Chat { id, repl, offset }) => {
                refresh_chat(&*source, *id, repl, offset)?;
                repl.tick_animations();
                let actions = repl.renderer.tick_tool();
                repl.apply_actions(&actions)?;
                repl.draw_input()
            }
            None => Ok(()),
        }
    }

    /// Route input to the visible view. Esc never reaches the parent's cancel token.
    pub(super) fn handle_subagent_event(&mut self, event: &Event) -> io::Result<bool> {
        let Some(mut view) = self.subagent_view.take() else {
            return Ok(false);
        };
        let Some(source) = self.subagent_source.clone() else {
            return Ok(false);
        };
        let mut close = false;
        match &mut view {
            SubagentView::Picker { infos, picker, tui } => {
                match picker.handle_event(event.clone()) {
                    Some(ChoicePickerAction::Select(response)) => {
                        if let Some(info) = infos.get(response.selected) {
                            tui.suspend()?;
                            view = self.subagent_chat(info)?;
                        }
                    }
                    Some(ChoicePickerAction::Cancel | ChoicePickerAction::Quit) => close = true,
                    _ => {}
                }
            }
            SubagentView::Chat { id, repl, .. } => match event {
                Event::Key(key) => match repl.process_key(*key) {
                    KeyAction::BackgroundCommand => repl.background_command(),
                    KeyAction::Escape | KeyAction::Quit => close = true,
                    KeyAction::Interrupt => source.cancel(*id),
                    KeyAction::Submit { text, images } => match source.send(*id, &text, images) {
                        Ok(()) => repl.pending_inputs.push_back((text, Vec::new())),
                        Err(error) => repl.toast = Some((error, repl.tick)),
                    },
                    KeyAction::ClearScreen => repl.clear_screen()?,
                    _ => {}
                },
                Event::Paste(text) => repl.handle_paste(text),
                Event::Mouse(mouse) => repl.handle_mouse(*mouse),
                Event::Resize(width, _) => {
                    repl.renderer.set_width(width.saturating_sub(1) as usize)
                }
                _ => {}
            },
        }
        if !close {
            self.subagent_view = Some(view);
        }
        Ok(true)
    }

    fn subagent_chat(&self, info: &SubagentInfo) -> io::Result<SubagentView> {
        let mut repl = Repl::new(ReplConfig {
            prompt: self.config.prompt.clone(),
            placeholder: "Message this subagent...".into(),
            max_input_height: self.config.max_input_height,
            ..ReplConfig::default()
        });
        repl.conversation_title = Some(info.title.clone());
        repl.set_expand_reasoning(self.expand_reasoning())?;
        repl.set_expand_tools(self.expand_tools())?;
        Ok(SubagentView::Chat {
            id: info.id,
            repl: Box::new(repl),
            offset: 0,
        })
    }

    fn apply_conversation_event(&mut self, event: ConversationEvent) -> io::Result<()> {
        match event {
            ConversationEvent::User(text) => {
                if self
                    .pending_inputs
                    .front()
                    .is_some_and(|(queued, _)| queued == &text)
                {
                    self.pending_inputs.pop_front();
                }
                self.mark_turn_start();
                self.replay_user_input(&text)
            }
            ConversationEvent::Agent(event) => self.apply_event(&event),
        }
    }
}

fn refresh_picker(
    infos: &mut Vec<SubagentInfo>,
    picker: &mut ChoicePicker,
    updated: Vec<SubagentInfo>,
) {
    let selected_id = infos.get(picker.selected).map(|info| info.id);
    let changed = !infos
        .iter()
        .map(|info| info.id)
        .eq(updated.iter().map(|info| info.id));
    let options = updated.iter().map(choice).collect();
    if changed {
        let selected = updated
            .iter()
            .position(|info| Some(info.id) == selected_id)
            .unwrap_or_else(|| picker.selected.min(updated.len().saturating_sub(1)));
        *picker = ChoicePicker::new("Subagents".into(), options).esc_hint("back");
        picker.selected = selected;
    } else {
        picker.options = options;
    }
    *infos = updated;
}

fn choice(info: &SubagentInfo) -> ChoiceOption {
    ChoiceOption {
        label: format!("{} ({})", info.label, info.status),
        accepts_input: false,
    }
}

fn refresh_chat(
    source: &dyn SubagentSource,
    id: Uuid,
    repl: &mut Repl<'_>,
    offset: &mut usize,
) -> io::Result<()> {
    let events = source.events(id, *offset).map_err(io::Error::other)?;
    *offset += events.len();
    for event in events {
        repl.apply_conversation_event(event)?;
    }
    if source
        .list()
        .iter()
        .any(|info| info.id == id && info.running)
    {
        repl.set_activity("Working");
    } else {
        repl.clear_activity();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::super::ForegroundCommand;
    use flashmind_types::tool::BackgroundRequest;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[derive(Default)]
    struct Source {
        sent: Mutex<Vec<String>>,
        cancelled: AtomicBool,
    }

    impl SubagentSource for Source {
        fn list(&self) -> Vec<SubagentInfo> {
            vec![SubagentInfo {
                id: Uuid::nil(),
                title: "worker".into(),
                label: "worker: Inspect the code.".into(),
                status: "completed".into(),
                running: false,
            }]
        }
        fn events(&self, _: Uuid, offset: usize) -> Result<Vec<ConversationEvent>, String> {
            Ok(vec![
                ConversationEvent::User("Inspect the code.".into()),
                ConversationEvent::Agent(AgentEvent::TextDelta("Child reply.\n".into())),
                ConversationEvent::Agent(AgentEvent::Done("Child reply.".into())),
            ]
            .into_iter()
            .skip(offset)
            .collect())
        }
        fn send(&self, _: Uuid, text: &str, _: Vec<PastedImage>) -> Result<(), String> {
            self.sent.lock().unwrap().push(text.into());
            Ok(())
        }
        fn cancel(&self, _: Uuid) {
            self.cancelled.store(true, Ordering::SeqCst);
        }
    }

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::from(code))
    }

    #[test]
    fn picker_keeps_the_selected_agent_when_another_agent_finishes() {
        fn info(id: u128) -> SubagentInfo {
            SubagentInfo {
                id: Uuid::from_u128(id),
                title: "worker".into(),
                label: "Example task".into(),
                status: "running".into(),
                running: true,
            }
        }
        let mut infos = vec![info(1), info(2), info(3)];
        let mut picker = ChoicePicker::new("Subagents".into(), infos.iter().map(choice).collect());
        picker.selected = 1;

        refresh_picker(&mut infos, &mut picker, vec![info(2), info(3)]);
        assert_eq!(infos[picker.selected].id, Uuid::from_u128(2));

        refresh_picker(&mut infos, &mut picker, vec![info(3)]);
        assert_eq!(infos[picker.selected].id, Uuid::from_u128(3));

        refresh_picker(&mut infos, &mut picker, Vec::new());
        assert!(infos.is_empty());
        assert!(picker.options.is_empty());
    }

    #[test]
    fn messaging_and_escape_preserve_the_running_parent_and_its_draft() {
        let source = Arc::new(Source::default());
        let mut parent = Repl::new(ReplConfig::default());
        parent.set_subagent_source(source.clone());
        parent.textarea.set_text("Parent draft");
        parent.set_activity("Parent tool");
        let cancel = CancellationToken::new();
        parent.cancel_token = Some(cancel.clone());
        parent.open_subagents().unwrap();
        assert!(parent.handle_subagent_event(&key(KeyCode::Enter)).unwrap());
        assert!(matches!(
            parent.subagent_view,
            Some(SubagentView::Chat { .. })
        ));
        // The parent can keep receiving output while the child owns the screen.
        parent
            .apply_event(&AgentEvent::TextDelta("Parent reply.\n".into()))
            .unwrap();
        parent
            .apply_event(&AgentEvent::Done("Parent reply.".into()))
            .unwrap();
        parent
            .handle_subagent_event(&Event::Paste("Follow up".into()))
            .unwrap();
        parent.handle_subagent_event(&key(KeyCode::Enter)).unwrap();
        assert_eq!(*source.sent.lock().unwrap(), ["Follow up"]);
        parent.handle_subagent_event(&key(KeyCode::Esc)).unwrap();
        assert!(parent.subagent_view.is_none());
        assert!(!cancel.is_cancelled());
        assert!(!source.cancelled.load(Ordering::SeqCst));
        assert_eq!(parent.textarea.text(), "Parent draft");
        assert_eq!(parent.activity.as_deref(), Some("Parent tool"));
        assert!(
            parent
                .tui
                .transcript_lines()
                .iter()
                .any(|line| line.to_string().contains("Parent reply."))
        );
        assert!(
            !parent
                .tui
                .transcript_lines()
                .iter()
                .any(|line| line.to_string().contains("Follow up"))
        );
    }

    #[test]
    fn child_history_is_separate_and_refresh_does_not_duplicate_it() {
        let source = Arc::new(Source::default());
        let mut parent = Repl::new(ReplConfig::default());
        parent.set_subagent_source(source.clone());
        parent.open_subagents().unwrap();
        parent.handle_subagent_event(&key(KeyCode::Enter)).unwrap();
        let Some(SubagentView::Chat { id, repl, offset }) = parent.subagent_view.as_mut() else {
            panic!("Enter opens the selected conversation");
        };
        refresh_chat(&*source, *id, repl, offset).unwrap();
        let lines = repl.tui.transcript_lines();
        assert_eq!(*offset, 3);
        assert!(
            lines
                .iter()
                .any(|line| line.to_string().contains("Inspect the code."))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.to_string().contains("Child reply."))
        );
        assert!(repl.activity.is_none());
        refresh_chat(&*source, *id, repl, offset).unwrap();
        assert_eq!(repl.tui.transcript_lines(), lines);
        assert!(parent.tui.transcript_lines().is_empty());
        // Reopening retains the same transcript through the source.
        parent.handle_subagent_event(&key(KeyCode::Esc)).unwrap();
        parent.open_subagents().unwrap();
        parent.handle_subagent_event(&key(KeyCode::Enter)).unwrap();
        let Some(SubagentView::Chat { id, repl, offset }) = parent.subagent_view.as_mut() else {
            panic!("the retained conversation can be reopened");
        };
        refresh_chat(&*source, *id, repl, offset).unwrap();
        assert_eq!(repl.tui.transcript_lines().len(), lines.len());
    }

    #[test]
    fn browser_escape_leaves_parent_cancellation_untouched() {
        let source = Arc::new(Source::default());
        let mut parent = Repl::new(ReplConfig::default());
        parent.set_subagent_source(source.clone());
        let request = BackgroundRequest::default();
        parent.foreground_command = Some(ForegroundCommand {
            id: "command".into(),
            request,
        });
        assert!(matches!(
            parent.process_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL)),
            KeyAction::BackgroundCommand
        ));
        parent.background_command();
        assert!(matches!(
            parent.process_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL)),
            KeyAction::OpenSubagents
        ));
        parent.open_subagents().unwrap();
        parent.handle_subagent_event(&key(KeyCode::Esc)).unwrap();
        assert!(parent.subagent_view.is_none());
        assert!(!source.cancelled.load(Ordering::SeqCst));
        parent.open_subagents().unwrap();
        parent.handle_subagent_event(&key(KeyCode::Enter)).unwrap();
        parent
            .handle_subagent_event(&Event::Key(KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL,
            )))
            .unwrap();
        assert!(source.cancelled.load(Ordering::SeqCst));
        assert!(parent.subagent_view.is_some());
    }
}
