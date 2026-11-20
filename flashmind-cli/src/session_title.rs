//! Short session titles written by a model from the session's first prompt.

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use flashmind_core::{Agent, CancellationToken};
use flashmind_memory::session::SessionStore;
use flashmind_types::{CompletionRequest, LlmProvider, Message, Model, StreamEvent};
use futures_util::{FutureExt, StreamExt};
use tokio::task::JoinHandle;

use crate::Cli;
use crate::config::Config;
use crate::presence::Updates as PresenceUpdates;

/// The longest title kept, in characters.
const MAX_CHARS: usize = 50;
/// The most of the first prompt sent to the title model.
const PROMPT_MAX_BYTES: usize = 2_000;
/// Room for models that reason before they answer, even when asked not to.
const MAX_TOKENS: u32 = 1_024;
const TIMEOUT: Duration = Duration::from_secs(30);

const INSTRUCTIONS: &str = "Write a short title for a coding session that starts with \
the request below. Use at most six words and 50 characters, in sentence case and in the \
request's language. Name the task: keep file names, code terms and ticket references exact. \
Reply with the title only, without quotes, markdown or trailing punctuation. Do not answer \
the request.";

/// The `title_model` setting: `off`, or the model that writes titles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TitleModel {
    Off,
    Model(Model),
}

impl FromStr for TitleModel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "off" | "none" => Ok(Self::Off),
            "" => bail!("the title model is empty; use a model ID or \"off\""),
            model => model
                .parse()
                .map(Self::Model)
                .with_context(|| format!("invalid title model: {model}")),
        }
    }
}

/// Which model writes a session's title.
#[derive(Clone, Default)]
pub(crate) enum TitleWriter {
    Off,
    /// The model the session is using when its first prompt arrives.
    #[default]
    SessionModel,
    Model(Titler),
}

impl TitleWriter {
    /// `--title-model`, else `title_model` in the config, else the session's
    /// model. A named model's provider is built now, so a missing key fails
    /// at startup rather than silently later.
    pub(crate) fn new(cli: &Cli, config: &Config) -> Result<Self> {
        let Some(value) = cli.title_model.as_deref().or(config.title_model.as_deref()) else {
            return Ok(Self::SessionModel);
        };
        match value.parse()? {
            TitleModel::Off => Ok(Self::Off),
            TitleModel::Model(model) => {
                let provider = crate::provider::build_provider(&model.provider, config)
                    .with_context(|| format!("could not set up the title model {model}"))?;
                Ok(Self::Model(Titler::new(provider, model)))
            }
        }
    }

    /// The titler for a session running on `agent`'s current model.
    pub(crate) fn titler(&self, agent: &Agent) -> Option<Titler> {
        match self {
            Self::Off => None,
            Self::SessionModel => Some(Titler {
                provider: agent.provider_arc().clone(),
                model: agent.llm().model.clone(),
            }),
            Self::Model(titler) => Some(titler.clone()),
        }
    }
}

/// A model ready to write titles.
#[derive(Clone)]
pub(crate) struct Titler {
    provider: Arc<dyn LlmProvider>,
    model: Model,
}

impl Titler {
    pub(crate) fn new(provider: Arc<dyn LlmProvider>, model: Model) -> Self {
        Self { provider, model }
    }

    /// A title for a session that starts with `prompt`.
    pub(crate) async fn generate(&self, prompt: &str) -> Result<String> {
        let mut request = CompletionRequest::new(
            self.model.clone(),
            vec![
                Message::system(INSTRUCTIONS),
                Message::user(truncate(prompt.trim(), PROMPT_MAX_BYTES)),
            ],
        );
        request.max_tokens = Some(MAX_TOKENS);
        let answer = tokio::time::timeout(TIMEOUT, self.complete(request))
            .await
            .context("the title model timed out")??;
        clean(&answer).with_context(|| format!("{} returned no usable title", self.model))
    }

    async fn complete(&self, request: CompletionRequest) -> Result<String> {
        let stream = self.provider.complete(request);
        tokio::pin!(stream);
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::ContentDelta(delta) = event? {
                text.push_str(&delta);
            }
        }
        Ok(text)
    }
}

/// Where a generated title goes besides the caller.
pub(crate) struct Destination<'a> {
    pub key: &'a str,
    pub store: Option<&'a SessionStore>,
    pub presence: &'a PresenceUpdates,
}

/// Writes a title for the session that starts with `prompt`, saves it and
/// reports it to the presence. A failure is logged, not returned: a session
/// without a title still works.
pub(crate) async fn write(titler: &Titler, prompt: &str, to: Destination<'_>) -> Option<String> {
    let title = match titler.generate(prompt).await {
        Ok(title) => title,
        Err(error) => {
            tracing::warn!(session = to.key, "no session title: {error:#}");
            return None;
        }
    };
    if let Some(store) = to.store
        && let Err(error) = store.update_title(to.key, &title).await
    {
        tracing::warn!(
            session = to.key,
            "could not save the session title: {error:#}"
        );
    }
    to.presence.title(to.key, &title);
    Some(title)
}

/// The title of the session the terminal UI shows, and the one being written.
#[derive(Default)]
pub(crate) struct PendingTitle {
    current: Option<String>,
    writing: Option<Writing>,
    /// One attempt per session: a model that cannot title is not asked again.
    attempted: bool,
}

struct Writing {
    cancel: CancellationToken,
    task: JoinHandle<Option<String>>,
}

impl PendingTitle {
    pub(crate) fn new(current: Option<String>) -> Self {
        Self {
            current,
            writing: None,
            attempted: false,
        }
    }

    pub(crate) fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Another session is now shown. A title still being written finishes
    /// for its own session.
    pub(crate) fn switch(&mut self, title: Option<String>) {
        self.current = title;
        self.writing = None;
        self.attempted = false;
    }

    /// The user named the session, so a title being written is dropped.
    pub(crate) fn rename(&mut self, title: &str) {
        if let Some(writing) = self.writing.take() {
            writing.cancel.cancel();
        }
        self.current = Some(title.to_owned());
    }

    /// Starts writing a title from `prompt` unless the session has one or
    /// one is being written.
    pub(crate) fn start(&mut self, titler: Option<Titler>, prompt: &str, to: Destination<'_>) {
        if self.current.is_some() || self.attempted || prompt.trim().is_empty() {
            return;
        }
        let Some(titler) = titler else {
            return;
        };
        self.attempted = true;
        let cancel = CancellationToken::new();
        let (prompt, key) = (prompt.to_owned(), to.key.to_owned());
        let (store, presence) = (to.store.cloned(), to.presence.clone());
        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                let to = Destination {
                    key: &key,
                    store: store.as_ref(),
                    presence: &presence,
                };
                tokio::select! {
                    () = cancel.cancelled() => None,
                    title = write(&titler, &prompt, to) => title,
                }
            }
        });
        self.writing = Some(Writing { cancel, task });
    }

    /// The new title, once it is written.
    pub(crate) fn poll(&mut self) -> Option<&str> {
        let writing = self.writing.as_mut()?;
        let finished = (&mut writing.task).now_or_never()?;
        self.writing = None;
        self.current = Some(finished.ok().flatten()?);
        self.current.as_deref()
    }
}

/// The longest prefix of `text` within `max_bytes`, on a character boundary.
fn truncate(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// One clean line from a model's answer, or `None` when nothing is left.
fn clean(answer: &str) -> Option<String> {
    // Some local models put their reasoning in the answer.
    let answer = answer
        .rsplit_once("</think>")
        .map_or(answer, |(_, rest)| rest);
    let line = answer
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    let line = strip_label(line);
    let line = line
        .trim_matches(|c: char| {
            matches!(c, '"' | '\'' | '`' | '*' | '_' | '#') || c.is_whitespace()
        })
        .trim_end_matches(['.', '!', '?', ':', ';', ',']);
    let words: Vec<&str> = line
        .split(|c: char| c.is_whitespace() || c.is_control())
        .filter(|word| !word.is_empty())
        .collect();
    let mut title = String::new();
    for word in words {
        let needed = word.chars().count() + usize::from(!title.is_empty());
        if title.chars().count() + needed > MAX_CHARS {
            if title.is_empty() {
                title.extend(word.chars().take(MAX_CHARS));
            }
            break;
        }
        if !title.is_empty() {
            title.push(' ');
        }
        title.push_str(word);
    }
    (!title.is_empty()).then_some(title)
}

fn strip_label(line: &str) -> &str {
    let lower = line.to_lowercase();
    for label in ["title:", "**title:**"] {
        if lower.starts_with(label) {
            return line[label.len()..].trim_start();
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presence::Presence;
    use async_trait::async_trait;
    use flashmind_types::{CompletionStream, Provider};
    use futures_util::stream;
    use tokio::sync::Notify;

    /// Answers with `title` once `release` is notified.
    struct Held {
        title: &'static str,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl LlmProvider for Held {
        fn name(&self) -> &str {
            "held"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        fn complete(&self, _request: CompletionRequest) -> CompletionStream {
            let (title, release) = (self.title, self.release.clone());
            Box::pin(stream::once(async move {
                release.notified().await;
                Ok(StreamEvent::ContentDelta(title.into()))
            }))
        }
    }

    fn held(title: &'static str) -> (Titler, Arc<Notify>) {
        let release = Arc::new(Notify::new());
        let provider = Arc::new(Held {
            title,
            release: release.clone(),
        });
        (
            Titler::new(provider, "ollama:titles".parse().unwrap()),
            release,
        )
    }

    async fn settle(pending: &mut PendingTitle) -> Option<String> {
        while pending
            .writing
            .as_ref()
            .is_some_and(|writing| !writing.task.is_finished())
        {
            tokio::task::yield_now().await;
        }
        pending.poll().map(str::to_owned)
    }

    #[tokio::test]
    async fn a_rename_wins_over_a_title_being_written() {
        let directory = tempfile::tempdir().unwrap();
        let presence = Presence::with_locator(
            "session_example",
            directory.path(),
            Some(directory.path().join("absent.json")),
        );
        let updates = presence.updates();
        let to = || Destination {
            key: "session_example",
            store: None,
            presence: &updates,
        };

        let (titler, release) = held("Generated title");
        let mut pending = PendingTitle::default();
        pending.start(Some(titler.clone()), "Fix the parser", to());
        pending.rename("Chosen name");
        release.notify_one();
        assert_eq!(settle(&mut pending).await, None);
        assert_eq!(pending.current(), Some("Chosen name"));

        pending.switch(None);
        pending.start(Some(titler.clone()), "Fix the parser", to());
        release.notify_one();
        assert_eq!(
            settle(&mut pending).await.as_deref(),
            Some("Generated title")
        );

        // A session is asked once, even when that attempt gave no title.
        pending.switch(None);
        let (empty, release) = held("  ");
        pending.start(Some(empty), "Fix the parser", to());
        release.notify_one();
        assert_eq!(settle(&mut pending).await, None);
        pending.start(Some(titler), "Fix it again", to());
        assert!(pending.writing.is_none());
        presence.shutdown().await;
    }

    #[test]
    fn plain_titles_pass_through() {
        assert_eq!(
            clean("Title generation in flsh").as_deref(),
            Some("Title generation in flsh")
        );
    }

    #[test]
    fn decoration_is_removed() {
        assert_eq!(
            clean("\n  \"Fix the `parser` crash.\"\nExtra line").as_deref(),
            Some("Fix the `parser` crash")
        );
        assert_eq!(
            clean("Title: Add ACP titles").as_deref(),
            Some("Add ACP titles")
        );
        assert_eq!(
            clean("**Title:** Add ACP titles").as_deref(),
            Some("Add ACP titles")
        );
        assert_eq!(
            clean("<think>The user wants</think>\nRename sessions").as_deref(),
            Some("Rename sessions")
        );
    }

    #[test]
    fn long_titles_stop_at_a_word() {
        let title = clean(&"word ".repeat(30)).unwrap();
        assert!(title.chars().count() <= MAX_CHARS);
        assert!(title.ends_with("word"));
        assert_eq!(clean(&"界".repeat(80)).unwrap().chars().count(), MAX_CHARS);
    }

    #[test]
    fn empty_answers_give_no_title() {
        assert_eq!(clean(""), None);
        assert_eq!(clean(" \n\"\"\n"), None);
        assert_eq!(clean("<think>only reasoning</think>"), None);
    }

    #[test]
    fn control_characters_never_reach_the_title() {
        let title = clean("Fix\tthe\x07parser\x1b bug").unwrap();
        assert!(!title.chars().any(char::is_control));
    }

    #[test]
    fn prompts_are_cut_on_a_character_boundary() {
        assert_eq!(truncate("aé", 2), "a");
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn title_model_setting_parses() {
        assert_eq!("off".parse::<TitleModel>().unwrap(), TitleModel::Off);
        assert!(matches!(
            "anthropic:example".parse::<TitleModel>().unwrap(),
            TitleModel::Model(_)
        ));
        assert!("".parse::<TitleModel>().is_err());
    }
}
