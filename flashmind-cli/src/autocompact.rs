//! Argument parsing for the `/autocompact` command.

use flashmind_core::Agent;
use flashmind_types::Model;
use toml_edit::{DocumentMut, Item, value};

use crate::config::{AutoCompactSettings, Config};

/// What `/autocompact` was asked to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoCompactArg {
    /// Show the current setting.
    Show,
    /// Compact automatically.
    On,
    /// Never compact before the context window overflows.
    Off,
    /// Drop the token limit and use the model's default trigger.
    Default,
    /// Compact once the prompt passes this many tokens.
    Tokens(u32),
    /// Remove the current model's override.
    Inherit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Global,
    Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AutoCompactCommand {
    scope: Scope,
    arg: AutoCompactArg,
}

pub(crate) const USAGE: &str = "  Usage: /autocompact [global|model] [on|off|auto|<tokens>]\n  /autocompact model inherit removes the current model's override.";

impl AutoCompactCommand {
    pub(crate) fn parse(input: &str) -> Option<Self> {
        let mut words = input.split_whitespace();
        let first = words.next().unwrap_or("");
        let (scope, action) = if first.eq_ignore_ascii_case("global") {
            (Scope::Global, words.next().unwrap_or(""))
        } else if first.eq_ignore_ascii_case("model") {
            (Scope::Model, words.next().unwrap_or(""))
        } else {
            (Scope::Global, first)
        };
        if words.next().is_some() {
            return None;
        }
        let arg = AutoCompactArg::parse(action)?;
        if arg == AutoCompactArg::Inherit && scope != Scope::Model {
            return None;
        }
        Some(Self { scope, arg })
    }

    pub(crate) fn is_change(self) -> bool {
        self.arg != AutoCompactArg::Show
    }

    pub(crate) fn apply(self, config: &mut Config, model: &Model) {
        if !self.is_change() {
            return;
        }
        match self.scope {
            Scope::Global => {
                let mut settings = config.global_auto_compact();
                self.arg.apply(&mut settings);
                config.auto_compact = Some(settings.enabled);
                config.auto_compact_tokens = settings.tokens;
            }
            Scope::Model => {
                if self.arg == AutoCompactArg::Inherit {
                    config.auto_compact_models.remove(model);
                } else {
                    let mut settings = config.auto_compact_for(model);
                    self.arg.apply(&mut settings);
                    config.auto_compact_models.insert(model.clone(), settings);
                }
            }
        }
    }

    pub(crate) fn save(self, doc: &mut DocumentMut, config: &Config, model: &Model) {
        if !self.is_change() {
            return;
        }
        match self.scope {
            Scope::Global => self.arg.save(doc),
            Scope::Model => {
                let key = model.to_string();
                if let Some(settings) = config.auto_compact_models.get(model) {
                    let rule = &mut doc["auto_compact_models"][&key];
                    rule["enabled"] = value(settings.enabled);
                    if let Some(tokens) = settings.tokens {
                        rule["tokens"] = value(i64::from(tokens));
                    } else if let Some(table) = rule.as_table_like_mut() {
                        table.remove("tokens");
                    }
                } else if let Some(table) = doc
                    .get_mut("auto_compact_models")
                    .and_then(Item::as_table_like_mut)
                {
                    table.remove(&key);
                }
            }
        }
    }

    pub(crate) fn describe(self, config: &Config, model: &Model) -> String {
        let (scope, settings) = match self.scope {
            Scope::Global => ("global".to_string(), config.global_auto_compact()),
            Scope::Model => {
                let source = if config.auto_compact_models.contains_key(model) {
                    format!("{model}")
                } else {
                    format!("{model}, using global rule")
                };
                (source, config.auto_compact_for(model))
            }
        };
        let rule = if !settings.enabled {
            "off".to_string()
        } else if let Some(tokens) = settings.tokens {
            format!("at {tokens} tokens, capped by the model's default trigger")
        } else {
            "model default".to_string()
        };
        format!("  Auto-compact ({scope}): {rule}")
    }
}

impl AutoCompactArg {
    fn parse(input: &str) -> Option<Self> {
        let input = input.trim().to_ascii_lowercase();
        match input.as_str() {
            "" => Some(Self::Show),
            "on" => Some(Self::On),
            "off" => Some(Self::Off),
            "default" | "reset" | "auto" => Some(Self::Default),
            "inherit" => Some(Self::Inherit),
            _ => parse_tokens(&input).map(Self::Tokens),
        }
    }

    fn apply(self, settings: &mut AutoCompactSettings) {
        match self {
            Self::Show | Self::Inherit => {}
            Self::On => settings.enabled = true,
            Self::Off => settings.enabled = false,
            Self::Default => {
                *settings = AutoCompactSettings::default();
            }
            Self::Tokens(tokens) => {
                settings.enabled = true;
                settings.tokens = Some(tokens);
            }
        }
    }
}

impl AutoCompactArg {
    /// Record the change in the config document. `Show` changes nothing.
    fn save(self, doc: &mut DocumentMut) {
        match self {
            Self::Show | Self::Inherit => {}
            Self::On => doc["auto_compact"] = value(true),
            Self::Off => doc["auto_compact"] = value(false),
            Self::Default => {
                doc["auto_compact"] = value(true);
                doc.remove("auto_compact_tokens");
            }
            Self::Tokens(tokens) => {
                doc["auto_compact"] = value(true);
                doc["auto_compact_tokens"] = value(i64::from(tokens));
            }
        }
    }
}

/// Restore the active model's rule after a model or config change.
pub(crate) fn apply_settings(agent: &mut Agent, config: &Config) {
    let settings = config.auto_compact_for(&agent.llm().model);
    agent.set_auto_compact(settings.enabled);
    agent.set_compact_at_tokens(settings.tokens);
}

pub(crate) fn describe(agent: &Agent, config: &Config) -> String {
    let scope = if config.auto_compact_models.contains_key(&agent.llm().model) {
        "model override"
    } else {
        "global rule"
    };
    if !agent.auto_compact() {
        return format!("  Auto-compact: off ({scope})");
    }
    let trigger = agent.compaction_trigger_tokens();
    let window = agent.context_window();
    let source = match agent.compact_at_tokens() {
        Some(limit) if limit > trigger => {
            format!(" (limit {limit} is past the model default, so the default applies)")
        }
        Some(_) => String::new(),
        None => " (default)".into(),
    };
    format!("  Auto-compact: at {trigger} of {window} tokens{source} ({scope})")
}

fn parse_tokens(input: &str) -> Option<u32> {
    let (digits, scale) = match input.strip_suffix('k') {
        Some(digits) => (digits, 1_000),
        None => (input, 1),
    };
    let digits = digits.replace(['_', ','], "");
    let tokens = digits.parse::<u32>().ok()?.checked_mul(scale)?;
    (tokens > 0).then_some(tokens)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use flashmind_types::{CompletionRequest, CompletionStream, LlmProvider, Provider};

    use super::*;

    #[test]
    fn parses_words_and_token_counts() {
        assert_eq!(AutoCompactArg::parse(""), Some(AutoCompactArg::Show));
        assert_eq!(AutoCompactArg::parse(" OFF "), Some(AutoCompactArg::Off));
        assert_eq!(
            AutoCompactArg::parse("reset"),
            Some(AutoCompactArg::Default)
        );
        assert_eq!(
            AutoCompactArg::parse("80k"),
            Some(AutoCompactArg::Tokens(80_000))
        );
        assert_eq!(
            AutoCompactArg::parse("120_000"),
            Some(AutoCompactArg::Tokens(120_000))
        );
        assert_eq!(AutoCompactArg::parse("0"), None);
        assert_eq!(AutoCompactArg::parse("lots"), None);
        assert_eq!(AutoCompactArg::parse("9999999k"), None);
    }

    #[test]
    fn saving_keeps_comments_and_tables() {
        let mut doc: DocumentMut =
            "# a comment\nmodel = \"ollama:example\"\n\n[openrouter.routing]\nzdr = true\n"
                .parse()
                .unwrap();
        AutoCompactArg::Tokens(80_000).save(&mut doc);
        let config: crate::config::Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(config.auto_compact, Some(true));
        assert_eq!(config.auto_compact_tokens, Some(80_000));
        assert_eq!(config.openrouter.routing.unwrap().zdr, Some(true));
        assert!(doc.to_string().starts_with("# a comment\n"));

        AutoCompactArg::Default.save(&mut doc);
        AutoCompactArg::Off.save(&mut doc);
        let config: crate::config::Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(config.auto_compact, Some(false));
        assert_eq!(config.auto_compact_tokens, None);
    }

    #[test]
    fn parses_scopes_and_requires_model_scope_for_inherit() {
        for input in ["80k", "global 80k", " GLOBAL 80K "] {
            assert_eq!(
                AutoCompactCommand::parse(input),
                Some(AutoCompactCommand {
                    scope: Scope::Global,
                    arg: AutoCompactArg::Tokens(80_000),
                })
            );
        }
        assert_eq!(
            AutoCompactCommand::parse("model inherit"),
            Some(AutoCompactCommand {
                scope: Scope::Model,
                arg: AutoCompactArg::Inherit,
            })
        );
        for input in [
            "inherit",
            "global inherit",
            "model 0",
            "model on extra",
            "on off",
        ] {
            assert_eq!(AutoCompactCommand::parse(input), None, "{input}");
        }
    }

    fn change(config: &mut Config, doc: &mut DocumentMut, model: &Model, input: &str) {
        let command = AutoCompactCommand::parse(input).unwrap();
        command.apply(config, model);
        command.save(doc, config, model);
        let restored: Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(config.global_auto_compact(), restored.global_auto_compact());
        assert_eq!(config.auto_compact_models, restored.auto_compact_models);
    }

    #[test]
    fn model_rules_survive_global_changes_and_can_return_to_inheritance() {
        let model: Model = "openrouter:example/model".parse().unwrap();
        let other: Model = "ollama:example".parse().unwrap();
        let mut doc: DocumentMut = "# settings\n[openrouter.routing]\nzdr = true\n"
            .parse()
            .unwrap();
        let mut config: Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(
            config.auto_compact_for(&model),
            AutoCompactSettings::default()
        );

        change(&mut config, &mut doc, &model, "global 80k");
        change(&mut config, &mut doc, &model, "model 120k");
        change(&mut config, &mut doc, &model, "global off");
        assert_eq!(
            config.auto_compact_for(&model),
            AutoCompactSettings {
                enabled: true,
                tokens: Some(120_000)
            }
        );
        assert!(!config.auto_compact_for(&other).enabled);

        change(&mut config, &mut doc, &model, "model off");
        change(&mut config, &mut doc, &model, "model on");
        assert_eq!(config.auto_compact_for(&model).tokens, Some(120_000));
        change(&mut config, &mut doc, &model, "model auto");
        assert_eq!(
            config.auto_compact_for(&model),
            AutoCompactSettings::default()
        );
        change(&mut config, &mut doc, &model, "model inherit");
        assert_eq!(
            config.auto_compact_for(&model),
            config.global_auto_compact()
        );
        assert!(!config.auto_compact_models.contains_key(&model));
        assert!(doc.to_string().contains("# settings\n"));
        assert_eq!(config.openrouter.routing.unwrap().zdr, Some(true));
    }

    #[test]
    fn model_toggles_keep_the_inherited_limit_and_show_does_not_create_overrides() {
        let model: Model = "ollama:example".parse().unwrap();
        let mut doc: DocumentMut = "auto_compact_tokens = 80000\n".parse().unwrap();
        let mut config: Config = toml::from_str(&doc.to_string()).unwrap();
        let original = doc.to_string();
        change(&mut config, &mut doc, &model, "model");
        assert_eq!(doc.to_string(), original);
        assert!(config.auto_compact_models.is_empty());
        change(&mut config, &mut doc, &model, "model off");
        change(&mut config, &mut doc, &model, "model on");
        assert_eq!(config.auto_compact_for(&model).tokens, Some(80_000));
    }

    #[test]
    fn model_edits_preserve_other_models_and_support_inline_tables() {
        let model: Model = "ollama:example".parse().unwrap();
        let other: Model = "openrouter:example/model".parse().unwrap();
        let mut doc: DocumentMut = r#"auto_compact_models = { "ollama:example" = { tokens = 80000 }, "openrouter:example/model" = { enabled = false } }
"#.parse().unwrap();
        let mut config: Config = toml::from_str(&doc.to_string()).unwrap();
        change(&mut config, &mut doc, &model, "model auto");
        assert!(!config.auto_compact_for(&other).enabled);
        change(&mut config, &mut doc, &model, "model inherit");
        assert!(!config.auto_compact_for(&other).enabled);
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
            Box::pin(futures::stream::empty())
        }
    }

    #[test]
    fn changing_models_restores_each_rule_without_carrying_over_the_last_limit() {
        let mut agent = Agent::builder(Arc::new(UnusedProvider)).build_sync();
        let config: Config = toml::from_str(
            r#"
            auto_compact = false
            auto_compact_tokens = 80000
            [auto_compact_models."ollama:example"]
            tokens = 120000
            [auto_compact_models."openrouter:example/model"]
            enabled = true
        "#,
        )
        .unwrap();
        for (model, expected) in [
            (
                "ollama:example",
                AutoCompactSettings {
                    enabled: true,
                    tokens: Some(120_000),
                },
            ),
            ("openrouter:example/model", AutoCompactSettings::default()),
            (
                "ollama:other",
                AutoCompactSettings {
                    enabled: false,
                    tokens: Some(80_000),
                },
            ),
        ] {
            agent.llm_mut().model = model.parse().unwrap();
            apply_settings(&mut agent, &config);
            assert_eq!(agent.auto_compact(), expected.enabled);
            assert_eq!(agent.compact_at_tokens(), expected.tokens);
        }
    }
}
