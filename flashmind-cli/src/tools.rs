use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock as SyncRwLock};

use flashmind_core::subagent::AgentManager;
use flashmind_skills::{DiskSkillProvider, SkillLoadTool, SkillProvider};
use flashmind_tools::protected::ProtectedPaths;
use flashmind_tools::{ToolBuilder, ToolSync};
use flashmind_types::tool::{CommandAllowList, ToolRegistry};
use flashmind_types::{AgentEvent, AgentLlmConfig, LlmProvider, Model};
use tokio::sync::{
    RwLock,
    mpsc::{self, UnboundedReceiver},
};

use crate::config::Config;
use crate::file_tools::register_file_tools;
use crate::paths::{claude_config_dir, config_dir};
use flashmind_tools::lsp::{LspManager, LspTool};

struct CommandPolicy {
    patterns: SyncRwLock<Vec<glob::Pattern>>,
    skip_permissions: bool,
}

impl CommandPolicy {
    fn from_config(patterns: Option<&Vec<String>>, skip_permissions: bool) -> Self {
        let patterns = patterns
            .into_iter()
            .flatten()
            .filter_map(|pattern| match glob::Pattern::new(pattern) {
                Ok(pattern) => Some(pattern),
                Err(error) => {
                    tracing::warn!(%error, "ignored invalid command allowlist pattern");
                    None
                }
            })
            .collect();
        Self {
            patterns: SyncRwLock::new(patterns),
            skip_permissions,
        }
    }
}

impl CommandAllowList for CommandPolicy {
    fn is_allowed(&self, command: &str) -> bool {
        if self.skip_permissions {
            return true;
        }
        self.patterns
            .read()
            .map(|patterns| patterns.iter().any(|pattern| pattern.matches(command)))
            .unwrap_or(false)
    }

    fn add_session_pattern(&self, pattern: &str) {
        let Ok(pattern) = glob::Pattern::new(pattern) else {
            tracing::warn!("ignored invalid command approval pattern");
            return;
        };
        if let Ok(mut patterns) = self.patterns.write() {
            patterns.push(pattern);
        }
    }
}

fn compute_skill_dirs(config: &Config) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(ref skill_dirs) = config.skill_dirs {
        for d in skill_dirs {
            let expanded = if let Some(rest) = d.strip_prefix("~/") {
                if let Some(home) = dirs::home_dir() {
                    home.join(rest)
                } else {
                    PathBuf::from(d)
                }
            } else {
                PathBuf::from(d)
            };
            dirs.push(expanded);
        }
    }

    if let Ok(cd) = config_dir() {
        let default_dir = cd.join("skills");
        let _ = std::fs::create_dir_all(&default_dir);
        if !dirs.contains(&default_dir) {
            dirs.push(default_dir);
        }
    }

    let cwd_skills = std::env::current_dir().unwrap_or_default().join("skills");
    if cwd_skills.is_dir() && !dirs.contains(&cwd_skills) {
        dirs.push(cwd_skills);
    }

    let mut roots = vec![std::env::current_dir().unwrap_or_default()];
    if let Some(home) = dirs::home_dir() {
        roots.push(home);
    }
    for root in roots {
        for relative in [".agents/skills", ".claude/skills", ".claude/commands"] {
            let path = root.join(relative);
            if path.is_dir() && !dirs.contains(&path) {
                dirs.push(path);
            }
        }
    }

    if let Some(claude_dir) = claude_config_dir() {
        let cwd = std::env::current_dir().unwrap_or_default();
        match crate::plugins::skill_dirs(&claude_dir, &cwd) {
            Ok(plugin_dirs) => {
                for dir in plugin_dirs {
                    if !dirs.contains(&dir) {
                        dirs.push(dir);
                    }
                }
            }
            Err(error) => tracing::warn!(%error, "Claude plugin skill discovery failed"),
        }
    }

    dirs
}

pub struct ToolSetup {
    pub subagents: Arc<AgentManager>,
    pub registry: ToolRegistry,
    pub events: UnboundedReceiver<AgentEvent>,
    pub sync: ToolSync,
    pub skill_index: String,
    pub skill_provider: Arc<RwLock<DiskSkillProvider>>,
}

pub fn skill_instructions(index: &str, can_load: bool) -> String {
    if index.is_empty() || !can_load {
        return String::new();
    }
    format!(
        "\n\n## Skills\n\nUse a listed skill when its description matches the task. Load its full instructions by name with `skill_load` before using it. Follow its instructions, use `read` for supporting files and `exec` for scripts, setting the working directory to the skill's directory when required. The user can activate a skill with `/{{skill}}`.\n{index}"
    )
}

struct CodingTools {
    subagents: Arc<AgentManager>,
    registry: ToolRegistry,
    sync: ToolSync,
    events: UnboundedReceiver<AgentEvent>,
}

async fn build_coding_tools(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    skip_permissions: bool,
    skill_provider: Arc<RwLock<DiskSkillProvider>>,
) -> CodingTools {
    let instructions = skill_instructions(&skill_index(&*skill_provider.read().await), true);
    let protected = Arc::new(ProtectedPaths::new(Path::new("/")));
    let vision_model: Option<Model> = config.vision_model.as_deref().and_then(|s| s.parse().ok());
    let (sender, events) = mpsc::unbounded_channel();
    let manager = Arc::new(AgentManager::new(config.max_agents(), 2).with_event_sender(sender));
    let policy: Arc<dyn CommandAllowList> = Arc::new(CommandPolicy::from_config(
        config.command_allowlist.as_ref(),
        skip_permissions,
    ));
    let lsp = Arc::new(LspManager::with_servers(config.lsp.clone()));
    let mut builder = ToolBuilder::new()
        .file_ops(vision_model.clone(), &protected)
        .bash(vec![], &protected, vec![], Some(policy.clone()))
        .worktrees(Some(policy))
        .subagents_with_instructions(manager.clone(), provider, Some(llm), instructions)
        // No Brave key: Firecrawl covers search as well as scraping, and one
        // provider with a key is worth more than two with a half-configured one.
        .search(None, config.firecrawl_api_key.clone());

    if let Some(ref vm) = vision_model
        && let Ok(provider) = crate::provider::build_provider(&vm.provider, config)
    {
        let mut providers = HashMap::new();
        providers.insert(vm.provider.clone(), provider);
        builder = builder.with_providers(Arc::new(providers));
    }

    let (mut registry, sync) = builder.models().build_with_sync().await;
    registry.register(Arc::new(LspTool::with_manager(lsp.clone())));
    register_file_tools(&mut registry, &protected, Some(lsp));
    registry.register(Arc::new(SkillLoadTool {
        provider: skill_provider,
    }));
    sync.sync(&mut registry).await;
    CodingTools {
        subagents: manager,
        registry,
        sync,
        events,
    }
}

pub async fn build_tools(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    skip_permissions: bool,
    allowed: Option<&[String]>,
) -> ToolSetup {
    let skill_provider = match DiskSkillProvider::discover(compute_skill_dirs(config)).await {
        Ok(provider) => provider,
        Err(error) => {
            tracing::warn!("skill discovery failed: {error}");
            DiskSkillProvider::discover(vec![])
                .await
                .expect("Skill discovery with no directories cannot fail.")
        }
    };
    let skill_index = skill_index(&skill_provider);
    let skill_provider = Arc::new(RwLock::new(skill_provider));
    let CodingTools {
        subagents,
        mut registry,
        sync,
        events,
    } = build_coding_tools(
        config,
        provider,
        llm,
        skip_permissions,
        skill_provider.clone(),
    )
    .await;
    if let Some(allowed) = allowed
        && !allowed.iter().any(|name| name == "all")
    {
        let names: Vec<String> = allowed
            .iter()
            .filter_map(|name| registry.get(name).map(|tool| tool.name().to_string()))
            .collect();
        registry.retain(|name| names.iter().any(|allowed| allowed == name));
    }
    sync.sync(&mut registry).await;

    ToolSetup {
        subagents,
        registry,
        events,
        sync,
        skill_index,
        skill_provider,
    }
}

pub fn skill_index(provider: &impl SkillProvider) -> String {
    provider
        .list()
        .iter()
        .map(|skill| {
            format!(
                "- /{}: {} (source: `{}`)",
                skill.meta.name,
                skill
                    .meta
                    .description
                    .as_deref()
                    .unwrap_or("no description"),
                skill.source_path.display(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Instructions injected into the system prompt and their load status.
#[derive(Default)]
pub struct AlwaysOnSkills {
    pub prompt: String,
    pub loaded: Vec<String>,
    pub missing: Vec<String>,
}

/// The full instructions of configured and flagged skills, once per name.
/// Names with no matching skill are logged and skipped.
pub fn always_on_skills(
    provider: &impl SkillProvider,
    names: &[String],
    flag_dirs: &[PathBuf],
) -> AlwaysOnSkills {
    let mut names = names.to_vec();
    let mut flagged = Vec::new();
    for dir in flag_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(name) = file_name
                .to_str()
                .and_then(|name| name.strip_prefix('.'))
                .and_then(|name| name.strip_suffix("-always"))
            else {
                continue;
            };
            if !name.is_empty() && entry.path().is_file() {
                flagged.push(name.to_string());
            }
        }
    }
    flagged.sort();
    names.extend(flagged);
    let mut seen = HashSet::new();
    names.retain(|name| seen.insert(name.clone()));

    let mut out = AlwaysOnSkills::default();
    for name in &names {
        let Some(skill) = provider.get(name) else {
            tracing::warn!(skill = %name, "always-on skill not found");
            out.missing.push(name.clone());
            continue;
        };
        out.loaded.push(name.clone());
        out.prompt.push_str(&format!(
            "\n\n## Always-on skill: {name}\n\nThe rules below apply to every response in this session. The user can turn them off for the session by asking to stop them.\n\n{}",
            skill.body.trim()
        ));
    }
    out
}

#[cfg(test)]
mod permission_tests {
    use super::*;
    use flashmind_core::CancellationToken;
    use flashmind_llm::OllamaProvider;
    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, Provider, Role, SessionId, StreamEvent,
        TokenUsage, ToolCall,
    };
    use serde_json::json;
    use std::sync::Mutex;

    #[test]
    fn skill_catalog_is_hidden_when_the_loader_is_unavailable() {
        let index = "- /example: Example instructions";
        assert!(skill_instructions(index, false).is_empty());
        assert!(skill_instructions("", true).is_empty());
        let instructions = skill_instructions(index, true);
        assert!(instructions.contains(index));
        assert!(instructions.contains("`skill_load`"));
    }

    struct UsageProvider {
        requests: Mutex<Vec<CompletionRequest>>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for UsageProvider {
        fn name(&self) -> &str {
            "mock"
        }
        fn provider(&self) -> Provider {
            Provider::OpenRouter
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.requests.lock().unwrap().push(request);
            Box::pin(futures_util::stream::iter([
                Ok(StreamEvent::ContentDelta("Done.".into())),
                Ok(StreamEvent::Usage(TokenUsage {
                    prompt_tokens: 1000,
                    completion_tokens: 20,
                    total_tokens: 1020,
                    cache_read_tokens: 800,
                    cache_creation_tokens: 100,
                })),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    #[tokio::test]
    async fn subagents_share_cache_prefix_and_current_session_and_forward_usage_once() {
        use flashmind_core::{Agent, Conversation};

        let provider = Arc::new(UsageProvider {
            requests: Mutex::new(Vec::<CompletionRequest>::new()),
        });
        let llm = AgentLlmConfig::new("openrouter:anthropic/example".parse().unwrap())
            .with_session_id("original-session".parse::<SessionId>().unwrap());
        let skills = Arc::new(RwLock::new(
            DiskSkillProvider::discover(vec![]).await.unwrap(),
        ));
        let CodingTools {
            registry, events, ..
        } = build_coding_tools(
            &Config::default(),
            provider.clone(),
            llm.clone(),
            false,
            skills,
        )
        .await;
        let mut agent = Agent::builder(provider.clone())
            .llm(llm)
            .tools(registry)
            .background_events(events)
            .working_dir("/srv/project/.worktrees/task")
            .build_sync();
        let cancel = CancellationToken::new();
        let mut conversation = Conversation::with_system("Shared instructions.");
        let mut parent_turn = agent.run_turn(&mut conversation, &cancel);
        let mut parent_usage = 0;
        while let Some(event) = parent_turn.next().await {
            parent_usage += usize::from(matches!(event, AgentEvent::Usage(_)));
        }
        assert!(parent_turn.take_result().unwrap().is_ok());
        assert_eq!(parent_usage, 1);
        drop(parent_turn);

        for (index, task) in ["First task.", "Second task.", "New session task."]
            .iter()
            .enumerate()
        {
            if index == 2 {
                agent.llm_mut().session_id = Some("new-session".parse().unwrap());
            }
            let mut usage_events = Vec::<TokenUsage>::new();
            let mut finished = false;
            for call in [
                ToolCall {
                    id: "spawn".into(),
                    name: "delegate".into(),
                    arguments: json!({"task": task, "name": "worker"}),
                },
                ToolCall {
                    id: "wait".into(),
                    name: "agent_wait".into(),
                    arguments: json!({"agent": "worker", "timeout_secs": 5}),
                },
            ] {
                let expected_model = agent.llm().model.clone();
                for batch in agent.tool_batches(&[call]) {
                    let mut execution = agent.execute_tool_batch(&batch, &cancel);
                    while let Some(event) = execution.next().await {
                        if let AgentEvent::SpawnedEvent { model, event, .. } = event {
                            assert_eq!(model.as_ref(), Some(&expected_model));
                            match *event {
                                AgentEvent::Usage(usage) => usage_events.push(usage),
                                AgentEvent::Done(_) => finished = true,
                                _ => {}
                            }
                        }
                    }
                    for done in execution.take_result().unwrap() {
                        assert!(done.result.is_success(), "{}", done.result.output());
                    }
                }
            }
            assert!(finished);
            assert_eq!(usage_events.len(), 1);
            assert_eq!(usage_events[0].cache_read_tokens, 800);
            assert_eq!(usage_events[0].cache_creation_tokens, 100);
            assert!(agent.drain_background_events().is_empty());
        }
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0].session_id, requests[1].session_id);
        assert_eq!(requests[1].session_id, requests[2].session_id);
        assert_eq!(
            requests[3].session_id.as_ref().unwrap().as_str(),
            "new-session"
        );
        let system = |request: &CompletionRequest| {
            request
                .messages
                .iter()
                .find(|message| message.role == Role::System)
                .unwrap()
                .content
                .clone()
        };
        assert!(system(&requests[1]).contains("Working directory: /srv/project/.worktrees/task"));
        assert_eq!(system(&requests[1]), system(&requests[2]));
        assert_eq!(system(&requests[2]), system(&requests[3]));
        assert!(!system(&requests[1]).contains("First task."));
        assert_eq!(requests[1].messages.last().unwrap().content, "First task.");
        assert_eq!(
            serde_json::to_value(&requests[1].tools).unwrap(),
            serde_json::to_value(&requests[2].tools).unwrap()
        );
    }

    #[tokio::test]
    async fn coding_toolkit_contains_files_shell_skills_and_subagents() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let provider = Arc::new(OllamaProvider::new(None, None).unwrap());
        let llm = AgentLlmConfig::new("ollama:example".parse().unwrap());
        let skills = Arc::new(RwLock::new(
            DiskSkillProvider::discover(vec![]).await.unwrap(),
        ));
        let CodingTools { registry, .. } =
            build_coding_tools(&Config::default(), provider, llm, false, skills).await;
        assert!(registry.get("lsp").is_some());
        let mut names = registry.list();
        names.retain(|name| *name != "lsp");
        assert_eq!(
            names,
            vec![
                "agent_model",
                "agent_status",
                "agent_terminate",
                "agent_wait",
                "delegate",
                "edit",
                "exec",
                "find",
                "image_read",
                "list_models",
                "process",
                "read",
                "skill_load",
                "worktree",
                "write",
            ],
        );
        assert!(!registry.get("exec").unwrap().concurrent());
    }

    #[tokio::test]
    async fn skill_load_uses_the_shared_provider_and_reports_missing_skills() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("example");
        std::fs::create_dir(&skill_dir).unwrap();
        let source = skill_dir.join("SKILL.md");
        std::fs::write(&source, "---\nname: example\n---\nRead the code.").unwrap();
        let skills = Arc::new(RwLock::new(
            DiskSkillProvider::discover(vec![dir.path().to_path_buf()])
                .await
                .unwrap(),
        ));
        let _ = rustls::crypto::ring::default_provider().install_default();
        let provider = Arc::new(OllamaProvider::new(None, None).unwrap());
        let llm = AgentLlmConfig::new("ollama:example".parse().unwrap());
        let CodingTools { registry, .. } =
            build_coding_tools(&Config::default(), provider, llm, false, skills.clone()).await;
        let mut call = ToolCall {
            id: "load-example".into(),
            name: "skill_load".into(),
            arguments: json!({"name": "example"}),
        };
        let cancel = CancellationToken::new();
        let result = registry.execute(&call, None, &cancel).await;
        assert!(result.is_success());
        assert_eq!(result.output(), "Read the code.");

        std::fs::write(&source, "---\nname: example\n---\nCheck the code.").unwrap();
        skills.write().await.refresh().await.unwrap();
        let result = registry.execute(&call, None, &cancel).await;
        assert!(result.is_success());
        assert_eq!(result.output(), "Check the code.");

        call.arguments = json!({"name": "missing"});
        let result = registry.execute(&call, None, &cancel).await;
        assert!(!result.is_success());
        assert_eq!(result.output(), "Skill not found: missing");
    }

    #[test]
    fn skip_permissions_bypasses_shell_allowlist_only_when_enabled() {
        assert!(!CommandPolicy::from_config(None, false).is_allowed("echo example"));
        assert!(CommandPolicy::from_config(None, true).is_allowed("echo example"));
        let patterns = vec!["cargo test*".to_string()];
        let policy = CommandPolicy::from_config(Some(&patterns), false);
        assert!(policy.is_allowed("cargo test -p example"));
        assert!(!policy.is_allowed("echo example"));
    }

    #[tokio::test]
    async fn always_on_skills_inline_bodies_and_skip_unknown_names() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("terse");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: terse\ndescription: Short answers\n---\nAnswer in one line.\n",
        )
        .unwrap();
        let provider = DiskSkillProvider::discover(vec![dir.path().to_path_buf()])
            .await
            .unwrap();

        let loaded = always_on_skills(&provider, &["terse".into(), "missing".into()], &[]);
        assert_eq!(loaded.loaded, ["terse"]);
        assert_eq!(loaded.missing, ["missing"]);
        let prompt = loaded.prompt;

        assert!(prompt.contains("## Always-on skill: terse"));
        assert!(prompt.contains("Answer in one line."));
        assert!(!prompt.contains("name: terse"));
        assert!(!prompt.contains("missing"));
    }

    #[tokio::test]
    async fn always_on_skills_merge_config_and_flags_without_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let skills = dir.path().join("skills");
        for name in ["manual", "terse", "detailed", "off"] {
            let skill_dir = skills.join(name);
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(skill_dir.join("SKILL.md"), format!("Rules for {name}.")).unwrap();
        }
        let provider = DiskSkillProvider::discover(vec![skills]).await.unwrap();
        let flag_dirs = vec![dir.path().join("config"), dir.path().join("claude")];
        for flag_dir in &flag_dirs {
            std::fs::create_dir(flag_dir).unwrap();
            std::fs::write(flag_dir.join(".terse-always"), "").unwrap();
        }
        let detailed_flag = flag_dirs[1].join(".detailed-always");
        std::fs::write(&detailed_flag, "").unwrap();
        std::fs::write(flag_dirs[0].join(".missing-always"), "").unwrap();
        std::fs::write(flag_dirs[0].join(".off"), "").unwrap();
        std::fs::create_dir(flag_dirs[1].join(".off-always")).unwrap();

        let config: Config =
            toml::from_str("always_skills = ['manual', 'terse', 'manual', 'terse']").unwrap();
        let names = config.always_skills.as_deref().unwrap_or_default();
        let loaded = always_on_skills(&provider, names, &flag_dirs);
        assert_eq!(loaded.loaded, ["manual", "terse", "detailed"]);
        assert_eq!(loaded.missing, ["missing"]);
        let prompt = loaded.prompt;
        for name in ["manual", "terse", "detailed"] {
            assert_eq!(
                prompt
                    .matches(&format!("## Always-on skill: {name}"))
                    .count(),
                1
            );
            assert!(prompt.contains(&format!("Rules for {name}.")));
        }
        assert!(!prompt.contains("Rules for off."));
        assert!(!prompt.contains("missing"));

        std::fs::remove_file(detailed_flag).unwrap();
        let prompt = always_on_skills(&provider, names, &flag_dirs).prompt;
        assert!(!prompt.contains("Rules for detailed."));
        assert!(prompt.contains("Rules for manual."));
        assert!(prompt.contains("Rules for terse."));

        let prompt = always_on_skills(&provider, &[], &flag_dirs).prompt;
        assert!(prompt.contains("Rules for terse."));
        assert!(!prompt.contains("Rules for manual."));
    }

    #[tokio::test]
    async fn default_prompt_has_rules_for_every_coding_tool() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let provider = Arc::new(OllamaProvider::new(None, None).unwrap());
        let llm = AgentLlmConfig::new("ollama:example".parse().unwrap());
        let skills = Arc::new(RwLock::new(
            DiskSkillProvider::discover(vec![]).await.unwrap(),
        ));
        let CodingTools { registry, .. } =
            build_coding_tools(&Config::default(), provider, llm, false, skills).await;

        let prompt = crate::prompt::default_prompt(&registry, Path::new("/"));
        let reminder = crate::prompt::tool_reminder(&registry);
        assert!(reminder.contains("tail: 40"));
        assert!(reminder.contains("Read a file before editing"));
        assert!(reminder.contains("independent tool calls in parallel"));
        assert!(!reminder.contains("instead of shell"));
        for tool in [
            "read",
            "find",
            "edit",
            "write",
            "exec",
            "delegate",
            "list_models",
            "image_read",
            "skill_load",
        ] {
            assert!(prompt.contains(&format!("- `{tool}`:")), "{tool}");
        }
    }
}
