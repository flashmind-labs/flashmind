//! Composable tool registry builder for shared tools.
//!
//! Registers tools from the `flashmind-tools` crate. Binary-specific tools
//! (canvas, cron, slack, telegram, webhooks, memory) are added by the agent
//! binary after calling [`ToolBuilder::build`].

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use flashmind_types::Tool;

#[cfg(feature = "subagent")]
use flashmind_core::AgentManager;
use flashmind_types::llm::ProviderRegistry;
use flashmind_types::model::Model;
use flashmind_types::tool::ToolRegistry;
#[cfg(feature = "subagent")]
use flashmind_types::{AgentLlmConfig, LlmProvider};
use tokio::sync::RwLock;

#[cfg(feature = "subagent")]
use crate::subagent::{AgentStatusTool, AgentTerminateTool, AgentWaitTool, DelegateTool};

use crate::audio::{AudioConfig, ListVoicesTool, TranscribeTool, TtsTool};
use crate::bash::BashTool;
use crate::brave::BraveSearchTool;
use crate::file_cache::FileCache;
use crate::file_ops::{FileDeleteTool, FileListTool, FileReadTool, FileWriteTool, ReadLinesTool};
use crate::firecrawl::{WebCrawlTool, WebMapTool, WebScrapeTool, WebSearchTool};
use crate::glob::GlobTool;
#[cfg(any(
    feature = "gmail",
    feature = "google-calendar",
    feature = "google-contacts"
))]
use crate::google::client::GoogleConfig;
use crate::grep::GrepTool;
use crate::http::HttpRequestTool;
use crate::image_edit::ImageEditTool;
use crate::image_gen::GenerateImageTool;
use crate::image_read::ImageReadTool;
use crate::json_query::JsonQueryTool;
use crate::list_models::ListModelsTool;
#[cfg(feature = "mcp")]
use crate::mcp::{McpAuthHandler, McpConfigProvider, McpRegistry};
#[cfg(feature = "outlook")]
use crate::outlook::OutlookConfig;
use crate::process::{ProcessRegistry, ProcessTool};
use crate::protected::ProtectedPaths;
use crate::search_cache::{SearchCacheRef, SearchResultCache};
use crate::search_read::WebSearchReadTool;
use crate::sqlite::SqliteQueryTool;
use crate::str_diff::StrDiffTool;
use crate::text_replace::StrReplaceTool;
use crate::text_replace_regex::StrReplaceRegexTool;
use crate::time::TimeTool;
use crate::tool_sync::ToolSync;
use crate::video_gen::GenerateVideoTool;
/// Shared queue for tools that should be registered on the next sync.
///
/// Used by auth tools (Google, Outlook) to dynamically add service tools
/// after a successful OAuth flow.
pub type PendingTools = Arc<Mutex<Vec<Arc<dyn Tool>>>>;

/// Composable builder for [`ToolRegistry`].
///
/// ```rust,ignore
/// let registry = ToolBuilder::new()
///     .with_providers(providers)
///     .file_ops(ocr_model, &protected)
///     .bash(secrets, &protected, forbidden_cmds, None)
///     .search(brave_key, firecrawl_key)
///     .time()
///     .sqlite()
///     .http()
///     .json()
///     .audio(model, voice, audio_dir)
///     .models()
///     .generate(image_model, video_model, output_dir)
///     .subagents(manager, provider, llm)
///     .mcp(provider, auth_handler)
///     .build();
/// ```
pub struct ToolBuilder {
    registry: ToolRegistry,
    file_cache: FileCache,
    providers: ProviderRegistry,
    offline: bool,
    #[cfg(feature = "mcp")]
    mcp_registry: Option<McpRegistry>,
    pending_tools: PendingTools,
}

impl Default for ToolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolBuilder {
    /// Create a new tool builder with no tools registered yet.
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
            file_cache: FileCache::new(),
            providers: Arc::new(std::collections::HashMap::new()),
            offline: false,
            #[cfg(feature = "mcp")]
            mcp_registry: None,
            pending_tools: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set the provider registry for audio, image, video, and model-listing tools.
    pub fn with_providers(mut self, providers: ProviderRegistry) -> Self {
        self.providers = providers;
        self
    }

    /// Enable offline mode — skips registration of network-dependent tools (web, search).
    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Register file operation tools: `file_read`, `file_write`, `file_delete`,
    /// `file_list`, `read_lines`, `glob`, `grep`, `str_replace`,
    /// `str_replace_regex`, `image_read`, `str_diff`.
    pub fn file_ops(mut self, ocr_model: Option<Model>, protected: &Arc<ProtectedPaths>) -> Self {
        self.registry.register(Arc::new(FileReadTool {
            protected: protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(FileWriteTool {
            protected: protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(FileDeleteTool {
            protected: protected.clone(),
        }));
        self.registry.register(Arc::new(FileListTool));
        self.registry.register(Arc::new(ReadLinesTool {
            protected: protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(GlobTool {
            protected: protected.clone(),
        }));
        self.registry.register(Arc::new(GrepTool));
        self.registry.register(Arc::new(StrReplaceTool {
            protected: protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(StrReplaceRegexTool {
            protected: protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(ImageReadTool {
            providers: Arc::clone(&self.providers),
            ocr_model,
        }));
        self.registry.register(Arc::new(StrDiffTool));
        self
    }

    /// Register shell and process tools: `exec` (alias: `bash_exec`) and
    /// `process` for managing background tasks.
    pub fn bash(
        mut self,
        secrets: Vec<String>,
        protected: &Arc<ProtectedPaths>,
        forbidden_cmds: Vec<flashmind_types::tool::ForbiddenCmd>,
        allowlist: Option<Arc<dyn flashmind_types::tool::CommandAllowList>>,
    ) -> Self {
        let process_registry = ProcessRegistry::new();

        self.registry.register(Arc::new(BashTool {
            protected: protected.clone(),
            secrets,
            process_registry: process_registry.clone(),
            forbidden_cmds,
            allowlist,
        }));
        self.registry.alias("bash_exec", "exec");
        self.registry.register(Arc::new(ProcessTool {
            registry: process_registry,
        }));
        self
    }

    /// Register web search tools: `brave_search`, `firecrawl_search`, and
    /// `web_search_read`. Also registers `web_crawl`, `web_scrape`, and
    /// `web_map` when Firecrawl is configured. Skipped in offline mode.
    pub fn search(
        mut self,
        brave_api_key: Option<String>,
        firecrawl_api_key: Option<String>,
    ) -> Self {
        if self.offline {
            return self;
        }

        if let Some(api_key) = brave_api_key {
            self.registry
                .register(Arc::new(BraveSearchTool::new(api_key)));
        }

        if let Some(api_key) = firecrawl_api_key {
            let search_cache: SearchCacheRef = Arc::new(RwLock::new(SearchResultCache::new()));
            self.registry.register(Arc::new(WebSearchTool::new(
                api_key.clone(),
                search_cache.clone(),
            )));
            self.registry
                .register(Arc::new(WebSearchReadTool::new(search_cache.clone())));
            self.registry.register(Arc::new(WebCrawlTool::new(
                api_key.clone(),
                search_cache.clone(),
            )));
            self.registry
                .register(Arc::new(WebScrapeTool::new(api_key.clone())));
            self.registry.register(Arc::new(WebMapTool::new(api_key)));
        }

        self
    }

    /// Register the `get_time` utility tool.
    pub fn time(mut self) -> Self {
        self.registry.register(Arc::new(TimeTool));
        self
    }

    /// Register the `sqlite_query` tool for read-only SQL access.
    pub fn sqlite(mut self) -> Self {
        self.registry.register(Arc::new(SqliteQueryTool));
        self
    }

    /// Register HTTP and web tools: `http_request`, `web_fetch`, `web_scrape`,
    /// `web_crawl`, `web_map`.
    pub fn http(mut self) -> Self {
        self.registry.register(Arc::new(HttpRequestTool::new()));
        self
    }

    /// Register the `json_query` tool for querying JSON data.
    pub fn json(mut self) -> Self {
        self.registry.register(Arc::new(JsonQueryTool));
        self
    }

    /// Register audio tools: `tts`, `transcribe`, and `list_voices`.
    pub fn audio(
        mut self,
        model: Option<Model>,
        voice: Option<String>,
        audio_dir: PathBuf,
    ) -> Self {
        let audio_config = AudioConfig { model, voice };
        self.registry.register(Arc::new(TtsTool::new(
            audio_config.clone(),
            audio_dir,
            Arc::clone(&self.providers),
        )));
        self.registry.register(Arc::new(TranscribeTool::new(
            None,
            Arc::clone(&self.providers),
        )));
        self.registry.register(Arc::new(ListVoicesTool::new(
            audio_config,
            Arc::clone(&self.providers),
        )));
        self
    }

    /// Register the `list_models` tool for discovering available LLM models.
    pub fn models(mut self) -> Self {
        self.registry.register(Arc::new(ListModelsTool {
            providers: Arc::clone(&self.providers),
        }));
        self
    }

    /// Register media generation tools: `image_gen`, `image_edit`, `video_gen`.
    pub fn generate(
        mut self,
        image_model: Option<Model>,
        video_model: Option<Model>,
        output_dir: PathBuf,
    ) -> Self {
        self.registry.register(Arc::new(ImageEditTool::new(
            image_model.clone(),
            output_dir.clone(),
            Arc::clone(&self.providers),
        )));
        self.registry.register(Arc::new(GenerateImageTool::new(
            image_model,
            output_dir.clone(),
            Arc::clone(&self.providers),
        )));
        self.registry.register(Arc::new(GenerateVideoTool::new(
            video_model,
            output_dir,
            Arc::clone(&self.providers),
        )));
        self
    }

    /// Register subagent management tools: `delegate`, `communicate`,
    /// `agent_status`, `agent_wait`, `agent_terminate`.
    #[cfg(feature = "subagent")]
    pub fn subagents(
        mut self,
        manager: Arc<AgentManager>,
        provider: Arc<dyn LlmProvider>,
        llm: Option<AgentLlmConfig>,
    ) -> Self {
        let mut delegate = DelegateTool::new(manager.clone(), provider);
        if let Some(llm) = llm {
            delegate = delegate.with_llm(llm);
        }
        self.registry.register(Arc::new(delegate));
        self.registry
            .register(Arc::new(AgentStatusTool::new(manager.clone())));
        self.registry
            .register(Arc::new(AgentWaitTool::new(manager.clone())));
        self.registry
            .register(Arc::new(AgentTerminateTool::new(manager)));
        self
    }

    /// Google API tools (Gmail, Calendar, Contacts) — all tools including write.
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `google_auth` tool which handles the
    /// OAuth flow and dynamically adds service tools on successful auth.
    #[cfg(any(
        feature = "gmail",
        feature = "google-calendar",
        feature = "google-contacts"
    ))]
    pub fn google(self, config: &GoogleConfig) -> Self {
        self.google_impl(config, false)
    }

    /// Google API tools (Gmail, Calendar, Contacts) — read-only tools.
    #[cfg(any(
        feature = "gmail",
        feature = "google-calendar",
        feature = "google-contacts"
    ))]
    pub fn google_readonly(self, config: &GoogleConfig) -> Self {
        self.google_impl(config, true)
    }

    #[cfg(any(
        feature = "gmail",
        feature = "google-calendar",
        feature = "google-contacts"
    ))]
    fn google_impl(mut self, config: &GoogleConfig, readonly: bool) -> Self {
        use crate::google::auth::Credentials;
        use crate::google::auth_tool::GoogleAuthTool;
        use crate::oauth;

        if self.offline {
            return self;
        }

        // Collect scopes for all enabled services
        let scopes: Vec<&'static str> = [
            #[cfg(feature = "gmail")]
            crate::google::gmail::SCOPE,
            #[cfg(feature = "google-calendar")]
            crate::google::calendar::SCOPE,
            #[cfg(feature = "google-contacts")]
            crate::google::contacts::SCOPE,
        ]
        .to_vec();

        // Check if we have a valid cached token
        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token || matches!(config.credentials, Credentials::ServiceAccount { .. }) {
            self = self.google_register_services(config, readonly);
        } else {
            // No valid token + user OAuth — register auth tool
            self.registry.register(Arc::new(GoogleAuthTool {
                config: config.clone(),
                scopes,
                readonly,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    /// Register Google service tools directly (when token is available).
    #[cfg(any(
        feature = "gmail",
        feature = "google-calendar",
        feature = "google-contacts"
    ))]
    fn google_register_services(mut self, config: &GoogleConfig, readonly: bool) -> Self {
        #[cfg(feature = "gmail")]
        {
            use crate::google::gmail::{self, tools::*};
            match gmail::new_client(config) {
                Ok(c) => {
                    let client = Arc::new(c);
                    self.registry.register(Arc::new(GmailSearchThreadsTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GmailGetThreadTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GmailListDraftsTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GmailListLabelsTool {
                        client: client.clone(),
                    }));
                    if !readonly {
                        self.registry.register(Arc::new(GmailSendTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailCreateDraftTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailCreateLabelTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailLabelMessageTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailUnlabelMessageTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailLabelThreadTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailUnlabelThreadTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailBatchModifyTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GmailBatchDeleteTool {
                            client: client.clone(),
                        }));
                    }
                }
                Err(e) => tracing::warn!("skipping Gmail tools: {e:#}"),
            }
        }

        #[cfg(feature = "google-calendar")]
        {
            use crate::google::calendar::{self, tools::*};
            match calendar::new_client(config) {
                Ok(c) => {
                    let client = Arc::new(c);
                    self.registry.register(Arc::new(GcalListCalendarsTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GcalListEventsTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GcalGetEventTool {
                        client: client.clone(),
                    }));
                    if !readonly {
                        self.registry.register(Arc::new(GcalCreateEventTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GcalUpdateEventTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GcalDeleteEventTool {
                            client: client.clone(),
                        }));
                    }
                }
                Err(e) => tracing::warn!("skipping Google Calendar tools: {e:#}"),
            }
        }

        #[cfg(feature = "google-contacts")]
        {
            use crate::google::contacts::{self, tools::*};
            match contacts::new_client(config) {
                Ok(c) => {
                    let client = Arc::new(c);
                    self.registry.register(Arc::new(GcontactsListTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GcontactsSearchTool {
                        client: client.clone(),
                    }));
                    self.registry.register(Arc::new(GcontactsGetTool {
                        client: client.clone(),
                    }));
                    if !readonly {
                        self.registry.register(Arc::new(GcontactsCreateTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GcontactsUpdateTool {
                            client: client.clone(),
                        }));
                        self.registry.register(Arc::new(GcontactsDeleteTool {
                            client: client.clone(),
                        }));
                    }
                }
                Err(e) => tracing::warn!("skipping Google Contacts tools: {e:#}"),
            }
        }

        self
    }

    /// Microsoft Outlook tools (Mail, Calendar, Contacts) — all tools including write.
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `outlook_auth` tool for interactive OAuth.
    #[cfg(feature = "outlook")]
    pub fn outlook(self, config: OutlookConfig) -> Self {
        self.outlook_impl(config, false)
    }

    /// Microsoft Outlook tools (Mail, Calendar, Contacts) — read-only tools.
    #[cfg(feature = "outlook")]
    pub fn outlook_readonly(self, config: OutlookConfig) -> Self {
        self.outlook_impl(config, true)
    }

    #[cfg(feature = "outlook")]
    fn outlook_impl(mut self, config: OutlookConfig, readonly: bool) -> Self {
        use crate::oauth;
        use crate::outlook::auth_tool::OutlookAuthTool;

        if self.offline {
            return self;
        }

        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token {
            self = self.outlook_register_services(config, readonly);
        } else {
            self.registry.register(Arc::new(OutlookAuthTool {
                config,
                readonly,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    #[cfg(feature = "outlook")]
    fn outlook_register_services(mut self, config: OutlookConfig, readonly: bool) -> Self {
        use crate::outlook::calendar::tools::*;
        use crate::outlook::contacts::tools::*;
        use crate::outlook::mail::tools::*;

        let mut scopes = vec!["offline_access"];
        if readonly {
            scopes.extend_from_slice(&["Mail.Read", "Calendars.Read", "Contacts.Read"]);
        } else {
            scopes.extend_from_slice(&[
                "Mail.Read",
                "Mail.Send",
                "Calendars.ReadWrite",
                "Contacts.Read",
                "Contacts.ReadWrite",
            ]);
        }

        let client = match crate::outlook::OutlookClient::new(config, &scopes) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Outlook tools: {e:#}");
                return self;
            }
        };

        // Mail (read)
        self.registry.register(Arc::new(OutlookListMessagesTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(OutlookGetMessageTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(OutlookListFoldersTool {
            client: client.clone(),
        }));

        // Calendar (read)
        self.registry.register(Arc::new(OutlookListEventsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(OutlookGetEventTool {
            client: client.clone(),
        }));

        // Contacts (read)
        self.registry.register(Arc::new(OutlookListContactsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(OutlookGetContactTool {
            client: client.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(OutlookSendMailTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookCreateDraftTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookBatchUpdateTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookBatchMoveTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookBatchDeleteTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookCreateEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookUpdateEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookDeleteEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(OutlookCreateContactTool {
                client: client.clone(),
            }));
        }

        self
    }

    /// GitHub tools (repos, issues, PRs, notifications) — all tools including write.
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `github_auth` tool for interactive OAuth.
    #[cfg(feature = "github")]
    pub fn github(self, config: crate::github::GitHubConfig) -> Self {
        self.github_impl(config, false)
    }

    /// GitHub tools (repos, issues, PRs, notifications) — read-only tools.
    #[cfg(feature = "github")]
    pub fn github_readonly(self, config: crate::github::GitHubConfig) -> Self {
        self.github_impl(config, true)
    }

    #[cfg(feature = "github")]
    fn github_impl(mut self, config: crate::github::GitHubConfig, readonly: bool) -> Self {
        use crate::github::auth_tool::GitHubAuthTool;
        use crate::oauth;

        if self.offline {
            return self;
        }

        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token {
            self = self.github_register_services(config, readonly);
        } else {
            self.registry.register(Arc::new(GitHubAuthTool {
                config,
                readonly,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    #[cfg(feature = "github")]
    fn github_register_services(
        mut self,
        config: crate::github::GitHubConfig,
        readonly: bool,
    ) -> Self {
        use crate::github::GitHubClient;
        use crate::github::tools::*;

        let client = match GitHubClient::new(config) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping GitHub tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(GitHubListReposTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(GitHubSearchIssuesTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(GitHubGetIssueTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(GitHubListPrsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(GitHubGetPrTool {
            client: client.clone(),
        }));
        self.registry
            .register(Arc::new(GitHubListNotificationsTool {
                client: client.clone(),
            }));

        if !readonly {
            self.registry.register(Arc::new(GitHubCreateIssueTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(GitHubCommentOnIssueTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(GitHubCommentOnPrTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(GitHubMergePrTool {
                client: client.clone(),
            }));
        }

        self
    }

    /// Slack tools (channels, messages, search, reactions) — all tools including write.
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `slack_auth` tool for interactive OAuth.
    #[cfg(feature = "slack")]
    pub fn slack(self, config: crate::slack::SlackConfig) -> Self {
        self.slack_impl(config, false)
    }

    /// Slack tools (channels, messages, search) — read-only tools.
    #[cfg(feature = "slack")]
    pub fn slack_readonly(self, config: crate::slack::SlackConfig) -> Self {
        self.slack_impl(config, true)
    }

    #[cfg(feature = "slack")]
    fn slack_impl(mut self, config: crate::slack::SlackConfig, readonly: bool) -> Self {
        use crate::oauth;
        use crate::slack::auth_tool::SlackAuthTool;

        if self.offline {
            return self;
        }

        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token {
            self = self.slack_register_services(config, readonly);
        } else {
            self.registry.register(Arc::new(SlackAuthTool {
                config,
                readonly,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    #[cfg(feature = "slack")]
    fn slack_register_services(
        mut self,
        config: crate::slack::SlackConfig,
        readonly: bool,
    ) -> Self {
        use crate::slack::SlackClient;
        use crate::slack::tools::*;

        let client = match SlackClient::new(config) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Slack tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(SlackListChannelsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(SlackReadChannelTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(SlackReadThreadTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(SlackSearchMessagesTool {
            client: client.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(SlackSendMessageTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(SlackAddReactionTool {
                client: client.clone(),
            }));
        }

        self
    }

    /// Cloudflare tools (zones, DNS, workers, cache) — all tools including write.
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `cloudflare_auth` tool for interactive
    /// token setup.
    #[cfg(feature = "cloudflare")]
    pub fn cloudflare(self, config: crate::cloudflare::CloudflareConfig) -> Self {
        self.cloudflare_impl(config, false)
    }

    /// Cloudflare tools (zones, DNS, workers, cache) — read-only tools.
    #[cfg(feature = "cloudflare")]
    pub fn cloudflare_readonly(self, config: crate::cloudflare::CloudflareConfig) -> Self {
        self.cloudflare_impl(config, true)
    }

    #[cfg(feature = "cloudflare")]
    fn cloudflare_impl(
        mut self,
        config: crate::cloudflare::CloudflareConfig,
        readonly: bool,
    ) -> Self {
        use crate::cloudflare::auth_tool::CloudflareAuthTool;
        use crate::oauth;

        if self.offline {
            return self;
        }

        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token {
            self = self.cloudflare_register_services(config, readonly);
        } else {
            self.registry.register(Arc::new(CloudflareAuthTool {
                config,
                readonly,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    #[cfg(feature = "cloudflare")]
    fn cloudflare_register_services(
        mut self,
        config: crate::cloudflare::CloudflareConfig,
        readonly: bool,
    ) -> Self {
        use crate::cloudflare::CloudflareClient;
        use crate::cloudflare::tools::*;
        use crate::oauth;

        let token = match oauth::load_token(&config.token_path) {
            Ok(Some(t)) => t.access_token,
            _ => {
                tracing::warn!("skipping Cloudflare tools: could not load token");
                return self;
            }
        };

        let client = match CloudflareClient::new(token) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Cloudflare tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(CloudflareListZonesTool {
            client: client.clone(),
        }));
        self.registry
            .register(Arc::new(CloudflareListDnsRecordsTool {
                client: client.clone(),
            }));
        self.registry.register(Arc::new(CloudflareGetDnsRecordTool {
            client: client.clone(),
        }));
        self.registry
            .register(Arc::new(CloudflareListWorkerRoutesTool {
                client: client.clone(),
            }));

        if !readonly {
            self.registry
                .register(Arc::new(CloudflareCreateDnsRecordTool {
                    client: client.clone(),
                }));
            self.registry
                .register(Arc::new(CloudflareUpdateDnsRecordTool {
                    client: client.clone(),
                }));
            self.registry
                .register(Arc::new(CloudflareDeleteDnsRecordTool {
                    client: client.clone(),
                }));
            self.registry.register(Arc::new(CloudflarePurgeCacheTool {
                client: client.clone(),
            }));
        }

        self
    }

    /// CalDAV tools (calendars, events) for any CalDAV-compliant server — all
    /// tools including write.
    ///
    /// For Basic auth, registers service tools directly.
    /// For OAuth, checks for a cached token and registers either service tools
    /// or just the `caldav_auth` tool.
    #[cfg(feature = "caldav")]
    pub fn caldav(self, config: crate::caldav::CalDavConfig) -> Self {
        self.caldav_impl(config, false)
    }

    /// CalDAV tools (calendars, events) — read-only tools.
    #[cfg(feature = "caldav")]
    pub fn caldav_readonly(self, config: crate::caldav::CalDavConfig) -> Self {
        self.caldav_impl(config, true)
    }

    #[cfg(feature = "caldav")]
    fn caldav_impl(mut self, config: crate::caldav::CalDavConfig, readonly: bool) -> Self {
        use crate::caldav::CalDavAuth;

        if self.offline {
            return self;
        }

        match &config.auth {
            CalDavAuth::Basic { .. } => {
                self = self.caldav_register_services(config, readonly);
            }
            CalDavAuth::OAuth { .. } => {
                if let Some(token_path) = &config.token_path {
                    let has_token = crate::oauth::load_token(token_path)
                        .ok()
                        .flatten()
                        .is_some_and(|t| !t.is_expired());

                    if has_token {
                        self = self.caldav_register_services(config, readonly);
                    } else {
                        use crate::caldav::auth_tool::CalDavAuthTool;
                        self.registry.register(Arc::new(CalDavAuthTool {
                            config,
                            readonly,
                            pending_tools: self.pending_tools.clone(),
                        }));
                    }
                } else {
                    tracing::warn!("skipping CalDAV tools: OAuth mode requires token_path");
                }
            }
        }

        self
    }

    #[cfg(feature = "caldav")]
    fn caldav_register_services(
        mut self,
        config: crate::caldav::CalDavConfig,
        readonly: bool,
    ) -> Self {
        use crate::caldav::CalDavClient;
        use crate::caldav::tools::*;

        let client = match CalDavClient::new(&config) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping CalDAV tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(CalDavListCalendarsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(CalDavListEventsTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(CalDavGetEventTool {
            client: client.clone(),
        }));
        self.registry.register(Arc::new(CalDavSearchEventsTool {
            client: client.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(CalDavCreateEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(CalDavUpdateEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(CalDavDeleteEventTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(CalDavCreateCalendarTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(CalDavDeleteCalendarTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(CalDavRenameCalendarTool {
                client: client.clone(),
            }));
        }

        self
    }

    // ---------------------------------------------------------------------------
    // Redis
    // ---------------------------------------------------------------------------

    /// Redis tools — all commands including write.
    #[cfg(feature = "redis")]
    pub fn redis(self, config: crate::redis_tools::RedisConfig) -> Self {
        self.redis_impl(config, false)
    }

    /// Redis tools — read-only commands only.
    #[cfg(feature = "redis")]
    pub fn redis_readonly(self, config: crate::redis_tools::RedisConfig) -> Self {
        self.redis_impl(config, true)
    }

    #[cfg(feature = "redis")]
    fn redis_impl(mut self, config: crate::redis_tools::RedisConfig, readonly: bool) -> Self {
        use crate::redis_tools::*;

        if self.offline {
            return self;
        }

        self.registry.register(Arc::new(RedisQueryTool {
            url: config.url.clone(),
            readonly,
        }));
        self.registry
            .register(Arc::new(RedisInfoTool { url: config.url }));

        self
    }

    // ---------------------------------------------------------------------------
    // Postgres
    // ---------------------------------------------------------------------------

    /// Postgres query tool — read and write.
    #[cfg(feature = "postgres")]
    pub fn postgres(self, config: crate::postgres::PostgresConfig) -> Self {
        self.postgres_impl(config, false)
    }

    /// Postgres query tool — read-only.
    #[cfg(feature = "postgres")]
    pub fn postgres_readonly(self, config: crate::postgres::PostgresConfig) -> Self {
        self.postgres_impl(config, true)
    }

    #[cfg(feature = "postgres")]
    fn postgres_impl(mut self, config: crate::postgres::PostgresConfig, readonly: bool) -> Self {
        use crate::postgres::PostgresQueryTool;

        if self.offline {
            return self;
        }

        self.registry.register(Arc::new(PostgresQueryTool {
            connection_string: config.connection_string,
            readonly,
        }));

        self
    }

    // ---------------------------------------------------------------------------
    // MySQL
    // ---------------------------------------------------------------------------

    /// MySQL query tool — read and write.
    #[cfg(feature = "mysql")]
    pub fn mysql(self, config: crate::mysql::MysqlConfig) -> Self {
        self.mysql_impl(config, false)
    }

    /// MySQL query tool — read-only.
    #[cfg(feature = "mysql")]
    pub fn mysql_readonly(self, config: crate::mysql::MysqlConfig) -> Self {
        self.mysql_impl(config, true)
    }

    #[cfg(feature = "mysql")]
    fn mysql_impl(mut self, config: crate::mysql::MysqlConfig, readonly: bool) -> Self {
        use crate::mysql::MysqlQueryTool;

        if self.offline {
            return self;
        }

        self.registry.register(Arc::new(MysqlQueryTool {
            connection_string: config.connection_string,
            readonly,
        }));

        self
    }

    // ---------------------------------------------------------------------------
    // ClickHouse
    // ---------------------------------------------------------------------------

    /// ClickHouse query tool — read and write.
    #[cfg(feature = "clickhouse")]
    pub fn clickhouse(self, config: crate::clickhouse::ClickHouseConfig) -> Self {
        self.clickhouse_impl(config, false)
    }

    /// ClickHouse query tool — read-only.
    #[cfg(feature = "clickhouse")]
    pub fn clickhouse_readonly(self, config: crate::clickhouse::ClickHouseConfig) -> Self {
        self.clickhouse_impl(config, true)
    }

    #[cfg(feature = "clickhouse")]
    fn clickhouse_impl(
        mut self,
        config: crate::clickhouse::ClickHouseConfig,
        readonly: bool,
    ) -> Self {
        use crate::clickhouse::ClickHouseQueryTool;

        if self.offline {
            return self;
        }

        self.registry
            .register(Arc::new(ClickHouseQueryTool { config, readonly }));

        self
    }

    // ---------------------------------------------------------------------------
    // Docker
    // ---------------------------------------------------------------------------

    /// Docker tools — all operations including write.
    #[cfg(feature = "docker")]
    pub fn docker(self, config: crate::docker::DockerConfig) -> Self {
        self.docker_impl(config, false)
    }

    /// Docker tools — read-only (list, inspect, logs).
    #[cfg(feature = "docker")]
    pub fn docker_readonly(self, config: crate::docker::DockerConfig) -> Self {
        self.docker_impl(config, true)
    }

    #[cfg(feature = "docker")]
    fn docker_impl(mut self, config: crate::docker::DockerConfig, readonly: bool) -> Self {
        use crate::docker::tools::*;

        if self.offline {
            return self;
        }

        let docker = match if let Some(ref endpoint) = config.endpoint {
            bollard::Docker::connect_with_http(endpoint, 120, bollard::API_DEFAULT_VERSION)
        } else {
            bollard::Docker::connect_with_local_defaults()
        } {
            Ok(d) => Arc::new(d),
            Err(e) => {
                tracing::warn!("skipping Docker tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(DockerListContainersTool {
            client: docker.clone(),
        }));
        self.registry.register(Arc::new(DockerInspectContainerTool {
            client: docker.clone(),
        }));
        self.registry.register(Arc::new(DockerContainerLogsTool {
            client: docker.clone(),
        }));
        self.registry.register(Arc::new(DockerListImagesTool {
            client: docker.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(DockerContainerExecTool {
                client: docker.clone(),
            }));
            self.registry.register(Arc::new(DockerCreateContainerTool {
                client: docker.clone(),
            }));
            self.registry.register(Arc::new(DockerStopContainerTool {
                client: docker.clone(),
            }));
            self.registry.register(Arc::new(DockerRemoveContainerTool {
                client: docker.clone(),
            }));
            self.registry.register(Arc::new(DockerPullImageTool {
                client: docker.clone(),
            }));
        }

        self
    }

    // ---------------------------------------------------------------------------
    // Messaging (Twilio / SMTP)
    // ---------------------------------------------------------------------------

    /// Messaging tools (Twilio SMS/WhatsApp, SMTP email) — all operations.
    #[cfg(feature = "messaging")]
    pub fn messaging(self, config: crate::messaging::MessagingConfig) -> Self {
        self.messaging_impl(config, false)
    }

    /// Messaging tools — read-only (Twilio list/get only, no SMTP).
    #[cfg(feature = "messaging")]
    pub fn messaging_readonly(self, config: crate::messaging::MessagingConfig) -> Self {
        self.messaging_impl(config, true)
    }

    #[cfg(feature = "messaging")]
    fn messaging_impl(mut self, config: crate::messaging::MessagingConfig, readonly: bool) -> Self {
        if self.offline {
            return self;
        }

        if let Some(twilio) = config.twilio {
            // Read tools (always registered)
            self.registry
                .register(Arc::new(crate::messaging::twilio::TwilioListMessagesTool {
                    account_sid: twilio.account_sid.clone(),
                    auth_token: twilio.auth_token.clone(),
                }));
            self.registry
                .register(Arc::new(crate::messaging::twilio::TwilioGetMessageTool {
                    account_sid: twilio.account_sid.clone(),
                    auth_token: twilio.auth_token.clone(),
                }));

            if !readonly {
                self.registry
                    .register(Arc::new(crate::messaging::twilio::TwilioSendSmsTool {
                        account_sid: twilio.account_sid.clone(),
                        auth_token: twilio.auth_token.clone(),
                        from_number: twilio.from_number.clone(),
                    }));
                self.registry.register(Arc::new(
                    crate::messaging::twilio::TwilioSendWhatsappTool {
                        account_sid: twilio.account_sid.clone(),
                        auth_token: twilio.auth_token.clone(),
                        from_number: twilio.from_number,
                    },
                ));
            }
        }

        if !readonly && let Some(smtp) = config.smtp {
            self.registry
                .register(Arc::new(crate::messaging::smtp::SmtpSendEmailTool {
                    config: smtp,
                }));
        }

        self
    }

    // ---------------------------------------------------------------------------
    // SSH
    // ---------------------------------------------------------------------------

    /// SSH tools — all operations including upload.
    #[cfg(feature = "ssh")]
    pub fn ssh(self, config: crate::ssh::SshConfig) -> Self {
        self.ssh_impl(config, false)
    }

    /// SSH tools — read-only (exec with read-only description, download).
    #[cfg(feature = "ssh")]
    pub fn ssh_readonly(self, config: crate::ssh::SshConfig) -> Self {
        self.ssh_impl(config, true)
    }

    #[cfg(feature = "ssh")]
    fn ssh_impl(mut self, config: crate::ssh::SshConfig, readonly: bool) -> Self {
        use crate::ssh::tools::*;

        if self.offline {
            return self;
        }

        let profiles = Arc::new(config.profiles);

        self.registry.register(Arc::new(SshExecTool {
            profiles: profiles.clone(),
            readonly,
        }));
        self.registry.register(Arc::new(SshDownloadTool {
            profiles: profiles.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(SshUploadTool {
                profiles: profiles.clone(),
            }));
        }

        self
    }

    // ---------------------------------------------------------------------------
    // Kubernetes
    // ---------------------------------------------------------------------------

    /// Kubernetes tools — all operations including apply/delete/scale.
    #[cfg(feature = "kubernetes")]
    pub fn kubernetes(self, config: crate::kubernetes::KubernetesConfig) -> Self {
        self.kubernetes_impl(config, false)
    }

    /// Kubernetes tools — read-only (list, get, logs, events).
    #[cfg(feature = "kubernetes")]
    pub fn kubernetes_readonly(self, config: crate::kubernetes::KubernetesConfig) -> Self {
        self.kubernetes_impl(config, true)
    }

    #[cfg(feature = "kubernetes")]
    fn kubernetes_impl(
        mut self,
        config: crate::kubernetes::KubernetesConfig,
        readonly: bool,
    ) -> Self {
        use crate::kubernetes::tools::*;

        let default_ns = config
            .namespace
            .clone()
            .unwrap_or_else(|| "default".to_string());

        let client = match futures::executor::block_on(crate::kubernetes::make_client(&config)) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Kubernetes tools: {e:#}");
                return self;
            }
        };

        // Read tools
        self.registry.register(Arc::new(K8sListPodsTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sGetPodTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sPodLogsTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sListDeploymentsTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sGetDeploymentTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sListServicesTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));
        self.registry.register(Arc::new(K8sListEventsTool {
            client: client.clone(),
            default_namespace: default_ns.clone(),
        }));

        if !readonly {
            self.registry.register(Arc::new(K8sApplyManifestTool {
                client: client.clone(),
                default_namespace: default_ns.clone(),
            }));
            self.registry.register(Arc::new(K8sDeleteResourceTool {
                client: client.clone(),
                default_namespace: default_ns.clone(),
            }));
            self.registry.register(Arc::new(K8sScaleDeploymentTool {
                client: client.clone(),
                default_namespace: default_ns.clone(),
            }));
            self.registry.register(Arc::new(K8sExecInPodTool {
                client: client.clone(),
                default_namespace: default_ns.clone(),
            }));
        }

        self
    }

    /// mcp_add, mcp_remove, mcp_list, mcp_auth.
    ///
    /// Registers MCP management tools. After calling `.build()`, the caller
    /// should run `mcp_registry().load_saved().await` and drain pending ops
    /// to register wrapper tools for cached MCP server tools.
    #[cfg(feature = "mcp")]
    pub fn mcp(
        mut self,
        provider: Arc<dyn McpConfigProvider>,
        auth_handler: Option<Arc<dyn McpAuthHandler>>,
    ) -> Self {
        use crate::mcp::McpRegistry;

        let registry = McpRegistry::new(provider, auth_handler);
        self.mcp_registry = Some(registry.clone());

        self.registry
            .register(Arc::new(crate::mcp::tools::McpAddTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpRemoveTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpListTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpAuthTool { mcp: registry }));
        self
    }

    /// Like [`mcp`](Self::mcp), but accepts a pre-constructed [`McpRegistry`]
    /// instead of creating one. Use this when the registry is shared across
    /// multiple `build_tools` calls (e.g. stored in Tauri state).
    #[cfg(feature = "mcp")]
    pub fn mcp_with_registry(mut self, registry: crate::mcp::McpRegistry) -> Self {
        self.mcp_registry = Some(registry.clone());

        self.registry
            .register(Arc::new(crate::mcp::tools::McpAddTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpRemoveTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpListTool {
                mcp: registry.clone(),
            }));
        self.registry
            .register(Arc::new(crate::mcp::tools::McpAuthTool { mcp: registry }));
        self
    }

    /// Access the MCP registry (if `.mcp()` was called).
    #[cfg(feature = "mcp")]
    pub fn mcp_registry(&self) -> Option<&McpRegistry> {
        self.mcp_registry.as_ref()
    }

    /// Consume the builder and return both the tool registry and a
    /// [`ToolSync`] handle for ongoing dynamic tool sync.
    ///
    /// If MCP was configured via [`.mcp()`](Self::mcp), loads saved MCP
    /// configs and registers their wrapper tools. Also drains any pending
    /// tools from auth flows (Google, Outlook).
    pub async fn build_with_sync(self) -> (ToolRegistry, ToolSync) {
        #[cfg(feature = "mcp")]
        let mcp_registry = {
            if let Some(reg) = self.mcp_registry {
                reg.load_saved().await;
                Some(reg)
            } else {
                None
            }
        };

        let mut tools = self.registry;
        let tool_sync = ToolSync::new(
            #[cfg(feature = "mcp")]
            mcp_registry,
            self.pending_tools,
        );
        tool_sync.sync(&mut tools);

        (tools, tool_sync)
    }

    /// Access the pending tools queue (for auth tools to push into).
    pub fn pending_tools(&self) -> &PendingTools {
        &self.pending_tools
    }

    /// Build the final tool registry from all registered tools.
    pub fn build(self) -> ToolRegistry {
        self.registry
    }

    /// Return the names of all currently registered tools.
    pub fn tool_names(&self) -> Vec<String> {
        self.registry
            .list()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }
}
