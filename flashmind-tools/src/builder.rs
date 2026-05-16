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
///     .bash(secrets, &protected, forbidden_cmds)
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

    /// file_read, file_write, file_delete, file_list, read_lines, glob, grep,
    /// str_replace, str_replace_regex, image_read, str_diff.
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

    /// bash, process management.
    pub fn bash(
        mut self,
        secrets: Vec<String>,
        protected: &Arc<ProtectedPaths>,
        forbidden_cmds: Vec<flashmind_types::tool::ForbiddenCmd>,
    ) -> Self {
        let process_registry = ProcessRegistry::new();

        self.registry.register(Arc::new(BashTool {
            protected: protected.clone(),
            secrets,
            process_registry: process_registry.clone(),
            forbidden_cmds,
        }));
        self.registry.alias("bash_exec", "exec");
        self.registry.register(Arc::new(ProcessTool {
            registry: process_registry,
        }));
        self
    }

    /// brave_search, firecrawl tools (skipped in offline mode).
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

    /// time.
    pub fn time(mut self) -> Self {
        self.registry.register(Arc::new(TimeTool));
        self
    }

    /// sqlite_query.
    pub fn sqlite(mut self) -> Self {
        self.registry.register(Arc::new(SqliteQueryTool));
        self
    }

    /// http_request.
    pub fn http(mut self) -> Self {
        self.registry.register(Arc::new(HttpRequestTool::new()));
        self
    }

    /// json_query.
    pub fn json(mut self) -> Self {
        self.registry.register(Arc::new(JsonQueryTool));
        self
    }

    /// tts, transcribe, list_voices.
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

    /// list_models.
    pub fn models(mut self) -> Self {
        self.registry.register(Arc::new(ListModelsTool {
            providers: Arc::clone(&self.providers),
        }));
        self
    }

    /// image_edit, generate_image, generate_video.
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

    /// delegate, agent_status, agent_wait, agent_terminate.
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

    /// Google API tools (Gmail, Calendar, Contacts).
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `google_auth` tool which handles the
    /// OAuth flow and dynamically adds service tools on successful auth.
    #[cfg(any(
        feature = "gmail",
        feature = "google-calendar",
        feature = "google-contacts"
    ))]
    pub fn google(mut self, config: &GoogleConfig, readonly: bool) -> Self {
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

    /// Microsoft Outlook tools (Mail, Calendar, Contacts).
    ///
    /// If a valid cached token exists, registers service tools immediately.
    /// Otherwise, registers only the `outlook_auth` tool for interactive OAuth.
    #[cfg(feature = "outlook")]
    pub fn outlook(mut self, config: OutlookConfig) -> Self {
        use crate::oauth;
        use crate::outlook::auth_tool::OutlookAuthTool;

        if self.offline {
            return self;
        }

        // Check if we have a valid cached token
        let has_token = oauth::load_token(&config.token_path)
            .ok()
            .flatten()
            .is_some_and(|t| !t.is_expired());

        if has_token {
            self = self.outlook_register_services(config);
        } else {
            self.registry.register(Arc::new(OutlookAuthTool {
                config,
                pending_tools: self.pending_tools.clone(),
            }));
        }

        self
    }

    /// Register Outlook service tools directly (when token is available).
    #[cfg(feature = "outlook")]
    fn outlook_register_services(mut self, config: OutlookConfig) -> Self {
        use crate::outlook::calendar::tools::*;
        use crate::outlook::contacts::tools::*;
        use crate::outlook::mail::tools::*;

        let readonly = config.readonly;
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

    /// Consume the builder and return the registry.
    pub fn build(self) -> ToolRegistry {
        self.registry
    }

    /// Return the names of all registered tools without consuming the builder.
    pub fn tool_names(&self) -> Vec<String> {
        self.registry
            .list()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }
}
