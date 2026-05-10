//! Composable tool registry builder for shared tools.
//!
//! Registers tools from the `flashmind-tools` crate. Binary-specific tools
//! (canvas, cron, slack, telegram, webhooks, memory) are added by the agent
//! binary after calling [`ToolBuilder::build`].

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "subagent")]
use flashmind_core::AgentManager;
#[cfg(feature = "subagent")]
use flashmind_types::LlmProvider;
use flashmind_types::llm::ProviderRegistry;
use flashmind_types::model::Model;
use flashmind_types::tool::ToolRegistry;
use tokio::sync::RwLock;

#[cfg(feature = "subagent")]
use crate::subagent::{
    AgentStatusTool, AgentTerminateTool, AgentWaitTool, CommunicateTool, DelegateTool,
};

use crate::audio::{AudioConfig, ListVoicesTool, TranscribeTool, TtsTool};
use crate::bash::BashTool;
use crate::brave::BraveSearchTool;
use crate::file_cache::FileCache;
use crate::file_ops::{FileDeleteTool, FileListTool, FileReadTool, FileWriteTool, ReadLinesTool};
use crate::firecrawl::{WebCrawlTool, WebMapTool, WebScrapeTool, WebSearchTool};
use crate::glob::GlobTool;
#[cfg(feature = "gmail")]
use crate::google::gmail::GmailConfig;
#[cfg(any(feature = "google-calendar", feature = "google-contacts"))]
use crate::google::client::GoogleConfig;
#[cfg(feature = "outlook")]
use crate::outlook::OutlookConfig;
use crate::grep::GrepTool;
use crate::http::HttpRequestTool;
use crate::image_edit::ImageEditTool;
use crate::image_gen::GenerateImageTool;
use crate::image_read::ImageReadTool;
use crate::json_query::JsonQueryTool;
use crate::list_models::ListModelsTool;
#[cfg(feature = "mcp")]
use crate::mcp::{McpAuthHandler, McpConfigProvider, McpRegistry, McpToolSet};
use crate::process::{ProcessRegistry, ProcessTool};
use crate::protected::ProtectedPaths;
use crate::search_cache::{SearchCacheRef, SearchResultCache};
use crate::search_read::WebSearchReadTool;
use crate::sqlite::SqliteQueryTool;
use crate::str_diff::StrDiffTool;
use crate::text_replace::StrReplaceTool;
use crate::text_replace_regex::StrReplaceRegexTool;
use crate::time::TimeTool;
use crate::video_gen::GenerateVideoTool;
/// Composable builder for [`ToolRegistry`].
///
/// ```rust,ignore
/// let registry = ToolBuilder::new()
///     .with_providers(providers)
///     .file_ops(ocr_model, &protected)
///     .bash(secrets, &protected)
///     .search(brave_key, firecrawl_key)
///     .time()
///     .sqlite()
///     .http()
///     .json()
///     .audio(model, voice, audio_dir)
///     .models()
///     .generate(image_model, video_model, output_dir)
///     .subagents(manager, provider)
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
    pub fn bash(mut self, secrets: Vec<String>, protected: &Arc<ProtectedPaths>) -> Self {
        let process_registry = ProcessRegistry::new();

        self.registry.register(Arc::new(BashTool {
            protected: protected.clone(),
            secrets,
            process_registry: process_registry.clone(),
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

    /// delegate, communicate, agent_status, agent_wait, agent_terminate.
    #[cfg(feature = "subagent")]
    pub fn subagents(mut self, manager: Arc<AgentManager>, provider: Arc<dyn LlmProvider>) -> Self {
        self.registry
            .register(Arc::new(DelegateTool::new(manager.clone(), provider)));
        self.registry
            .register(Arc::new(CommunicateTool::new(manager.clone())));
        self.registry
            .register(Arc::new(AgentStatusTool::new(manager.clone())));
        self.registry
            .register(Arc::new(AgentWaitTool::new(manager.clone())));
        self.registry
            .register(Arc::new(AgentTerminateTool::new(manager)));
        self
    }

    /// Gmail tools (skipped in offline mode).
    ///
    /// Registers all Gmail API tools behind the `gmail` feature flag.
    /// The `GmailClient` is constructed eagerly — credential file errors
    /// surface at builder time rather than at first tool call.
    #[cfg(feature = "gmail")]
    pub fn gmail(mut self, config: GmailConfig) -> Self {
        use crate::google::gmail::{self, tools::*};

        if self.offline {
            return self;
        }

        let readonly = config.readonly;
        let client = match gmail::new_client(&config.google) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Gmail tools: {e:#}");
                return self;
            }
        };

        // Read-only tools (always registered).
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
            self.registry.register(Arc::new(GmailLabelThreadTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(GmailUnlabelMessageTool {
                client: client.clone(),
            }));
            self.registry.register(Arc::new(GmailUnlabelThreadTool {
                client: client.clone(),
            }));
        }

        self
    }

    /// Google Calendar tools (skipped in offline mode).
    #[cfg(feature = "google-calendar")]
    pub fn google_calendar(mut self, config: &GoogleConfig, readonly: bool) -> Self {
        use crate::google::calendar::{self, tools::*};

        if self.offline {
            return self;
        }

        let client = match crate::google::client::GoogleClient::new(
            config.clone(),
            calendar::BASE_URL,
            calendar::SCOPE,
        ) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Google Calendar tools: {e:#}");
                return self;
            }
        };

        self.registry.register(Arc::new(GcalListCalendarsTool { client: client.clone() }));
        self.registry.register(Arc::new(GcalListEventsTool { client: client.clone() }));
        self.registry.register(Arc::new(GcalGetEventTool { client: client.clone() }));

        if !readonly {
            self.registry.register(Arc::new(GcalCreateEventTool { client: client.clone() }));
            self.registry.register(Arc::new(GcalUpdateEventTool { client: client.clone() }));
            self.registry.register(Arc::new(GcalDeleteEventTool { client: client.clone() }));
        }

        self
    }

    /// Google Contacts tools (skipped in offline mode).
    #[cfg(feature = "google-contacts")]
    pub fn google_contacts(mut self, config: &GoogleConfig, readonly: bool) -> Self {
        use crate::google::contacts::{self, tools::*};

        if self.offline {
            return self;
        }

        let client = match crate::google::client::GoogleClient::new(
            config.clone(),
            contacts::BASE_URL,
            contacts::SCOPE,
        ) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Google Contacts tools: {e:#}");
                return self;
            }
        };

        self.registry.register(Arc::new(GcontactsListTool { client: client.clone() }));
        self.registry.register(Arc::new(GcontactsSearchTool { client: client.clone() }));
        self.registry.register(Arc::new(GcontactsGetTool { client: client.clone() }));

        if !readonly {
            self.registry.register(Arc::new(GcontactsCreateTool { client: client.clone() }));
            self.registry.register(Arc::new(GcontactsUpdateTool { client: client.clone() }));
            self.registry.register(Arc::new(GcontactsDeleteTool { client: client.clone() }));
        }

        self
    }

    /// Microsoft Outlook tools: mail, calendar, contacts (skipped in offline mode).
    #[cfg(feature = "outlook")]
    pub fn outlook(mut self, config: OutlookConfig) -> Self {
        use crate::outlook::mail::tools::*;
        use crate::outlook::calendar::tools::*;
        use crate::outlook::contacts::tools::*;

        if self.offline {
            return self;
        }

        let mut scopes = vec!["offline_access"];
        if config.readonly {
            scopes.extend_from_slice(&["Mail.Read", "Calendars.Read", "Contacts.Read"]);
        } else {
            scopes.extend_from_slice(&[
                "Mail.Read", "Mail.Send",
                "Calendars.ReadWrite",
                "Contacts.Read", "Contacts.ReadWrite",
            ]);
        }

        let readonly = config.readonly;
        let client = match crate::outlook::OutlookClient::new(config, &scopes) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!("skipping Outlook tools: {e:#}");
                return self;
            }
        };

        // Mail (read-only)
        self.registry.register(Arc::new(OutlookListMessagesTool { client: client.clone() }));
        self.registry.register(Arc::new(OutlookGetMessageTool { client: client.clone() }));
        self.registry.register(Arc::new(OutlookListFoldersTool { client: client.clone() }));

        // Calendar (read-only)
        self.registry.register(Arc::new(OutlookListEventsTool { client: client.clone() }));
        self.registry.register(Arc::new(OutlookGetEventTool { client: client.clone() }));

        // Contacts (read-only)
        self.registry.register(Arc::new(OutlookListContactsTool { client: client.clone() }));
        self.registry.register(Arc::new(OutlookGetContactTool { client: client.clone() }));

        if !readonly {
            // Mail (write)
            self.registry.register(Arc::new(OutlookSendMailTool { client: client.clone() }));
            self.registry.register(Arc::new(OutlookCreateDraftTool { client: client.clone() }));

            // Calendar (write)
            self.registry.register(Arc::new(OutlookCreateEventTool { client: client.clone() }));
            self.registry.register(Arc::new(OutlookUpdateEventTool { client: client.clone() }));
            self.registry.register(Arc::new(OutlookDeleteEventTool { client: client.clone() }));

            // Contacts (write)
            self.registry.register(Arc::new(OutlookCreateContactTool { client: client.clone() }));
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

    /// Access the MCP registry (if `.mcp()` was called).
    #[cfg(feature = "mcp")]
    pub fn mcp_registry(&self) -> Option<&McpRegistry> {
        self.mcp_registry.as_ref()
    }

    /// Consume the builder, load saved MCP configs, and return both the
    /// tool registry (pre-populated with MCP wrapper tools) and an
    /// [`McpToolSet`](crate::mcp::McpToolSet) handle for ongoing sync.
    ///
    /// # Panics
    ///
    /// Panics if [`.mcp()`](Self::mcp) was not called on this builder.
    #[cfg(feature = "mcp")]
    pub async fn build_with_mcp(self) -> (ToolRegistry, McpToolSet) {
        let mcp_registry = self
            .mcp_registry
            .expect("build_with_mcp() requires .mcp() to have been called");

        mcp_registry.load_saved().await;

        let mut tools = self.registry;
        let tool_set = McpToolSet::new(mcp_registry);
        tool_set.sync(&mut tools);

        (tools, tool_set)
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
