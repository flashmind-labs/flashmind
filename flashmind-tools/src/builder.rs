//! Composable tool registry builder for shared tools.
//!
//! Registers tools from the `flashmind-tools` crate. Binary-specific tools
//! (canvas, cron, slack, telegram, webhooks, memory, subagents) are
//! added by the agent binary after calling [`ToolBuilder::build`].

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::info;

use flashmind_types::llm::ProviderRegistry;
use flashmind_types::model::Model;
use flashmind_types::tool::{ForbiddenCmd, ToolRegistry};

use crate::audio::{AudioConfig, ListVoicesTool, TranscribeTool, TtsTool};
use crate::bash::BashTool;
use crate::brave::BraveSearchTool;
use crate::file_cache::FileCache;
use crate::file_ops::{FileDeleteTool, FileListTool, FileReadTool, FileWriteTool, ReadLinesTool};
use crate::firecrawl::{WebCrawlTool, WebMapTool, WebScrapeTool, WebSearchTool};
use crate::glob::GlobTool;
use crate::grep::GrepTool;
use crate::http::HttpRequestTool;
use crate::image_edit::ImageEditTool;
use crate::image_gen::GenerateImageTool;
use crate::image_read::ImageReadTool;
use crate::json_query::JsonQueryTool;
use crate::list_models::ListModelsTool;
#[cfg(feature = "mcp")]
use crate::mcp::{McpAuthHandler, McpConfigProvider};
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
use crate::web_fetch::WebFetchTool;

/// Composable builder for [`ToolRegistry`].
///
/// ```rust,ignore
/// let registry = ToolBuilder::new(protected)
///     .with_providers(providers)
///     .file_ops(ocr_model)
///     .bash(secrets, forbidden)
///     .web(browser_engine)
///     .search(brave_key, firecrawl_key)
///     .time()
///     .sqlite()
///     .http()
///     .json()
///     .audio(model, voice, audio_dir)
///     .models()
///     .generate(image_model, video_model, output_dir)
///     .mcp(provider)
///     .build();
/// ```
pub struct ToolBuilder {
    registry: ToolRegistry,
    protected: Arc<ProtectedPaths>,
    file_cache: FileCache,
    providers: ProviderRegistry,
    offline: bool,
    #[cfg(feature = "mcp")]
    mcp_registry: Option<crate::mcp::McpRegistry>,
}

impl ToolBuilder {
    pub fn new(protected: &Arc<ProtectedPaths>) -> Self {
        Self {
            registry: ToolRegistry::new(),
            protected: protected.clone(),
            file_cache: FileCache::new(),
            providers: Arc::new(std::collections::HashMap::new()),
            offline: false,
            #[cfg(feature = "mcp")]
            mcp_registry: None,
        }
    }

    pub fn with_providers(mut self, providers: ProviderRegistry) -> Self {
        self.providers = providers;
        self
    }

    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// file_read, file_write, file_delete, file_list, read_lines, glob, grep,
    /// str_replace, str_replace_regex, image_read, str_diff.
    pub fn file_ops(mut self, ocr_model: Option<Model>) -> Self {
        self.registry.register(Arc::new(FileReadTool {
            protected: self.protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(FileWriteTool {
            protected: self.protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(FileDeleteTool {
            protected: self.protected.clone(),
        }));
        self.registry.register(Arc::new(FileListTool));
        self.registry.register(Arc::new(ReadLinesTool {
            protected: self.protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(GlobTool {
            protected: self.protected.clone(),
        }));
        self.registry.register(Arc::new(GrepTool));
        self.registry.register(Arc::new(StrReplaceTool {
            protected: self.protected.clone(),
            file_cache: self.file_cache.clone(),
        }));
        self.registry.register(Arc::new(StrReplaceRegexTool {
            protected: self.protected.clone(),
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
    pub fn bash(mut self, secrets: Vec<String>, forbidden: Vec<ForbiddenCmd>) -> Self {
        let process_registry = ProcessRegistry::new();

        if !forbidden.is_empty() {
            info!("Forbidden commands configured:");
            for fc in &forbidden {
                info!("  - {}: {}", fc.command, fc.reason);
            }
        }

        self.registry.set_forbidden(forbidden);
        self.registry.register(Arc::new(BashTool {
            protected: self.protected.clone(),
            secrets,
            process_registry: process_registry.clone(),
        }));
        self.registry.alias("bash_exec", "exec");
        self.registry.register(Arc::new(ProcessTool {
            registry: process_registry,
        }));
        self
    }

    /// web_fetch (skipped in offline mode).
    pub fn web(mut self, browser_engine: Option<String>) -> Self {
        if !self.offline {
            self.registry.register(Arc::new(WebFetchTool::new(
                browser_engine.unwrap_or_default(),
            )));
        }
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
    pub fn mcp_registry(&self) -> Option<&crate::mcp::McpRegistry> {
        self.mcp_registry.as_ref()
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
