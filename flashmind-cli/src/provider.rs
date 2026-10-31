use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use flashmind_llm::{
    AnthropicProvider, MistralProvider, OllamaProvider, OpenAiCompatProfile, OpenAiProvider,
    OpenRouterProvider,
};
use flashmind_types::{LlmProvider, Model, ModelPricing, Provider};

use crate::Cli;
use crate::config::Config;

pub fn resolve_model(cli: &Cli, config: &Config) -> Result<Model> {
    let raw = cli
        .model
        .as_deref()
        .or(config.model.as_deref())
        .unwrap_or("ollama:llama3.2");
    raw.parse()
        .with_context(|| format!("invalid model string: {raw}"))
}

pub fn build_provider(provider: &Provider, config: &Config) -> Result<Arc<dyn LlmProvider>> {
    match provider {
        Provider::Ollama => {
            let p = OllamaProvider::new(config.ollama_url.clone(), None)?;
            Ok(Arc::new(p))
        }
        Provider::OpenRouter => {
            let key = config
                .api_key_for(provider)
                .context("providers.openrouter.api_key required for OpenRouter models")?;
            let mut provider = OpenRouterProvider::new(key.to_owned());
            if let Some(routing) = &config.openrouter.routing {
                provider = provider.with_provider_preferences(routing.clone());
            }
            if let Some(seconds) = config.openrouter.stream_start_timeout_secs {
                provider = provider.with_stream_start_timeout(Duration::from_secs(seconds.get()));
            }
            if let Some(seconds) = config.openrouter.stream_idle_timeout_secs {
                provider = provider.with_stream_idle_timeout(Duration::from_secs(seconds.get()));
            }
            Ok(Arc::new(provider))
        }
        Provider::Anthropic => {
            let key = config
                .api_key_for(provider)
                .context("providers.anthropic.api_key required for Anthropic models")?;
            Ok(Arc::new(AnthropicProvider::new(key.to_owned())))
        }
        Provider::OpenAi => {
            let key = config
                .api_key_for(provider)
                .context("providers.openai.api_key required for OpenAI models")?;
            let base = config
                .endpoint_override
                .url
                .as_ref()
                .or_else(|| {
                    config
                        .providers
                        .get("openai")
                        .and_then(|entry| entry.url.as_ref())
                })
                .map(|url| url.as_str())
                .or(config.openai_base_url.as_deref())
                .unwrap_or("https://api.openai.com/v1");
            let p = OpenAiProvider::builder(base).api_key(key).build()?;
            Ok(Arc::new(p))
        }
        Provider::Mistral => {
            let key = config
                .api_key_for(provider)
                .context("providers.mistral.api_key required for Mistral models")?;
            Ok(Arc::new(MistralProvider::new(key.to_owned())))
        }
        Provider::Custom(name) => {
            let endpoint = config.compatible_provider(name).with_context(|| {
                format!("provider '{name}' is not configured; add [providers.{name}]")
            })?;
            let default_profile = endpoint
                .profile
                .or_else(|| name.parse::<OpenAiCompatProfile>().ok());
            let url = config
                .endpoint_override
                .url
                .as_ref()
                .or(endpoint.url.as_ref())
                .map(|url| url.as_str())
                .or_else(|| default_profile.and_then(OpenAiCompatProfile::default_url))
                .with_context(|| {
                    format!("URL required for custom provider '{name}'; set providers.{name}.url")
                })?;
            let key = config
                .api_key_for(provider)
                .with_context(|| format!("API key required; set providers.{name}.api_key"))?;
            let mut builder = OpenAiProvider::builder(url)
                .name(name)
                .provider(provider.clone())
                .api_key(key);
            if let Some(profile) = endpoint.profile {
                builder = builder.profile(profile);
            }
            if let Some(enabled) = endpoint.cache_routing {
                builder = builder.cache_routing(enabled);
            }
            let provider = builder.build()?;
            Ok(Arc::new(provider))
        }
        other => bail!("unsupported provider: {other}"),
    }
}

pub async fn fetch_pricing(provider: &Arc<dyn LlmProvider>, model: &Model) -> ModelPricing {
    let name = model.capability_name();
    let models = provider.list_models().await.unwrap_or_default();
    models
        .iter()
        .find(|m| m.id == name)
        .map(|m| m.pricing.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const CONFIG: &str = r#"
        model = "first:example-model"
        [openrouter.openai_compat.first]
        url = "https://first.example/v1"
        api_key = "first-key"
        [openrouter.openai_compat.second]
        url = "https://second.example/v1"
        api_key = "second-key"
    "#;

    #[test]
    fn known_providers_need_only_an_api_key() {
        for name in [
            "venice",
            "groq",
            "together",
            "fireworks",
            "deepseek",
            "openai",
            "openrouter",
            "anthropic",
            "mistral",
        ] {
            let config: Config =
                toml::from_str(&format!("[providers.{name}]\napi_key = \"example-key\"\n"))
                    .unwrap();
            let provider: Provider = name.parse().unwrap();
            assert!(build_provider(&provider, &config).is_ok(), "{name}");
        }
        let config: Config =
            toml::from_str("[providers.custom]\napi_key = \"example-key\"\n").unwrap();
        assert!(build_provider(&Provider::Custom("custom".into()), &config).is_err());
        assert!(toml::from_str::<Config>("[providers.custom]\nprofile = \"unknown\"\n").is_err());
    }

    #[test]
    fn named_providers_round_trip_and_keep_their_identity() {
        let config: Config = toml::from_str(CONFIG).unwrap();
        let config: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        for name in ["first", "second"] {
            let raw = format!("{name}:example-model");
            let cli = Cli::try_parse_from(["flsh", "--model", &raw]).unwrap();
            let model = resolve_model(&cli, &config).unwrap();
            let provider = build_provider(&model.provider, &config).unwrap();
            assert_eq!(provider.name(), name);
            assert_eq!(provider.provider(), model.provider);
            assert_eq!(model.name(), "example-model");
        }
        let cli = Cli::try_parse_from(["flsh"]).unwrap();
        assert_eq!(
            resolve_model(&cli, &config).unwrap().to_string(),
            "first:example-model"
        );
        let error = build_provider(&Provider::Custom("missing".into()), &config)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("[providers.missing]"));
    }

    #[tokio::test]
    async fn named_providers_use_their_own_endpoint_and_key_with_cli_overrides() {
        for name in ["first", "second"] {
            for override_flags in [false, true] {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let url = format!("http://{}/v1", listener.local_addr().unwrap());
                let mut config: Config = toml::from_str(CONFIG).unwrap();
                let model = format!("{name}:example-model");
                let mut args = vec!["flsh", "--model", &model];
                if override_flags {
                    args.extend(["--base-url", &url, "--key", "override-key"]);
                } else {
                    config.openrouter.openai_compat.get_mut(name).unwrap().url =
                        Some(url.parse().unwrap());
                }
                let cli = Cli::try_parse_from(args).unwrap();
                let config = cli.apply_provider_overrides(config);
                let provider =
                    build_provider(&resolve_model(&cli, &config).unwrap().provider, &config)
                        .unwrap();
                let serve = async {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    while !request.ends_with(b"\r\n\r\n") {
                        request.push(stream.read_u8().await.unwrap());
                        assert!(request.len() < 8192);
                    }
                    let request = String::from_utf8(request).unwrap();
                    assert!(request.starts_with("GET /v1/models HTTP/1.1\r\n"));
                    let key = if override_flags {
                        "override-key".into()
                    } else {
                        format!("{name}-key")
                    };
                    assert!(
                        request
                            .to_ascii_lowercase()
                            .contains(&format!("authorization: bearer {key}\r\n"))
                    );
                    let body = r#"{"data":[{"id":"example-model"}]}"#;
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                };
                let (models, ()) = tokio::time::timeout(Duration::from_secs(5), async {
                    tokio::join!(provider.list_models(), serve)
                })
                .await
                .expect("The local endpoint should receive the request.");
                assert_eq!(models.unwrap()[0].id, "example-model");
            }
        }
    }
}
