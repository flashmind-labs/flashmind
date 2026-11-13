use std::sync::Arc;

use anyhow::Result;
use flashmind_tools::{ToolBuilder, list_models::list_models};
use flashmind_types::{ApiKey, ModelCategory, Provider, ProviderRegistry};

use crate::config::Config;
use crate::interactive::configured_providers;
use crate::provider::build_provider;

pub(crate) async fn run(
    filter: Option<&Provider>,
    feature: Option<ModelCategory>,
    config: &Config,
) -> Result<()> {
    let registry = catalogs(filter, config)?;
    let mut models = list_models(&registry, feature, filter, None).await?;
    models.sort_by(|a, b| a.id.cmp(&b.id));
    for model in models {
        println!("{}", model.id);
    }
    Ok(())
}

fn catalogs(filter: Option<&Provider>, config: &Config) -> Result<ProviderRegistry> {
    let providers: Vec<_> = match filter {
        Some(provider) => vec![provider.clone()],
        None => configured_providers(config)
            .into_iter()
            .map(|(_, provider)| provider)
            .collect(),
    };
    let mut registry = ToolBuilder::new()
        .media(
            |kind| match kind {
                ApiKey::Meshy | ApiKey::Tripo => config
                    .api_key_for(&Provider::Custom(kind.to_string()))
                    .map(str::to_owned),
                _ => None,
            },
            Default::default(),
        )
        .providers()
        .as_ref()
        .clone();
    for provider in providers {
        if registry.contains_key(&provider) {
            continue;
        }
        match build_provider(&provider, config) {
            Ok(llm) => {
                registry.insert(provider, llm);
            }
            Err(error) if filter.is_none() => eprintln!("{provider}: {error:#}"),
            Err(error) => return Err(error),
        }
    }
    Ok(Arc::new(registry))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cli_catalog_uses_configured_three_d_keys() {
        let config: Config = toml::from_str(
            "meshy_api_key = 'example-meshy-key'\ntripo_api_key = 'example-tripo-key'\n",
        )
        .unwrap();
        for provider in ["meshy", "tripo"] {
            let provider: Provider = provider.parse().unwrap();
            let registry = catalogs(Some(&provider), &config).unwrap();
            let models = list_models(
                &registry,
                Some(ModelCategory::ThreeD),
                Some(&provider),
                None,
            )
            .await
            .unwrap();
            assert_eq!(models.len(), 4);
            assert!(
                models
                    .iter()
                    .all(|model| model.id.starts_with(&format!("{provider}:")))
            );
        }
    }
}
