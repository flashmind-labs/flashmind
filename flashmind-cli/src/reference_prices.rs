//! Reference prices from the public OpenRouter model catalog.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use flashmind_types::llm::{ReferencePricing, ReferencePricingSource};
use flashmind_types::{Model, ModelInfo, Provider};
use serde::Deserialize;
use tokio::sync::OnceCell;

static CATALOG: OnceCell<HashMap<String, ReferencePricing>> = OnceCell::const_new();

#[derive(Deserialize)]
struct Catalog {
    data: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    pricing: Option<Prices>,
}

#[derive(Deserialize)]
struct Prices {
    prompt: Option<String>,
    completion: Option<String>,
}

fn model_id(model: &Model) -> Option<String> {
    let author = match &model.provider {
        Provider::OpenAi => "openai",
        Provider::Anthropic => "anthropic",
        Provider::Mistral => "mistralai",
        Provider::Custom(name) if name == "google" => "google",
        Provider::Custom(name) if name == "moonshot" => "moonshotai",
        Provider::Custom(name) if name == "zai" => "z-ai",
        _ => return None,
    };
    let name = model.capability_name();
    if name.contains('/') || name.contains(':') {
        return None;
    }
    Some(format!("{author}/{name}"))
}

async fn fetch_catalog() -> Result<HashMap<String, ReferencePricing>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let catalog: Catalog = client
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await
        .context("reference price request failed")?
        .error_for_status()?
        .json()
        .await
        .context("invalid reference price catalog")?;
    Ok(parse_catalog(catalog))
}

fn parse_catalog(catalog: Catalog) -> HashMap<String, ReferencePricing> {
    catalog
        .data
        .into_iter()
        .filter_map(|entry| {
            let pricing = entry.pricing?;
            let prompt = pricing
                .prompt
                .as_deref()
                .and_then(|value| value.parse::<rust_decimal::Decimal>().ok())
                .filter(|price| !price.is_sign_negative());
            let completion = pricing
                .completion
                .as_deref()
                .and_then(|value| value.parse::<rust_decimal::Decimal>().ok())
                .filter(|price| !price.is_sign_negative());
            if prompt.is_none() && completion.is_none() {
                return None;
            }
            Some((
                entry.id,
                ReferencePricing {
                    source: ReferencePricingSource::OpenRouter,
                    prompt,
                    completion,
                },
            ))
        })
        .collect()
}

/// Return OpenRouter prices as a reference, never as provider billing rates.
pub(crate) async fn lookup(model: &Model) -> Option<ReferencePricing> {
    let id = model_id(model)?;
    let catalog = CATALOG.get_or_try_init(fetch_catalog).await.ok()?;
    catalog.get(&id).cloned()
}

/// Add reference rates without changing the rates used for cost accounting.
pub(crate) async fn enrich(provider: &Provider, models: &mut [ModelInfo]) {
    if !models
        .iter()
        .any(|model| model.pricing.prompt.is_none() || model.pricing.completion.is_none())
    {
        return;
    }
    if !models.iter().any(|model| {
        format!("{provider}:{}", model.id)
            .parse::<Model>()
            .ok()
            .and_then(|model| model_id(&model))
            .is_some()
    }) {
        return;
    }
    let Ok(catalog) = CATALOG.get_or_try_init(fetch_catalog).await else {
        return;
    };
    for model in models {
        if model.pricing.prompt.is_some() && model.pricing.completion.is_some() {
            continue;
        }
        if let Ok(id) = format!("{provider}:{}", model.id).parse::<Model>() {
            model.pricing.reference = model_id(&id).and_then(|key| catalog.get(&key).cloned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_only_known_authors_and_exact_model_ids() {
        assert_eq!(
            model_id(&"openai:example-model".parse().unwrap()).as_deref(),
            Some("openai/example-model")
        );
        assert!(model_id(&"ollama:example-model".parse().unwrap()).is_none());
        assert!(model_id(&"openai:other/example-model".parse().unwrap()).is_none());
    }

    #[test]
    fn catalog_ignores_missing_and_invalid_prices() {
        let catalog: Catalog = serde_json::from_str(r#"{"data":[{"id":"openai/example","pricing":{"prompt":"0.000001","completion":"0.000005"}},{"id":"openai/missing","pricing":null},{"id":"openai/invalid","pricing":{"prompt":"-1","completion":"bad"}}]}"#).unwrap();
        let prices = parse_catalog(catalog);
        assert_eq!(prices.len(), 1);
        assert_eq!(
            prices["openai/example"].prompt,
            Some(rust_decimal::Decimal::new(1, 6))
        );
        assert_eq!(
            prices["openai/example"].completion,
            Some(rust_decimal::Decimal::new(5, 6))
        );
    }
}
