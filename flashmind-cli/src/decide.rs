use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use flashmind_llm::{JEV_LATEST_MODEL, JevRequest, OpenRouterProvider};
use flashmind_types::Provider;

use crate::config::Config;

#[derive(Args)]
pub(crate) struct DecideArgs {
    /// JSON request file. Reads stdin when omitted.
    input: Option<PathBuf>,
}

pub(crate) async fn run(args: &DecideArgs, model: Option<&str>, config: &Config) -> Result<()> {
    let request = request(args, model)?;
    let key = config
        .api_key_for(&Provider::OpenRouter)
        .filter(|key| !key.trim().is_empty())
        .context("OpenRouter API key required")?;
    let response = OpenRouterProvider::new(key.to_owned())
        .decide(request)
        .await?;
    serde_json::to_writer(io::stdout(), &response).context("cannot write response")?;
    println!();
    Ok(())
}

fn request(args: &DecideArgs, model: Option<&str>) -> Result<JevRequest> {
    let input = if let Some(path) = &args.input {
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?
    } else {
        if io::stdin().is_terminal() {
            bail!("pass a JSON file or pipe a JSON request to stdin");
        }
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .context("cannot read stdin")?;
        input
    };
    parse_request(&input, model)
}

fn parse_request(input: &str, model: Option<&str>) -> Result<JevRequest> {
    let mut value: serde_json::Value =
        serde_json::from_str(input).context("invalid request JSON")?;
    let object = value
        .as_object_mut()
        .context("request must be a JSON object")?;
    if let Some(model) = model {
        object.insert("model".into(), model.into());
    } else if !object.contains_key("model") {
        object.insert("model".into(), JEV_LATEST_MODEL.into());
    }
    serde_json::from_value(value).context("invalid Jev request")
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: &str = r#"{
        "state": "An invoice arrived.",
        "questions": {
            "is_invoice": {
                "type": "noul",
                "instructions": "Is this an invoice?"
            }
        }
    }"#;

    #[test]
    fn uses_default_model_when_omitted() {
        let request = parse_request(INPUT, None).expect("valid sample request");
        assert_eq!(request.model, JEV_LATEST_MODEL);
        assert!(request.questions.contains_key("is_invoice"));
    }

    #[test]
    fn command_line_model_overrides_request() {
        let input = INPUT.replace("\"state\"", "\"model\": \"jev-1.0\", \"state\"");
        let request = parse_request(&input, Some("jev-1.13")).expect("valid sample request");
        assert_eq!(request.model, "jev-1.13");
    }
}
