use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use flashmind_llm::{JEV_LATEST_MODEL, JevRequest, OpenRouterProvider};
use serde::Deserialize;

#[derive(Parser)]
#[command(about = "Run typed Jev decisions through OpenRouter")]
struct Args {
    /// JSON request file. Reads stdin when omitted.
    input: Option<PathBuf>,

    /// Override the model in the request.
    #[arg(long)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct Config {
    openrouter_key: String,
}

fn api_key() -> Result<String> {
    let path = dirs::home_dir()
        .context("cannot determine home directory")?
        .join(".jev-cli");
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return std::env::var("OPENROUTER_API_KEY")
                .ok()
                .filter(|key| !key.trim().is_empty())
                .context("set openrouter_key in ~/.jev-cli or OPENROUTER_API_KEY");
        }
        Err(error) => return Err(error).with_context(|| format!("cannot read {}", path.display())),
    };
    let config: Config = toml::from_str(&contents)
        .with_context(|| format!("invalid config in {}", path.display()))?;
    if config.openrouter_key.trim().is_empty() {
        bail!("openrouter_key in {} is empty", path.display());
    }
    Ok(config.openrouter_key)
}

fn request(args: &Args) -> Result<JevRequest> {
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
    parse_request(&input, args.model.as_deref())
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let request = request(&args)?;
    let provider = OpenRouterProvider::new(api_key()?);
    let response = provider.decide(request).await?;
    serde_json::to_writer(io::stdout(), &response).context("cannot write response")?;
    println!();
    Ok(())
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
