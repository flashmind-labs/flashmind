use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use flashmind_llm::three_d::{
    ThreeDClient, ThreeDImage, ThreeDInput, ThreeDProvider, ThreeDRequest,
};
use flashmind_types::{Model, Provider};
use url::Url;

use crate::config::Config;

#[derive(Args, Clone)]
pub(crate) struct ThreeDArgs {
    /// Text describing the model (omit when using --image)
    #[arg(required_unless_present = "image", conflicts_with = "image")]
    prompt: Option<String>,
    /// Reference images (up to four); for Tripo, use front, left, back, right order
    #[arg(long, value_hint = clap::ValueHint::FilePath, num_args = 1)]
    image: Vec<String>,
    /// Write a new GLB file at this path
    #[arg(short, long)]
    output: PathBuf,
    /// Generate textures; Meshy text requests also create a separate refine task
    #[arg(long)]
    texture: bool,
}

pub(crate) async fn run(model: &Model, config: &Config, args: ThreeDArgs) -> Result<()> {
    let provider = match &model.provider {
        Provider::Custom(name) if name == "meshy" => ThreeDProvider::Meshy,
        Provider::Custom(name) if name == "tripo" => ThreeDProvider::Tripo,
        _ => bail!("3D generation requires a meshy:model or tripo:model"),
    };
    if !args
        .output
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"))
    {
        bail!("output filename must end in .glb");
    }
    let key = config
        .api_key_for(&model.provider)
        .with_context(|| format!("API key required for {}", model.provider))?;
    let mut client = ThreeDClient::new(provider, key.to_owned())?;
    if let Some(url) = config.endpoint_override.url.as_ref().or_else(|| {
        config
            .compatible_provider(&model.provider.to_string())
            .and_then(|entry| entry.url.as_ref())
    }) {
        client = client.with_base_url(url.clone());
    }
    let input = if args.image.is_empty() {
        ThreeDInput::Text(args.prompt.context("provide a prompt or --image")?)
    } else {
        let images = args
            .image
            .into_iter()
            .map(|image| match Url::parse(&image) {
                Ok(url) if matches!(url.scheme(), "http" | "https") => Ok(ThreeDImage::Url(url)),
                Ok(_) => bail!("image URL must use HTTP or HTTPS"),
                Err(url::ParseError::RelativeUrlWithoutBase) => {
                    Ok(ThreeDImage::File(PathBuf::from(image)))
                }
                Err(error) => Err(error).context("invalid image URL"),
            })
            .collect::<Result<Vec<_>>>()?;
        ThreeDInput::Images(images)
    };
    let request = ThreeDRequest {
        model: model.clone(),
        input,
        texture: args.texture,
        extra_args: config.endpoint_override.extra_args.clone(),
    };
    let mut output = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .await
        .with_context(|| format!("cannot create {}", args.output.display()))?;
    let result = tokio::select! {
        result = async {
            let asset = client.generate(request, |id| eprintln!("3D task: {id}")).await?;
            client.save(&asset, &mut output).await
        } => result,
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => Err(anyhow::anyhow!("interrupted; a submitted task may still be running with the provider")),
                Err(error) => Err(error).context("cannot listen for interruption"),
            }
        }
    };
    if let Err(error) = result {
        drop(output);
        if let Err(cleanup_error) = tokio::fs::remove_file(&args.output).await {
            return Err(error.context(format!(
                "cannot remove incomplete {}: {cleanup_error}",
                args.output.display()
            )));
        }
        return Err(error);
    }
    println!("{}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_existing_output_without_overwriting_it() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model.glb");
        tokio::fs::write(&output, b"original").await.unwrap();
        let config: Config =
            toml::from_str("[providers.meshy]\napi_key = \"example-key\"\n").unwrap();
        let model = "meshy:example-model".parse().unwrap();
        let args = ThreeDArgs {
            prompt: Some("example".into()),
            image: Vec::new(),
            output: output.clone(),
            texture: false,
        };
        assert!(run(&model, &config, args).await.is_err());
        assert_eq!(tokio::fs::read(output).await.unwrap(), b"original");
    }

    #[tokio::test]
    async fn removes_reserved_output_if_input_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model.glb");
        let config: Config =
            toml::from_str("[providers.tripo]\napi_key = \"example-key\"\n").unwrap();
        let model = "tripo:example-model".parse().unwrap();
        let args = ThreeDArgs {
            prompt: None,
            image: vec![
                dir.path()
                    .join("missing.png")
                    .to_string_lossy()
                    .into_owned(),
            ],
            output: output.clone(),
            texture: false,
        };
        assert!(run(&model, &config, args).await.is_err());
        assert!(!tokio::fs::try_exists(output).await.unwrap());
    }
}
