use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use flashmind_llm::three_d::{
    RigInput, ThreeDClient, ThreeDImage, ThreeDInput, ThreeDProvider, ThreeDRequest,
};
use flashmind_types::{Model, Provider};
use url::Url;

use crate::config::Config;

#[derive(Args, Clone)]
pub(crate) struct ThreeDArgs {
    /// Text describing the model (omit when using --image)
    #[arg(required_unless_present_any = ["image", "rig_model"], conflicts_with_all = ["image", "rig_model"])]
    prompt: Option<String>,
    /// Reference images (up to four); for Tripo, use front, left, back, right order
    #[arg(long, value_hint = clap::ValueHint::FilePath, num_args = 1, conflicts_with = "rig_model")]
    image: Vec<String>,
    /// Write a new GLB file at this path
    #[arg(short, long)]
    output: PathBuf,
    /// Generate textures; Meshy text requests also create a separate refine task
    #[arg(long)]
    texture: bool,
    /// Rig a textured humanoid with Meshy after generation
    #[arg(long)]
    rig: bool,
    /// Rig an existing GLB file or HTTP(S) URL instead of generating a model
    #[arg(long, requires = "rig", conflicts_with = "texture")]
    rig_model: Option<String>,
    /// Remesh to roughly 300,000 triangles before rigging
    #[arg(long, requires = "rig")]
    remesh: bool,
    /// Character height in meters (default: 1.7)
    #[arg(long, requires = "rig", value_parser = clap::value_parser!(f64))]
    height: Option<f64>,
}

pub(crate) async fn run(model: &Model, config: &Config, args: ThreeDArgs) -> Result<()> {
    let provider = match &model.provider {
        Provider::Custom(name) if name == "meshy" => ThreeDProvider::Meshy,
        Provider::Custom(name) if name == "tripo" => ThreeDProvider::Tripo,
        _ => bail!("3D generation requires a meshy:model or tripo:model"),
    };
    if args.rig && !matches!(provider, ThreeDProvider::Meshy) {
        bail!("rigging is only supported for Meshy models");
    }
    if args.rig && args.rig_model.is_none() && !args.texture {
        bail!("Meshy rigging requires a textured model; add --texture");
    }
    if args.rig_model.is_some() && !config.endpoint_override.extra_args.is_empty() {
        bail!("extra arguments apply only to model generation");
    }
    if args
        .height
        .is_some_and(|height| !height.is_finite() || height <= 0.0)
    {
        bail!("character height must be a positive number of meters");
    }
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
    let rig_input = args
        .rig_model
        .map(|model| match Url::parse(&model) {
            Ok(url) if matches!(url.scheme(), "http" | "https") => Ok(RigInput::Url(url)),
            Ok(_) => bail!("model URL must use HTTP or HTTPS"),
            Err(url::ParseError::RelativeUrlWithoutBase) => {
                Ok(RigInput::File(PathBuf::from(model)))
            }
            Err(error) => Err(error).context("invalid model URL"),
        })
        .transpose()?;
    let input = if rig_input.is_some() {
        None
    } else if args.image.is_empty() {
        Some(ThreeDInput::Text(
            args.prompt.context("provide a prompt or --image")?,
        ))
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
        Some(ThreeDInput::Images(images))
    };
    let request = input.map(|input| ThreeDRequest {
        model: model.clone(),
        input,
        texture: args.texture,
        extra_args: config.endpoint_override.extra_args.clone(),
    });
    let mut output = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .await
        .with_context(|| format!("cannot create {}", args.output.display()))?;
    let result = tokio::select! {
        result = async {
            let asset = match (rig_input, request) {
                (Some(input), None) => client.rig(input, args.remesh, args.height.unwrap_or(1.7), |id| eprintln!("3D task: {id}")).await?,
                (None, Some(request)) => {
                    let asset = client.generate(request, |id| eprintln!("3D task: {id}")).await?;
                    if args.rig {
                        client.rig(RigInput::Url(asset), args.remesh, args.height.unwrap_or(1.7), |id| eprintln!("3D task: {id}")).await?
                    } else { asset }
                }
                _ => bail!("provide a model or generation input"),
            };
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
            rig: false,
            rig_model: None,
            remesh: false,
            height: None,
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
            rig: false,
            rig_model: None,
            remesh: false,
            height: None,
        };
        assert!(run(&model, &config, args).await.is_err());
        assert!(!tokio::fs::try_exists(output).await.unwrap());
    }
}
