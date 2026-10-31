use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use clap::{Args, Subcommand};
use flashmind_llm::OpenRouterProvider;
use flashmind_llm::openrouter::{
    TimestampGranularity, TranscriptionOptions, TranscriptionResponse,
};
use flashmind_types::llm::{
    AudioFormat, CompletionRequest, CompletionStream, FinishReason, FrameImage, ImageEditRequest,
    ImageGenConfig, ImageGenRequest, ImageOutputFormat, LlmProvider, StreamEvent, SttRequest,
    TtsRequest, VideoGenRequest, mime_from_extension,
};
use flashmind_types::message::{ContentPart, Message};
use flashmind_types::model::{Model, ReasoningLevel};
use futures_util::StreamExt;
use rust_decimal::Decimal;
use serde_json::{Map, Value};

#[derive(Args, Clone)]
pub(crate) struct TextArgs {
    /// Prompt text. Reads stdin when omitted.
    prompt: Option<String>,
    /// System prompt.
    #[arg(short, long)]
    system: Option<String>,
    /// Attach an image, PDF, audio, video or text file. Repeatable.
    #[arg(short, long)]
    attach: Vec<PathBuf>,
    #[arg(long)]
    max_tokens: Option<u32>,
    #[arg(long)]
    temperature: Option<Decimal>,
    /// Reasoning effort: off, low, medium or high.
    #[arg(long, default_value = "off", value_parser = crate::shell_completion::reasoning_levels())]
    reasoning: ReasoningLevel,
}

#[derive(Args, Clone)]
struct OutputArgs {
    /// Output file, or - for stdout. Defaults to a new file in the current directory.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Replace an existing output file.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Args, Clone)]
pub(crate) struct ImageArgs {
    #[command(subcommand)]
    pub(crate) command: Option<ImageCommand>,
    #[command(flatten)]
    pub(crate) generation: ImageGenArgs,
}

#[derive(Args, Clone)]
pub(crate) struct ImageGenArgs {
    /// Prompt text. Reads stdin when omitted.
    prompt: Option<String>,
    #[command(flatten)]
    output: OutputArgs,
    /// Size, such as 1K or 1024x1024.
    #[arg(long)]
    size: Option<String>,
    /// Aspect ratio, such as 16:9.
    #[arg(long)]
    aspect_ratio: Option<String>,
    #[arg(long)]
    quality: Option<String>,
    /// Reference image as a file path or URL. Repeatable.
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    reference: Vec<String>,
    /// Number of images.
    #[arg(short)]
    n: Option<u32>,
    /// Output format, when supported by the selected model.
    #[arg(long, value_parser = parse_name::<ImageOutputFormat>)]
    format: Option<ImageOutputFormat>,
}

#[derive(Subcommand, Clone)]
pub(crate) enum ImageCommand {
    /// Generate an image from a prompt.
    Gen(ImageGenArgs),
    /// Edit an existing image with an edit-capable model.
    Edit(ImageEditArgs),
}

#[derive(Args, Clone)]
pub(crate) struct ImageEditArgs {
    /// Edit instructions. Reads stdin when omitted.
    prompt: Option<String>,
    /// Input image path or URL. Repeat for multiple images, base image first.
    #[arg(long, required = true, value_hint = clap::ValueHint::FilePath)]
    image: Vec<String>,
    #[command(flatten)]
    output: OutputArgs,
    /// Output resolution tier, such as 1K or 2K.
    #[arg(long, alias = "size")]
    resolution: Option<String>,
    /// Output aspect ratio, such as 3:4.
    #[arg(long)]
    aspect_ratio: Option<String>,
    /// Provider-specific quality tier.
    #[arg(long)]
    quality: Option<String>,
    /// PNG mask: transparent areas are edited (OpenAI only).
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    mask: Option<String>,
    /// Output format, when supported by the selected model.
    #[arg(long, value_parser = parse_name::<ImageOutputFormat>)]
    format: Option<ImageOutputFormat>,
}

#[derive(Args, Clone)]
pub(crate) struct VideoArgs {
    /// Prompt text. Reads stdin when omitted.
    prompt: Option<String>,
    #[command(flatten)]
    output: OutputArgs,
    /// Resolution, such as 720p.
    #[arg(long)]
    resolution: Option<String>,
    /// Aspect ratio, such as 16:9.
    #[arg(long)]
    aspect_ratio: Option<String>,
    /// Duration in seconds.
    #[arg(long)]
    duration: Option<u32>,
    /// Ask the model to generate an audio track.
    #[arg(long)]
    audio: bool,
    /// First frame image, as a file path or URL.
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    first_frame: Option<String>,
    /// Last frame image, as a file path or URL.
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    last_frame: Option<String>,
    /// Style reference image, as a file path or URL. Repeatable.
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    reference: Vec<String>,
    /// Reference video as a file path or URL. Repeatable.
    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    video_reference: Vec<String>,
}

#[derive(Args, Clone)]
pub(crate) struct SpeechArgs {
    /// Text to speak. Reads stdin when omitted.
    text: Option<String>,
    #[command(flatten)]
    output: OutputArgs,
    #[arg(long)]
    voice: String,
    /// Audio format: mp3, wav, opus, aac or flac.
    #[arg(long, default_value = "mp3", value_parser = crate::shell_completion::audio_formats())]
    format: AudioFormat,
}

#[derive(Args, Clone)]
pub(crate) struct TranscribeArgs {
    /// Audio file to transcribe.
    file: PathBuf,
    /// Language hint, such as en.
    #[arg(long)]
    language: Option<String>,
    /// Request verbose JSON and print the full response, including speaker metadata.
    #[arg(long)]
    json: bool,
    /// Request segment or word timestamps (comma-separated). Enables JSON output.
    #[arg(long, value_delimiter = ',', value_parser = parse_name::<TimestampGranularity>)]
    timestamps: Vec<TimestampGranularity>,
    /// Provider options as a JSON object keyed by endpoint slug. Enables JSON output.
    #[arg(long, value_parser = parse_provider_options)]
    provider_options: Option<Map<String, Value>>,
}

fn parse_provider_options(value: &str) -> Result<Map<String, Value>, String> {
    let options: Map<String, Value> = serde_json::from_str(value)
        .map_err(|error| format!("expected a JSON object keyed by provider slug: {error}"))?;
    if options.values().any(|options| !options.is_object()) {
        return Err("each provider's options must be a JSON object".into());
    }
    Ok(options)
}

impl TranscribeArgs {
    fn structured_output(&self) -> bool {
        self.json || !self.timestamps.is_empty() || self.provider_options.is_some()
    }
}

/// Parses a strum enum. Its error type is not a std error, which clap needs.
fn parse_name<T: FromStr>(value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("unknown value {value}"))
}

/// Returns the argument, or stdin when it is absent.
fn input_text(arg: Option<String>) -> Result<String> {
    let text = match arg {
        Some(text) => text,
        None => {
            if io::stdin().is_terminal() {
                bail!("pass a prompt argument or pipe it to stdin");
            }
            let mut text = String::new();
            io::stdin()
                .read_to_string(&mut text)
                .context("cannot read stdin")?;
            text
        }
    };
    if text.trim().is_empty() {
        bail!("prompt is empty");
    }
    Ok(text)
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn attachment(path: &Path) -> Result<ContentPart> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let part = match extension(path).as_deref() {
        Some("png" | "jpg" | "jpeg" | "gif" | "webp") => ContentPart::Image {
            media_type: mime_from_extension(path).into(),
            data: STANDARD.encode(&bytes),
        },
        Some("pdf") => ContentPart::Document {
            media_type: "application/pdf".into(),
            filename,
            data: STANDARD.encode(&bytes),
        },
        Some(ext @ ("mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac")) => ContentPart::Audio {
            media_type: format!("audio/{ext}"),
            filename,
            data: STANDARD.encode(&bytes),
        },
        Some(ext @ ("mp4" | "webm" | "mov")) => ContentPart::Video {
            media_type: if ext == "mov" {
                "video/quicktime".into()
            } else {
                format!("video/{ext}")
            },
            filename,
            data: STANDARD.encode(&bytes),
        },
        _ => match String::from_utf8(bytes) {
            Ok(text) => ContentPart::Text {
                text: format!("File {filename}:\n{text}"),
            },
            Err(_) => bail!("unsupported attachment type: {}", path.display()),
        },
    };
    Ok(part)
}

/// Accepts a URL or data URI as is and turns a local path into a data URI.
fn image_url(value: &str) -> Result<String> {
    if ["http://", "https://", "data:"]
        .iter()
        .any(|prefix| value.starts_with(prefix))
    {
        return Ok(value.to_owned());
    }
    ImageGenConfig::data_uri_from_path(Path::new(value))
        .with_context(|| format!("cannot read {value}"))
}

fn video_url(value: &str) -> Result<String> {
    if ["http://", "https://", "data:"]
        .iter()
        .any(|prefix| value.starts_with(prefix))
    {
        return Ok(value.to_owned());
    }
    let path = Path::new(value);
    let media_type = match extension(path).as_deref() {
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        _ => bail!("reference video must be an MP4, WebM, or MOV file"),
    };
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    Ok(format!(
        "data:{media_type};base64,{}",
        STANDARD.encode(bytes)
    ))
}

fn file_extension(media_type: &str) -> &str {
    match media_type {
        "image/jpeg" => "jpg",
        "image/svg+xml" => "svg",
        "audio/mpeg" => "mp3",
        "video/quicktime" => "mov",
        other => other
            .split(';')
            .next()
            .and_then(|mime| mime.split('/').nth(1))
            .filter(|ext| !ext.is_empty())
            .unwrap_or("bin"),
    }
}

/// Path for the file at `index` when the user passed `-o path`: the first
/// file keeps the path, later ones get `-2`, `-3` before the extension.
fn numbered_path(path: &Path, index: usize) -> PathBuf {
    if index == 0 {
        return path.to_path_buf();
    }
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = match path.extension() {
        Some(ext) => format!("{stem}-{}.{}", index + 1, ext.to_string_lossy()),
        None => format!("{stem}-{}", index + 1),
    };
    path.with_file_name(name)
}

/// Writes generated files and prints where each one went.
struct Saver {
    output: Option<PathBuf>,
    prefix: &'static str,
    count: usize,
    overwrite: bool,
}

impl Saver {
    fn new(output: OutputArgs, prefix: &'static str) -> Self {
        Self {
            output: output.output,
            prefix,
            count: 0,
            overwrite: output.overwrite,
        }
    }

    fn save(&mut self, media_type: &str, data: &[u8]) -> Result<()> {
        let index = self.count;
        self.count += 1;
        let path = match &self.output {
            Some(path) if path.as_os_str() == "-" => {
                if index > 0 {
                    bail!("the model returned more than one file; pass a file path to -o");
                }
                let mut stdout = io::stdout().lock();
                stdout.write_all(data).context("cannot write to stdout")?;
                return stdout.flush().context("cannot write to stdout");
            }
            Some(path) => numbered_path(path, index),
            None => {
                let millis = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or_default();
                let suffix = if index == 0 {
                    String::new()
                } else {
                    format!("-{}", index + 1)
                };
                PathBuf::from(format!(
                    "{}-{millis}{suffix}.{}",
                    self.prefix,
                    file_extension(media_type)
                ))
            }
        };
        let mut options = fs::OpenOptions::new();
        options.write(true);
        if self.overwrite {
            options.create(true).truncate(true);
        } else {
            options.create_new(true);
        }
        let mut file = options.open(&path).with_context(|| {
            format!(
                "cannot create {} (use --overwrite to replace an existing file)",
                path.display()
            )
        })?;
        file.write_all(data)
            .with_context(|| format!("cannot write {}", path.display()))?;
        println!("{}", path.display());
        Ok(())
    }
}

/// Reads a provider stream to the end. Text goes to stdout when `text_to_stdout`
/// is set and to stderr as progress otherwise. Files go through `saver`.
async fn drain(
    mut stream: CompletionStream,
    saver: &mut Saver,
    text_to_stdout: bool,
) -> Result<()> {
    let mut wrote_text = false;
    let mut ends_with_newline = true;
    let mut audio: Option<(String, Vec<u8>)> = None;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) if text_to_stdout => {
                if text.is_empty() {
                    continue;
                }
                let mut stdout = io::stdout().lock();
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
                wrote_text = true;
                ends_with_newline = text.ends_with('\n');
            }
            StreamEvent::ContentDelta(text) => eprint!("{text}"),
            StreamEvent::AudioDelta { data, format } => {
                audio
                    .get_or_insert_with(|| (format, Vec::new()))
                    .1
                    .extend(data);
            }
            StreamEvent::FileAttachment {
                media_type, data, ..
            } => saver.save(&media_type, &data)?,
            StreamEvent::Finished(
                reason @ (FinishReason::Length | FinishReason::ContentFilter),
            ) => {
                eprintln!("warning: model stopped early ({reason})");
            }
            _ => {}
        }
    }
    if wrote_text && !ends_with_newline {
        println!();
    }
    if let Some((format, data)) = audio {
        saver.save(&format!("audio/{format}"), &data)?;
    }
    if !wrote_text && saver.count == 0 {
        bail!("the model returned no output");
    }
    Ok(())
}

pub(crate) async fn text(provider: &dyn LlmProvider, model: Model, args: TextArgs) -> Result<()> {
    let mut messages = Vec::new();
    if let Some(system) = args.system {
        messages.push(Message::system(system));
    }
    let mut message = Message::user(input_text(args.prompt)?);
    if !args.attach.is_empty() {
        let parts = args
            .attach
            .iter()
            .map(|path| attachment(path))
            .collect::<Result<Vec<_>>>()?;
        message.parts = Some(parts);
    }
    messages.push(message);
    let mut request = CompletionRequest::new(model, messages);
    request.max_tokens = args.max_tokens;
    request.reasoning = args.reasoning;
    request.sampling.temperature = args.temperature;
    let mut saver = Saver::new(
        OutputArgs {
            output: None,
            overwrite: false,
        },
        "output",
    );
    drain(provider.complete(request), &mut saver, true).await
}

pub(crate) async fn image(
    provider: &dyn LlmProvider,
    model: &Model,
    args: ImageGenArgs,
) -> Result<()> {
    check_output(&args.output)?;
    if !args.reference.is_empty() {
        check_image_capabilities(provider, model, args.reference.len(), false).await?;
    }
    let request = ImageGenRequest {
        model: model.capability_name().to_owned(),
        prompt: input_text(args.prompt)?,
        size: args.size,
        aspect_ratio: args.aspect_ratio,
        quality: args.quality,
        style: None,
        n: args.n,
        input_references: args
            .reference
            .iter()
            .map(|value| image_url(value))
            .collect::<Result<Vec<_>>>()?,
        output_format: args.format,
    };
    let mut saver = Saver::new(args.output, "image");
    drain(provider.generate_image(request), &mut saver, false).await
}

pub(crate) async fn image_edit(
    provider: &dyn LlmProvider,
    model: &Model,
    args: ImageEditArgs,
) -> Result<()> {
    check_output(&args.output)?;
    check_image_capabilities(provider, model, args.image.len(), true).await?;
    if let Some(mask) = &args.mask
        && !mask.starts_with("https://")
        && !mask.starts_with("http://")
        && !mask.starts_with("data:image/png;base64,")
        && extension(Path::new(mask)).as_deref() != Some("png")
    {
        bail!("mask must be a PNG file or PNG data URL");
    }
    let request = ImageEditRequest {
        model: model.capability_name().to_owned(),
        prompt: input_text(args.prompt)?,
        images: args
            .image
            .iter()
            .map(|value| image_url(value))
            .collect::<Result<Vec<_>>>()?,
        resolution: args.resolution.map(|value| {
            if value.contains(['x', 'X']) {
                value.to_lowercase()
            } else {
                value.to_uppercase()
            }
        }),
        aspect_ratio: args.aspect_ratio,
        quality: args.quality,
        mask: args.mask.as_deref().map(image_url).transpose()?,
        output_format: args.format,
    };
    let mut saver = Saver::new(args.output, "image");
    drain(provider.edit_image(request), &mut saver, false).await
}

fn check_output(output: &OutputArgs) -> Result<()> {
    if let Some(path) = &output.output
        && path.as_os_str() != "-"
        && !output.overwrite
    {
        match fs::symlink_metadata(path) {
            Ok(_) => bail!(
                "{} already exists; use --overwrite to replace it",
                path.display()
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("cannot check {}", path.display()));
            }
        }
    }
    Ok(())
}

async fn check_image_capabilities(
    provider: &dyn LlmProvider,
    model: &Model,
    count: usize,
    editing: bool,
) -> Result<()> {
    if provider.name() != "venice" {
        return Ok(());
    }
    let catalog =
        tokio::time::timeout(std::time::Duration::from_secs(15), provider.list_models()).await;
    let Ok(Some(models)) = catalog else {
        return Ok(());
    };
    let Some(info) = models
        .iter()
        .find(|info| info.id == model.capability_name())
    else {
        return Ok(());
    };
    if !editing && info.image.style_references == Some(false) {
        bail!(
            "{} does not support style references; choose a model with style-reference support or use an edit model with flsh image edit",
            model
        );
    }
    if editing
        && let Some(max) = info.image.max_edit_images
        && u32::try_from(count).map_or(true, |count| count > max)
    {
        bail!("{} supports at most {max} input images", model);
    }
    Ok(())
}

pub(crate) async fn video(
    provider: &dyn LlmProvider,
    model: &Model,
    args: VideoArgs,
) -> Result<()> {
    check_output(&args.output)?;
    let mut frame_images = Vec::new();
    for (value, frame_type) in [
        (&args.first_frame, "first_frame"),
        (&args.last_frame, "last_frame"),
    ] {
        if let Some(value) = value {
            frame_images.push(FrameImage {
                url: image_url(value)?,
                frame_type: frame_type.into(),
            });
        }
    }
    let input_references = args
        .reference
        .iter()
        .map(|value| image_url(value))
        .collect::<Result<Vec<_>>>()?;
    let request = VideoGenRequest {
        model: model.capability_name().to_owned(),
        description: input_text(args.prompt)?,
        resolution: args.resolution,
        aspect_ratio: args.aspect_ratio,
        duration: args.duration,
        generate_audio: args.audio.then_some(true),
        frame_images,
        input_references,
        video_references: args
            .video_reference
            .iter()
            .map(|value| video_url(value))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut saver = Saver::new(args.output, "video");
    drain(provider.generate_video(request), &mut saver, false).await
}

pub(crate) async fn speech(
    provider: &dyn LlmProvider,
    model: &Model,
    args: SpeechArgs,
) -> Result<()> {
    check_output(&args.output)?;
    let request = TtsRequest {
        model: model.capability_name().to_owned(),
        input: input_text(args.text)?,
        voice: args.voice,
        response_format: args.format,
    };
    let mut saver = Saver::new(args.output, "speech");
    drain(provider.text_to_speech(request), &mut saver, false).await
}

pub(crate) async fn transcribe(
    provider: &OpenRouterProvider,
    model: &Model,
    args: TranscribeArgs,
) -> Result<()> {
    let structured = args.structured_output();
    let audio =
        fs::read(&args.file).with_context(|| format!("cannot read {}", args.file.display()))?;
    let format = extension(&args.file).unwrap_or_else(|| "mp3".into());
    let request = SttRequest {
        model: model.capability_name().to_owned(),
        audio,
        media_type: format!("audio/{format}"),
        language: args.language,
    };
    let response = provider
        .transcribe_detailed(
            request,
            TranscriptionOptions {
                verbose_json: structured,
                timestamp_granularities: args.timestamps,
                provider_options: args.provider_options.unwrap_or_default(),
            },
        )
        .await?;
    write_transcription(&mut io::stdout().lock(), &response, structured)
}

pub(crate) async fn transcribe_compatible(
    provider: &dyn LlmProvider,
    model: &Model,
    args: TranscribeArgs,
) -> Result<()> {
    if args.structured_output() {
        bail!(
            "--json, --timestamps and --provider-options are only supported for OpenRouter transcription"
        );
    }
    let audio =
        fs::read(&args.file).with_context(|| format!("cannot read {}", args.file.display()))?;
    let format = extension(&args.file).unwrap_or_else(|| "mp3".into());
    let request = SttRequest {
        model: model.capability_name().to_owned(),
        audio,
        media_type: format!("audio/{format}"),
        language: args.language,
    };
    let mut saver = Saver::new(
        OutputArgs {
            output: None,
            overwrite: false,
        },
        "transcription",
    );
    drain(provider.transcribe(request), &mut saver, true).await
}

fn write_transcription(
    output: &mut impl Write,
    response: &TranscriptionResponse,
    structured: bool,
) -> Result<()> {
    if structured {
        serde_json::to_writer_pretty(&mut *output, response)?;
        writeln!(output)?;
    } else {
        output.write_all(response.text.as_bytes())?;
        if !response.text.is_empty() && !response.text.ends_with('\n') {
            writeln!(output)?;
        }
    }
    output.flush().context("cannot write transcription")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_files_are_not_overwritten_without_the_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("result.png");
        fs::write(&path, b"original").unwrap();
        let args = OutputArgs {
            output: Some(path.clone()),
            overwrite: false,
        };
        assert!(check_output(&args).is_err());
        assert!(Saver::new(args, "image").save("image/png", b"new").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        let args = OutputArgs {
            output: Some(path.clone()),
            overwrite: true,
        };
        assert!(check_output(&args).is_ok());
        Saver::new(args, "image").save("image/png", b"new").unwrap();
        assert_eq!(fs::read(path).unwrap(), b"new");
    }

    #[test]
    fn video_references_support_urls_and_local_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("example.mp4");
        fs::write(&path, b"video").unwrap();
        assert_eq!(
            video_url(path.to_str().unwrap()).unwrap(),
            "data:video/mp4;base64,dmlkZW8="
        );
        assert_eq!(
            video_url("https://example.com/example.mp4").unwrap(),
            "https://example.com/example.mp4"
        );
        assert!(video_url("example.txt").is_err());
    }

    #[test]
    fn numbers_extra_output_files() {
        let path = Path::new("out/cat.png");
        assert_eq!(numbered_path(path, 0), PathBuf::from("out/cat.png"));
        assert_eq!(numbered_path(path, 2), PathBuf::from("out/cat-3.png"));
    }

    #[test]
    fn maps_media_types_to_extensions() {
        assert_eq!(file_extension("image/jpeg"), "jpg");
        assert_eq!(file_extension("video/mp4"), "mp4");
        assert_eq!(file_extension("audio/mp3"), "mp3");
        assert_eq!(file_extension("image/webp; charset=binary"), "webp");
        assert_eq!(file_extension("garbage"), "bin");
    }

    #[test]
    fn keeps_remote_image_urls() {
        let url = "https://example.com/frame.png";
        assert_eq!(image_url(url).expect("URLs pass through"), url);
    }

    #[test]
    fn json_transcription_keeps_speakers_timings_usage_and_extensions() {
        let original = serde_json::json!({
            "text": "Hello. Hi.",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": 0},
                {"start": 1.5, "end": 2.5, "text": "Hi.", "speaker": 1},
            ],
            "words": [{"word": "Hello.", "start": 0.0, "end": 1.0, "speaker": 0}],
            "usage": {"seconds": 2.5, "cost": 0.0001},
            "extension": {"example": true},
        });
        let response = serde_json::from_value(original.clone()).unwrap();
        let mut output = Vec::new();
        write_transcription(&mut output, &response, true).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), original);

        output.clear();
        write_transcription(&mut output, &response, false).unwrap();
        assert_eq!(output, b"Hello. Hi.\n");
    }
}
