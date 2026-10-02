use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use clap::{Args, Parser, Subcommand, ValueEnum};
use flashmind_llm::OpenRouterProvider;
use flashmind_types::llm::{
    AudioFormat, CompletionRequest, CompletionStream, FinishReason, FrameImage, ImageGenConfig,
    ImageGenRequest, LlmProvider, StreamEvent, SttRequest, TtsRequest, VideoGenRequest,
    mime_from_extension,
};
use flashmind_types::message::{ContentPart, Message};
use flashmind_types::model::{Model, ReasoningLevel, SamplingParams};
use futures::StreamExt;
use rust_decimal::Decimal;
use serde::Deserialize;

#[derive(Parser)]
#[command(
    name = "openrouter",
    about = "Call any OpenRouter model for text, images, video or audio"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text. Prints the reply to stdout.
    Text(TextArgs),
    /// Generate images. Prints each saved path to stdout.
    Image(ImageArgs),
    /// Generate a video. Prints the saved path to stdout.
    Video(VideoArgs),
    /// Turn text into speech. Prints the saved path to stdout.
    Speech(SpeechArgs),
    /// Transcribe an audio file. Prints the text to stdout.
    Transcribe(TranscribeArgs),
    /// List model IDs, one per line.
    Models(ModelsArgs),
}

#[derive(Args)]
struct TextArgs {
    /// OpenRouter model ID, such as google/gemini-2.5-flash.
    #[arg(short, long)]
    model: String,
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
    #[arg(long, default_value = "off", value_parser = parse_name::<ReasoningLevel>)]
    reasoning: ReasoningLevel,
}

#[derive(Args)]
struct OutputArgs {
    /// Output file, or - for stdout. Defaults to a new file in the current directory.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct ImageArgs {
    #[arg(short, long)]
    model: String,
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
    /// Number of images.
    #[arg(short)]
    n: Option<u32>,
}

#[derive(Args)]
struct VideoArgs {
    #[arg(short, long)]
    model: String,
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
    #[arg(long)]
    first_frame: Option<String>,
    /// Last frame image, as a file path or URL.
    #[arg(long)]
    last_frame: Option<String>,
    /// Style reference image, as a file path or URL. Repeatable.
    #[arg(long)]
    reference: Vec<String>,
}

#[derive(Args)]
struct SpeechArgs {
    #[arg(short, long)]
    model: String,
    /// Text to speak. Reads stdin when omitted.
    text: Option<String>,
    #[command(flatten)]
    output: OutputArgs,
    #[arg(long)]
    voice: String,
    /// Audio format: mp3, wav, opus, aac or flac.
    #[arg(long, default_value = "mp3", value_parser = parse_name::<AudioFormat>)]
    format: AudioFormat,
}

#[derive(Args)]
struct TranscribeArgs {
    #[arg(short, long)]
    model: String,
    /// Audio file to transcribe.
    file: PathBuf,
    /// Language hint, such as en.
    #[arg(long)]
    language: Option<String>,
}

#[derive(Args)]
struct ModelsArgs {
    #[arg(value_enum, default_value = "text")]
    kind: ModelKind,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelKind {
    Text,
    Image,
    Video,
    Speech,
    Transcription,
}

/// Parses a strum enum. Its error type is not a std error, which clap needs.
fn parse_name<T: FromStr>(value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("unknown value {value}"))
}

#[derive(Deserialize)]
struct Config {
    openrouter_key: String,
}

fn api_key() -> Result<String> {
    let path = dirs::home_dir()
        .context("cannot determine home directory")?
        .join(".openrouter-cli");
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return std::env::var("OPENROUTER_API_KEY")
                .ok()
                .filter(|key| !key.trim().is_empty())
                .context("set openrouter_key in ~/.openrouter-cli or OPENROUTER_API_KEY");
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
}

impl Saver {
    fn new(output: OutputArgs, prefix: &'static str) -> Self {
        Self {
            output: output.output,
            prefix,
            count: 0,
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
        fs::write(&path, data).with_context(|| format!("cannot write {}", path.display()))?;
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

fn model(id: &str) -> Result<Model> {
    format!("openrouter:{id}")
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid model {id}: {error}"))
}

async fn text(provider: &OpenRouterProvider, args: TextArgs) -> Result<()> {
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
    let request = CompletionRequest {
        model: model(&args.model)?,
        messages,
        tools: Vec::new(),
        max_tokens: args.max_tokens,
        reasoning: args.reasoning,
        sampling: SamplingParams {
            temperature: args.temperature,
            ..SamplingParams::default()
        },
        modalities: Vec::new(),
        audio_config: None,
        image_config: None,
        user: None,
        provider_preferences: None,
    };
    let mut saver = Saver::new(OutputArgs { output: None }, "output");
    drain(provider.complete(request), &mut saver, true).await
}

async fn image(provider: &OpenRouterProvider, args: ImageArgs) -> Result<()> {
    let request = ImageGenRequest {
        model: args.model,
        prompt: input_text(args.prompt)?,
        size: args.size,
        aspect_ratio: args.aspect_ratio,
        quality: args.quality,
        style: None,
        n: args.n,
    };
    let mut saver = Saver::new(args.output, "image");
    drain(provider.generate_image(request), &mut saver, false).await
}

async fn video(provider: &OpenRouterProvider, args: VideoArgs) -> Result<()> {
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
        model: args.model,
        description: input_text(args.prompt)?,
        resolution: args.resolution,
        aspect_ratio: args.aspect_ratio,
        duration: args.duration,
        generate_audio: args.audio.then_some(true),
        frame_images,
        input_references,
    };
    let mut saver = Saver::new(args.output, "video");
    drain(provider.generate_video(request), &mut saver, false).await
}

async fn speech(provider: &OpenRouterProvider, args: SpeechArgs) -> Result<()> {
    let request = TtsRequest {
        model: args.model,
        input: input_text(args.text)?,
        voice: args.voice,
        response_format: args.format,
    };
    let mut saver = Saver::new(args.output, "speech");
    drain(provider.text_to_speech(request), &mut saver, false).await
}

async fn transcribe(provider: &OpenRouterProvider, args: TranscribeArgs) -> Result<()> {
    let audio =
        fs::read(&args.file).with_context(|| format!("cannot read {}", args.file.display()))?;
    let format = extension(&args.file).unwrap_or_else(|| "mp3".into());
    let request = SttRequest {
        model: args.model,
        audio,
        media_type: format!("audio/{format}"),
        language: args.language,
    };
    let mut saver = Saver::new(OutputArgs { output: None }, "output");
    drain(provider.transcribe(request), &mut saver, true).await
}

async fn models(provider: &OpenRouterProvider, args: ModelsArgs) -> Result<()> {
    let ids: Vec<String> = match args.kind {
        ModelKind::Image => provider
            .list_image_models()
            .await?
            .into_iter()
            .map(|model| model.id)
            .collect(),
        ModelKind::Video => provider
            .list_video_models()
            .await?
            .into_iter()
            .map(|model| model.id)
            .collect(),
        ModelKind::Speech => provider
            .list_speech_models()
            .await?
            .into_iter()
            .map(|model| model.id)
            .collect(),
        ModelKind::Transcription => provider
            .list_transcription_models()
            .await?
            .into_iter()
            .map(|model| model.id)
            .collect(),
        ModelKind::Text => provider
            .list_models()
            .await
            .context("cannot list models")?
            .into_iter()
            .map(|model| model.id)
            .collect(),
    };
    for id in ids {
        println!("{id}");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let provider = OpenRouterProvider::new(api_key()?);
    match cli.command {
        Command::Text(args) => text(&provider, args).await,
        Command::Image(args) => image(&provider, args).await,
        Command::Video(args) => video(&provider, args).await,
        Command::Speech(args) => speech(&provider, args).await,
        Command::Transcribe(args) => transcribe(&provider, args).await,
        Command::Models(args) => models(&provider, args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parses_text_command() {
        let cli = Cli::try_parse_from([
            "openrouter",
            "text",
            "-m",
            "openai/gpt-5",
            "--reasoning",
            "high",
            "--temperature",
            "0.2",
            "hello",
        ])
        .expect("valid command line");
        let Command::Text(args) = cli.command else {
            panic!("expected the text command");
        };
        assert_eq!(args.reasoning, ReasoningLevel::High);
        assert_eq!(args.prompt.as_deref(), Some("hello"));
        assert_eq!(
            model(&args.model).expect("valid model").name(),
            "openai/gpt-5"
        );
    }
}
