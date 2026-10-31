use std::time::Duration;

use anyhow::{Context, Result, bail};
use flashmind_types::{CompletionStream, FinishReason, StreamEvent, VideoGenRequest};
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use tokio::time::{Instant, sleep, timeout_at};
use url::Url;

use crate::http::http_client;

const JOB_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const POLL_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Serialize)]
struct VideoMedia {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    frame_images: Vec<VideoFrame>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    reference_images: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    reference_videos: Vec<VideoReference>,
}

#[derive(Serialize)]
struct VideoReference {
    video: String,
}

#[derive(Serialize)]
struct VideoFrame {
    input_image: String,
    frame: u32,
}

#[derive(Serialize)]
struct CreateVideo {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media: Option<VideoMedia>,
}

impl TryFrom<VideoGenRequest> for CreateVideo {
    type Error = anyhow::Error;

    fn try_from(request: VideoGenRequest) -> Result<Self> {
        let mut frames = Vec::new();
        for frame in request.frame_images {
            if frame.frame_type != "first_frame" {
                bail!("Together video currently supports --first-frame, but not --last-frame");
            }
            frames.push(VideoFrame {
                input_image: frame.url,
                frame: 0,
            });
        }
        let media = if frames.is_empty()
            && request.input_references.is_empty()
            && request.video_references.is_empty()
        {
            None
        } else {
            Some(VideoMedia {
                frame_images: frames,
                reference_images: request.input_references,
                reference_videos: request
                    .video_references
                    .into_iter()
                    .map(|video| VideoReference { video })
                    .collect(),
            })
        };
        Ok(Self {
            model: request.model,
            prompt: request.description,
            resolution: request.resolution,
            ratio: request.aspect_ratio,
            seconds: request.duration.map(|seconds| seconds.to_string()),
            generate_audio: request.generate_audio,
            media,
        })
    }
}

#[derive(Deserialize)]
struct VideoJob {
    id: String,
    status: String,
    outputs: Option<VideoOutput>,
    error: Option<VideoError>,
}

#[derive(Deserialize)]
struct VideoOutput {
    video_url: Url,
}

#[derive(Deserialize)]
struct VideoError {
    message: String,
}

pub(crate) fn generate(
    create: RequestBuilder,
    retrieve: RequestBuilder,
    request: VideoGenRequest,
    extra: serde_json::Map<String, serde_json::Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        let request = CreateVideo::try_from(request)?;
        let request = crate::http::add_extra_args(serde_json::to_value(request)?, &extra)?;
        let response = create.json(&request).send().await.context("Together video request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read Together video response")?;
        if !status.is_success() { Err(anyhow::anyhow!("Together video request failed ({status}): {}", String::from_utf8_lossy(&bytes)))?; }
        let mut job: VideoJob = serde_json::from_slice(&bytes).context("invalid Together video response")?;
        yield StreamEvent::ContentDelta(format!("Together video job: {}\n", job.id));
        let deadline = Instant::now() + JOB_TIMEOUT;
        loop {
            match job.status.as_str() {
                "completed" => {
                    let output = job.outputs.context("Together completed the job without a video URL")?;
                    if output.video_url.scheme() != "https" { Err(anyhow::anyhow!("Together returned an insecure video URL"))?; }
                    let response = timeout_at(deadline, http_client().get(output.video_url).send()).await.context("Together video download timed out")?.map_err(|error| error.without_url()).context("Together video download failed")?.error_for_status().map_err(|error| error.without_url()).context("Together video download failed")?;
                    let bytes = timeout_at(deadline, response.bytes()).await.context("Together video download timed out")?.map_err(|error| error.without_url()).context("cannot read Together video")?;
                    yield StreamEvent::FileAttachment { filename: String::new(), media_type: "video/mp4".into(), data: bytes.to_vec() };
                    break;
                }
                "failed" | "cancelled" => {
                    let detail = job.error.map(|error| error.message).unwrap_or_else(|| job.status.clone());
                    Err(anyhow::anyhow!("Together video job {} failed: {detail}", job.id))?;
                }
                "queued" | "in_progress" => {},
                other => Err(anyhow::anyhow!("unknown Together video status: {other}"))?,
            }
            timeout_at(deadline, sleep(POLL_INTERVAL)).await.context("Together video polling timed out")?;
            let mut request = retrieve.try_clone().context("cannot clone Together video status request")?.build().context("cannot build Together video status request")?;
            request.url_mut().path_segments_mut().map_err(|_| anyhow::anyhow!("invalid Together video URL"))?.push(&job.id);
            let response = timeout_at(deadline, http_client().execute(request)).await.context("Together video polling timed out")?.context("Together video status request failed")?;
            let status = response.status();
            let bytes = timeout_at(deadline, response.bytes()).await.context("Together video polling timed out")?.context("cannot read Together video status")?;
            if !status.is_success() { Err(anyhow::anyhow!("Together video status failed ({status}): {}", String::from_utf8_lossy(&bytes)))?; }
            job = serde_json::from_slice(&bytes).context("invalid Together video status")?;
        }
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn sends_native_video_parameters_and_reports_failure() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = format!("http://{}/v2/videos", listener.local_addr().unwrap());
        let client = reqwest::Client::new();
        let request = VideoGenRequest {
            model: "example/model".into(),
            description: "example".into(),
            resolution: Some("720p".into()),
            aspect_ratio: Some("16:9".into()),
            duration: Some(5),
            generate_audio: Some(true),
            frame_images: vec![],
            input_references: vec![],
            video_references: vec![],
        };
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
            assert!(headers.starts_with("post /v2/videos http/1.1"));
            let len: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0; len];
            socket.read_exact(&mut body).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["seconds"], "5");
            assert_eq!(body["ratio"], "16:9");
            let response =
                r#"{"id":"example-job","status":"failed","error":{"message":"example failure"}}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
        };
        let (events, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(
                generate(
                    client.post(&url),
                    client.get(&url),
                    request,
                    Default::default()
                )
                .collect::<Vec<_>>(),
                server
            )
        })
        .await
        .expect("The local endpoint should complete the request.");
        assert!(events.iter().any(|event| {
            event
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("example failure"))
        }));
    }
}
