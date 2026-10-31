use std::time::Duration;

use anyhow::{Context, Result, bail};
use flashmind_types::{CompletionStream, FinishReason, StreamEvent, VideoGenRequest};
use reqwest::{RequestBuilder, header};
use serde::{Deserialize, Serialize};
use tokio::time::{Instant, sleep, timeout_at};
use url::Url;

use crate::http::http_client;

const POLL_INTERVAL: Duration = Duration::from_secs(5);
const JOB_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Serialize)]
struct QueueRequest {
    model: String,
    prompt: String,
    duration: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

impl TryFrom<VideoGenRequest> for QueueRequest {
    type Error = anyhow::Error;

    fn try_from(request: VideoGenRequest) -> Result<Self> {
        let duration = request
            .duration
            .context("Venice video requires --duration in seconds")?;
        if !request.input_references.is_empty()
            || !request.video_references.is_empty()
            || request.frame_images.len() > 1
        {
            bail!("Venice video supports only one first frame in this command");
        }
        let image_url = match request.frame_images.into_iter().next() {
            Some(frame) if frame.frame_type == "first_frame" => Some(frame.url),
            Some(_) => bail!("Venice video supports only a first frame in this command"),
            None => None,
        };
        Ok(Self {
            model: request.model,
            prompt: request.description,
            duration: format!("{duration}s"),
            resolution: request.resolution,
            aspect_ratio: request.aspect_ratio,
            audio: request.generate_audio,
            image_url,
        })
    }
}

#[derive(Deserialize)]
struct QueuedVideo {
    model: String,
    queue_id: String,
    download_url: Option<Url>,
}

#[derive(Deserialize)]
struct VideoStatus {
    status: String,
}

pub(crate) fn generate(
    queue: RequestBuilder,
    retrieve: RequestBuilder,
    request: VideoGenRequest,
    extra: serde_json::Map<String, serde_json::Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        let request = QueueRequest::try_from(request)?;
        let request = crate::http::add_extra_args(serde_json::to_value(request)?, &extra)?;
        let response = queue.json(&request).send().await.context("Venice video queue request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read Venice queue response")?;
        if !status.is_success() {
            Err(anyhow::anyhow!("Venice video queue failed ({status}): {}", String::from_utf8_lossy(&bytes)))?;
        }
        let job: QueuedVideo = serde_json::from_slice(&bytes).context("invalid Venice queue response")?;
        yield StreamEvent::ContentDelta(format!("Venice video job: {} (model {})\n", job.queue_id, job.model));
        let deadline = Instant::now() + JOB_TIMEOUT;
        let body = serde_json::json!({"model": job.model, "queue_id": job.queue_id});
        loop {
            let builder = retrieve.try_clone().context("cannot clone Venice video status request")?.json(&body);
            let response = timeout_at(deadline, builder.send()).await.context("Venice video polling timed out")?.context("Venice video status request failed")?;
            let status = response.status();
            let is_video = response.headers().get(header::CONTENT_TYPE).and_then(|value| value.to_str().ok()).is_some_and(|value| value.starts_with("video/mp4"));
            let bytes = timeout_at(deadline, response.bytes()).await.context("Venice video download timed out")?.context("cannot read Venice video response")?;
            if !status.is_success() {
                Err(anyhow::anyhow!("Venice video retrieval failed ({status}): {}", String::from_utf8_lossy(&bytes)))?;
            }
            if is_video {
                yield StreamEvent::FileAttachment { filename: String::new(), media_type: "video/mp4".into(), data: bytes.to_vec() };
                break;
            }
            let progress: VideoStatus = serde_json::from_slice(&bytes).context("invalid Venice video status")?;
            match progress.status.as_str() {
                "COMPLETED" => {
                    let url = job.download_url.as_ref().context("Venice completed the job without a download URL")?;
                    if url.scheme() != "https" { Err(anyhow::anyhow!("Venice returned an insecure video download URL"))?; }
                    let response = timeout_at(deadline, http_client().get(url.clone()).send()).await.context("Venice video download timed out")?.map_err(|error| error.without_url()).context("Venice video download failed")?.error_for_status().map_err(|error| error.without_url()).context("Venice video download failed")?;
                    let bytes = timeout_at(deadline, response.bytes()).await.context("Venice video download timed out")?.context("cannot read Venice video")?;
                    yield StreamEvent::FileAttachment { filename: String::new(), media_type: "video/mp4".into(), data: bytes.to_vec() };
                    break;
                }
                "FAILED" | "ERROR" | "CANCELLED" => Err(anyhow::anyhow!("Venice video job {}: {}", job.queue_id, progress.status))?,
                "QUEUED" | "PENDING" | "PROCESSING" | "IN_PROGRESS" => {},
                other => Err(anyhow::anyhow!("unknown Venice video status: {other}"))?,
            }
            timeout_at(deadline, sleep(POLL_INTERVAL)).await.context("Venice video polling timed out")?;
        }
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;
    use flashmind_types::FrameImage;
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn queues_and_retrieves_video_without_repeating_generation() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let client = reqwest::Client::new();
        let request = VideoGenRequest {
            model: "example-model".into(),
            description: "example prompt".into(),
            resolution: Some("720p".into()),
            aspect_ratio: Some("3:4".into()),
            duration: Some(5),
            generate_audio: Some(true),
            frame_images: vec![FrameImage {
                frame_type: "first_frame".into(),
                url: "https://example.com/image.png".into(),
            }],
            input_references: vec![],
            video_references: vec![],
        };
        let server = async {
            for (path, response_type, response) in [
                (
                    "/queue",
                    "application/json",
                    r#"{"model":"example-model","queue_id":"example-job"}"#,
                ),
                ("/retrieve", "video/mp4", "video-data"),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    headers.push(socket.read_u8().await.unwrap());
                }
                let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
                assert!(headers.starts_with(&format!("post {path} http/1.1")));
                let length: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap()
                    .parse()
                    .unwrap();
                let mut body = vec![0; length];
                socket.read_exact(&mut body).await.unwrap();
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                if path == "/queue" {
                    assert_eq!(body["duration"], "5s");
                    assert_eq!(body["aspect_ratio"], "3:4");
                    assert_eq!(body["image_url"], "https://example.com/image.png");
                } else {
                    assert_eq!(body["queue_id"], "example-job");
                }
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {response_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
            }
        };
        let (events, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(
                generate(
                    client.post(format!("{base}/queue")),
                    client.post(format!("{base}/retrieve")),
                    request,
                    Default::default(),
                )
                .collect::<Vec<_>>(),
                server
            )
        })
        .await
        .expect("The local endpoint should complete the request.");
        let events: Vec<_> = events.into_iter().collect::<Result<_, _>>().unwrap();
        assert!(events.iter().any(|event| matches!(event, StreamEvent::FileAttachment { data, .. } if data == b"video-data")));
    }
}
