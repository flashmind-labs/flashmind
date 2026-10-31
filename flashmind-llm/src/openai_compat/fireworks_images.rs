use anyhow::Context;
use flashmind_types::{
    CompletionStream, FinishReason, ImageGenRequest, ImageOutputFormat, StreamEvent,
};
use reqwest::RequestBuilder;
use serde::Serialize;

#[derive(Serialize)]
struct FireworksImageRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
}

pub(crate) fn generate(
    builder: RequestBuilder,
    request: ImageGenRequest,
    extra: serde_json::Map<String, serde_json::Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        if request.model != "flux-1-schnell-fp8" {
            Err(anyhow::anyhow!("Fireworks image generation currently supports flux-1-schnell-fp8"))?;
        }
        if !request.input_references.is_empty() { Err(anyhow::anyhow!("Fireworks FLUX image generation does not support reference images"))?; }
        if request.output_format.is_some_and(|format| format != ImageOutputFormat::Png) { Err(anyhow::anyhow!("Fireworks FLUX supports only PNG output"))?; }
        if request.size.is_some() || request.quality.is_some() || request.style.is_some() || request.n.is_some_and(|n| n != 1) {
            Err(anyhow::anyhow!("Fireworks FLUX supports --aspect-ratio and one image per request; size, quality and style are unavailable"))?;
        }
        let body = FireworksImageRequest { prompt: request.prompt, aspect_ratio: request.aspect_ratio };
        let body = crate::http::add_extra_args(serde_json::to_value(body)?, &extra)?;
        let response = builder.header(reqwest::header::ACCEPT, "image/png").json(&body).send().await.context("Fireworks image request failed")?;
        let status = response.status();
        let content_type = response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|h| h.to_str().ok()).unwrap_or("").to_owned();
        let bytes = response.bytes().await.context("cannot read Fireworks image response")?;
        if !status.is_success() {
            Err(anyhow::anyhow!("Fireworks image generation failed ({status}): {}", String::from_utf8_lossy(&bytes)))?;
        }
        if !content_type.starts_with("image/png") {
            Err(anyhow::anyhow!("Fireworks returned an unexpected image content type: {content_type}"))?;
        }
        yield StreamEvent::FileAttachment { filename: String::new(), media_type: "image/png".into(), data: bytes.to_vec() };
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
    async fn sends_ratio_and_returns_png() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let request = ImageGenRequest {
            output_format: None,
            model: "flux-1-schnell-fp8".into(),
            prompt: "example".into(),
            size: None,
            aspect_ratio: Some("3:4".into()),
            quality: None,
            style: None,
            n: None,
            input_references: Vec::new(),
        };
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
            assert!(headers.contains("accept: image/png\r\n"));
            let len: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0; len];
            socket.read_exact(&mut body).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["aspect_ratio"], "3:4");
            let response = "example-png";
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
        };
        let (events, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(
                generate(
                    reqwest::Client::new().post(url),
                    request,
                    Default::default()
                )
                .collect::<Vec<_>>(),
                server
            )
        })
        .await
        .expect("The local endpoint should complete the request.");
        let events: Vec<_> = events
            .into_iter()
            .collect::<anyhow::Result<_, _>>()
            .unwrap();
        assert!(events.iter().any(|event| matches!(event, StreamEvent::FileAttachment { data, .. } if data == b"example-png")));
    }
}
