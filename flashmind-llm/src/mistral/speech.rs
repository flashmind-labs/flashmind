use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_types::{AudioFormat, CompletionStream, FinishReason, StreamEvent, TtsRequest};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct SpeechResponse {
    audio_data: String,
}

pub(super) fn generate(builder: RequestBuilder, request: TtsRequest) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        if matches!(request.response_format, AudioFormat::Aac) {
            Err(anyhow::anyhow!("Mistral speech does not support AAC"))?;
        }
        let format = request.response_format.to_string();
        let body = json!({"model": request.model, "input": request.input, "voice_id": request.voice, "response_format": format, "stream": false});
        let response = builder.json(&body).send().await.context("Mistral speech request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read Mistral speech response")?;
        if !status.is_success() {
            Err(anyhow::anyhow!("Mistral speech failed ({status}): {}", String::from_utf8_lossy(&bytes)))?;
        }
        let response: SpeechResponse = serde_json::from_slice(&bytes).context("invalid Mistral speech response")?;
        let data = STANDARD.decode(response.audio_data).context("invalid Mistral speech audio")?;
        yield StreamEvent::AudioDelta { data, format };
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn sends_voice_id_and_decodes_audio_response() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = format!("http://{}/v1/audio/speech", listener.local_addr().unwrap());
        let request = TtsRequest {
            model: "example-model".into(),
            input: "hello".into(),
            voice: "example-voice".into(),
            response_format: AudioFormat::Mp3,
        };
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
            assert!(headers.starts_with("post /v1/audio/speech http/1.1"));
            let len: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0; len];
            socket.read_exact(&mut body).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["voice_id"], "example-voice");
            assert!(body.get("voice").is_none());
            let response = r#"{"audio_data":"aGVsbG8="}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
        };
        let (events, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(
                generate(reqwest::Client::new().post(url), request).collect::<Vec<_>>(),
                server
            )
        })
        .await
        .expect("The local endpoint should complete the request.");
        let events: Vec<_> = events
            .into_iter()
            .collect::<anyhow::Result<_, _>>()
            .unwrap();
        assert!(events.iter().any(|event| matches!(event, StreamEvent::AudioDelta { data, format } if data == b"hello" && format == "mp3")));
    }
}
