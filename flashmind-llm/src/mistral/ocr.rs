use anyhow::{Context, ensure};
use reqwest::RequestBuilder;
use serde::Serialize;
use serde_json::{Map, Value};

/// A publicly accessible URL or data URL for Mistral OCR.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl { document_url: String },
    ImageUrl { image_url: String },
}

/// A Mistral OCR request with optional provider-specific fields.
#[derive(Serialize)]
pub struct OcrRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub options: Map<String, Value>,
}

pub(super) async fn process(
    builder: RequestBuilder,
    request: &OcrRequest,
) -> anyhow::Result<Value> {
    ensure!(
        !request.options.contains_key("model") && !request.options.contains_key("document"),
        "model and document are controlled by the OCR command"
    );
    let response = builder
        .json(request)
        .send()
        .await
        .context("Mistral OCR request failed")?;
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .context("cannot read Mistral OCR response")?;
    ensure!(
        status.is_success(),
        "Mistral OCR failed ({status}): {}",
        String::from_utf8_lossy(&bytes)
    );
    serde_json::from_slice(&bytes).context("invalid Mistral OCR response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn sends_native_ocr_request_and_returns_pages() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = format!("http://{}/v1/ocr", listener.local_addr().unwrap());
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
            assert!(headers.starts_with("post /v1/ocr "));
            let len: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0; len];
            socket.read_exact(&mut body).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["document"]["type"], "image_url");
            assert_eq!(
                body["document"]["image_url"],
                "https://example.com/image.png"
            );
            assert_eq!(body["table_format"], "html");
            let response = r#"{"pages":[{"markdown":"hello"}]}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
        };
        let request = OcrRequest {
            model: "example-model".into(),
            document: OcrDocument::ImageUrl {
                image_url: "https://example.com/image.png".into(),
            },
            options: serde_json::from_value(serde_json::json!({"table_format":"html"})).unwrap(),
        };
        let (response, ()) =
            tokio::join!(process(reqwest::Client::new().post(url), &request), server);
        assert_eq!(response.unwrap()["pages"][0]["markdown"], "hello");
    }
}
