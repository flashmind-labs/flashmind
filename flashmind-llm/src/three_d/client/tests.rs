use std::net::Ipv4Addr;

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct Request {
    line: String,
    headers: String,
    body: Vec<u8>,
}

async fn read_request(socket: &mut TcpStream) -> Request {
    let mut header_bytes = Vec::new();
    while !header_bytes.ends_with(b"\r\n\r\n") {
        header_bytes.push(socket.read_u8().await.unwrap());
        assert!(header_bytes.len() < 16384);
    }
    let headers = String::from_utf8(header_bytes)
        .unwrap()
        .to_ascii_lowercase();
    let len: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length: "))
        .unwrap_or("0")
        .parse()
        .unwrap();
    let mut body = vec![0; len];
    socket.read_exact(&mut body).await.unwrap();
    Request {
        line: headers.lines().next().unwrap().to_owned(),
        headers,
        body,
    }
}

async fn respond(socket: &mut TcpStream, json: &str) {
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}", json.len()).as_bytes()).await.unwrap();
}

fn client(provider: ThreeDProvider, listener: &TcpListener) -> ThreeDClient {
    let url = Url::parse(&format!(
        "http://{}/gateway",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    ThreeDClient::new(provider, "example-key")
        .unwrap()
        .with_base_url(url)
}

fn test_model(provider: ThreeDProvider) -> Model {
    let name = match provider {
        ThreeDProvider::Meshy => "meshy:example-model",
        ThreeDProvider::Tripo => "tripo:example-model",
    };
    name.parse().unwrap()
}

#[tokio::test]
async fn meshy_text_preview_and_explicit_refine_produce_glb() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let client = client(ThreeDProvider::Meshy, &listener);
    let request = ThreeDRequest {
        model: "meshy:example-model".parse().unwrap(),
        input: ThreeDInput::Text("example prompt".into()),
        texture: true,
        extra_args: serde_json::from_value(json!({"target_polycount": 5000})).unwrap(),
    };
    let server = async {
        for step in 0..4 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let req = read_request(&mut socket).await;
            assert!(
                req.headers
                    .contains("authorization: bearer example-key\r\n")
            );
            let response = match step {
                0 => {
                    assert_eq!(req.line, "post /gateway/openapi/v2/text-to-3d http/1.1");
                    let body: Value = serde_json::from_slice(&req.body).unwrap();
                    assert_eq!(body["mode"], "preview");
                    assert_eq!(body["target_polycount"], 5000);
                    r#"{"result":"preview-id"}"#
                }
                1 => {
                    assert_eq!(
                        req.line,
                        "get /gateway/openapi/v2/text-to-3d/preview-id http/1.1"
                    );
                    r#"{"status":"SUCCEEDED","model_urls":{"glb":"https://example.com/preview.glb"}}"#
                }
                2 => {
                    let body: Value = serde_json::from_slice(&req.body).unwrap();
                    assert_eq!(body["mode"], "refine");
                    assert_eq!(body["preview_task_id"], "preview-id");
                    r#"{"result":"refine-id"}"#
                }
                _ => {
                    assert_eq!(
                        req.line,
                        "get /gateway/openapi/v2/text-to-3d/refine-id http/1.1"
                    );
                    r#"{"status":"SUCCEEDED","model_urls":{"glb":"https://example.com/final.glb"}}"#
                }
            };
            respond(&mut socket, response).await;
        }
    };
    let run = async {
        let mut ids = Vec::new();
        let url = client
            .generate(request, |id| ids.push(id.to_owned()))
            .await
            .unwrap();
        assert_eq!(url.as_str(), "https://example.com/final.glb");
        assert_eq!(ids, ["preview-id", "refine-id"]);
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(run, server);
    })
    .await
    .expect("The local endpoint should complete the tasks.");
}

#[tokio::test]
async fn tripo_uploads_local_image_then_submits_and_retrieves_model() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let client = client(ThreeDProvider::Tripo, &listener);
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("example.png");
    fs::write(&image, b"example-image").await.unwrap();
    let request = ThreeDRequest {
        model: "tripo:example-model".parse().unwrap(),
        input: ThreeDInput::Images(vec![ThreeDImage::File(image)]),
        texture: false,
        extra_args: Map::new(),
    };
    let server = async {
        for step in 0..3 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let req = read_request(&mut socket).await;
            assert!(
                req.headers
                    .contains("authorization: bearer example-key\r\n")
            );
            let response = match step {
                0 => {
                    assert_eq!(req.line, "post /gateway/v3/files http/1.1");
                    assert!(req.headers.contains("multipart/form-data"));
                    assert!(
                        req.body
                            .windows(b"example-image".len())
                            .any(|window| window == b"example-image")
                    );
                    r#"{"code":0,"data":{"file_token":"example-token"}}"#
                }
                1 => {
                    assert_eq!(
                        req.line,
                        "post /gateway/v3/generation/image-to-model http/1.1"
                    );
                    let body: Value = serde_json::from_slice(&req.body).unwrap();
                    assert_eq!(body["input"], "example-token");
                    assert_eq!(body["texture"], false);
                    assert_eq!(body["pbr"], false);
                    r#"{"code":0,"data":{"task_id":"example-task"}}"#
                }
                _ => {
                    assert_eq!(req.line, "get /gateway/v3/tasks/example-task http/1.1");
                    r#"{"code":0,"data":{"status":"success","output":{"model_url":"https://example.com/model.glb"}}}"#
                }
            };
            respond(&mut socket, response).await;
        }
    };
    let run = async {
        let mut ids = Vec::new();
        let url = client
            .generate(request, |id| ids.push(id.to_owned()))
            .await
            .unwrap();
        assert_eq!(url.as_str(), "https://example.com/model.glb");
        assert_eq!(ids, ["example-task"]);
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(run, server);
    })
    .await
    .expect("The local endpoint should complete the task.");
}

#[tokio::test]
async fn meshy_image_and_tripo_text_use_their_native_fields() {
    for provider in [ThreeDProvider::Meshy, ThreeDProvider::Tripo] {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let client = client(provider, &listener);
        let input = match provider {
            ThreeDProvider::Meshy => ThreeDInput::Images(vec![ThreeDImage::Url(
                Url::parse("https://example.com/image.png").unwrap(),
            )]),
            ThreeDProvider::Tripo => ThreeDInput::Text("example prompt".into()),
        };
        let request = ThreeDRequest {
            model: test_model(provider),
            input,
            texture: true,
            extra_args: Map::new(),
        };
        let server = async {
            for step in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let req = read_request(&mut socket).await;
                let response = match (provider, step) {
                    (ThreeDProvider::Meshy, 0) => {
                        assert_eq!(req.line, "post /gateway/openapi/v1/image-to-3d http/1.1");
                        let body: Value = serde_json::from_slice(&req.body).unwrap();
                        assert_eq!(body["image_url"], "https://example.com/image.png");
                        assert_eq!(body["should_texture"], true);
                        r#"{"result":"example-task"}"#
                    }
                    (ThreeDProvider::Meshy, _) => {
                        r#"{"status":"SUCCEEDED","model_urls":{"glb":"https://example.com/model.glb"}}"#
                    }
                    (ThreeDProvider::Tripo, 0) => {
                        assert_eq!(
                            req.line,
                            "post /gateway/v3/generation/text-to-model http/1.1"
                        );
                        let body: Value = serde_json::from_slice(&req.body).unwrap();
                        assert_eq!(body["prompt"], "example prompt");
                        assert_eq!(body["texture"], true);
                        r#"{"code":0,"data":{"task_id":"example-task"}}"#
                    }
                    (ThreeDProvider::Tripo, _) => {
                        r#"{"code":0,"data":{"status":"success","output":{"model_url":"https://example.com/model.glb"}}}"#
                    }
                };
                respond(&mut socket, response).await;
            }
        };
        let run = async {
            let url = client.generate(request, |_| {}).await.unwrap();
            assert_eq!(url.as_str(), "https://example.com/model.glb");
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(run, server);
        })
        .await
        .expect("The local endpoint should complete the task.");
    }
}

#[tokio::test]
async fn multi_image_requests_use_each_providers_native_shape() {
    for provider in [ThreeDProvider::Meshy, ThreeDProvider::Tripo] {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let client = client(provider, &listener);
        let images = [
            "https://example.com/front.png",
            "https://example.com/left.png",
        ]
        .into_iter()
        .map(|url| ThreeDImage::Url(Url::parse(url).unwrap()))
        .collect();
        let request = ThreeDRequest {
            model: test_model(provider),
            input: ThreeDInput::Images(images),
            texture: false,
            extra_args: Map::new(),
        };
        let server = async {
            for step in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let req = read_request(&mut socket).await;
                let response = match (provider, step) {
                    (ThreeDProvider::Meshy, 0) => {
                        assert_eq!(
                            req.line,
                            "post /gateway/openapi/v1/multi-image-to-3d http/1.1"
                        );
                        let body: Value = serde_json::from_slice(&req.body).unwrap();
                        assert_eq!(body["image_urls"].as_array().unwrap().len(), 2);
                        r#"{"result":"example-task"}"#
                    }
                    (ThreeDProvider::Meshy, _) => {
                        r#"{"status":"SUCCEEDED","model_urls":{"glb":"https://example.com/model.glb"}}"#
                    }
                    (ThreeDProvider::Tripo, 0) => {
                        assert_eq!(
                            req.line,
                            "post /gateway/v3/generation/multiview-to-model http/1.1"
                        );
                        let body: Value = serde_json::from_slice(&req.body).unwrap();
                        assert_eq!(body["inputs"][0]["front"], "https://example.com/front.png");
                        assert_eq!(body["inputs"][1]["left"], "https://example.com/left.png");
                        r#"{"code":0,"data":{"task_id":"example-task"}}"#
                    }
                    (ThreeDProvider::Tripo, _) => {
                        r#"{"code":0,"data":{"status":"success","output":{"model_url":"https://example.com/model.glb"}}}"#
                    }
                };
                respond(&mut socket, response).await;
            }
        };
        let run = async {
            assert!(client.generate(request, |_| {}).await.is_ok());
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(run, server);
        })
        .await
        .expect("The local endpoint should complete the task.");
    }
}

#[tokio::test]
async fn downloads_are_streamed_and_insecure_urls_are_rejected() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let url = format!("http://{}/model", listener.local_addr().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("model.glb");
    let client = ThreeDClient::new(ThreeDProvider::Meshy, "example-key").unwrap();
    let mut file = File::create(&output).await.unwrap();
    assert!(
        client
            .save(&Url::parse(&url).unwrap(), &mut file)
            .await
            .is_err()
    );
    let server = async {
        let (mut socket, _) = listener.accept().await.unwrap();
        let req = read_request(&mut socket).await;
        assert!(!req.headers.contains("authorization:"));
        let data = "example-glb";
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: model/gltf-binary\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len()).as_bytes()).await.unwrap();
    };
    let run = async {
        let response = client.http.get(url).send().await.unwrap();
        write_model(response, &mut file).await.unwrap();
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(run, server);
    })
    .await
    .expect("The local endpoint should complete the download.");
    drop(file);
    assert_eq!(fs::read(output).await.unwrap(), b"example-glb");
}

#[tokio::test]
async fn validates_every_local_image_before_uploading() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.png");
    fs::write(&first, b"image").await.unwrap();
    let missing = dir.path().join("missing.png");
    let client = ThreeDClient::new(ThreeDProvider::Tripo, "example-key")
        .unwrap()
        .with_base_url(Url::parse("https://example.invalid/").unwrap());
    let request = ThreeDRequest {
        model: "tripo:example-model".parse().unwrap(),
        input: ThreeDInput::Images(vec![ThreeDImage::File(first), ThreeDImage::File(missing)]),
        texture: false,
        extra_args: Map::new(),
    };
    let error = client.generate(request, |_| {}).await.unwrap_err();
    assert!(error.to_string().contains("cannot read"));
}

#[tokio::test]
async fn rejects_conflicting_fields_before_submitting() {
    let client = ThreeDClient::new(ThreeDProvider::Tripo, "example-key").unwrap();
    let request = ThreeDRequest {
        model: "tripo:example-model".parse().unwrap(),
        input: ThreeDInput::Images(vec![ThreeDImage::File(PathBuf::from("missing.png"))]),
        texture: false,
        extra_args: serde_json::from_value(json!({"pbr": true})).unwrap(),
    };
    let error = client.generate(request, |_| {}).await.unwrap_err();
    assert!(error.to_string().contains("pbr requires --texture"));
}
