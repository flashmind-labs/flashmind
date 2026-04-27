//! JSON-RPC 2.0 wire types for MCP (Model Context Protocol) communication.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request identifier — either a number or a string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RequestId {
    Num(u64),
    Str(String),
}

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: RequestId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC 2.0 request.
    pub fn new(id: RequestId, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub id: Option<RequestId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// Client capabilities sent during initialization (currently empty).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {}

/// Information about the MCP client.
#[derive(Debug, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Parameters for the `initialize` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Information about the MCP server returned during initialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Result returned from the `initialize` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Value>,
}

/// A tool definition returned by the `tools/list` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Result returned from the `tools/list` method.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<McpToolDef>,
}

/// Parameters for the `tools/call` method.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: Value,
}

/// Content item returned in a tool call result.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Resource {
        resource: Value,
    },
}

impl Content {
    /// Return the text content if this is a `Text` variant, otherwise `None`.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }
}

/// Result returned from the `tools/call` method.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<Content>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_serialize_request() {
        let req = JsonRpcRequest::new(
            RequestId::Num(1),
            "initialize",
            Some(json!({"protocolVersion": "2024-11-05"})),
        );
        let serialized = serde_json::to_value(&req).unwrap();

        assert_eq!(serialized["jsonrpc"], "2.0");
        assert_eq!(serialized["id"], 1);
        assert_eq!(serialized["method"], "initialize");
        assert_eq!(serialized["params"]["protocolVersion"], "2024-11-05");
    }

    #[test]
    fn test_serialize_request_no_params() {
        let req = JsonRpcRequest::new(RequestId::Str("abc".into()), "tools/list", None);
        let serialized = serde_json::to_value(&req).unwrap();

        assert_eq!(serialized["id"], "abc");
        assert!(serialized.get("params").is_none());
    }

    #[test]
    fn test_deserialize_initialize_result() {
        let json = json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "test-server",
                "version": "1.0.0"
            },
            "capabilities": {}
        });

        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.protocol_version, "2024-11-05");
        assert_eq!(result.server_info.name, "test-server");
        assert_eq!(result.server_info.version.as_deref(), Some("1.0.0"));
        assert!(result.capabilities.is_some());
    }

    #[test]
    fn test_deserialize_tools_list_result() {
        let json = json!({
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file from disk",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" }
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "list_dir",
                    "inputSchema": { "type": "object" }
                }
            ]
        });

        let result: ToolsListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.tools.len(), 2);
        assert_eq!(result.tools[0].name, "read_file");
        assert_eq!(
            result.tools[0].description.as_deref(),
            Some("Read a file from disk")
        );
        assert_eq!(result.tools[1].name, "list_dir");
        assert!(result.tools[1].description.is_none());
    }

    #[test]
    fn test_deserialize_tool_call_result() {
        let json = json!({
            "content": [
                { "type": "text", "text": "Hello, world!" },
                {
                    "type": "image",
                    "data": "base64encodeddata",
                    "mimeType": "image/png"
                }
            ],
            "isError": false
        });

        let result: ToolCallResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.content.len(), 2);
        assert!(!result.is_error);

        assert_eq!(result.content[0].as_text(), Some("Hello, world!"));

        match &result.content[1] {
            Content::Image { data, mime_type } => {
                assert_eq!(data, "base64encodeddata");
                assert_eq!(mime_type, "image/png");
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn test_deserialize_error_response() {
        let json = json!({
            "id": 1,
            "error": {
                "code": -32601,
                "message": "Method not found",
                "data": null
            }
        });

        let response: JsonRpcResponse = serde_json::from_value(json).unwrap();
        assert!(response.result.is_none());
        let err = response.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    #[test]
    fn test_request_id_string_and_num() {
        let num_id = RequestId::Num(42);
        let str_id = RequestId::Str("my-id".to_string());

        let num_json = serde_json::to_value(&num_id).unwrap();
        let str_json = serde_json::to_value(&str_id).unwrap();

        assert_eq!(num_json, json!(42));
        assert_eq!(str_json, json!("my-id"));

        let roundtrip_num: RequestId = serde_json::from_value(num_json).unwrap();
        let roundtrip_str: RequestId = serde_json::from_value(str_json).unwrap();

        assert_eq!(roundtrip_num, RequestId::Num(42));
        assert_eq!(roundtrip_str, RequestId::Str("my-id".to_string()));
    }
}
