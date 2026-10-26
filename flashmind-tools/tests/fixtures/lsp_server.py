"""A stdio language server fixture with deterministic diagnostics."""

import json
import sys


def send(message):
    body = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()


def diagnostics(document):
    if "broken" not in document["text"]:
        return []
    return [{
        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
        "severity": 1,
        "message": f"type error in {document['languageId']} version {document['version']}",
    }]


documents = {}
pull = "--pull" in sys.argv
while True:
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        if line == b"\r\n":
            break
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    message = json.loads(sys.stdin.buffer.read(length))
    method = message.get("method")
    params = message.get("params", {})
    result = None
    if method == "initialize":
        result = {"capabilities": {"diagnosticProvider": {}} if pull else {}}
    elif method in ("textDocument/didOpen", "textDocument/didChange"):
        document = params["textDocument"]
        uri = document["uri"]
        if method == "textDocument/didOpen":
            documents[uri] = document
        else:
            documents[uri].update(document)
            documents[uri]["text"] = params["contentChanges"][0]["text"]
        if not pull:
            send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {
                "uri": uri, "version": documents[uri]["version"], "diagnostics": diagnostics(documents[uri]),
            }})
    elif method == "textDocument/diagnostic":
        result = {"kind": "full", "items": diagnostics(documents[params["textDocument"]["uri"]])}
    elif method == "textDocument/hover":
        result = {"contents": {"kind": "plaintext", "value": documents[params["textDocument"]["uri"]]["languageId"]}}
    elif method == "exit":
        sys.exit(0)
    if "id" in message:
        send({"jsonrpc": "2.0", "id": message["id"], "result": result})
