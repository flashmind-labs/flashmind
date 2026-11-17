"""Local MCP regression fixture using newline-delimited JSON-RPC."""
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(str(os.getpid()))

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    response = {"jsonrpc": "2.0", "id": request["id"]}
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fixture", "version": "1"},
        }
    elif method == "tools/list":
        if os.environ.get("FAIL_DISCOVERY"):
            response["error"] = {"code": -32603, "message": "discovery failed"}
            print(json.dumps(response), flush=True)
            continue
        params = request.get("params") or {}
        name = "fail" if params.get("cursor") else "start_agent"
        result = {"tools": [{
            "name": name,
            "description": "Local regression tool",
            "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}}},
        }]}
        if name == "start_agent":
            result["nextCursor"] = "second"
    elif method == "tools/call":
        params = request["params"]
        if params.get("arguments", {}).get("task") == "hang":
            continue
        if params.get("arguments", {}).get("task") == "rpc-error":
            response["error"] = {"code": -32603, "message": "call failed"}
            print(json.dumps(response), flush=True)
            continue
        result = {
            "content": [{"type": "text", "text": json.dumps({
                "task": params.get("arguments", {}).get("task"),
                "marker": os.environ.get("MCP_MARKER"),
                "cwd": os.getcwd(),
            })}],
            "isError": params["name"] == "fail",
        }
    else:
        result = {}
    response["result"] = result
    print(json.dumps(response), flush=True)
