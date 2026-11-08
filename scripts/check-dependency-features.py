#!/usr/bin/env python3
"""Check that minimal builds exclude optional dependency stacks."""

import subprocess


def check(package, features, forbidden, *, defaults=False):
    command = [
        "cargo", "tree", "--locked", "-p", package,
        "--edges", "normal,build", "--prefix", "none", "--format", "{p}",
    ]
    if not defaults:
        command.append("--no-default-features")
    if features:
        command.extend(["--features", features])
    output = subprocess.check_output(command, text=True)
    packages = {tuple(line.split()[:2]) for line in output.splitlines() if line.strip()}
    unexpected = {name for name, _ in packages} & forbidden
    if unexpected:
        raise SystemExit(f"{package} ({features or 'minimal'}): unexpected {sorted(unexpected)}")
    label = features or ("default" if defaults else "minimal")
    print(f"{package} ({label}): {len(packages)} packages")


SQLITE = {"rusqlite", "libsqlite3-sys", "sqlite-vec", "tokio-rusqlite"}
HTTP = {"reqwest", "hyper", "rustls", "ring", "webpki-roots"}
TRANSPORT = {"h2", "async-compression", "flate2", "yawc"}
UI = {"arboard", "image", "ratatui", "comfy-table", "ansi-to-tui"}

check("flashmind-cli", "", SQLITE, defaults=True)
check("flashmind-cli", "full", SQLITE)
check("flashmind-cli", "", SQLITE | TRANSPORT | {
    "agent-client-protocol", "jsonwebtoken", "arboard", "image", "self-replace",
    "comfy-table", "ansi-to-tui", "nucleo-matcher",
})
check("flashmind-tools", "", SQLITE | HTTP | {"flashmind-core", "flashmind-skills", "serde_yaml"})
check("flashmind-tools", "http2,http-compression", SQLITE | HTTP)
check("flashmind-memory", "", SQLITE | HTTP | {"flashmind-types", "tokio", "uuid", "sha2"})
check("flashmind-memory", "session", SQLITE | HTTP | {"sqlx", "pgvector", "metrics"})
check("flashmind-memory", "session-sqlite", HTTP | UI | {"sqlite-vec", "sqlx", "pgvector", "sha2", "metrics"})
check("flashmind", "session", SQLITE | HTTP | UI)
check("flashmind", "session-sqlite", HTTP | UI | {"sqlite-vec", "sqlx", "pgvector", "sha2", "metrics"})
check("flashmind", "tools", SQLITE | HTTP | UI | {"flashmind-core", "flashmind-skills", "serde_yaml"})
check("flashmind", "llm", SQLITE | TRANSPORT | UI)
check("flashmind", "tui", SQLITE | HTTP | {"arboard", "image", "comfy-table", "ansi-to-tui"})
check("flashmind", "http2,http-compression,mistral-realtime,request-compression,markdown,clipboard,tool-http,tool-skills,subagent",
      SQLITE | HTTP | UI | {"flashmind-tools", "flashmind-memory", "flashmind-llm", "flashmind-tui"})
