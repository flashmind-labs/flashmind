# flashmind-memory

Vector memory store with hybrid search for [Flashmind](https://github.com/flashmind-labs/flashmind).

Combines cosine similarity (sqlite-vec) and BM25 keyword search (FTS5) via Reciprocal Rank Fusion.

## Usage

```rust
use flashmind_memory::{MemoryStore, OllamaEmbedding};

let embedder = Arc::new(OllamaEmbedding::new(None, "nomic-embed-text".into()));
let store = MemoryStore::connect("memory.db", embedder).await?;

// Store with metadata and TTL
store.store("user prefers vim keybindings")
    .meta("scope", "preferences")
    .await?;

// Hybrid search
let results = store.search("editor preferences")
    .filter("scope", "preferences")
    .limit(5)
    .await?;
```

## Features

- `sqlite` (default): SQLite vector storage with sqlite-vec and FTS5.
- `postgres`: PostgreSQL vector storage with pgvector.
- `http-embeddings` (default): HTTP embedding clients.
- `http-compression`, `http2` (default): HTTP transport options.
- `session`: JSONL conversation history, with no database dependency.
- `session-sqlite`: SQLite conversation history without vector storage or HTTP.
- `session-sqlite-import`: import legacy SQLite conversation history.

For JSONL history alone, use `default-features = false, features = ["session"]`.
For SQLite history alone, use `default-features = false, features = ["session-sqlite"]`
and open it with `SqliteSessionStore::open("sessions.db").await?`.
Both stores expose the same entry, metadata, rewrite, branch, and prune operations.
SQLite writes use transactions; opening a database enables WAL and a five-second
busy timeout. Session SQLite storage is opt-in and independent of the CLI.

## Embedding Providers

- `OllamaEmbedding`  -  local, no API key
- `OpenAIEmbedding`  -  OpenAI embeddings API
- `OpenRouterEmbedding`  -  via OpenRouter

## License

MPL-2.0
