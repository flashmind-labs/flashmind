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

- `session` — conversation session persistence (SQLite-backed)

## Embedding Providers

- `OllamaEmbedding` — local, no API key
- `OpenAIEmbedding` — OpenAI embeddings API
- `OpenRouterEmbedding` — via OpenRouter

## License

MIT
