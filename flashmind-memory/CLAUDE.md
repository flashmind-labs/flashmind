# flashmind-memory

Vector memory with hybrid search (cosine similarity + BM25 via FTS5) backed by SQLite + sqlite-vec.

## Key Types

- `VectorMemory` (`provider.rs`) — implements `MemoryProvider` trait. Main entry point.
- `DbStore` (`store.rs`) — SQLite database layer. Handles schema, CRUD, vector ops.
- `EmbeddingProvider` trait + implementations (`embeddings/`) — `OllamaEmbedding`, `OpenAIEmbedding`, `OpenRouterEmbedding`
- `Scope` / `Source` / `Tag` — memory organization primitives
- `MemoryRecord` / `MemorySearchResult` — storage and retrieval types

## Architecture

```
VectorMemory
  ├── DbStore (SQLite + sqlite-vec + FTS5)
  └── EmbeddingProvider (Ollama / OpenAI / OpenRouter)
```

Search uses Reciprocal Rank Fusion to merge cosine similarity results with BM25 keyword matches.

## Schema

Schema is in `schema.rs`. Migrations run automatically on `DbStore::open()`.

## Testing

```bash
cargo test -p flashmind-memory
```

Uses `test_util` module for test fixtures.
