# docindex-server

Semantic + BM25 search server for personal docs/notes. Indexes a folder of markdown, serves a tiny HTTP API over Tailscale. Built to power an Obsidian mobile plugin and anything else that wants ranked retrieval against a personal vault.

## Stack
- **Language:** Go
- **Storage:** SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vectors) + FTS5 (BM25)
- **Embeddings:** Google `gemini-embedding-001` (Matryoshka, dim 768)
- **Watcher:** `fsnotify` with 5s debounce
- **Transport:** Tailscale-only bind, bearer-token auth

## Endpoints
```
GET  /health                          -> { ok, indexedChunks, lastReindex, embeddingModel, dim }
POST /search   { query, limit=10 }    -> { hits: [{ path, title, headingPath, snippet, score, chunkId }] }
POST /similar  { path,  limit=10 }    -> same shape
```

## Ranking
Hybrid: top-30 cosine (sqlite-vec) + top-30 BM25 (FTS5), fused with Reciprocal Rank Fusion (k=60), top-10 returned.

## Chunking
Heading-aware (H1/H2/H3), ~500-token fallback, 50-token overlap.
Stored per chunk: `(path, chunk_idx, heading_path, content, content_hash, mtime_ns)`.

## Reindex
- Startup: full walk, diff against stored `content_hash`.
- Live: `fsnotify` + 5s debounce, batch dirty files.
- Embedding cache keyed by `content_hash` (survives renames/moves).

## Deployment
- Host: Hetzner VPS, bound to Tailscale interface.
- Process: systemd unit.
- Config: env vars (see `.env.example`).

## Status
Scaffolding. See `docs/ARCHITECTURE.md` for the full plan.

## License
Private — for personal use.
