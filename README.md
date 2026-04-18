# docindex-server

Semantic + BM25 search server for personal docs/notes. Indexes a folder of markdown, serves a tiny HTTP API over Tailscale. Built to power an Obsidian mobile plugin and anything else that wants ranked retrieval against a personal vault.

## Stack
- **Language:** Rust (edition 2024, MSRV 1.90)
- **Storage:** SQLite (`rusqlite`, bundled) + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vectors, `vec0`) + FTS5 (BM25)
- **Embeddings:** Google `gemini-embedding-001` (Matryoshka, dim 768)
- **Async runtime:** `tokio` (current-thread)
- **HTTP client:** `reqwest` + rustls
- **Watcher (Phase 2):** `notify` with 5s debounce
- **Transport (Phase 2):** Tailscale-only bind, bearer-token auth

## Endpoints (Phase 2)
```
GET  /health                          -> { ok, indexedChunks, lastReindex, embeddingModel, dim }
POST /search   { query, limit=10 }    -> { hits: [{ path, title, headingPath, snippet, score, chunkId }] }
POST /similar  { path,  limit=10 }    -> same shape
```

## Ranking
Hybrid: top-30 cosine (sqlite-vec `vec_distance_cosine`) + top-30 BM25 (FTS5), fused with Reciprocal Rank Fusion (k=60), top-10 returned.

## Chunking
Heading-aware (H1/H2/H3), ~500-token fallback, 50-token overlap.
Stored per chunk: `(path, chunk_idx, heading_path, content, content_hash, mtime_ns)`.

## Reindex
- Startup: full walk, diff against stored `content_hash`.
- Live (Phase 2): `notify` + 5s debounce, batch dirty files.
- Embedding cache keyed by `content_hash` (survives renames/moves).

## Deployment
- Host: Hetzner VPS, bound to Tailscale interface.
- Process: systemd user service.
- Config: env vars (see `.env.example`).

## Status
Phase 1 (scaffolding): config parsing, walker, chunker, embedder (Gemini + Fake), store (rusqlite + **real sqlite-vec `vec0`** + FTS5). HTTP, watcher, and hybrid search land in Phase 2.

## Quick start

```sh
cp .env.example .env          # then edit values
make test                     # cargo test --all
make build-release            # -> target/release/docindex
set -a; source .env; set +a   # export env vars
./target/release/docindex     # logs "ready" on stderr and exits 0 (Phase 1)
```

Run the Python end-to-end harness:
```sh
make pytest
```

See `docs/ARCHITECTURE.md` for the full plan and `CLAUDE.md` for coding standards.

## License
Private — for personal use.
