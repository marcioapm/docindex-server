# docindex-server

Semantic + BM25 search server for personal docs/notes. Indexes a folder of markdown, serves a tiny HTTP API over Tailscale. Built to power an Obsidian mobile plugin and anything else that wants ranked retrieval against a personal vault.

## What it indexes
Files with extension `.md` or `.txt` (case-insensitive) anywhere under `DOCINDEX_VAULT_DIR`. Dot-files, `.git`, `.obsidian`, and `node_modules` are skipped. Heading-less files (plain `.txt`) flow through the chunker's 500-word fallback path.

## Stack
- **Language:** Rust (edition 2024, MSRV 1.90)
- **Storage:** SQLite (`rusqlite`, bundled) + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vectors, `vec0` cosine) + FTS5 (BM25)
- **Embeddings:** Google `gemini-embedding-001` at the configured dim (default 3072 native; Matryoshka-truncatable via `DOCINDEX_EMBED_DIM`) — or deterministic `fake` backend for tests
- **Async runtime:** `tokio` (multi-thread)
- **HTTP server:** `axum` 0.8
- **HTTP client:** `reqwest` + rustls
- **Watcher:** `notify` with a debounced dirty-set (default 5s, configurable)
- **Transport:** Tailscale-only bind (config rejects `0.0.0.0` / `[::]`), bearer-token auth

## Endpoints
```
GET  /health                          -> { ok, indexed_chunks, last_reindex_ms, embedding_model, dim }
POST /search   { query, limit=10 }    -> { hits: [{ path, title, heading_path, snippet, score, score_rrf, score_normalized, chunk_id }] }
POST /similar  { path,  limit=10 }    -> same shape
```
`/search` and `/similar` require `Authorization: Bearer $DOCINDEX_BEARER`. `limit` is clamped to [1, 50]. Errors return `{error, code}` JSON.

Every `hit.path` (and the `path` accepted by `/similar`) is **vault-relative** — e.g. `"notes/foo.md"`, never `/home/…/vault/notes/foo.md`. This matches Obsidian's `TFile.path` so the plugin can feed paths straight into `openLinkText()` / `getAbstractFileByPath()` without rewriting. Databases created before v0.2.0 are migrated in place on first open: any `chunks.path` / `files.path` that are absolute but still inside `DOCINDEX_VAULT_DIR` get rewritten atomically, and rows pointing outside the vault cause the migration to refuse (logged, not fatal) so operators can reconcile. Once complete, `meta.path_schema_version = 1` short-circuits the scan on every subsequent boot.

## Ranking
Hybrid: top-30 cosine (sqlite-vec `chunks_vec` cosine) + top-30 BM25 (FTS5), fused with Reciprocal Rank Fusion (`k=60`). Every hit also carries `score_normalized` (0..1, query-independent) for display + threshold filtering — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the `DOCINDEX_DISPLAY_K` / `DOCINDEX_WEIGHT_VEC` / `DOCINDEX_WEIGHT_BM25` env vars.

## Chunking
Heading-aware (H1/H2/H3), ~500-token fallback, 50-token overlap. Stored per chunk: `(path, chunk_idx, heading_path, content, content_hash, mtime_ns)`.

## Reindex
- Startup: full walk, diff against `files.content_hash`.
- Live: `notify` + debounce (default 5s, `DOCINDEX_DEBOUNCE_MS`), batch dirty files.
- Embedding cache keyed by `content_hash` (survives renames/moves).

## Deployment
- Host: Hetzner VPS, bound to Tailscale interface.
- Process: systemd user service.
- Config: env vars (see `.env.example`).
- Full guide: [`docs/deployment.md`](docs/deployment.md).

## Quick start

```sh
cp .env.example .env          # then edit values
make test                     # cargo test --all
make build-release            # -> target/release/docindex
set -a; source .env; set +a   # export env vars
./target/release/docindex     # serves HTTP until SIGINT/SIGTERM
```

Local dev without a Gemini key:
```sh
export DOCINDEX_VAULT_DIR=/tmp/vault DOCINDEX_DB_PATH=/tmp/index.db \
       DOCINDEX_LISTEN=127.0.0.1:7777 DOCINDEX_ALLOW_LOOPBACK=true \
       DOCINDEX_BEARER=dev DOCINDEX_EMBED=fake
./target/release/docindex
curl -H 'Authorization: Bearer dev' -X POST http://127.0.0.1:7777/search \
     -H 'Content-Type: application/json' -d '{"query":"hello"}'
```

Run the Python end-to-end harness:
```sh
make pytest
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`CLAUDE.md`](CLAUDE.md) for coding standards.

## License
Private — for personal use.
