# Architecture

## Goals
- Semantic + BM25 search over a markdown vault.
- Mobile-reachable over Tailscale.
- Single static binary, SQLite storage, minimal ops.

## Components
- **Walker** (`src/walk.rs`): initial full scan, content-hash diff -> dirty set.
- **Chunker** (`src/chunk.rs`): heading-aware (H1/H2/H3) + ~500-token fallback with 50-token overlap. Pure.
- **Embedder** (`src/embed/`): Gemini `gemini-embedding-001`, Matryoshka dim 768, task-asymmetric (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search). `Fake` embedder for tests.
- **Store** (`src/store/`): SQLite via `rusqlite` (`bundled` + `load_extension`) with `sqlite-vec` loaded as a real extension (`vec0` virtual table) and FTS5 for BM25.
- **Watcher** (Phase 2): `notify`, 5s debounce, batched per file.
- **API** (Phase 2): `/health`, `/search`, `/similar` + bearer auth middleware, Tailscale-only bind.

## Schema
See `src/store/schema.sql`.

## Ranking (Hybrid, Phase 2)
1. Embed query with `RETRIEVAL_QUERY`.
2. Top-30 via cosine (`vec_distance_cosine` from sqlite-vec).
3. Top-30 via BM25 (FTS5 `bm25(chunks_fts)`).
4. Fuse with Reciprocal Rank Fusion (k=60).
5. Return top-10 with snippet.

## Embedding cache
Keyed by `content_hash`. Rename/move of a chunk with identical text skips the API call.

## sqlite-vec loading
`src/store/mod.rs` registers the `sqlite3_vec_init` C function via `rusqlite::ffi::sqlite3_auto_extension` exactly once per process (`OnceLock`). Every subsequent `Connection::open` loads the extension before any SQL runs, so the schema can `CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding FLOAT[768])`. The store calls `SELECT vec_version()` right after open and errors hard if the extension isn't present — no silent fallback.

## Build / deployment
- Static-ish Rust binary (rustls, no OpenSSL system dep; `rusqlite` bundles libsqlite3).
- systemd user service on Hetzner.
- Binds to Tailscale IP.
- UFW continues to block public ingress on the service port.

## Test harness
- `cargo test --all` — Rust unit + integration tests (chunker, walker, config, store with real sqlite-vec, Gemini client via wiremock, fake embedder).
- `tests/run_tests.py` — Python `pytest` harness: builds the release binary, spins it up against a fixture vault, validates the DB schema and phase-1 startup contracts.
