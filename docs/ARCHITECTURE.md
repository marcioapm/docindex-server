# Architecture

## Goals
- Semantic + BM25 search over a markdown vault.
- Mobile-reachable over Tailscale.
- Single static binary, SQLite storage, minimal ops.

## Components
- **Walker** (`src/walk.rs`): initial full scan, content-hash diff → dirty set. Accepts the extensions in `walk::INDEXABLE_EXTENSIONS` (currently `.md` and `.txt`, case-insensitive).
- **Chunker** (`src/chunk.rs`): heading-aware (H1/H2/H3) + ~500-token fallback with 50-token overlap. Pure. Heading-less inputs (plain `.txt`) flow through the fallback path.
- **Embedder** (`src/embed/`): Gemini `gemini-embedding-001`, Matryoshka-truncatable — the configured embedding dim (default 3072) is requested via `outputDimensionality` and baked into storage, task-asymmetric (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search). `Fake` embedder for tests (SHA256-seeded, L2-normalized). `AnyEmbedder` enum for static dispatch (native async fn in traits → not dyn-compatible).
- **Store** (`src/store/`): SQLite via `rusqlite` (`bundled` + `load_extension`) with `sqlite-vec` loaded as a real extension (`vec0` virtual table, `distance_metric=cosine`) and FTS5 for BM25. Shared across tasks via `Arc<Mutex<Store>>`; every SQL call runs inside `spawn_blocking`.
- **Indexer** (`src/indexer/`): single pipeline consuming dirty paths from an `mpsc::UnboundedReceiver`. Computes per-file SHA256, short-circuits on unchanged files, splits into chunks, consults `embedding_cache` before calling the embedder (batched), persists chunks+vectors+FTS atomically, bumps `last_reindex_ms`. Both the startup walker and the live watcher feed the same channel.
- **Watcher** (`src/watch/`): `notify` recursive watcher with a 500ms-polled in-house debounce map (`HashMap<PathBuf, Instant>`). Filters to `walk::INDEXABLE_EXTENSIONS` (`.md`, `.txt`), rejects `.git` / `.obsidian` / `node_modules` / dot-files. Debounce window comes from `DOCINDEX_DEBOUNCE_MS` (default 5s).
- **Search** (`src/search/`): hybrid BM25+semantic retrieval with Reciprocal Rank Fusion (k=60, candidate pool 30 per ranker). Pure `fuse_rrf` is unit-tested. FTS queries are sanitized (`fts_query_from_user`) — tokenized to alphanumerics+`_-`, each token double-quoted, implicit AND. Snippets are heading-stripped, whitespace-collapsed, 240 chars.
- **API** (`src/api/`): axum 0.8 router with `/health` (public), `/search` + `/similar` (bearer-protected via middleware). Structured errors as `{error, code}` JSON. `AppState` is `Send+Sync` (statically asserted in `server.rs`).
- **Server** (`src/server.rs`): wires everything; spawns indexer first, then watcher + initial scan, serves HTTP with `with_graceful_shutdown` that listens for SIGINT + SIGTERM (Unix) / Ctrl-C (otherwise). Bind is Tailscale-IP only (config rejects `0.0.0.0`/`[::]`); dev/test can opt in via `DOCINDEX_ALLOW_LOOPBACK=true`.

## Schema
See `src/store/schema.sql`. Notable tables:
- `chunks` — canonical chunk rows; uniqueness on `(path, chunk_idx)`.
- `chunks_fts` — contentless FTS5 over `chunks`, `tokenize='porter unicode61'`.
- `chunks_vec` — `vec0` virtual table, `embedding FLOAT[<DOCINDEX_EMBED_DIM>] distance_metric=cosine` (the dim literal is rendered at `Store::open` time from config).
- `embedding_cache` — content-hashed embedding cache (rename-safe).
- `files` — per-path `{content_hash, mtime_ns, indexed_at}` for the startup diff.
- `meta` — `schema_version=2`, `embedding_model`, `embedding_dim`, `last_full_scan`.

## Ranking
1. Embed query with `RETRIEVAL_QUERY`.
2. Top-30 via cosine (`chunks_vec MATCH ? AND k = ?`).
3. Top-30 via BM25 (`bm25(chunks_fts)`).
4. Fuse with RRF: `score(d) = Σ 1/(k+rank_i(d))`, `k=60`. Sorted desc; ties broken by id ascending (deterministic).
5. Hydrate top-N (clamped to [1, 50]) with snippet + `heading_path`.

`search::similar(path, limit)` uses the average of the path's stored chunk vectors (L2-normalized) as the semantic query, and the concatenated first-4-chunk content as the FTS bag. Excludes the source path from hydration.

## Embedding cache
Keyed by `content_hash`. Rename/move of a chunk with identical text skips the API call.

## sqlite-vec loading
`src/store/mod.rs` registers the `sqlite3_vec_init` C function via `rusqlite::ffi::sqlite3_auto_extension` exactly once per process (`OnceLock`). Every subsequent `Connection::open` loads the extension before any SQL runs. `Store::open` first applies the base `schema.sql` (chunks, FTS5, cache, files, meta), then renders the `chunks_vec` DDL from the configured dim — `CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding FLOAT[<embed_dim>] distance_metric=cosine)` — and executes it, then checks `meta.embedding_dim` and refuses to start on mismatch. The store also calls `SELECT vec_version()` right after open and errors hard if the extension isn't present — no silent fallback.

## Concurrency model
- `main.rs` builds a multi-thread tokio runtime.
- The HTTP executor, the indexer task, and the watcher task run on tokio workers.
- `rusqlite` is `!Sync` — every SQL call is wrapped in `tokio::task::spawn_blocking(move || { let guard = store.lock()...; guard.method() })`.
- Dirty paths flow through `mpsc::UnboundedChannel<PathBuf>`. The indexer batches bursts by draining the queue with `try_recv` after each blocking `recv`.
- Graceful shutdown: a `watch::Sender<bool>` notifies the watcher to stop; axum's `with_graceful_shutdown` drains in-flight requests; a 5s `tokio::time::timeout` joins the background tasks.

## Bind / auth / TLS posture
- `validate_listen` rejects `0.0.0.0` and `[::]`. Loopback is rejected unless `DOCINDEX_ALLOW_LOOPBACK=true` (dev/tests only).
- Every non-`/health` endpoint is bearer-gated. Comparison is constant-time.
- TLS terminates at Tailscale. No server-side TLS; no public ingress.

## Build / deployment
- Static-ish Rust binary (rustls, no OpenSSL system dep; `rusqlite` bundles libsqlite3).
- systemd user service on Hetzner. See `docs/deployment.md`.
- Binds to Tailscale IP.
- UFW continues to block public ingress on the service port.

## Test harness
- `cargo test --all` — Rust unit + integration tests (chunker, walker, config, store with real sqlite-vec, Gemini client via wiremock, fake embedder, RRF, FTS sanitization, snippet, auth middleware, indexer end-to-end, watcher relevance/debounce).
- `tests/run_tests.py` — Python `pytest` harness: builds the release binary, spins it up against a fixture vault via the `spawn_server` fixture, then runs the `test_health`, `test_auth`, `test_search`, `test_similar`, and `test_watcher` suites.
