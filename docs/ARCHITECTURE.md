# Architecture

## Goals
- Semantic + BM25 search over a markdown vault.
- Mobile-reachable over Tailscale.
- Single static binary, SQLite storage, minimal ops.

## Components
- **Walker** (`src/walk.rs`): initial full scan, content-hash diff → dirty set. Accepts the extensions in `walk::INDEXABLE_EXTENSIONS` (currently `.md` and `.txt`, case-insensitive). Emits **vault-relative** paths only — entries whose canonical location escapes the canonicalized vault root (e.g. via a symlink) are logged and skipped.
- **Chunker** (`src/chunk.rs`): heading-aware (H1/H2/H3) + ~500-token fallback with 50-token overlap. Pure. Heading-less inputs (plain `.txt`) flow through the fallback path.
- **Embedder** (`src/embed/`): provider/model registry (`src/embed/registry.rs`) resolves `(EmbedProvider, model, dim)` and validates against a static table. Gemini `gemini-embedding-001` (native 3072, Matryoshka to 768/1536) and Voyage AI's `voyage-4` family (native 1024, Matryoshka to 256/512/2048) are both Matryoshka-truncatable — the configured dim is requested at embed time and baked into storage, task-asymmetric (doc vs. query task labels come from the registry entry). `Fake` embedder for tests (SHA256-seeded, L2-normalized). `AnyEmbedder` enum for static dispatch (native async fn in traits → not dyn-compatible).
- **Store** (`src/store/`): SQLite via `rusqlite` (`bundled` + `load_extension`) with `sqlite-vec` loaded as a real extension (`vec0` virtual table, `distance_metric=cosine`) and FTS5 for BM25. Shared across tasks via `Arc<Mutex<Store>>`; every SQL call runs inside `spawn_blocking`. On open the server calls `migrate_paths_to_relative(vault_dir)` which idempotently rewrites any pre-0.2.0 absolute paths in `chunks.path` / `files.path` to vault-relative form (or refuses, logged-not-fatal, if any row points outside the vault); success is recorded in `meta.path_schema_version = 1`.
- **Indexer** (`src/indexer/`): single pipeline consuming **relative** dirty paths from an `mpsc::UnboundedReceiver<PathBuf>`. `reindex_one` rejects absolute paths defensively, then joins them onto `ctx.vault_dir` for I/O while using the relative string as the DB key. Computes per-file SHA256, short-circuits on unchanged files, splits into chunks, consults `embedding_cache` before calling the embedder (batched), persists chunks+vectors+FTS atomically, bumps `last_reindex_ms`. Both the startup walker and the live watcher feed the same channel.
- **Watcher** (`src/watch/`): `notify` recursive watcher with a 500ms-polled in-house debounce map (`HashMap<PathBuf, Instant>`). Strips the canonicalized vault prefix before inserting into the debounce map, so the indexer only ever sees relative paths. Filters to `walk::INDEXABLE_EXTENSIONS` (`.md`, `.txt`), rejects `.git` / `.obsidian` / `node_modules` / dot-files. Debounce window comes from `DOCINDEX_DEBOUNCE_MS` (default 5s).
- **Search** (`src/search/`): hybrid BM25+semantic retrieval with Reciprocal Rank Fusion (k=60, candidate pool 30 per ranker). Pure `fuse_rrf` is unit-tested. FTS queries are sanitized (`fts_query_from_user`) — tokenized to alphanumerics+`_-`, each token double-quoted, implicit AND. Snippets are heading-stripped, whitespace-collapsed, 240 chars.
- **API** (`src/api/`): axum 0.8 router with `/health` (public), `/search` + `/similar` (bearer-protected via middleware). Structured errors as `{error, code}` JSON. `AppState` is `Send+Sync` (statically asserted in `server.rs`).
- **Server** (`src/server.rs`): wires everything; resolves the index fingerprint (see "Index fingerprint guard" below) before opening the store, spawns indexer first, then watcher + initial scan, serves HTTP with `with_graceful_shutdown` that listens for SIGINT + SIGTERM (Unix) / Ctrl-C (otherwise). Bind is Tailscale-IP only (config rejects `0.0.0.0`/`[::]`); dev/test can opt in via `allow_loopback=true`.
- **Config** (`src/config.rs`, `src/config/file.rs`): layered — CLI flags > env vars > TOML file > built-in defaults. `Config::load` takes an injectable `Lookup` (env) and `FileReader` (filesystem) so tests never touch real env/`$HOME`; `Config::from_env` is a thin wrapper using the real process env and no file/flag layer, preserving today's env-only production behavior exactly. TOML file discovery: `--config` > `$DOCINDEX_CONFIG` > `~/.config/docindex/server.toml` (or `$XDG_CONFIG_HOME`) > `/etc/docindex/server.toml`. `*_env` indirection keys (`bearer_env`, `api_key_env`) read a named env var instead of an inline secret; a world-readable file with an inline secret logs a warning (doesn't refuse to start).
- **CLI** (`src/cli/`, `src/bin/search.rs`): `docindex-search` binary sharing the library crate. `cli::config::CliConfig` mirrors the server's layering with a much smaller schema (server URL, token, limit, format). `cli::client::Client` wraps `/health` `/search` `/similar` over `reqwest`, mapping HTTP status to `ClientError` (401/403 → auth, other non-2xx → server, transport failures → network). `cli::output` formats hits for the terminal (rank, `score_normalized`, path, heading, snippet truncated to terminal width).

## Index fingerprint guard
The index is only valid for the `(provider, model, dim)` it was embedded with — mixing vectors from two different embedding spaces silently corrupts ranking (no error, just garbage cosine distances). `meta.embedding_provider` / `embedding_model` / `embedding_dim` record the fingerprint on first successful open.

On every boot, `server::open_store_with_fingerprint_check`:
1. Calls `Store::peek_fingerprint(db_path)` — reads `meta` via the base schema only, **before** the configured dim gets baked into `chunks_vec`'s DDL (a `CREATE VIRTUAL TABLE IF NOT EXISTS` would otherwise silently keep whatever dim an existing table already has).
2. No fingerprint recorded (fresh DB, or a pre-fingerprint upgrade): opens normally, adopts the current config as the fingerprint.
3. Fingerprint matches: opens normally.
4. Fingerprint mismatches, no `--reembed`: refuses to start with an error naming every changed field and both values, e.g. `index built with provider=gemini model=gemini-embedding-001 dim=3072, config says provider=voyage model=voyage-4 dim=1024; re-embed required: run with --reembed`.
5. Fingerprint mismatches, `--reembed` set: opens via `Store::open_for_reembed` (skips the low-level dim-only refusal in `Store::open`) then calls `Store::wipe_and_rebuild`, which drops+recreates `chunks_fts`/`chunks_vec`, deletes every row of `chunks`/`files`/`embedding_cache`, and writes the new fingerprint in one transaction. The normal startup scan then re-embeds every file.

`base_url` (provider API base URL override, for proxies/mocks) is deliberately **not** part of the fingerprint — pointing at a different endpoint for the same provider/model/dim must not invalidate a good index.

## Schema
See `src/store/schema.sql`. Notable tables:
- `chunks` — canonical chunk rows; uniqueness on `(path, chunk_idx)`.
- `chunks_fts` — contentless FTS5 over `chunks`, `tokenize='porter unicode61'`.
- `chunks_vec` — `vec0` virtual table, `embedding FLOAT[<DOCINDEX_EMBED_DIM>] distance_metric=cosine` (the dim literal is rendered at `Store::open` time from config).
- `embedding_cache` — content-hashed embedding cache (rename-safe).
- `files` — per-path `{content_hash, mtime_ns, indexed_at}` for the startup diff.
- `meta` — `schema_version=2`, `path_schema_version=1` (after the absolute→relative path migration), `embedding_provider`, `embedding_model`, `embedding_dim` (the index fingerprint), `last_full_scan`.

## Path invariants
- The mpsc channel, `chunks.path`, `files.path`, and every `/search` / `/similar` `hit.path` are **always vault-relative** (e.g. `notes/foo.md`). Absolute paths never leave the walker / watcher boundary.
- `walk::relativize_inside(vault, path)` and `watch::strip_vault_prefix` canonicalize the candidate and the vault root before stripping; any candidate that would escape (via symlinks or `..`) is rejected with a warning, never silently written through.
- `indexer::reindex_one` accepts relative paths only and errors if given an absolute one — this is the defensive backstop that guarantees the channel contract.
- Migration (`Store::migrate_paths_to_relative(vault_dir)`): idempotent, transactional, refuses when any absolute row lies outside `vault_dir`. Logs `migrated N rows from absolute to relative paths` on success and sets `meta.path_schema_version = 1` so subsequent boots short-circuit.
- Clients (notably the Obsidian plugin) treat `hit.path` as directly compatible with `TFile.path` — no stripping, no prefixing.

## Ranking
1. Embed query with `RETRIEVAL_QUERY`.
2. Top-30 via cosine (`chunks_vec MATCH ? AND k = ?`).
3. Top-30 via BM25 (`bm25(chunks_fts)`).
4. Fuse with RRF: `score(d) = Σ 1/(k+rank_i(d))`, `k=60`. Sorted desc; ties broken by id ascending (deterministic).
5. Hydrate top-N (clamped to [1, 50]) with snippet + `heading_path`.

Every hit carries three score fields:
- `score` — the RRF score, kept for API back-compat.
- `score_rrf` — same value as `score`, duplicated for clarity. This is the field the ranker orders on.
- `score_normalized` — a query-independent display score in `[0, 1]` derived from per-branch ranks. See below.

### Why two `k`s (ranking k=60, display k=10)

RRF's strength is stability in the long tail: `k=60` deliberately dampens differences between rank 1 and rank 2 so a doc that appears in both lists beats one that only appears in one even if it's not top-1 anywhere. That's great for ranking, terrible for a threshold — the absolute RRF score of the top hit drifts with list size and overlap, so "score ≥ 0.3" means different things across queries.

The plugin needs a stable `"is this result good enough to show?"` threshold, so we add a second normalization only for display:

```
branch_norm(rank, k) = (k + 1) / (k + rank)    if present, else 0
score_normalized     = W_VEC  * branch_norm(v_rank, k)
                     + W_BM25 * branch_norm(b_rank, k)
```

At `k=10`, rank-1 in both branches → `0.55 + 0.45 = 1.0`; rank-10 in both → `0.55`; rank-20 in both → `~0.37`. The default threshold of `0.40` ≈ "rank ≤ 15 in at least one branch, ideally both." Tunable via env (see `docs/deployment.md`, "Tuning display + threshold").

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
- systemd user service. See `docs/deployment.md`.
- Binds to Tailscale IP.
- UFW continues to block public ingress on the service port.

## Test harness
- `cargo test --all` — Rust unit + integration tests (chunker, walker, config layering + `*_env` indirection, provider registry validation, store with real sqlite-vec, Gemini + Voyage clients via wiremock, fake embedder, fingerprint compare, RRF, FTS sanitization, snippet, auth middleware, indexer end-to-end, watcher relevance/debounce).
- `tests/run_tests.py` — Python `pytest` harness: builds both release binaries (`docindex`, `docindex-search`), spins up the server against a fixture vault via the `spawn_server` fixture, then runs `test_health`, `test_auth`, `test_search`, `test_similar`, `test_watcher`, `test_phase1_smoke`, `test_config_file`, `test_voyage` (against a local mock HTTP server), `test_cli`, and `test_fingerprint`.
