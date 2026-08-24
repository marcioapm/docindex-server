# docindex-server — Agent Guide

> Tiny Rust server that indexes a markdown vault and serves semantic + BM25 search over a Tailscale-only HTTP API. Powers an Obsidian mobile plugin (and anything else that wants ranked retrieval).

Read `README.md` for setup. Run `make run` to start locally; `systemctl --user status docindex-server` on the deployed host. Deployment details in `docs/deployment.md`.

## Architecture at a Glance

```
docindex-server/
├── Cargo.toml              # crate manifest (edition 2024, MSRV 1.90)
├── rust-toolchain.toml     # pin stable channel
├── src/
│   ├── main.rs             # docindex binary entry point (calls server::run)
│   ├── lib.rs              # re-exports public modules
│   ├── config.rs           # layered config: flags > env > TOML file > defaults
│   ├── config/
│   │   └── file.rs         # TOML discovery, injectable file-reader, *_env indirection
│   ├── server.rs           # wires config → store → embedder → indexer → watcher → HTTP
│   ├── walk.rs             # initial full-tree scan + content-hash diff
│   ├── chunk.rs            # heading-aware markdown chunker (pure)
│   ├── embed/
│   │   ├── mod.rs          # Embedder trait + AnyEmbedder enum, task-type constants
│   │   ├── registry.rs     # EmbedProvider enum + static model registry, validation
│   │   ├── gemini.rs       # Gemini REST client (reqwest + rustls)
│   │   ├── voyage.rs       # Voyage AI REST client (reqwest + rustls)
│   │   └── fake.rs         # Deterministic fake for tests
│   ├── indexer/
│   │   └── mod.rs          # single pipeline: walk/watch → chunk → embed → store
│   ├── watch/
│   │   └── mod.rs          # notify watcher + debounced dirty-set emitter
│   ├── search/
│   │   └── mod.rs          # hybrid BM25 + semantic search, RRF fusion
│   ├── api/
│   │   ├── mod.rs          # axum router, AppState
│   │   ├── auth.rs         # bearer middleware (constant-time compare)
│   │   ├── error.rs        # ApiError + IntoResponse
│   │   └── handlers.rs     # /health /search /similar
│   ├── store/
│   │   ├── mod.rs          # rusqlite + sqlite-vec wiring, upsert/delete/meta, fingerprint guard
│   │   ├── schema.sql      # canonical schema (schema_version=3)
│   │   └── vec.rs          # little-endian f32 (de)serialization
│   ├── cli/
│   │   ├── mod.rs          # re-exports for the docindex-search binary
│   │   ├── config.rs       # CLI config layering (flags > env > cli.toml > defaults)
│   │   ├── client.rs       # HTTP client against a running docindex server
│   │   └── output.rs       # human-readable hit formatting
│   └── bin/
│       └── search.rs       # docindex-search binary entry point
├── tests/                  # Python/pytest harness (spawn_server + E2E suites)
├── docs/
│   ├── ARCHITECTURE.md     # system design
│   └── deployment.md       # systemd user unit, Tailscale, UFW, upgrades, reembed
└── Makefile                # cargo wrappers

Obsidian mobile ──Tailscale──►  docindex-server  ──►  SQLite (index.db)
                                      │
                                      ├─ watches vault (Syncthing-synced)
                                      ├─ chunks markdown
                                      ├─ calls Gemini or Voyage embeddings
                                      └─ serves /health /search /similar
                docindex-search CLI ──Tailscale──►  docindex-server
```

**Deployment:** single static Rust binary (musl or aarch64-linux), systemd user service, bound to the Tailscale interface.

## Quick Reference

| Path | Purpose |
|---|---|
| `src/main.rs` | Thin binary: parse `--config`/`--reembed`, build tokio runtime, init tracing, call `server::run` |
| `src/server.rs` | Wires store + embedder + indexer + watcher + axum; fingerprint guard on open; graceful SIGTERM/SIGINT shutdown |
| `src/config.rs` | Layered config (flags > env > TOML file > defaults); `Config::from_env` is a thin env-only wrapper; rejects `0.0.0.0` and bare loopback unless `allow_loopback` |
| `src/config/file.rs` | TOML file discovery (`--config`/`$DOCINDEX_CONFIG`/well-known paths), injectable `FileReader`, `*_env` indirection, world-readable secret warning |
| `src/embed/registry.rs` | `EmbedProvider` enum + static model table (gemini/voyage/fake); validates provider/model/dim/key with actionable errors |
| `src/walk.rs` | Full-tree scan, `content_hash` diff, feeds dirty set to indexer |
| `src/chunk.rs` | Heading-aware chunker (H1/H2/H3 + ~500-token fallback, 50-token overlap) |
| `src/embed/mod.rs` | `Embedder` trait (native async fn), `AnyEmbedder` enum (Gemini/Voyage/Fake), `EmbedError`, task-type constants |
| `src/embed/gemini.rs` | Gemini embeddings client; retries on 429/5xx, x-goog-api-key header |
| `src/embed/voyage.rs` | Voyage embeddings client; Retry-After-aware 429 handling, 128-input batch chunking, index-ordered response |
| `src/embed/fake.rs` | Deterministic fake embedder for tests (sha256-seeded, L2-normalized) |
| `src/media_prepare.rs` | Byte-based image/PDF detection, provider-aware preparation (GIF/WebP→PNG, downscale, PDF subset/raster), deterministic cache keys |
| `src/watch/mod.rs` | `notify` recursive watcher + in-house debounce (500ms polled) with relevance filter |
| `src/search/mod.rs` | `search`, `similar`, `fuse_rrf` (pure), `fts_query_from_user`, snippet, limit clamp |
| `src/api/mod.rs` | axum router: public `/health` + bearer-gated sub-router |
| `src/api/auth.rs` | Constant-time bearer check |
| `src/api/error.rs` | `ApiError` → `{error, code}` JSON; `From<SearchError>` maps to 400/404 |
| `src/api/handlers.rs` | `/health`, `/search`, `/similar` handlers |
| `src/store/mod.rs` | SQLite handle + `sqlite-vec` auto-extension load, chunk/FTS/vec upsert, `files` diff, search helpers, `peek_fingerprint`/`wipe_and_rebuild` |
| `src/store/schema.sql` | Canonical schema (chunks, chunks_fts, chunks_vec `vec0` cosine, embedding_cache, files, meta) |
| `src/store/vec.rs` | Little-endian f32 encode/decode for vector BLOBs |
| `src/cli/config.rs` | `docindex-search` config layering (flags > env > `cli.toml` > defaults) |
| `src/cli/client.rs` | HTTP client wrapping `/health` `/search` `/similar`; maps status → `ClientError` |
| `src/cli/output.rs` | Human-readable hit formatting (rank, score, path, heading, truncated snippet) |
| `src/bin/search.rs` | `docindex-search` CLI: search/similar/health subcommands, exit codes |
| `tests/conftest.py` | Shared fixtures + `spawn_server` context manager |
| `tests/suites/test_*.py` | E2E suites: health, auth, search, similar, watcher, phase1 smoke, config file, voyage, cli, fingerprint |
| `tests/run_tests.py` | Python pytest runner; builds both bins, passes `DOCINDEX_BIN`/`DOCINDEX_SEARCH_BIN`, runs suites/ |
| `docs/ARCHITECTURE.md` | Full system design |
| `docs/deployment.md` | systemd unit, Tailscale, UFW, build + upgrade, reembed |

> The table above reflects the intended layout. When files are added/moved, update this section **in the same commit**.

## Tech Stack

- **Language:** Rust (edition 2024, MSRV 1.90)
- **Async runtime:** `tokio` multi-thread (main.rs). SQL is offloaded via `spawn_blocking` — `rusqlite` is `!Sync`.
- **HTTP server:** `axum` 0.8 + `tower` 0.5 / `tower-http` 0.6
- **HTTP client:** `reqwest` with `rustls-tls` (no OpenSSL system dep)
- **SQLite:** `rusqlite` 0.34 with `bundled` + `load_extension` features (statically linked libsqlite3)
- **Vector search:** `sqlite-vec` 0.1.x — **loaded as a real SQLite extension** via `sqlite3_auto_extension`, exposing the `vec0` virtual table (`distance_metric=cosine`)
- **FTS:** SQLite FTS5 (compiled into the bundled libsqlite3), `tokenize='porter unicode61'`
- **Embeddings:** provider/model registry (`src/embed/registry.rs`) — Google `gemini-embedding-2` (default, text+image+PDF, native 3072, Matryoshka to 768/1536, no `taskType`) and `gemini-embedding-001` (text only, legacy), Voyage AI's `voyage-multimodal-3.5` (text+image+PDF, native 1024) and `voyage-4` text family (native 1024, Matryoshka to 256/512/2048), task-asymmetric where supported. `Fake` embedder for tests (SHA256-seeded, L2-normalized, media-capable). `AnyEmbedder` enum for static dispatch (native async fn in traits → not dyn-compatible). Typed inputs: `EmbedInput::Text` / `EmbedInput::Media(Vec<MediaPart>)` — bytes never logged.
- **CLI parsing:** `clap` (derive) for both binaries
- **Config files:** `toml` (server.toml / cli.toml), layered under env vars per `src/config.rs` / `src/cli/config.rs`
- **Hashing:** `sha2` + `hex`
- **Filesystem walker:** `walkdir`
- **File watcher:** `notify` 8 with in-house debounce (default 5s, overridable via `DOCINDEX_DEBOUNCE_MS`)
- **Errors:** `thiserror::Error` per module (`ConfigError`, `WalkError`, `EmbedError`, `RegistryError`, `StoreError`, `IndexerError`, `SearchError`, `ApiError`, `ClientError`); `anyhow` only in `main.rs`/`server.rs`
- **Logging:** `tracing` + `tracing-subscriber` (JSON in prod via `DOCINDEX_LOG_FORMAT=json`, text in dev)
- **Config:** layered — CLI flags > env vars > TOML file > built-in defaults; env-only mode (today's production) keeps working unchanged
- **Tests:** `cargo test` for unit/integration; Python `pytest` harness in `tests/` for end-to-end via `spawn_server`
- **Deployment:** single static binary, systemd user service

## Endpoints

```
GET  /health                          → { ok, indexed_chunks, last_reindex_ms, embedding_model, dim }
POST /search   { query, limit=10 }    → { hits: [{ path, title, heading_path, snippet, score, score_rrf, score_normalized, chunk_id }] }
POST /similar  { path,  limit=10 }    → same shape
```

- Auth: every non-`/health` endpoint requires `Authorization: Bearer <DOCINDEX_BEARER>`. Constant-time compare.
- Bind: `DOCINDEX_LISTEN` **must be a Tailscale IP**, never `0.0.0.0` or `[::]` (enforced at startup in `config.rs`). Loopback is rejected unless `DOCINDEX_ALLOW_LOOPBACK=true` — dev/tests only; production MUST leave it unset/false.
- Errors: JSON `{ "error": "...", "code": "..." }`. `code` values: `bad_request`, `unauthorized`, `not_found`, `internal`.
- `limit` is clamped to `[1, 50]`.

### Score fields
Every hit carries three scores:
- `score` — the RRF fusion score (kept for back-compat with older plugin versions).
- `score_rrf` — same value as `score`. This is the field the ranker orders on.
- `score_normalized` — 0..1, query-independent, derived from per-branch ranks via `W_VEC * branch_norm(v_rank, K) + W_BM25 * branch_norm(b_rank, K)` where `branch_norm(r, K) = (K+1)/(K+r)` (or 0 if absent from that branch). Used by the plugin for "% relevance" and threshold filtering. Defaults: `K = DOCINDEX_DISPLAY_K = 10`, `W_VEC = DOCINDEX_WEIGHT_VEC = 0.55`, `W_BM25 = DOCINDEX_WEIGHT_BM25 = 0.45`.

`score_rrf` is what the server ranks on; `score_normalized` is what the plugin displays + thresholds. See `docs/ARCHITECTURE.md` ("Ranking") and `docs/deployment.md` ("Tuning display + threshold") for the rationale behind two `k`s (ranking k=60, display k=10).

## Lifecycle

1. `main.rs` parses `--config`/`--reembed` flags, builds a multi-thread tokio runtime, loads the layered `Config`, inits `tracing`.
2. `server::run(cfg)`:
   - Resolves the store's embedding fingerprint via `Store::peek_fingerprint` before opening — a mismatch without `--reembed` refuses startup naming every changed field; with `--reembed` it opens via `Store::open_for_reembed` + `wipe_and_rebuild`.
   - Opens the `Store`, wraps in `Arc<Mutex<Store>>`.
   - Builds an `AnyEmbedder` (`Gemini`/`Voyage`/`Fake`) from `cfg.embed_provider` via the registry.
   - Creates `mpsc::unbounded_channel::<PathBuf>` for dirty paths.
   - Spawns the indexer task (`indexer::run`, drains rx).
   - Spawns the watcher (`watch::run`, emits into tx) and the initial scan (`indexer::initial_scan`, emits into tx).
   - Drops the original tx so the channel closes when both sources are done.
   - Binds the TCP listener and serves axum with `with_graceful_shutdown` (SIGINT/SIGTERM on Unix).
3. On shutdown: the watch signal stops the watcher; axum drains in-flight requests; a 5s timeout joins the background tasks.

## Ranking (Hybrid)

1. Embed query with task type `RETRIEVAL_QUERY` (Gemini) / `query` (Voyage).
2. Top-30 via cosine (`chunks_vec MATCH ? AND k = ?`).
3. Top-30 via BM25 (`FTS5`, `bm25(chunks_fts)`).
4. **Reciprocal Rank Fusion** (k=60, `search::RRF_K`) over the two lists. Ties broken by id ascending (deterministic).
5. Return top-N (clamped 1..=50) with snippet + metadata.

Don't normalize raw scores across the two lists — RRF avoids that rabbit hole.

The `score_normalized` field attached to each hit is a separate, query-independent 0..1 score used only for display + the plugin's relevance-threshold filter; it uses a different smoothing constant (`DOCINDEX_DISPLAY_K`, default 10) so rank-1 pins to ~1.0 and a fixed threshold like 0.40 is meaningful across queries. See the "Score fields" sub-section of "Endpoints" above.

`search::similar(path, limit)` uses the mean of the path's chunk vectors (L2-normalized) as the semantic query, concatenated first-4-chunk content as the FTS bag, and excludes the source path from hydration.

## Schema (canonical, v2)

```sql
CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY,
  path         TEXT    NOT NULL,
  chunk_idx    INTEGER NOT NULL,
  heading      TEXT,
  heading_path TEXT,                     -- "Parent > Child > This"
  content      TEXT    NOT NULL,
  content_hash TEXT    NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  tokens       INTEGER,
  UNIQUE(path, chunk_idx)
);
CREATE INDEX idx_chunks_path ON chunks(path);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  content, heading_path,
  content=chunks, content_rowid=id,
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
  embedding FLOAT[<embed_dim>] distance_metric=cosine  -- rendered from DOCINDEX_EMBED_DIM at open
);

CREATE TABLE embedding_cache (
  content_hash TEXT PRIMARY KEY,
  model        TEXT    NOT NULL,
  task_type    TEXT    NOT NULL,
  dim          INTEGER NOT NULL,
  embedding    BLOB    NOT NULL,
  created_at   INTEGER NOT NULL
);

CREATE TABLE files (
  path         TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  indexed_at   INTEGER NOT NULL
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- meta keys: embedding_provider, embedding_model, embedding_dim,
-- schema_version, last_full_scan, path_schema_version
```

**Embedding cache** is keyed by `content_hash` — renaming/moving a file with identical content never re-embeds. **`files`** keeps a per-path SHA256 + mtime so the startup scan can skip unchanged files without touching `chunks`. **`meta.embedding_provider`/`embedding_model`/`embedding_dim`** are the index fingerprint checked on every boot — see "Index fingerprint guard" in `docs/ARCHITECTURE.md`.

## Coding Standards

### Rust

- `cargo fmt --all` clean. `cargo clippy --all-targets --all-features -- -D warnings` before push.
- **No `unwrap()` / `expect()` / `panic!()` in library code.** Exceptions: `#[cfg(test)]` modules, `main.rs` terminating a startup failure, and places where an invariant is *proven* locally and documented.
- Propagate errors with `?`. Use `thiserror::Error` per module. Reserve `anyhow::Result` for `main.rs` and `server.rs`.
- Prefer small free functions over god-structs. Types and modules should have one reason to change.
- Native async fn in traits (stable Rust ≥ 1.75). For places needing dyn-style dispatch without `async_trait`, use an enum wrapper (e.g. `AnyEmbedder`).
- Tokio multi-thread runtime. SQL calls always wrapped in `spawn_blocking(move || { let guard = store.lock()...; guard.method() })` — `rusqlite` is `!Sync`.
- `tracing::{info, warn, error, debug}` with structured fields — never `println!` / `eprintln!` in production paths.
- Every external call (DB, HTTP, embedder) has a timeout (config). Per-request cancellation via tokio.
- Feature flags in `Cargo.toml` are fine, but don't create them speculatively.

### Naming Conventions

- Rust files: `snake_case.rs`; one concept per file.
- Types: `PascalCase`. Traits: capability-oriented (`Embedder`, `Store`) — no `-Impl` / `-Service`.
- Functions / methods: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Env vars: `UPPER_SNAKE_CASE`, all prefixed `DOCINDEX_*` (except `GEMINI_API_KEY`).
- SQL tables/columns: `snake_case`.
- HTTP routes: `kebab-case`, lowercase.

### Error Handling

- Wrap external errors with `#[from]` in the module's `thiserror` enum. Add variants for semantic failures (e.g. `StoreError::CacheDimMismatch { got, want }`).
- HTTP handlers return structured JSON: `{ "error": "...", "code": "..." }`. `ApiError` owns this mapping.
- Never expose internal error details or stack traces to clients.
- Log at `error` for 5xx; `warn` for 4xx that indicates misconfiguration.
- Request timeouts: 30s default, configurable via `DOCINDEX_HTTP_TIMEOUT_MS`.

### Testing

- Every feature must have tests. No exceptions.
- Three layers:
  1. **Unit tests** colocated with source (`#[cfg(test)] mod tests { ... }`): chunker, RRF, FTS sanitization, snippet, auth middleware, config parsing, vec encode/decode, watcher relevance/debounce.
  2. **Integration tests** using a real on-disk SQLite file (via `tempfile::TempDir`): walker + chunker + store + indexer roundtrip, Gemini client via `wiremock`.
  3. **End-to-end Python harness** under `tests/` (`uv`-managed, `pytest`): `spawn_server` spins up `target/release/docindex` against a fixture vault on a random loopback port with `DOCINDEX_ALLOW_LOOPBACK=true` + `DOCINDEX_EMBED=fake`, asserts health/auth/search/similar/watcher behavior.
- Use `tempfile::TempDir` (Rust) / `tmp_path` (pytest) — no shared state across tests.
- Mock Gemini at the boundary (`fake` embedder or `wiremock`): deterministic vectors keyed by content.
- `cargo test --all` must pass cleanly; `cargo clippy --all-targets -- -D warnings` must be clean.

### Git

- Commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`.
- One concern per commit. Explain *why* in the body, not just *what*.
- Update `README.md`, `CLAUDE.md`, and `docs/` **in the same commit** as the code change.

## Architecture Patterns

### Walker + Watcher (single source of truth)

`walk.rs` runs once at startup and feeds a "dirty set" to the indexer through an `mpsc` channel. `watch/mod.rs` emits dirty paths on fs events (debounced). **Both paths feed the same `indexer::run` pipeline** — there is no divergence.

When you change how a file is indexed, change it once in `indexer::reindex_one` / `resolve_embeddings`; never duplicate logic between walker and watcher.

### Path invariant: everything downstream of the walker/watcher is vault-relative

The mpsc channel, `chunks.path`, `files.path`, and every `/search` / `/similar` `hit.path` MUST be vault-relative (e.g. `notes/foo.md`, never `/home/.../vault/notes/foo.md`). Absolute paths are rejected at the indexer boundary (`reindex_one` errors if given one) and stripped at the walker/watcher boundary (via `canonicalize()` + `strip_prefix` against the canonicalized vault root, rejecting symlink escapes with a warning).

This matches Obsidian's `TFile.path` so the plugin can pass `hit.path` straight into `openLinkText()` / `getAbstractFileByPath()` with no rewriting. Existing DBs are migrated in place on first open via `Store::migrate_paths_to_relative(vault_dir)` — idempotent via `meta.path_schema_version = 1`, refuses (logged, not fatal) if any row is absolute but outside the vault.

### Chunker contract

Input: raw markdown bytes. Output: an ordered `Vec<Chunk>`. Chunks are deterministic — same input → same output, including byte-identical content and identical `content_hash`. The chunker does **not** call the embedder or the store; it's pure.

Indexable extensions live in `walk::INDEXABLE_EXTENSIONS` (`md`, `txt`; case-insensitive) and are shared by the walker and the watcher. `.txt` files have no ATX headings, so they naturally flow through the 500-word fallback path — one chunk when short, heading-less sub-chunks with 50-word overlap when long.

### Embedding cache

Before calling Gemini, the indexer hashes the chunk content (`sha256`) and consults `embedding_cache`. A hit returns the cached vector; a miss is batched into a single embedder call, then stored keyed by hash. **Renames and reorganizations never re-embed** — this is load-bearing for cost.

### Hybrid search + RRF

Entry points: `search::search(store, embedder, dim, query, limit)` and `search::similar(store, dim, path, limit)`:
1. Embed query with task type `RETRIEVAL_QUERY`.
2. Run vec + FTS queries concurrently via `tokio::join!(spawn_blocking, spawn_blocking)`.
3. Fuse via `fuse_rrf(vec_hits, fts_hits, 60)` — pure, unit-tested.
4. Hydrate snippets + heading paths in a single blocking hop.

All ranking logic lives in `search::`; handlers never touch SQL directly. Invalid FTS queries fall back to an empty candidate list so the semantic side still runs.

### Tailscale-only bind + bearer auth

The bind address is validated at startup in `config.rs`: `0.0.0.0:*` and `[::]:*` are rejected, and bare loopback binds require `DOCINDEX_ALLOW_LOOPBACK=true`. Bearer auth is belt-and-suspenders; Tailscale is the primary boundary.

### sqlite-vec extension loading

`src/store/mod.rs` registers `sqlite-vec`'s C init function via `rusqlite::ffi::sqlite3_auto_extension` exactly once per process (`OnceLock`). Every subsequent `Connection::open` gets the extension loaded before any SQL runs, so `CREATE VIRTUAL TABLE ... USING vec0(...)` works in the schema bootstrap. `verify_vec_loaded` calls `vec_version()` right after open and errors hard if the extension isn't there — we never silently fall back to BLOB math.

## Common Pitfalls

- **sqlite-vec load order:** The extension MUST be registered (`register_sqlite_vec` in `store/mod.rs`) before the first `Connection::open`. `sqlite3_auto_extension` is process-global; guard it with `OnceLock`. Verify with `SELECT vec_version()` after opening — any error here means the rest of the store is broken.
- **`vec0` has no `INSERT OR REPLACE`:** To update a row in `chunks_vec`, `DELETE` then `INSERT` inside a transaction. A naive `UPSERT` will fail.
- **FTS5 sync:** `chunks_fts` is a contentless FTS5 table indexed on `chunks`. You must manually `INSERT`/`DELETE` into the FTS table when `chunks` changes — it doesn't auto-sync in SQLite. Use the `INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ...)` form for deletes.
- **FTS5 MATCH is picky:** Parens, colons, quotes, backslashes are operators. Sanitize user input through `fts_query_from_user`: tokenize on alphanumerics/`_-`, wrap each in double quotes, implicit AND. Drop tokens ≤ 1 char. Errors from FTS fall back to an empty candidate list so the semantic side still ranks.
- **rusqlite is !Sync:** Don't call it from async code directly. Wrap every SQL call in `tokio::task::spawn_blocking(move || { let guard = store.lock()...; guard.method() })`. Keep the guard lifetime short.
- **Gemini task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search. Getting this wrong silently degrades quality.
- **Matryoshka dim:** Request `outputDimensionality: <DOCINDEX_EMBED_DIM>` at embed time (default 3072 — the model's native size). Smaller dims (768, 1536) are Matryoshka-valid and save disk/ANN cost; larger than 3072 is not meaningful for this model. The dim is baked into the `chunks_vec` DDL *and* cached under `meta.embedding_dim` — the store refuses to open if the on-disk dim doesn't match `DOCINDEX_EMBED_DIM`. Wipe `index.db` to change dim.
- **Debounce state:** The watcher's debounce map is keyed by the event path as `notify` reports it; the relevance filter rejects `.git`, `.obsidian`, `node_modules`, and dot-files. Symlink-containing vaults may dedupe unexpectedly — don't rely on symlinks inside the vault.
- **Dev-loopback bypass:** `DOCINDEX_ALLOW_LOOPBACK=true` is **dev/test only**. Production MUST leave it unset/false — the Tailscale boundary is not optional.
- **Unicode in FTS5:** Use `tokenize='porter unicode61'`, not the default — the default strips non-ASCII.
- **Headings with pipes:** `heading_path` uses `" > "` as a separator; don't use `"|"` or `"/"` (both appear in markdown headings).
- **Graceful shutdown:** axum's `with_graceful_shutdown` drains in-flight requests, but background tasks (indexer, watcher) only join within a 5s timeout. Long-running Gemini calls may be truncated on shutdown — that's fine, they re-queue on next boot via the content-hash diff.
- **No CGo-equivalent foot-guns:** `rusqlite` with `bundled` compiles libsqlite3 inside the crate — you get FTS5, JSON1, and a consistent version. Don't disable `bundled` without a strong reason (it breaks reproducibility).

## Rules

**Every commit:**
- `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all` — all must pass.
- Update `README.md`, `CLAUDE.md`, and `docs/ARCHITECTURE.md` when architecture or structure changes.
- Write tests for every new feature or bug fix.

**After significant changes** (chunker, ranking, schema, API shape):
- Run the Python pytest harness against a fixture vault.
- Run the subagent reviewers in `.claude/agents/` (at minimum `code-reviewer` + `code-simplifier`).

**Pre-push CI check (mandatory):**
1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test --all`
4. `python3 tests/run_tests.py`  (or `uv run --project tests pytest`)
5. `cargo build --release`

- Never bind to `0.0.0.0` or `[::]`.
- Never hard-code the bearer token; always from env.
- Never log the bearer, the Gemini API key, or the full content of a chunk at info level.
- Every new endpoint requires: bearer auth (unless public like `/health`), timeout, structured error response, a test.
- No `unwrap()` / `expect()` / `panic!()` outside of `#[cfg(test)]`, `main.rs` startup errors, and provably-safe invariants.
