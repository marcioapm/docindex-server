# docindex-server — Agent Guide

> Tiny Rust server that indexes a markdown vault and serves semantic + BM25 search over a Tailscale-only HTTP API. Powers an Obsidian mobile plugin (and anything else that wants ranked retrieval).

Read `README.md` for setup. Run `make run` to start locally; `systemctl --user status docindex-server` on Hetzner.

## Architecture at a Glance

```
docindex-server/
├── Cargo.toml              # crate manifest (edition 2024, MSRV 1.90)
├── rust-toolchain.toml     # pin stable channel
├── src/
│   ├── main.rs             # binary entry point; wires config → store → (phase 2: api)
│   ├── lib.rs              # re-exports public modules
│   ├── config.rs           # env parsing + validation
│   ├── walk.rs             # initial full-tree scan + content-hash diff
│   ├── chunk.rs            # heading-aware markdown chunker (pure)
│   ├── embed/
│   │   ├── mod.rs          # Embedder trait, task-type constants, EmbedError
│   │   ├── gemini.rs       # Gemini REST client (reqwest + rustls)
│   │   └── fake.rs         # Deterministic fake for tests
│   └── store/
│       ├── mod.rs          # rusqlite + sqlite-vec wiring, upsert/delete/meta
│       ├── schema.sql      # canonical schema
│       └── vec.rs          # little-endian f32 (de)serialization
├── tests/                  # Python/pytest harness (invokes the bin + smoke tests)
├── docs/
│   └── ARCHITECTURE.md     # system design
└── Makefile                # cargo wrappers

Obsidian mobile ──Tailscale──►  docindex-server  ──►  SQLite (index.db)
                                      │
                                      ├─ watches vault (Syncthing-synced)
                                      ├─ chunks markdown
                                      ├─ calls Gemini embeddings
                                      └─ serves /health /search /similar   (phase 2)
```

**Deployment:** single static Rust binary (musl or aarch64-linux), systemd user service on Hetzner, bound to Tailscale interface.

## Quick Reference

| Path | Purpose |
|---|---|
| `src/main.rs` | Binary entry point; parses config, opens store, emits structured ready log |
| `src/config.rs` | Env var parsing + validation (`DOCINDEX_*`, `GEMINI_API_KEY`), refuses `0.0.0.0` binds |
| `src/walk.rs` | Full-tree scan, `content_hash` diff, feeds dirty set to indexer |
| `src/chunk.rs` | Heading-aware chunker (H1/H2/H3 + ~500-token fallback, 50-token overlap) |
| `src/embed/mod.rs` | `Embedder` trait (native async fn), `EmbedError`, task-type constants |
| `src/embed/gemini.rs` | Gemini embeddings client; retries on 429/5xx, x-goog-api-key header |
| `src/embed/fake.rs` | Deterministic fake embedder for tests (sha256-seeded, L2-normalized) |
| `src/store/mod.rs` | SQLite handle + `sqlite-vec` auto-extension load, chunk/FTS/vec upsert |
| `src/store/schema.sql` | Canonical schema (chunks, chunks_fts, chunks_vec `vec0`, embedding_cache, meta) |
| `src/store/vec.rs` | Little-endian f32 encode/decode for vector BLOBs |
| `tests/run_tests.py` | Python pytest runner; builds the bin, runs suites/ |
| `docs/ARCHITECTURE.md` | Full system design |

> The table above reflects the intended layout. When files are added/moved, update this section **in the same commit**.

## Tech Stack

- **Language:** Rust (edition 2024, MSRV 1.90)
- **Async runtime:** `tokio` (current-thread; expand to multi-thread only if benchmarked need)
- **HTTP server:** TBD in Phase 2 (`axum` is the likely pick)
- **HTTP client:** `reqwest` with `rustls-tls` (no OpenSSL system dep)
- **SQLite:** `rusqlite` 0.34 with `bundled` + `load_extension` features (statically linked libsqlite3)
- **Vector search:** `sqlite-vec` 0.1.x — **loaded as a real SQLite extension** via `sqlite3_auto_extension`, exposing the `vec0` virtual table (`vec_distance_cosine` etc.)
- **FTS:** SQLite FTS5 (compiled into the bundled libsqlite3), `tokenize='porter unicode61'`
- **Embeddings:** Google `gemini-embedding-001`, Matryoshka dim 768, task-asymmetric (doc/query)
- **Hashing:** `sha2` + `hex`
- **Filesystem walker:** `walkdir`
- **File watcher:** TBD in Phase 2 (likely `notify` with a 5s debounce)
- **Errors:** `thiserror::Error` per module (`ConfigError`, `WalkError`, `EmbedError`, `StoreError`); `anyhow` only in `main.rs`
- **Logging:** `tracing` + `tracing-subscriber` (JSON in prod via `DOCINDEX_LOG_FORMAT=json`, text in dev)
- **Config:** env vars only (12-factor); no config files
- **Tests:** `cargo test` for unit/integration; Python `pytest` harness in `tests/` for end-to-end parity checks (spins up the binary, talks to it, validates the DB)
- **Deployment:** single static binary, systemd user service on Hetzner

## Endpoints

Phase 2 targets (not yet wired):

```
GET  /health                          → { ok, indexedChunks, lastReindex, embeddingModel, dim }
POST /search   { query, limit=10 }    → { hits: [{ path, title, headingPath, snippet, score, chunkId }] }
POST /similar  { path,  limit=10 }    → same shape
```

Auth: every non-`/health` endpoint will require `Authorization: Bearer <DOCINDEX_BEARER>`.
Bind: `DOCINDEX_LISTEN` **must be a Tailscale IP**, never `0.0.0.0` or `[::]` (enforced at startup in `config.rs`).

## Ranking (Hybrid, Phase 2)

1. Embed query with task type `RETRIEVAL_QUERY` (Gemini).
2. Top-30 via cosine (`sqlite-vec`, `vec_distance_cosine`).
3. Top-30 via BM25 (`FTS5`, `bm25(chunks_fts)`).
4. **Reciprocal Rank Fusion** (k=60) over the two lists.
5. Return top-10 with a snippet + metadata.

Don't normalize raw scores across the two lists — RRF avoids that rabbit hole.

## Schema (canonical)

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
  embedding FLOAT[768]
);

CREATE TABLE embedding_cache (
  content_hash TEXT PRIMARY KEY,
  model        TEXT    NOT NULL,
  task_type    TEXT    NOT NULL,
  dim          INTEGER NOT NULL,
  embedding    BLOB    NOT NULL,
  created_at   INTEGER NOT NULL
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- meta keys: embedding_model, embedding_dim, schema_version, last_full_scan
```

**Embedding cache** is keyed by `content_hash` — renaming/moving a file with identical content never re-embeds.

## Coding Standards

### Rust

- `cargo fmt --all` clean. `cargo clippy --all-targets --all-features -- -D warnings` before push.
- **No `unwrap()` / `expect()` / `panic!()` in library code.** Exceptions: `#[cfg(test)]` modules, `main.rs` terminating a startup failure, and places where an invariant is *proven* locally and documented.
- Propagate errors with `?`. Use `thiserror::Error` per module (`ConfigError`, `StoreError`, `EmbedError`, `WalkError`). Reserve `anyhow::Result` for `main.rs`.
- Prefer small free functions over god-structs. Types and modules should have one reason to change.
- Native async fn in traits (stable Rust ≥ 1.75). Don't pull in `async_trait` unless dyn-dispatch forces it.
- Tokio current-thread runtime by default. Justify multi-thread with a benchmark.
- `tracing::{info, warn, error, debug}` with structured fields — never `println!` / `eprintln!` in production paths. `main.rs` may emit a single ready line at info.
- `context::Context`-equivalent: every external call (DB, HTTP, embedder) accepts a timeout via config; per-request cancellation via tokio.
- Keep `main.rs` minimal: parse config, init tracing, open store, (phase 2) wire components and serve.
- Feature flags in `Cargo.toml` are fine, but don't create them speculatively.

### Naming Conventions

- Rust files: `snake_case.rs`; one concept per file.
- Types: `PascalCase`. Traits: capability-oriented names (`Embedder`, `Store`) — no `-Impl` / `-Service` suffixes.
- Functions / methods: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Env vars: `UPPER_SNAKE_CASE`, all prefixed `DOCINDEX_*` (except `GEMINI_API_KEY`).
- SQL tables/columns: `snake_case`.
- HTTP routes (Phase 2): `kebab-case`, lowercase.

### Error Handling

- Wrap external errors with `#[from]` in the module's `thiserror` enum. Add variants for semantic failures (e.g. `StoreError::DimMismatch { got, want }`).
- Return structured JSON errors from HTTP handlers (Phase 2): `{ "error": "...", "code": "..." }`.
- Never expose internal error details or stack traces to clients.
- Log at `error` for 5xx; `warn` for 4xx that indicates misconfiguration.
- Request timeouts: 30s default, configurable via `DOCINDEX_HTTP_TIMEOUT_MS`.

### Testing

- Every feature must have tests. No exceptions.
- Three layers:
  1. **Unit tests** colocated with source (`#[cfg(test)] mod tests { ... }`): chunker, RRF (Phase 2), auth middleware (Phase 2), config parsing, vec encode/decode.
  2. **Integration tests** using a real on-disk SQLite file (via `tempfile::TempDir`): walker + chunker + store + (Phase 2) search roundtrip.
  3. **End-to-end Python harness** under `tests/` (`uv`-managed, `pytest`): spins up `target/release/docindex` against a fixture vault, asserts behavior.
- Use `tempfile::TempDir` for temp vault/DB paths — no shared state across tests.
- Mock Gemini at the boundary (`src/embed/fake.rs` or `wiremock`): deterministic vectors keyed by content.
- `cargo test --all -- --nocapture` must pass cleanly; `cargo clippy --all-targets -- -D warnings` must be clean.

### Git

- Commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`.
- One concern per commit. Explain *why* in the body, not just *what*.
- Update `README.md`, `CLAUDE.md`, and `docs/` **in the same commit** as the code change.

## Architecture Patterns

### Walker + Watcher (single source of truth)

`walk.rs` runs once at startup and feeds a "dirty set" to the indexer. Phase 2 will add a `watch` module (likely `notify`-based) that emits dirty paths on fs events (debounced 5s). Both paths feed the **same** indexing pipeline in `store`/`embed` — there is no divergence.

When you change how a file is indexed, change it once in the pipeline; never duplicate logic between walker and watcher.

### Chunker contract

Input: raw markdown bytes. Output: an ordered `Vec<Chunk>`. Chunks are deterministic — same input → same output, including byte-identical content and identical `content_hash`. The chunker does **not** call the embedder or the store; it's pure. The only randomness allowed is `None`.

### Embedding cache

Before calling Gemini, the indexer hashes the chunk content (`sha256`) and consults `embedding_cache`. A hit returns the cached vector; a miss calls the API, then stores the vector keyed by hash. **Renames and reorganizations never re-embed** — this is load-bearing for cost.

### Hybrid search + RRF (Phase 2)

A single `search::hybrid` entry point will:
1. Call `embed::Embedder::embed_query` with task type `RETRIEVAL_QUERY`.
2. Run vec + FTS queries concurrently (`tokio::join!` or `try_join!`).
3. Fuse via `fuse_rrf(vec_hits, fts_hits, 60)` — pure, unit-tested.
4. Hydrate snippets + heading paths.

All ranking logic lives here; handlers never touch SQL directly.

### Tailscale-only bind + bearer auth

The bind address is validated at startup in `config.rs`: `0.0.0.0:*` and `[::]:*` are rejected. Bearer auth (Phase 2) is belt-and-suspenders; Tailscale is the primary boundary.

### sqlite-vec extension loading

`src/store/mod.rs` registers `sqlite-vec`'s C init function via `rusqlite::ffi::sqlite3_auto_extension` exactly once per process (`OnceLock`). Every subsequent `Connection::open` gets the extension loaded before any SQL runs, so `CREATE VIRTUAL TABLE ... USING vec0(...)` works in the schema bootstrap. `verify_vec_loaded` calls `vec_version()` right after open and errors hard if the extension isn't there — we never silently fall back to BLOB math.

## Common Pitfalls

- **sqlite-vec load order:** The extension MUST be registered (`register_sqlite_vec` in `store/mod.rs`) before the first `Connection::open`. `sqlite3_auto_extension` is process-global; guard it with `OnceLock`. Verify with `SELECT vec_version()` after opening — any error here means the rest of the store is broken.
- **`vec0` has no `INSERT OR REPLACE`:** To update a row in `chunks_vec`, `DELETE` then `INSERT` inside a transaction. A naive `UPSERT` will fail.
- **FTS5 sync:** `chunks_fts` is a contentless FTS5 table indexed on `chunks`. You must manually `INSERT`/`DELETE` into the FTS table when `chunks` changes — it doesn't auto-sync in SQLite. Use the `INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ...)` form for deletes.
- **Gemini task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search. Getting this wrong silently degrades quality — the vectors still work, they're just mis-calibrated.
- **Matryoshka dim:** Request `outputDimensionality: 768` at embed time. Storing 3072 "just in case" quadruples disk and slows ANN for zero recall gain at this scale.
- **Debounce state (Phase 2):** The watcher's debounce map must be keyed by absolute path, not relative. Symlink-containing vaults will dedupe incorrectly otherwise.
- **Unicode in FTS5:** Use `tokenize='porter unicode61'`, not the default — the default strips non-ASCII.
- **Headings with pipes:** `heading_path` uses `" > "` as a separator; don't use `"|"` or `"/"` (both appear in markdown headings).
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
- Every new endpoint (Phase 2) requires: bearer auth, timeout, structured error response, a test.
- No `unwrap()` / `expect()` / `panic!()` outside of `#[cfg(test)]`, `main.rs` startup errors, and provably-safe invariants.
