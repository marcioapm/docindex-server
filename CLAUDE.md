# docindex-server — Agent Guide

> Tiny Go server that indexes a markdown vault and serves semantic + BM25 search over a Tailscale-only HTTP API. Powers an Obsidian mobile plugin (and anything else that wants ranked retrieval).

Read `README.md` for setup. Run `make run` (once a Makefile exists) to start locally; `systemctl --user status docindex-server` on Hetzner.

## Architecture at a Glance

```
docindex-server/
├── cmd/docindex/           # main.go — entry point
├── internal/
│   ├── config/             # env parsing
│   ├── walk/               # initial full-tree scan + content-hash diff
│   ├── watch/              # fsnotify + 5s debounce
│   ├── chunk/              # heading-aware markdown chunker
│   ├── embed/              # Gemini embeddings client (doc/query task types)
│   ├── store/              # SQLite + sqlite-vec + FTS5 wiring
│   │   └── schema.sql      # canonical schema
│   ├── search/             # hybrid ranking + RRF fusion
│   └── api/                # http handlers, bearer auth, Tailscale bind
├── docs/
│   ├── ARCHITECTURE.md     # system design
│   └── deployment.md       # systemd unit, env vars, firewall
├── scripts/                # deploy/release helpers
└── Makefile

Obsidian mobile ──Tailscale──►  docindex-server  ──►  SQLite (index.db)
                                      │
                                      ├─ watches vault (Syncthing-synced)
                                      ├─ chunks markdown
                                      ├─ calls Gemini embeddings
                                      └─ serves /health /search /similar
```

**Deployment:** single static Go binary, systemd user service on Hetzner, bound to Tailscale interface.

## Quick Reference

| Path | Purpose |
|---|---|
| `cmd/docindex/main.go` | Binary entry point; wires config → store → walker → watcher → api |
| `internal/config/config.go` | Env var parsing + validation (`DOCINDEX_*`, `GEMINI_API_KEY`) |
| `internal/walk/walker.go` | Full-tree scan, `content_hash` diff, feeds dirty set to indexer |
| `internal/watch/watcher.go` | `fsnotify` wrapper with 5s debounce, batched per-file events |
| `internal/chunk/markdown.go` | Heading-aware chunker (H1/H2/H3 + 500-token fallback, 50-token overlap) |
| `internal/embed/gemini.go` | Gemini embeddings client; task types `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` |
| `internal/store/sqlite.go` | SQLite handle, migrations, sqlite-vec extension load |
| `internal/store/schema.sql` | Canonical schema (chunks, chunks_fts, chunks_vec, embedding_cache, meta) |
| `internal/search/hybrid.go` | Vec top-30 ∪ BM25 top-30 → RRF (k=60) → top-10 |
| `internal/api/router.go` | HTTP routes: `GET /health`, `POST /search`, `POST /similar` |
| `internal/api/auth.go` | Bearer-token middleware |
| `docs/ARCHITECTURE.md` | Full system design |
| `docs/deployment.md` | systemd unit, env vars, production notes |

> The table above reflects the intended layout. When files are added/moved, update this section **in the same commit**.

## Tech Stack

- **Language:** Go (1.22+)
- **HTTP:** `net/http` + `chi` router (small, idiomatic)
- **SQLite:** `modernc.org/sqlite` (pure-Go, no CGo) + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension + FTS5
- **File watcher:** `fsnotify`
- **Embeddings:** Google `gemini-embedding-001`, Matryoshka dim 768, task-asymmetric (doc/query)
- **Logging:** `log/slog` (structured, JSON in prod, text in dev)
- **Config:** env vars only (12-factor); no config files
- **Tests:** `go test ./...` (standard library + `testify` where it helps)
- **Deployment:** single static binary, systemd user service on Hetzner

## Endpoints

```
GET  /health                          → { ok, indexedChunks, lastReindex, embeddingModel, dim }
POST /search   { query, limit=10 }    → { hits: [{ path, title, headingPath, snippet, score, chunkId }] }
POST /similar  { path,  limit=10 }    → same shape
```

Auth: every non-`/health` endpoint requires `Authorization: Bearer <DOCINDEX_BEARER>`.
Bind: `DOCINDEX_LISTEN` **must be a Tailscale IP**, never `0.0.0.0`.

## Ranking (Hybrid)

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

### Go

- `gofmt` + `go vet` clean, no exceptions. `golangci-lint run` before push.
- Return `error` explicitly everywhere — never panic in request handlers.
- Use `context.Context` on every external call (DB, HTTP, embedding API) with per-request timeouts.
- Structured logging via `log/slog` — never `fmt.Println` / `log.Printf` in production code.
- Keep handlers thin — extract business logic into `internal/*` packages.
- No global state except `slog` default logger and the `http.Client` for outbound calls (reused for connection pooling).
- Prefer small interfaces at call sites, not giant service structs.
- Decimal/money: N/A here, but be careful with floats — cosine scores are fine, but never round-trip floats through strings.
- Keep `main.go` minimal: parse config, wire components, `http.ListenAndServe`.

### Naming Conventions

- Go files: `snake_case.go` **no**, Go uses `lowercase.go`; one concept per file.
- Go types: `PascalCase`. Interfaces: `-er` suffix when it fits.
- Go functions: `PascalCase` (exported) / `camelCase` (unexported).
- Env vars: `UPPER_SNAKE_CASE`, all prefixed `DOCINDEX_*` (except `GEMINI_API_KEY`).
- SQL tables/columns: `snake_case`.
- HTTP routes: `kebab-case`, lowercase.

### Error Handling

- Wrap errors with `fmt.Errorf("context: %w", err)` — always preserve the chain.
- Return structured JSON errors from HTTP handlers: `{ "error": "...", "code": "..." }`.
- Never expose internal error details or stack traces to clients.
- Log at `error` level for 5xx; `warn` for 4xx that indicates misconfiguration.
- Request timeouts: 30s default, configurable; handler-level deadlines, not just client-level.

### Testing

- Every feature must have tests. No exceptions.
- Three layers:
  1. **Unit tests** colocated with source (`*_test.go`): chunker, RRF, auth middleware, config parsing.
  2. **Integration tests** in `internal/*/*_test.go` using a temp SQLite file: walker + chunker + store + search roundtrip.
  3. **End-to-end test** in `cmd/docindex/e2e_test.go`: spin up server on ephemeral port with a fixture vault, assert `/search` returns expected paths.
- Use `t.TempDir()` for temp vault/DB paths — no shared state across tests.
- Mock Gemini at the boundary (`internal/embed/`): provide a fake that returns deterministic vectors keyed by content.
- `go test ./... -race` must pass.

### Git

- Commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`.
- One concern per commit. Explain *why* in the body, not just *what*.
- Update `README.md`, `CLAUDE.md`, and `docs/` **in the same commit** as the code change.

## Architecture Patterns

### Walker + Watcher (single source of truth)

The `walk` package runs once at startup and feeds a "dirty set" to the indexer. The `watch` package then takes over, emitting dirty paths on fs events (debounced 5s). Both paths feed the **same** indexing pipeline in `internal/store`/`internal/embed` — there is no divergence.

When you change how a file is indexed, change it once in the pipeline; never duplicate logic between walker and watcher.

### Chunker contract

Input: raw markdown bytes. Output: an ordered slice of `Chunk` structs. Chunks are deterministic — same input → same output, including byte-identical content and identical `content_hash`. The chunker does **not** call the embedder or the store; it's pure.

### Embedding cache

Before calling Gemini, the indexer hashes the chunk content (`sha256`) and consults `embedding_cache`. A hit returns the cached vector; a miss calls the API, then stores the vector keyed by hash. **Renames and reorganizations never re-embed** — this is load-bearing for cost.

### Hybrid search + RRF

`internal/search/hybrid.go` is the single entry point. It:
1. Calls `internal/embed` with task type `RETRIEVAL_QUERY`.
2. Runs two SQL queries in parallel (errgroup).
3. Fuses via `fuseRRF(vecHits, ftsHits, k=60)` — pure function, unit-tested independently.
4. Hydrates snippets + heading paths.

All ranking logic lives here; handlers never touch SQL directly.

### Tailscale-only bind + bearer auth

The bind address is validated at startup: if `DOCINDEX_LISTEN` is `0.0.0.0:*` or a non-Tailscale IP, the server refuses to start. Bearer auth is belt-and-suspenders; Tailscale is the primary boundary.

## Common Pitfalls

- **CGo:** We deliberately use `modernc.org/sqlite` (pure Go) so the binary is static. Don't add `mattn/go-sqlite3` — it re-introduces CGo + a cross-compile toolchain.
- **sqlite-vec loading:** The extension must be loaded per-connection. Use a connection pool of size 1 or load on every `*sql.Conn` via `Raw(...)`.
- **FTS5 sync:** `chunks_fts` is a contentless FTS5 table indexed on `chunks`. You must manually `INSERT`/`DELETE` into the FTS table when `chunks` changes — it doesn't auto-sync in SQLite.
- **Gemini task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search. Getting this wrong silently degrades quality — the vectors still work, they're just mis-calibrated.
- **Matryoshka dim:** Request dim=768 at embed time. Storing 3072 "just in case" quadruples disk and slows ANN for zero recall gain at this scale.
- **Debounce state:** The watcher's debounce map must be keyed by absolute path, not relative. Symlink-containing vaults will dedupe incorrectly otherwise.
- **Unicode in FTS5:** Use `tokenize='porter unicode61'`, not the default — the default strips non-ASCII.
- **Headings with pipes:** `heading_path` uses `" > "` as a separator; don't use `"|"` or `"/"` (both appear in markdown headings).

## Rules

**Every commit:**
- `go fmt ./...`, `go vet ./...`, `golangci-lint run`, `go test ./... -race` — all must pass.
- Update `README.md`, `CLAUDE.md`, and `docs/ARCHITECTURE.md` when architecture or structure changes.
- Write tests for every new feature or bug fix.

**After significant changes** (chunker, ranking, schema, API shape):
- Run the full E2E test against a fixture vault.
- Run the subagent reviewers in `.claude/agents/` (at minimum `code-reviewer` + `code-simplifier`).

**Pre-push CI check (mandatory):**
1. `go fmt ./...`
2. `go vet ./...`
3. `golangci-lint run`
4. `go test ./... -race -cover`
5. `go build ./cmd/docindex`

- Never bind to `0.0.0.0`.
- Never hard-code the bearer token; always from env.
- Never log the bearer, the Gemini API key, or the full content of a chunk at info level.
- Every new endpoint requires: bearer auth, timeout, structured error response, a test.
- No unwrap-equivalents in Go (`panic(err)`, `must(...)`) outside of `init()` and tests.
