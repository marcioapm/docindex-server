# docindex-server

Semantic + BM25 search server for personal docs/notes. Indexes a folder of markdown, serves a tiny HTTP API over Tailscale. Built to power the [`docindex-obsidian`](https://github.com/marcioapm/docindex-obsidian) plugin and the `docindex-search` CLI — anything that wants ranked retrieval against a personal vault.

## What it indexes
Files with extension `.md` or `.txt` (case-insensitive) anywhere under the configured vault directory. Dot-files, `.git`, `.obsidian`, and `node_modules` are skipped. Heading-less files (plain `.txt`) flow through the chunker's 500-word fallback path.

## Stack
- **Language:** Rust (edition 2024, MSRV 1.90)
- **Storage:** SQLite (`rusqlite`, bundled) + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vectors, `vec0` cosine) + FTS5 (BM25)
- **Embeddings:** provider/model registry (below) — Google `gemini-embedding-2` (default; text+image+PDF), `gemini-embedding-001` (text-only, explicit legacy), Voyage AI `voyage-multimodal-3.5` (text+image+PDF), Voyage text family (`voyage-4` family), or a deterministic `fake` backend for tests
- **Async runtime:** `tokio` (multi-thread)
- **HTTP server:** `axum` 0.8
- **HTTP client:** `reqwest` + rustls
- **Watcher:** `notify` with a debounced dirty-set (default 5s, configurable)
- **CLI:** `clap` (derive), `terminal_size` for snippet wrapping
- **Config:** `toml` for file-based config, layered under env vars
- **Transport:** Tailscale-only bind (config rejects `0.0.0.0` / `[::]`), bearer-token auth

## Supported providers / models

| Provider | Model | Native dim | Allowed dims | Media | Doc task | Query task |
|---|---|---|---|---|---|---|
| `gemini` | `gemini-embedding-2` (**default**) | 3072 | 768, 1536, 3072 | text+image+PDF | text prefix | text prefix |
| `gemini` | `gemini-embedding-001` | 3072 | 768, 1536, 3072 | text only | `RETRIEVAL_DOCUMENT` | `RETRIEVAL_QUERY` |
| `voyage` | `voyage-4` (**default**) | 1024 | 256, 512, 1024, 2048 | text only | `document` | `query` |
| `voyage` | `voyage-4-lite` | 1024 | 256, 512, 1024, 2048 | text only | `document` | `query` |
| `voyage` | `voyage-4-large` | 1024 | 256, 512, 1024, 2048 | text only | `document` | `query` |
| `voyage` | `voyage-context-4` | 1024 | 256, 512, 1024, 2048 | text only | `document` | `query` |
| `voyage` | `voyage-code-3` | 1024 | 256, 512, 1024, 2048 | text only | `document` | `query` |
| `voyage` | `voyage-multimodal-3.5` | 1024 | 256, 512, 1024, 2048 | text+image+PDF | `document` | `query` |
| `fake` | any name | — | any | text+image+PDF | `document` | `query` |

Smaller allowed dims are Matryoshka truncations — cheaper to store and search, small quality tradeoff. `fake` is deterministic (sha256-seeded, L2-normalized) and used only in tests / dev without a real API key.

Unknown provider, unknown model for a provider, or a `dim` outside a model's allowed set all fail startup with an error listing the valid choices. A provider that needs a key and has none resolved fails startup naming exactly which env var (or config key) to set.

## Config

**Precedence, highest wins:** CLI flags > environment variables > TOML file > built-in defaults. The `docindex` server binary is driven entirely by env vars today (see `.env.example`) and that keeps working unchanged — a config file is optional.

### Server config file search order (first existing file wins)
1. `--config <path>`
2. `$DOCINDEX_CONFIG`
3. `~/.config/docindex/server.toml` (or `$XDG_CONFIG_HOME/docindex/server.toml`)
4. `/etc/docindex/server.toml`

### Server TOML schema
```toml
vault_dir        = "/var/lib/docindex/vault"
db_path          = "/var/lib/docindex/index.db"
listen           = "100.64.0.1:7777"
bearer           = "..."            # or bearer_env = "DOCINDEX_BEARER"
debounce_ms      = 5000
http_timeout_ms  = 30000
log_format       = "json"           # json | text
allow_loopback   = false

[embed]
provider  = "gemini"                # gemini | voyage | fake
model     = "gemini-embedding-2"    # stable default; 001 remains explicit text-only compatibility
# Switching an existing gemini-embedding-001 DB requires --reembed.
dim       = 3072                    # optional; model native dim when omitted
api_key   = "..."                   # or api_key_env = "GEMINI_API_KEY"
base_url  = "https://..."           # optional override (proxy/mock); NOT part of the index fingerprint

[media]
# Opt-in image/PDF indexing. Requires a media-capable model.
enabled = false
include = ["Attachments/**", "Papers/**"]
exclude = ["**/Private/**", "**/Thumbnails/**"]
exclude_types = ["audio", "video"] # image | pdf | audio | video
max_file_mb = 20
pdf_pages_per_chunk = 1              # Google recommends one PDF page for quality
pdf_dpi = 150
```

### Media indexing
Media is opt-in and defaults off. Images (`png`, `jpg`, `jpeg`, `webp`, `gif`) and PDFs are indexed when `[media].enabled = true` and the configured model is media-capable: Gemini `gemini-embedding-2`, Voyage `voyage-multimodal-3.5`, or the offline fake provider. Startup rejects media with legacy Gemini `gemini-embedding-001` or Voyage text models rather than leaving files perpetually dirty. `gemini-embedding-2` is the stable Gemini default; migrating an existing 001 index requires a complete `--reembed`, because vector spaces never coexist.

Image format is detected from bytes, not filenames. PNG/JPEG inputs are retained when within the pixel limit; GIF/WebP use their first frame as PNG, and oversized images are downscaled before embedding. PDFs default to **one page per chunk** because Google recommends one page for quality and it avoids silent context truncation. Gemini receives page-subset PDFs; Voyage receives rendered page PNGs. These provider details are internal: all indexed results share the one vector space of the configured model.

Text (`md`, `txt`) remains always eligible. Paths are vault-relative and normalized to `/` before matching. An empty `include` admits all supported media; a nonempty `include` admits matching paths only; `exclude` removes matching paths and always wins. `exclude_types` can contain `image`, `pdf`, `audio`, and `video`; audio/video are accepted as reserved exclusion labels but are not indexed.

`max_file_mb` is an eligibility limit. The media processing profile used in effective media hashes contains `media-v1`, `max_file_mb`, `pdf_pages_per_chunk`, and `pdf_dpi`; changing it reindexes still-eligible media at the next scan. Include/exclude/type changes are eligibility-only. Provider/model/dim changes require `--reembed`. Provider pricing and usage tiers are account-specific; consult the provider's official pricing before enabling a paid model.

### CLI config file (`docindex-search`)
Search order: `--config <path>` > `$DOCINDEX_CLI_CONFIG` > `~/.config/docindex/cli.toml`.

```toml
server = "http://100.64.0.1:7777"
token  = "..."                      # or token_env = "DOCINDEX_BEARER"
limit  = 10
format = "text"                     # text | json
```

### Secrets
`*_env` indirection (`bearer_env`, `api_key_env`, `token_env`) reads the named environment variable instead of an inline value — keeps a token out of a file that might get committed by accident. If a config file is readable by group or other (`mode & 0o077 != 0`) **and** contains an inline secret, startup logs a warning naming the file and its mode; it does not refuse to start.

### Env vars
See `.env.example` for the full list. Provider/model/dim: `DOCINDEX_EMBED` (`gemini`/`voyage`/`fake`), `DOCINDEX_EMBED_MODEL`, `DOCINDEX_EMBED_DIM`, `GEMINI_API_KEY` / `VOYAGE_API_KEY`. `DOCINDEX_ALLOW_LOOPBACK=true` is dev/test only — production must leave it unset.

## CLI

```sh
docindex-search "some query"              # default subcommand = search
docindex-search "diagram architecture" --media # server-side all-media search
docindex-search "invoice total due" --media pdf # server-side PDF-only search
docindex-search search "q" -n 5 --json
docindex-search similar path/to/note.md
docindex-search health
```

Flags: `-n/--limit`, `--media [image,pdf,audio,video]` (server-side media ranking; omit its value for all media; search queries only), `--json` (emit the server response verbatim), `--server <url>`, `--token <tok>`, `--config <path>`, `--path-filter <prefix>` (client-side filter on returned `path`).

Human output, one hit per line, snippet truncated to terminal width (or 200 chars):
```
1. 0.79  Rax/holdouts-prompt.md › Holdouts Feature > 8. Key Design Decisions
        - Holdout split format: `[holdout_fraction, 1.0 - holdout_fraction]`...
```

**Exit codes:** `0` ok, `1` usage/config error, `2` network/server error, `3` authentication failure (401/403 or `health` returns `authenticated: false`), `4` no results. Errors go to stderr; stdout is reserved for results.

## Endpoints
```
GET  /health                          -> without/invalid bearer: { ok, authenticated: false }
                                      -> valid bearer: { ok, authenticated: true, indexed_chunks, last_reindex_ms, embedding_model, dim }
POST /search   { query, limit=10, media_only=false, media_types=[] } -> { hits: [{ path, title, heading_path, snippet, score, score_rrf, score_normalized, chunk_id, media_type, mime_type, media_start, media_end, media_unit, truncated }] }
POST /similar  { path,  limit=10 }                    -> same shape
```
`/health` always returns `200 OK`: without a valid bearer it is a liveness probe only and must not be used to verify credentials; with `Authorization: Bearer $DOCINDEX_BEARER`, it returns operational detail and `authenticated: true`. `/search` and `/similar` require the bearer and return `401` for missing or invalid credentials. `limit` is clamped to [1, 50]. Errors return `{error, code}` JSON.

`media_types` is optional and valid only with `media_only=true`. It accepts `image`, `pdf`, `audio`, and `video`; an empty list preserves all-media search.

Every `hit.path` (and the `path` accepted by `/similar`) is **vault-relative** — e.g. `"notes/foo.md"`, never `/home/…/vault/notes/foo.md`. This matches Obsidian's `TFile.path` so the plugin can feed paths straight into `openLinkText()` / `getAbstractFileByPath()` without rewriting. Databases created before v0.2.0 are migrated in place on first open: any `chunks.path` / `files.path` that are absolute but still inside the vault dir get rewritten atomically, and rows pointing outside the vault cause the migration to refuse (logged, not fatal) so operators can reconcile. Once complete, `meta.path_schema_version = 1` short-circuits the scan on every subsequent boot.

## Ranking
Hybrid: top-30 cosine (sqlite-vec `chunks_vec` cosine) + top-30 BM25 (FTS5), fused with Reciprocal Rank Fusion (`k=60`). Every hit also carries `score_normalized` (0..1, query-independent) for display + threshold filtering — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the `DOCINDEX_DISPLAY_K` / `DOCINDEX_WEIGHT_VEC` / `DOCINDEX_WEIGHT_BM25` env vars.

## Chunking
Heading-aware (H1/H2/H3), ~500-token fallback, 50-token overlap. Stored per chunk: `(path, chunk_idx, heading_path, content, content_hash, mtime_ns)`.

## Reindex + fingerprint
- Startup: full walk, diff against `files.content_hash`.
- Live: `notify` + debounce (default 5s, `DOCINDEX_DEBOUNCE_MS`), batch dirty files.
- Embedding cache keyed by `content_hash` (survives renames/moves).
- The index is fingerprinted with the `(provider, model, dim)` it was built with (`base_url` is excluded — pointing at a proxy/mock never invalidates a good index). A mismatch on startup refuses with an error naming every changed field and both values; pass `--reembed` to wipe chunks/vectors/FTS and rebuild at the new settings. See ["Changing the embedding dim, provider, or model"](docs/deployment.md#changing-the-embedding-dim-provider-or-model) in the deployment guide.

## Clients

- [`docindex-obsidian`](https://github.com/marcioapm/docindex-obsidian) — Obsidian desktop/mobile client with opt-in remote semantic + BM25 search.
- `docindex-search` — CLI shipped by this repository.

## Deployment
- Host: any Linux VPS, bound to the Tailscale interface.
- Process: systemd user service.
- Config: TOML file and/or env vars (see [Config](#config) above and `.env.example`).
- Full guide: [`docs/deployment.md`](docs/deployment.md).

## Quick start

```sh
cp .env.example .env          # then edit values
make test                     # cargo test --all
make build-release             # -> target/release/docindex, target/release/docindex-search
set -a; source .env; set +a   # export env vars
./target/release/docindex     # serves HTTP until SIGINT/SIGTERM
```

Local dev without a real embedding API key:
```sh
export DOCINDEX_VAULT_DIR=/tmp/vault DOCINDEX_DB_PATH=/tmp/index.db \
       DOCINDEX_LISTEN=127.0.0.1:7777 DOCINDEX_ALLOW_LOOPBACK=true \
       DOCINDEX_BEARER=dev DOCINDEX_EMBED=fake
./target/release/docindex
./target/release/docindex-search "hello" --server http://127.0.0.1:7777 --token dev
```

Run the Python end-to-end harness (builds both binaries, exercises config layering, Voyage against a local mock server, the CLI, and the fingerprint guard, plus the pre-existing suites):
```sh
make pytest
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`CLAUDE.md`](CLAUDE.md) for coding standards.

## License
MIT — see [LICENSE](LICENSE).
