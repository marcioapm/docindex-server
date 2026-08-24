# Deployment

Single static binary, systemd user service, Tailscale-bound.

## Build

Cross-compiling for Linux from macOS (aarch64 example):

```sh
# Install the target once:
rustup target add aarch64-unknown-linux-musl
brew install filosottile/musl-cross/musl-cross

# Build
CC_aarch64_unknown_linux_musl=aarch64-linux-musl-gcc \
    cargo build --release --target aarch64-unknown-linux-musl

# Binary at:
ls target/aarch64-unknown-linux-musl/release/docindex
```

Or build on the target host directly: `cargo build --release`.

## Copy

```sh
scp target/aarch64-unknown-linux-musl/release/docindex docindex-host:~/bin/
ssh docindex-host 'chmod +x ~/bin/docindex'
```

## Environment

Create `~/.config/docindex/env` on the host:

```sh
DOCINDEX_VAULT_DIR=/home/docindex/vault
DOCINDEX_DB_PATH=/home/docindex/index.db
# Use the Tailscale IP assigned to this host:
DOCINDEX_LISTEN=100.64.0.1:7777
DOCINDEX_BEARER=<random 32-char secret>
GEMINI_API_KEY=<from Google AI Studio>
DOCINDEX_EMBED_MODEL=gemini-embedding-001
# 3072 is gemini-embedding-001's native dim. 768/1536 are valid Matryoshka
# truncations if you want to trade a little quality for disk/ANN cost.
DOCINDEX_EMBED_DIM=3072
DOCINDEX_DEBOUNCE_MS=5000
DOCINDEX_HTTP_TIMEOUT_MS=30000
DOCINDEX_LOG_FORMAT=json
```

Never set `DOCINDEX_ALLOW_LOOPBACK=true` in production.

Get the Tailscale IP with `tailscale ip -4`.

## systemd user unit

`~/.config/systemd/user/docindex-server.service`:

```ini
[Unit]
Description=docindex-server — markdown semantic search
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/docindex/env
ExecStart=%h/bin/docindex
Restart=on-failure
RestartSec=3s

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/vault %h
PrivateTmp=true

[Install]
WantedBy=default.target
```

Enable + start:

```sh
loginctl enable-linger docindex    # run without an active session
systemctl --user daemon-reload
systemctl --user enable --now docindex-server
systemctl --user status docindex-server
journalctl --user -u docindex-server -f
```

## Firewall (UFW)

The server binds the Tailscale interface directly — public network never sees it. Belt-and-suspenders, block the port on the public interface:

```sh
sudo ufw deny 7777/tcp
sudo ufw reload
```

Traffic from Tailscale (`tailscale0`) is allowed by default interface policy; UFW's `deny 7777/tcp` only affects the public NIC.

## Smoke test

From any Tailscale peer:

```sh
# Public liveness only; this does not verify credentials.
curl http://100.64.0.1:7777/health

# Authenticated health returns index and embedding details and verifies the bearer.
curl -H "Authorization: Bearer $DOCINDEX_BEARER" \
    http://100.64.0.1:7777/health
curl -H "Authorization: Bearer $DOCINDEX_BEARER" \
    -X POST http://100.64.0.1:7777/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"hello","limit":5}'
```

`GET /health` always returns `200 OK`. A request without a valid bearer receives only `{ "ok": true, "authenticated": false }`; use that shape only for liveness monitoring, never to verify credentials. A valid bearer receives `{ "ok": true, "authenticated": true, "indexed_chunks", "last_reindex_ms", "embedding_model", "dim" }`.

## Upgrade

```sh
scp docindex docindex-host:~/bin/docindex.new
ssh docindex-host '
    mv ~/bin/docindex ~/bin/docindex.prev &&
    mv ~/bin/docindex.new ~/bin/docindex &&
    systemctl --user restart docindex-server
'
```

Roll back by restoring `~/bin/docindex.prev` and restarting.

## Schema migrations

The DB version is stored in `meta.schema_version`. Phase 2 is at `2`. When bumping the schema:

1. Back up `index.db` first (`cp index.db index.db.bak`).
2. Deploy a binary that handles the new version.
3. On first boot it will apply migrations (or error if the on-disk version is ahead).

## Path-normalization migration (v0.2.0+)

Databases indexed before v0.2.0 stored absolute paths in `chunks.path` / `files.path` (e.g. `/home/docindex/vault/notes/foo.md`). Since v0.2.0 all rows are vault-relative (`notes/foo.md`).

**You do not need to wipe the DB or re-index** — on first boot of a v0.2.0+ binary, `Store::migrate_paths_to_relative(vault_dir)` runs in a single transaction:

- If every absolute row lies inside the configured `DOCINDEX_VAULT_DIR`, `chunks.path` and `files.path` are rewritten with `substr(path, length(vault_dir) + 2)` and `meta.path_schema_version` is set to `1`.
- If any row points outside the vault (e.g. the vault was moved without also moving the DB), the migration **refuses** — it logs the offending row count and proceeds without touching data. The indexer will still start, but searches may miss those rows until you reconcile (either move the DB back next to the original vault, or accept the re-index and wipe).
- Once `path_schema_version = 1` is set, subsequent boots short-circuit the scan — the check is a single `meta` read.

What you'll see in the log on a successful first migration:

```
INFO migrated N rows from absolute to relative paths (chunks=A, files=B, vault=...)
```

And on an already-migrated DB:

```
DEBUG path migration: already at path_schema_version=1
```

## Changing the embedding dim, provider, or model

The index is fingerprinted on first boot: `meta.embedding_provider` /
`embedding_model` / `embedding_dim` record what it was built with. On every
subsequent boot the effective config (TOML + env + flags) is compared
against that fingerprint:

- match → starts normally.
- mismatch → refuses to start, naming every changed field and both values:

  ```
  index built with provider=gemini model=gemini-embedding-001 dim=3072,
  config says provider=voyage model=voyage-4 dim=1024; re-embed required:
  run with --reembed
  ```

To switch provider/model/dim (e.g. gemini 3072 → voyage-4 1024, or just a
dim change like 768 → 3072):

```sh
systemctl --user stop docindex-server
# Update ~/.config/docindex/env or server.toml with the new [embed] settings.
docindex --reembed   # or add --reembed to the systemd ExecStart temporarily
```

`--reembed` wipes `chunks`, `files`, `embedding_cache`, `chunks_fts`, and
`chunks_vec` in one transaction, recreates `chunks_vec` at the new dim, and
writes the new fingerprint — then the normal startup scan re-embeds every
file. Remove `--reembed` from the unit after the first successful boot; it
is not meant to stay in the permanent `ExecStart`. `base_url` (proxy/mock
overrides) is intentionally excluded from the fingerprint.

`(provider, model, dim)` is baked into `chunks_vec`'s DDL (dim as a SQL
literal) *and* cached in `meta`; there's no way to mix reads across dims —
the store refuses to open on a raw dim-only mismatch even before the
fingerprint check runs.

## Tuning display + threshold

Three env vars control the 0..1 `score_normalized` field the plugin uses for
"% relevance" and the relevance-threshold filter. None of them affect ranking
— ranking is always RRF with `k=60`.

- `DOCINDEX_DISPLAY_K` (default `10`) — smoothing constant for the display
  normalization. Smaller = steeper falloff past rank-1. At the default, rank-1
  in both branches scores `1.0`, rank-10 scores `~0.55`, rank-20 scores
  `~0.37`.
- `DOCINDEX_WEIGHT_VEC` (default `0.55`) — weight of the semantic branch.
- `DOCINDEX_WEIGHT_BM25` (derived as `1 - DOCINDEX_WEIGHT_VEC` when unset) —
  weight of the BM25 branch. If both are set explicitly they must sum to
  `1.0 ± 0.01` or startup fails.

The formula:

```
branch_norm(rank, K)   = (K + 1) / (K + rank)     if in list, else 0
score_normalized(doc)  = W_VEC  * branch_norm(v_rank, K)
                       + W_BM25 * branch_norm(b_rank, K)
```

Rule of thumb:
- Bump `DOCINDEX_WEIGHT_VEC` toward `0.7` if you want semantic retrieval to
  dominate (conceptual queries, exploratory search).
- Drop it to `~0.40` if your vault is very keyword-heavy (API references,
  code snippets, glossary-style notes).
- Lower `DOCINDEX_DISPLAY_K` (e.g. `5`) to make the default threshold behave
  stricter — only top-5ish results will clear 0.40.

Changes take effect on restart. No DB migration needed; the formula is
stateless.
