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
DOCINDEX_LISTEN=100.83.46.59:7777
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
curl http://100.83.46.59:7777/health
curl -H "Authorization: Bearer $DOCINDEX_BEARER" \
    -X POST http://100.83.46.59:7777/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"hello","limit":5}'
```

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

## Changing the embedding dim

`DOCINDEX_EMBED_DIM` is baked into the `chunks_vec` DDL *and* cached in `meta.embedding_dim`. The store refuses to open when the two disagree — you'll see:

```
store: embedding_dim on disk is <stored>, config says <new> — refusing to mix. Delete index.db to reindex at new dim.
```

To switch dim (e.g. 768 → 3072):

```sh
systemctl --user stop docindex-server
# Update the env file:
#   sed -i 's/^DOCINDEX_EMBED_DIM=.*/DOCINDEX_EMBED_DIM=3072/' ~/.config/docindex/env
rm ~/index.db ~/index.db-wal ~/index.db-shm 2>/dev/null || true
systemctl --user start docindex-server
journalctl --user -u docindex-server -f   # watch the reindex
```

The initial scan will re-embed every file at the new dim. Embedding cache rows at the previous dim are swept on open, so there's no chance of mixed-dim reads.
