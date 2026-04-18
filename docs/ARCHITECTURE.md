# Architecture

## Goals
- Semantic + BM25 search over a markdown vault.
- Mobile-reachable over Tailscale.
- Single binary, SQLite storage, minimal ops.

## Components
- **Walker**: initial full scan, diff against `content_hash` -> dirty set.
- **Chunker**: heading-aware (H1/H2/H3) + ~500-token fallback with 50-token overlap.
- **Embedder**: Gemini `gemini-embedding-001`, Matryoshka dim 768, task-asymmetric (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search).
- **Store**: SQLite + sqlite-vec + FTS5.
- **Watcher**: fsnotify, 5s debounce, batched per file.
- **API**: `/health`, `/search`, `/similar` + bearer auth middleware.

## Schema
See `internal/store/schema.sql`.

## Ranking (Hybrid)
1. Embed query with `RETRIEVAL_QUERY`.
2. Top-30 via cosine (sqlite-vec).
3. Top-30 via BM25 (FTS5).
4. Fuse with Reciprocal Rank Fusion (k=60).
5. Return top-10 with snippet.

## Embedding cache
Keyed by `content_hash`. Rename/move of a chunk with identical text skips the API call.

## Deployment
- systemd unit on Hetzner.
- Binds to Tailscale IP.
- UFW continues to block public ingress on the service port.
