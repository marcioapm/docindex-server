-- schema_version = 1
-- Storage schema for docindex-server. Apply with CREATE IF NOT EXISTS so
-- repeated opens are idempotent.

CREATE TABLE IF NOT EXISTS chunks (
  id           INTEGER PRIMARY KEY,
  path         TEXT    NOT NULL,
  chunk_idx    INTEGER NOT NULL,
  heading      TEXT,
  heading_path TEXT,
  content      TEXT    NOT NULL,
  content_hash TEXT    NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  tokens       INTEGER,
  UNIQUE(path, chunk_idx)
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- Contentless FTS5 index over chunks. Not auto-synced — the store layer
-- is responsible for INSERT/DELETE into chunks_fts whenever chunks changes.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content, heading_path,
  content=chunks, content_rowid=id,
  tokenize='porter unicode61'
);

-- Phase 1 vector storage: a plain BLOB table keyed by chunks.id. Phase 2
-- will replace this with `USING vec0(embedding FLOAT[768])` once the
-- sqlite-vec toolchain is pinned. The byte layout (little-endian float32)
-- is identical to what vec0 consumes, so the migration is a table swap.
CREATE TABLE IF NOT EXISTS chunks_vec (
  rowid     INTEGER PRIMARY KEY,
  embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_cache (
  content_hash TEXT PRIMARY KEY,
  model        TEXT    NOT NULL,
  task_type    TEXT    NOT NULL,
  dim          INTEGER NOT NULL,
  embedding    BLOB    NOT NULL,
  created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
