-- schema_version = 1
-- Storage schema for docindex-server. Applied with CREATE IF NOT EXISTS so
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

-- Contentless FTS5 index over chunks. Manually synced: every insert/update/
-- delete on `chunks` must also hit `chunks_fts`.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content, heading_path,
  content=chunks, content_rowid=id,
  tokenize='porter unicode61'
);

-- sqlite-vec vec0 virtual table: packed little-endian float32, 768-dim.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
  embedding FLOAT[768]
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
