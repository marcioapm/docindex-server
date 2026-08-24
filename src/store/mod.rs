//! SQLite + sqlite-vec + FTS5 store.

mod vec;

use std::path::Path;
use std::sync::OnceLock;

use rusqlite::{Connection, OptionalExtension, ffi::sqlite3_auto_extension, params};
use thiserror::Error;
use tracing::{info, warn};

use crate::chunk::Chunk;

pub use self::vec::{decode_f32, encode_f32, vec_schema_ddl};

const SCHEMA_SQL: &str = include_str!("schema.sql");

/// Current schema version, written to meta on open.
pub const SCHEMA_VERSION: &str = "3";

/// Version tag for the vault-relative-path normalization. Written to
/// `meta.path_schema_version` after a successful in-place migration; used
/// to make `migrate_paths_to_relative` idempotent across restarts.
pub const PATH_SCHEMA_VERSION: &str = "1";

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store: {0}")]
    Msg(String),
    #[error("store: sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("store: sqlite-vec extension failed to load: {0}")]
    VecExtension(String),
    #[error("store: embedding_cache row dim mismatch: got {got}, want {want}")]
    CacheDimMismatch { got: usize, want: usize },
    #[error("store: replace file has {chunks} chunks but {embeddings} embeddings")]
    ReplaceEmbeddingCount { chunks: usize, embeddings: usize },
    #[error(
        "store: embedding_dim on disk is {stored}, config says {config} — run with --reembed to reindex at the new dim, or delete the index DB as a fallback."
    )]
    SchemaDimMismatch { stored: usize, config: usize },
}

/// Wraps a `rusqlite::Connection` with the docindex schema applied and
/// the `sqlite-vec` extension loaded.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Read the embedding fingerprint (`embedding_provider` / `embedding_model`
    /// / `embedding_dim` in `meta`) without applying the dim-specific
    /// `chunks_vec` DDL. Used by callers that need to decide *how* to open
    /// (normal vs. `--reembed`) before the dim gets baked into a virtual
    /// table — `chunks_vec`'s `CREATE VIRTUAL TABLE IF NOT EXISTS` would
    /// otherwise silently keep whatever dim the table already has.
    ///
    /// Returns `None` for a DB file that doesn't exist yet, or exists but
    /// has never had a fingerprint written (fresh index).
    pub fn peek_fingerprint(
        path: impl AsRef<Path>,
    ) -> Result<Option<(String, String, usize)>, StoreError> {
        if !path.as_ref().exists() {
            return Ok(None);
        }
        register_sqlite_vec()?;
        let conn = Connection::open(path.as_ref())?;
        init_pragmas(&conn)?;
        // `meta` lives in the base schema, not the dim-parameterized part —
        // safe to apply without ever touching `chunks_vec`.
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| StoreError::Msg(format!("apply base schema: {e}")))?;
        let provider: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_provider'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        let model: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_model'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        let dim: Option<usize> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_dim'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?
            .and_then(|v| v.parse().ok());
        Ok(match (provider, model, dim) {
            (Some(p), Some(m), Some(d)) => Some((p, m, d)),
            _ => None,
        })
    }

    /// Open (or create) the SQLite DB at `path`. Registers `sqlite-vec` as
    /// an auto-extension exactly once per process, applies the base schema,
    /// renders + applies the `chunks_vec` DDL with `embed_dim` baked into
    /// `FLOAT[...]`, and enforces that `meta.embedding_dim` matches
    /// `embed_dim` (refusing to start on mismatch).
    ///
    /// Callers that need the unified `provider=.../model=.../dim=...`
    /// fingerprint mismatch message (rather than this lower-level dim-only
    /// refusal) should call [`Store::peek_fingerprint`] first and only
    /// reach `open` once they know the dim will match — see
    /// `server::open_store_with_fingerprint_check`.
    pub fn open(path: impl AsRef<Path>, embed_dim: usize) -> Result<Self, StoreError> {
        Self::open_internal(path, embed_dim, false)
    }

    /// Open for a `--reembed` run: skips the `embedding_dim` mismatch
    /// refusal so the caller can immediately call [`Store::wipe_and_rebuild`]
    /// to drop and recreate `chunks_vec` at the new dim. Opening this way
    /// without following up with a wipe leaves the store in an inconsistent
    /// state — callers must always pair it with `wipe_and_rebuild`.
    pub fn open_for_reembed(path: impl AsRef<Path>, embed_dim: usize) -> Result<Self, StoreError> {
        Self::open_internal(path, embed_dim, true)
    }

    fn open_internal(
        path: impl AsRef<Path>,
        embed_dim: usize,
        skip_dim_check: bool,
    ) -> Result<Self, StoreError> {
        if embed_dim == 0 {
            return Err(StoreError::Msg("embed_dim must be > 0".into()));
        }
        register_sqlite_vec()?;
        let conn = Connection::open(path.as_ref())?;
        init_pragmas(&conn)?;
        verify_vec_loaded(&conn)?;
        // Apply the base schema and the dim-parameterized `chunks_vec` DDL
        // in a single batch. vec0 requires the dim as a SQL literal, so it
        // can't live in the static schema.sql file. If `chunks_vec` already
        // exists at a different dim, `IF NOT EXISTS` makes this a silent
        // no-op — safe because the `skip_dim_check=true` (reembed) caller
        // is about to drop and recreate it anyway, and the normal caller
        // gets a hard refusal below.
        let full_schema = format!("{}\n{}", SCHEMA_SQL, vec_schema_ddl(embed_dim));
        conn.execute_batch(&full_schema)
            .map_err(|e| StoreError::Msg(format!("apply schema: {e}")))?;
        let s = Self { conn };
        s.migrate_schema()?;
        if !skip_dim_check {
            s.reconcile_embedding_dim(embed_dim)?;
        }
        Ok(s)
    }

    fn migrate_schema(&self) -> Result<(), StoreError> {
        let stored = self.get_meta("schema_version")?;
        match stored.as_deref() {
            None => self.set_meta("schema_version", SCHEMA_VERSION)?,
            Some("2") => {
                let tx = self.conn.unchecked_transaction()?;
                for column in [
                    "media_type TEXT NOT NULL DEFAULT 'text'",
                    "mime_type TEXT",
                    "media_start INTEGER",
                    "media_end INTEGER",
                    "media_unit TEXT",
                    "truncated INTEGER NOT NULL DEFAULT 0",
                ] {
                    tx.execute_batch(&format!("ALTER TABLE chunks ADD COLUMN {column}"))?;
                }
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![SCHEMA_VERSION],
                )?;
                tx.commit()?;
            }
            Some(version) if version == SCHEMA_VERSION => {}
            Some(version) => {
                return Err(StoreError::Msg(format!(
                    "schema version {version} is unsupported; this binary supports {SCHEMA_VERSION}"
                )));
            }
        }
        Ok(())
    }

    /// Drop and recreate `chunks_fts` and `chunks_vec`, delete every row of
    /// `chunks` / `files` / `embedding_cache`, and write the new embedding
    /// fingerprint. Used by `--reembed` after [`Store::open_for_reembed`] —
    /// the only supported way to change the embedding dim on an existing
    /// DB, since `chunks_vec`'s dim is baked into its DDL.
    ///
    /// The wipe and the fingerprint write are atomic: all three meta keys
    /// (`embedding_provider` / `embedding_model` / `embedding_dim`) are
    /// upserted inside the transaction before `commit`. A crash after commit
    /// leaves a wiped index whose fingerprint exactly matches the new config.
    pub fn wipe_and_rebuild(
        &self,
        embed_dim: usize,
        provider: &str,
        model: &str,
    ) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute_batch("DROP TABLE IF EXISTS chunks_fts; DROP TABLE IF EXISTS chunks_vec;")?;
        tx.execute("DELETE FROM chunks", [])?;
        tx.execute("DELETE FROM files", [])?;
        tx.execute("DELETE FROM embedding_cache", [])?;
        // Recreates chunks_fts (IF NOT EXISTS in SCHEMA_SQL; chunks/files/
        // embedding_cache/meta are no-ops since they still exist) and
        // chunks_vec at the new dim.
        tx.execute_batch(SCHEMA_SQL)
            .map_err(|e| StoreError::Msg(format!("reapply base schema: {e}")))?;
        tx.execute_batch(&vec_schema_ddl(embed_dim))
            .map_err(|e| StoreError::Msg(format!("recreate chunks_vec: {e}")))?;
        // Write the fingerprint inside the same transaction so the wipe and
        // the new fingerprint are committed atomically.
        write_fingerprint_in_tx(&tx, provider, model, embed_dim)?;
        tx.commit()?;
        Ok(())
    }

    /// Compare the stored embedding fingerprint (`embedding_provider` /
    /// `embedding_model` / `embedding_dim` in `meta`) against the effective
    /// config. A DB with no fingerprint recorded yet (pre-existing
    /// deployments, or a genuinely empty index) is [`FingerprintOutcome::Fresh`]
    /// — the caller should adopt the current config as the fingerprint via
    /// [`Store::set_fingerprint`] rather than error, so upgrading an
    /// existing production DB never breaks on its own.
    ///
    /// A partial fingerprint (one or two of the three keys present but not
    /// all) is also treated as `Fresh`. This can only arise from a crash
    /// mid-`set_fingerprint` on a very old build; the correct recovery is
    /// to adopt the current config and re-write a complete fingerprint.
    pub fn check_fingerprint(
        &self,
        provider: &str,
        model: &str,
        dim: usize,
    ) -> Result<FingerprintOutcome, StoreError> {
        let stored_provider = self.get_meta("embedding_provider")?;
        let stored_model = self.get_meta("embedding_model")?;
        // Require all three keys to constitute a real fingerprint. Any
        // missing key (including partial-write state) is treated as Fresh.
        let (stored_provider, stored_model) = match (stored_provider, stored_model) {
            (Some(p), Some(m)) => (p, m),
            _ => return Ok(FingerprintOutcome::Fresh),
        };
        let stored_dim: usize = match self.get_meta("embedding_dim")?.and_then(|v| v.parse().ok()) {
            Some(d) => d,
            None => return Ok(FingerprintOutcome::Fresh),
        };
        if stored_provider == provider && stored_model == model && stored_dim == dim {
            return Ok(FingerprintOutcome::Match);
        }
        Ok(FingerprintOutcome::Mismatch(format!(
            "index built with provider={stored_provider} model={stored_model} dim={stored_dim}, \
             config says provider={provider} model={model} dim={dim}; re-embed required: run with --reembed"
        )))
    }

    /// Record the embedding fingerprint atomically. Called once, on a
    /// genuinely fresh index (see [`FingerprintOutcome::Fresh`]) or
    /// immediately after [`Store::wipe_and_rebuild`] (which writes it
    /// directly). All three keys are upserted inside a single transaction
    /// so a crash between writes cannot leave a partial fingerprint.
    pub fn set_fingerprint(
        &self,
        provider: &str,
        model: &str,
        dim: usize,
    ) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        write_fingerprint_in_tx(&tx, provider, model, dim)?;
        tx.commit()?;
        Ok(())
    }

    /// Ensure `meta.embedding_dim` agrees with `embed_dim`:
    ///
    /// - missing: write it from config (first open).
    /// - match: nothing to do.
    /// - mismatch: refuse to open with `SchemaDimMismatch`.
    ///
    /// Also sweep any `embedding_cache` rows whose stored dim differs from
    /// the current config — belt-and-suspenders for partial writes from
    /// older builds, and keeps reads from ever returning a mismatched
    /// vector.
    fn reconcile_embedding_dim(&self, embed_dim: usize) -> Result<(), StoreError> {
        match self.get_meta("embedding_dim")? {
            None => self.set_meta("embedding_dim", &embed_dim.to_string())?,
            Some(v) => {
                let stored: usize = v
                    .parse()
                    .map_err(|e| StoreError::Msg(format!("parse meta.embedding_dim {v:?}: {e}")))?;
                if stored != embed_dim {
                    return Err(StoreError::SchemaDimMismatch {
                        stored,
                        config: embed_dim,
                    });
                }
            }
        }
        self.conn.execute(
            "DELETE FROM embedding_cache WHERE dim != ?1",
            params![embed_dim as i64],
        )?;
        Ok(())
    }

    /// Exposed for tests that want to run raw SQL.
    #[cfg(test)]
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Atomically replace every indexed row for one file. This includes old
    /// chunks, FTS rows, vectors, prepared embedding-cache entries, and the
    /// file-state bookkeeping row. If any statement fails, the prior file
    /// index remains intact.
    pub fn replace_file(&self, replacement: FileReplacement<'_>) -> Result<(), StoreError> {
        if replacement.chunks.len() != replacement.embeddings.len() {
            return Err(StoreError::ReplaceEmbeddingCount {
                chunks: replacement.chunks.len(),
                embeddings: replacement.embeddings.len(),
            });
        }
        for entry in replacement.cache_entries {
            validate_cache_entry(entry)?;
        }

        let tx = self.conn.unchecked_transaction()?;
        delete_chunks_for_path_in_tx(&tx, replacement.path)?;
        for (chunk, embedding) in replacement.chunks.iter().zip(replacement.embeddings) {
            let id = insert_chunk_in_tx(&tx, chunk, replacement.path, replacement.mtime_ns)?;
            tx.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
                params![id, encode_f32(embedding)],
            )?;
        }
        for entry in replacement.cache_entries {
            put_embedding_cache_in_tx(&tx, entry)?;
        }
        set_file_state_in_tx(
            &tx,
            replacement.path,
            replacement.content_hash,
            replacement.mtime_ns,
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Insert or update the chunk at (`path`, `chunk.idx`) and keep
    /// `chunks_fts` in sync. Returns the stable `chunks.id` rowid.
    pub fn upsert_chunk(&self, c: &Chunk, path: &str, mtime_ns: i64) -> Result<i64, StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        let existing: Option<(i64, String, Option<String>, String)> = tx
            .query_row(
                "SELECT id, content, heading_path, media_type FROM chunks WHERE path = ? AND chunk_idx = ?",
                params![path, c.idx as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .optional()?;

        let id = match existing {
            None => {
                tx.execute(
                    "INSERT INTO chunks(path, chunk_idx, heading, heading_path, content, content_hash, mtime_ns, tokens, media_type, mime_type, media_start, media_end, media_unit, truncated)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                    params![
                        path,
                        c.idx as i64,
                        null_if_empty(&c.heading),
                        null_if_empty(&c.heading_path),
                        c.content,
                        c.content_hash,
                        mtime_ns,
                        c.tokens as i64,
                        c.media_type.as_str(),
                        c.mime_type,
                        c.media_start,
                        c.media_end,
                        c.media_unit,
                        c.truncated as i64,
                    ],
                )?;
                tx.last_insert_rowid()
            }
            Some((id, old_content, old_path, old_media_type)) => {
                if old_media_type == "text" {
                    tx.execute(
                        "INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?1, ?2, ?3)",
                        params![id, old_content, old_path.unwrap_or_default()],
                    )?;
                }
                tx.execute(
                    "UPDATE chunks SET heading=?1, heading_path=?2, content=?3, content_hash=?4, mtime_ns=?5, tokens=?6, media_type=?7, mime_type=?8, media_start=?9, media_end=?10, media_unit=?11, truncated=?12
                     WHERE id=?13",
                    params![
                        null_if_empty(&c.heading), null_if_empty(&c.heading_path), c.content,
                        c.content_hash, mtime_ns, c.tokens as i64, c.media_type.as_str(),
                        c.mime_type, c.media_start, c.media_end, c.media_unit, c.truncated as i64, id,
                    ],
                )?;
                id
            }
        };

        if c.media_type.as_str() == "text" {
            tx.execute(
                "INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (?1, ?2, ?3)",
                params![id, c.content, c.heading_path],
            )?;
        }
        tx.commit()?;
        Ok(id)
    }

    /// Remove every chunk for `path`, including FTS and vec rows.
    pub fn delete_chunks_for_path(&self, path: &str) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        delete_chunks_for_path_in_tx(&tx, path)?;
        tx.commit()?;
        Ok(())
    }

    /// Atomically remove every indexed representation and bookkeeping state
    /// for `path`. If any deletion fails, all prior rows remain intact.
    pub fn remove_file(&self, path: &str) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        delete_chunks_for_path_in_tx(&tx, path)?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        tx.commit()?;
        Ok(())
    }

    /// Return the cached embedding for `content_hash`, if any.
    pub fn get_embedding_cache(&self, content_hash: &str) -> Result<Option<Vec<f32>>, StoreError> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT embedding FROM embedding_cache WHERE content_hash = ?1",
                params![content_hash],
                |r| r.get(0),
            )
            .optional()?;
        match blob {
            Some(b) => {
                Ok(Some(decode_f32(&b).map_err(|e| {
                    StoreError::Msg(format!("decode cached vec: {e}"))
                })?))
            }
            None => Ok(None),
        }
    }

    /// Store an embedding keyed by `content_hash`.
    pub fn put_embedding_cache(
        &self,
        content_hash: &str,
        model: &str,
        task_type: &str,
        dim: usize,
        embedding: &[f32],
    ) -> Result<(), StoreError> {
        if embedding.len() != dim {
            return Err(StoreError::CacheDimMismatch {
                got: embedding.len(),
                want: dim,
            });
        }
        let blob = encode_f32(embedding);
        self.conn.execute(
            "INSERT INTO embedding_cache(content_hash, model, task_type, dim, embedding, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s','now'))
             ON CONFLICT(content_hash) DO UPDATE SET
               model=excluded.model, task_type=excluded.task_type,
               dim=excluded.dim, embedding=excluded.embedding, created_at=excluded.created_at",
            params![content_hash, model, task_type, dim as i64, blob],
        )?;
        Ok(())
    }

    /// Insert or replace the vector for `chunk_id` in the `vec0` table.
    pub fn set_vector_for_chunk(&self, chunk_id: i64, embedding: &[f32]) -> Result<(), StoreError> {
        let blob = encode_f32(embedding);
        // vec0 virtual tables don't support INSERT OR REPLACE, so drop any
        // prior row for this rowid first, then insert. Wrap in a tx so the
        // two statements succeed or fail atomically.
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![chunk_id])?;
        tx.execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
            params![chunk_id, blob],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Read `meta[key]`.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>, StoreError> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    /// Upsert `meta[key] = value`.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Per-file state snapshot (content hash + mtime) from the last successful
    /// reindex. Used by the indexer's initial-scan diff.
    pub fn get_file_state(&self, path: &str) -> Result<Option<(String, i64)>, StoreError> {
        Ok(self
            .conn
            .query_row(
                "SELECT content_hash, mtime_ns FROM files WHERE path = ?1",
                params![path],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )
            .optional()?)
    }

    /// Record that `path` was fully reindexed at mtime=`mtime_ns`, hash=`hash`.
    pub fn set_file_state(
        &self,
        path: &str,
        content_hash: &str,
        mtime_ns: i64,
    ) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO files(path, content_hash, mtime_ns, indexed_at)
             VALUES (?1, ?2, ?3, strftime('%s','now'))
             ON CONFLICT(path) DO UPDATE SET
               content_hash = excluded.content_hash,
               mtime_ns = excluded.mtime_ns,
               indexed_at = excluded.indexed_at",
            params![path, content_hash, mtime_ns],
        )?;
        Ok(())
    }

    /// Remove the `files` bookkeeping row (used after deleting a path).
    pub fn delete_file_state(&self, path: &str) -> Result<(), StoreError> {
        self.conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])?;
        Ok(())
    }

    /// Every path we've fully indexed (used by the startup diff to detect
    /// files that disappeared from disk and must be pruned).
    pub fn list_indexed_paths(&self) -> Result<Vec<String>, StoreError> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let mapped = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in mapped {
            out.push(r?);
        }
        Ok(out)
    }

    /// Total number of chunks currently indexed.
    pub fn count_chunks(&self) -> Result<i64, StoreError> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?)
    }

    /// Top-`k` chunk rowids by cosine distance to `query` (ascending
    /// distance). Uses `sqlite-vec`'s kNN MATCH operator against `vec0`.
    pub fn search_vec(&self, query: &[f32], k: usize) -> Result<Vec<(i64, f32)>, StoreError> {
        let blob = encode_f32(query);
        let mut stmt = self.conn.prepare(
            "SELECT rowid, distance FROM chunks_vec
             WHERE embedding MATCH ?1 AND k = ?2
             ORDER BY distance",
        )?;
        let rows = stmt.query_map(params![blob, k as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)? as f32))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Top-`k` media chunk rowids by cosine distance to `query` (ascending
    /// distance). Streams every `(rowid, embedding)` pair from `chunks_vec`
    /// joined to `chunks` on the media-type predicate and retains only the
    /// best `k` results in a bounded max-heap — no `IN (…)` with unbounded
    /// placeholders, no materialisation of all vectors.
    ///
    /// The returned order is dense over media only: rank-1 is the closest
    /// media chunk, regardless of where it would rank among all chunk types.
    pub fn search_media_vec(&self, query: &[f32], k: usize) -> Result<Vec<(i64, f32)>, StoreError> {
        use std::collections::BinaryHeap;

        if k == 0 {
            return Ok(Vec::new());
        }

        // Stream media vectors via a join; never materialise all rows or ids.
        let mut stmt = self.conn.prepare(
            "SELECT v.rowid, v.embedding \
             FROM chunks_vec v \
             JOIN chunks c ON c.id = v.rowid \
             WHERE c.media_type != 'text'",
        )?;

        // Max-heap keyed by (dist_bits, rowid): the root holds the entry with
        // the LARGEST distance, so when the heap is full we can evict the
        // worst candidate to make room for a closer one. IEEE 754 positive
        // floats sort correctly by their bit representation; cosine distance
        // is always ≥ 0, so this encoding is safe.
        //
        // Tie-break: the eviction condition uses strict `<`, so a new entry
        // with the same distance as the current worst is not admitted. Among
        // equally-distant candidates the first-scanned (lowest rowid, earliest
        // in table order) is retained; higher-rowid duplicates are discarded.
        let mut heap: BinaryHeap<(u32, i64)> = BinaryHeap::with_capacity(k + 1);

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let rowid: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let embedding =
                decode_f32(&blob).map_err(|e| StoreError::Msg(format!("decode vec: {e}")))?;
            let dist = cosine_distance(query, &embedding);
            let dist_bits = dist.to_bits();

            if heap.len() < k {
                heap.push((dist_bits, rowid));
            } else if let Some(&(top_bits, _)) = heap.peek()
                && dist_bits < top_bits
            {
                // New entry is closer than the current worst; replace it.
                heap.pop();
                heap.push((dist_bits, rowid));
            }
        }

        let mut results: Vec<(i64, f32)> = heap
            .into_iter()
            .map(|(bits, id)| (id, f32::from_bits(bits)))
            .collect();
        // Sort ascending by distance, then ascending by rowid for determinism.
        results.sort_by(|(id_a, d_a), (id_b, d_b)| d_a.total_cmp(d_b).then(id_a.cmp(id_b)));
        Ok(results)
    }

    /// Top-`k` chunk rowids by BM25 over `chunks_fts` for the raw FTS5
    /// query string (ascending bm25 score — smaller is better per FTS5).
    pub fn search_fts(&self, query: &str, k: usize) -> Result<Vec<(i64, f64)>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![query, k as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Load a chunk's display fields for hit hydration.
    pub fn chunk_for_hit(&self, id: i64) -> Result<Option<HitRow>, StoreError> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, path, heading, heading_path, content, media_type, mime_type, media_start, media_end, media_unit, truncated FROM chunks WHERE id = ?1",
                params![id],
                |r| {
                    Ok(HitRow {
                        id: r.get::<_, i64>(0)?,
                        path: r.get::<_, String>(1)?,
                        heading: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        heading_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        content: r.get::<_, String>(4)?,
                        media_type: r.get::<_, String>(5)?,
                        mime_type: r.get::<_, Option<String>>(6)?,
                        media_start: r.get::<_, Option<i64>>(7)?,
                        media_end: r.get::<_, Option<i64>>(8)?,
                        media_unit: r.get::<_, Option<String>>(9)?,
                        truncated: r.get::<_, i64>(10)? != 0,
                    })
                },
            )
            .optional()?)
    }

    /// All chunks for `path` (id, content). Used by callers that need the
    /// stored content without media metadata.
    pub fn chunks_for_path(&self, path: &str) -> Result<Vec<(i64, String)>, StoreError> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM chunks WHERE path = ?1 ORDER BY chunk_idx")?;
        let rows = stmt.query_map(params![path], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// All chunks for `path` with media metadata. Used by /similar to restrict
    /// its lexical bag to text chunks while retaining all vectors.
    pub fn chunks_for_similar(&self, path: &str) -> Result<Vec<PathChunkRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, media_type FROM chunks WHERE path = ?1 ORDER BY chunk_idx",
        )?;
        let rows = stmt.query_map(params![path], |r| {
            Ok(PathChunkRow {
                id: r.get(0)?,
                content: r.get(1)?,
                media_type: r.get(2)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Stored vectors for a set of chunk ids. Order is not guaranteed to
    /// match the input — callers handle that.
    pub fn vectors_for_chunks(&self, ids: &[i64]) -> Result<Vec<(i64, Vec<f32>)>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql =
            format!("SELECT rowid, embedding FROM chunks_vec WHERE rowid IN ({placeholders})");
        let mut stmt = self.conn.prepare(&sql)?;
        let params_vec: Vec<rusqlite::types::Value> = ids
            .iter()
            .map(|i| rusqlite::types::Value::Integer(*i))
            .collect();
        let rows = stmt.query_map(rusqlite::params_from_iter(params_vec), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, blob) = r?;
            let v = decode_f32(&blob).map_err(|e| StoreError::Msg(format!("decode vec: {e}")))?;
            out.push((id, v));
        }
        Ok(out)
    }

    /// Rewrite `chunks.path` and `files.path` from absolute (pre-0.2.0) to
    /// vault-relative, in place. Idempotent: once
    /// `meta.path_schema_version == PATH_SCHEMA_VERSION`, this is a no-op.
    ///
    /// Safety: if any row's `path` starts with `/` but does **not** fall
    /// under `vault_dir`, we refuse the migration, log a warning with the
    /// offending count, and leave `meta.path_schema_version` unset. This
    /// leaves the DB readable in its old form but makes the operator decide
    /// between wiping it or restoring the original vault dir — far safer
    /// than silently mangling paths.
    pub fn migrate_paths_to_relative(
        &self,
        vault_dir: &Path,
    ) -> Result<MigrationOutcome, StoreError> {
        if let Some(v) = self.get_meta("path_schema_version")?
            && v == PATH_SCHEMA_VERSION
        {
            return Ok(MigrationOutcome::AlreadyCurrent);
        }

        // Canonicalize the vault dir so the prefix match survives trailing
        // slashes and symlinks. If canonicalize fails (missing dir) we bail
        // out without touching anything.
        let canonical = vault_dir.canonicalize().map_err(|e| {
            StoreError::Msg(format!(
                "migrate_paths: canonicalize({}): {e}",
                vault_dir.display()
            ))
        })?;
        let vault_str = canonical.to_string_lossy().into_owned();
        if !vault_str.starts_with('/') {
            return Err(StoreError::Msg(format!(
                "migrate_paths: vault_dir must be absolute, got {vault_str:?}"
            )));
        }
        let prefix_match = format!("{vault_str}/%");

        // Count offending rows: absolute paths that don't live under the
        // vault. If the DB is already fully relative, this is also 0 (the
        // second clause excludes them).
        let offending: i64 = self.conn.query_row(
            "SELECT
               (SELECT COUNT(*) FROM chunks WHERE path LIKE '/%' AND path NOT LIKE ?1)
             + (SELECT COUNT(*) FROM files  WHERE path LIKE '/%' AND path NOT LIKE ?1)",
            params![prefix_match],
            |r| r.get(0),
        )?;
        if offending > 0 {
            warn!(
                rows = offending,
                vault = %vault_str,
                "path migration: found rows with absolute paths outside the configured vault; refusing to migrate. Either restore DOCINDEX_VAULT_DIR to the original path or wipe index.db.",
            );
            return Ok(MigrationOutcome::Refused {
                offending_rows: offending as u64,
            });
        }

        // Safe to proceed. +2 on substr accounts for: +1 for 1-indexed SQL
        // and +1 to skip the "/" immediately following vault_dir.
        let tx = self.conn.unchecked_transaction()?;
        let chunks_updated = tx.execute(
            "UPDATE chunks
               SET path = substr(path, length(?1) + 2)
             WHERE path LIKE ?2",
            params![vault_str, prefix_match],
        )?;
        let files_updated = tx.execute(
            "UPDATE files
               SET path = substr(path, length(?1) + 2)
             WHERE path LIKE ?2",
            params![vault_str, prefix_match],
        )?;
        tx.execute(
            "INSERT INTO meta(key, value) VALUES ('path_schema_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![PATH_SCHEMA_VERSION],
        )?;
        tx.commit()?;

        let total = (chunks_updated + files_updated) as u64;
        if total > 0 {
            info!(
                chunks_rewritten = chunks_updated,
                files_rewritten = files_updated,
                vault = %vault_str,
                "migrated {total} rows from absolute to relative paths"
            );
        } else {
            info!(
                vault = %vault_str,
                "path migration: no absolute paths found; marking as current"
            );
        }
        Ok(MigrationOutcome::Migrated {
            chunks_rewritten: chunks_updated as u64,
            files_rewritten: files_updated as u64,
        })
    }
}

/// All data needed to atomically replace one file's indexed representation.
/// Embeddings must be ordered to match `chunks`; `cache_entries` can contain
/// newly resolved document embeddings to persist with the replacement.
#[derive(Debug, Clone, Copy)]
pub struct FileReplacement<'a> {
    pub path: &'a str,
    pub content_hash: &'a str,
    pub mtime_ns: i64,
    pub chunks: &'a [Chunk],
    pub embeddings: &'a [Vec<f32>],
    pub cache_entries: &'a [PreparedEmbeddingCacheEntry<'a>],
}

/// A validated-at-commit embedding-cache write prepared by an indexer.
#[derive(Debug, Clone, Copy)]
pub struct PreparedEmbeddingCacheEntry<'a> {
    pub content_hash: &'a str,
    pub model: &'a str,
    pub task_type: &'a str,
    pub dim: usize,
    pub embedding: &'a [f32],
}

/// Summary of a `migrate_paths_to_relative` call. Exposed so the caller (and
/// tests) can distinguish a no-op from a real rewrite from a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationOutcome {
    /// `meta.path_schema_version` already matches — nothing to do.
    AlreadyCurrent,
    /// Migration ran and updated these row counts.
    Migrated {
        chunks_rewritten: u64,
        files_rewritten: u64,
    },
    /// Migration was refused because some rows lie outside `vault_dir`. The
    /// DB is untouched; the operator must reconcile manually.
    Refused { offending_rows: u64 },
}

/// Result of comparing the stored embedding fingerprint against the
/// effective config, from [`Store::check_fingerprint`] or
/// [`FingerprintOutcome::from_peek`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FingerprintOutcome {
    /// No fingerprint recorded yet — caller should adopt the current config.
    Fresh,
    /// Stored fingerprint matches the effective config.
    Match,
    /// Mismatch; the `String` names exactly which field(s) changed and both
    /// old/new values, ready to surface as a startup error.
    Mismatch(String),
}

impl FingerprintOutcome {
    /// Derive an outcome from the raw `peek_fingerprint` result, using the
    /// same equality semantics as [`Store::check_fingerprint`]. Allows
    /// callers to decide how to open the store (normal vs reembed) based
    /// solely on the outcome variant, without a second independent comparison
    /// of the three fingerprint fields.
    pub fn from_peek(
        stored: Option<(String, String, usize)>,
        provider: &str,
        model: &str,
        dim: usize,
    ) -> Self {
        match stored {
            None => FingerprintOutcome::Fresh,
            Some((sp, sm, sd)) if sp == provider && sm == model && sd == dim => {
                FingerprintOutcome::Match
            }
            Some((sp, sm, sd)) => FingerprintOutcome::Mismatch(format!(
                "index built with provider={sp} model={sm} dim={sd}, \
                 config says provider={provider} model={model} dim={dim}; \
                 re-embed required: run with --reembed"
            )),
        }
    }
}

/// One stored chunk used by /similar to form a vector mean and a text-only
/// lexical bag.
#[derive(Debug, Clone)]
pub struct PathChunkRow {
    pub id: i64,
    pub content: String,
    pub media_type: String,
}

/// Minimal row projection used to hydrate a search hit. Kept in the store
/// module so SQL column order stays colocated with the schema.
#[derive(Debug, Clone)]
pub struct HitRow {
    pub id: i64,
    pub path: String,
    pub heading: String,
    pub heading_path: String,
    pub content: String,
    pub media_type: String,
    pub mime_type: Option<String>,
    pub media_start: Option<i64>,
    pub media_end: Option<i64>,
    pub media_unit: Option<String>,
    pub truncated: bool,
}

/// Upsert the three embedding fingerprint keys inside an open transaction.
/// Both `wipe_and_rebuild` and `set_fingerprint` share this body so the
/// SQL and key names live in one place.
fn write_fingerprint_in_tx(
    tx: &rusqlite::Transaction<'_>,
    provider: &str,
    model: &str,
    dim: usize,
) -> Result<(), StoreError> {
    let dim_str = dim.to_string();
    for (key, val) in [
        ("embedding_provider", provider),
        ("embedding_model", model),
        ("embedding_dim", dim_str.as_str()),
    ] {
        tx.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, val],
        )?;
    }
    Ok(())
}

fn delete_chunks_for_path_in_tx(
    tx: &rusqlite::Transaction<'_>,
    path: &str,
) -> Result<(), StoreError> {
    let rows: Vec<(i64, String, Option<String>, String)> = {
        let mut stmt =
            tx.prepare("SELECT id, content, heading_path, media_type FROM chunks WHERE path = ?")?;
        let mapped = stmt.query_map(params![path], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut rows = Vec::new();
        for row in mapped {
            rows.push(row?);
        }
        rows
    };
    for (id, content, heading_path, media_type) in rows {
        if media_type == "text" {
            tx.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?1, ?2, ?3)",
                params![id, content, heading_path.unwrap_or_default()],
            )?;
        }
        tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![id])?;
    }
    tx.execute("DELETE FROM chunks WHERE path = ?1", params![path])?;
    Ok(())
}

fn insert_chunk_in_tx(
    tx: &rusqlite::Transaction<'_>,
    chunk: &Chunk,
    path: &str,
    mtime_ns: i64,
) -> Result<i64, StoreError> {
    tx.execute(
        "INSERT INTO chunks(path, chunk_idx, heading, heading_path, content, content_hash, mtime_ns, tokens, media_type, mime_type, media_start, media_end, media_unit, truncated)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            path,
            chunk.idx as i64,
            null_if_empty(&chunk.heading),
            null_if_empty(&chunk.heading_path),
            chunk.content,
            chunk.content_hash,
            mtime_ns,
            chunk.tokens as i64,
            chunk.media_type.as_str(),
            chunk.mime_type,
            chunk.media_start,
            chunk.media_end,
            chunk.media_unit,
            chunk.truncated as i64,
        ],
    )?;
    let id = tx.last_insert_rowid();
    if chunk.media_type.as_str() == "text" {
        tx.execute(
            "INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (?1, ?2, ?3)",
            params![id, chunk.content, chunk.heading_path],
        )?;
    }
    Ok(id)
}

fn validate_cache_entry(entry: &PreparedEmbeddingCacheEntry<'_>) -> Result<(), StoreError> {
    if entry.embedding.len() != entry.dim {
        return Err(StoreError::CacheDimMismatch {
            got: entry.embedding.len(),
            want: entry.dim,
        });
    }
    Ok(())
}

fn put_embedding_cache_in_tx(
    tx: &rusqlite::Transaction<'_>,
    entry: &PreparedEmbeddingCacheEntry<'_>,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT INTO embedding_cache(content_hash, model, task_type, dim, embedding, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s','now'))
         ON CONFLICT(content_hash) DO UPDATE SET
           model=excluded.model, task_type=excluded.task_type,
           dim=excluded.dim, embedding=excluded.embedding, created_at=excluded.created_at",
        params![
            entry.content_hash,
            entry.model,
            entry.task_type,
            entry.dim as i64,
            encode_f32(entry.embedding),
        ],
    )?;
    Ok(())
}

fn set_file_state_in_tx(
    tx: &rusqlite::Transaction<'_>,
    path: &str,
    content_hash: &str,
    mtime_ns: i64,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT INTO files(path, content_hash, mtime_ns, indexed_at)
         VALUES (?1, ?2, ?3, strftime('%s','now'))
         ON CONFLICT(path) DO UPDATE SET
           content_hash = excluded.content_hash,
           mtime_ns = excluded.mtime_ns,
           indexed_at = excluded.indexed_at",
        params![path, content_hash, mtime_ns],
    )?;
    Ok(())
}

/// Cosine distance matching the `distance_metric=cosine` semantics of
/// `sqlite-vec`'s `vec0` table: `1.0 - dot(a, b)` for unit vectors, with
/// the result clamped to `[0.0, 2.0]`. For the Fake embedder and most real
/// providers, embeddings are L2-normalised, so this is numerically equivalent
/// to the kNN distance sqlite-vec computes internally.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (1.0 - dot).clamp(0.0, 2.0)
}

fn null_if_empty(s: &str) -> Option<&str> {
    if s.is_empty() { None } else { Some(s) }
}

fn init_pragmas(conn: &Connection) -> Result<(), StoreError> {
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// Verify the extension is loaded by calling `vec_version()`. If this fails
/// the rest of the store is unusable — return a clear error instead of
/// silently degrading, per the Phase 1 rules.
fn verify_vec_loaded(conn: &Connection) -> Result<(), StoreError> {
    conn.query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
        .map(|_| ())
        .map_err(|e| StoreError::VecExtension(format!("vec_version(): {e}")))
}

static VEC_REGISTERED: OnceLock<()> = OnceLock::new();

/// Register `sqlite-vec` as an auto-extension. Every subsequent
/// `Connection::open` gets it loaded before any SQL runs.
fn register_sqlite_vec() -> Result<(), StoreError> {
    let mut outcome: Result<(), StoreError> = Ok(());
    VEC_REGISTERED.get_or_init(|| {
        // SAFETY: sqlite3_auto_extension wants a C function pointer; the
        // sqlite-vec crate exports `sqlite3_vec_init` as exactly that.
        // Transmuting between two `unsafe extern "C" fn` of different
        // concrete signatures is what every sqlite-vec example uses and
        // what SQLite itself expects via its api_routines pointer.
        #[allow(clippy::missing_transmute_annotations)]
        let rc = unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )))
        };
        if rc != 0 {
            outcome = Err(StoreError::VecExtension(format!(
                "sqlite3_auto_extension returned {rc}"
            )));
        }
    });
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Dim used by the store's own tests. Small and cheap — the behavior
    /// under test (upsert, FTS sync, vec delete+insert, caching, reopen)
    /// doesn't depend on the embedding size as long as it's consistent.
    const TEST_DIM: usize = 8;

    fn open_temp() -> (TempDir, Store) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        let store = Store::open(&path, TEST_DIM).expect("open");
        (dir, store)
    }

    fn sample_chunk(idx: usize, content: &str, hash: &str) -> Chunk {
        Chunk {
            idx,
            heading: String::new(),
            heading_path: String::new(),
            content: content.to_string(),
            content_hash: hash.to_string(),
            tokens: content.split_whitespace().count(),
            media_type: crate::media::MediaType::Text,
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        }
    }

    #[test]
    fn migrates_v2_text_rows_additively_and_preserves_fts_search() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("v2.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT NOT NULL, chunk_idx INTEGER NOT NULL, heading TEXT, heading_path TEXT, content TEXT NOT NULL, content_hash TEXT NOT NULL, mtime_ns INTEGER NOT NULL, tokens INTEGER, UNIQUE(path, chunk_idx));
             CREATE VIRTUAL TABLE chunks_fts USING fts5(content, heading_path, content=chunks, content_rowid=id, tokenize='porter unicode61');
             CREATE TABLE embedding_cache (content_hash TEXT PRIMARY KEY, model TEXT NOT NULL, task_type TEXT NOT NULL, dim INTEGER NOT NULL, embedding BLOB NOT NULL, created_at INTEGER NOT NULL);
             CREATE TABLE files (path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, mtime_ns INTEGER NOT NULL, indexed_at INTEGER NOT NULL);
             CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);",
        ).unwrap();
        conn.execute("INSERT INTO chunks(id, path, chunk_idx, content, content_hash, mtime_ns) VALUES (1, 'note.md', 0, 'legacy searchable token', 'hash', 1)", []).unwrap();
        conn.execute("INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (1, 'legacy searchable token', '')", []).unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', '2')",
            [],
        )
        .unwrap();
        drop(conn);

        let store = Store::open(&path, TEST_DIM).unwrap();
        assert_eq!(
            store.get_meta("schema_version").unwrap().as_deref(),
            Some("3")
        );
        let row: (String, Option<String>, Option<i64>, Option<i64>, Option<String>, i64) = store.conn().query_row(
            "SELECT media_type, mime_type, media_start, media_end, media_unit, truncated FROM chunks WHERE id = 1", [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
        ).unwrap();
        assert_eq!(row, ("text".into(), None, None, None, None, 0));
        assert_eq!(store.search_fts("legacy", 10).unwrap().len(), 1);
        drop(store);
        assert!(Store::open(&path, TEST_DIM).is_ok());
    }

    #[test]
    fn open_applies_schema_and_sets_version() {
        let (_d, s) = open_temp();
        let v = s.get_meta("schema_version").unwrap();
        assert_eq!(v.as_deref(), Some(SCHEMA_VERSION));
    }

    #[test]
    fn vec_extension_loaded_with_vec0_table() {
        let (_d, s) = open_temp();
        let version: String = s
            .conn()
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .expect("vec_version");
        assert!(!version.is_empty(), "vec_version empty");
        // vec0 table must exist and accept inserts.
        s.conn()
            .execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
                params![1i64, encode_f32(&[0.0_f32; TEST_DIM])],
            )
            .expect("insert into vec0");
    }

    #[test]
    fn media_chunks_are_vector_only_and_metadata_hydrates() {
        let (_d, s) = open_temp();
        let mut image = sample_chunk(0, "image representation", "image-hash");
        image.media_type = crate::media::MediaType::Image;
        image.mime_type = Some("image/png".into());
        let id = s.upsert_chunk(&image, "Attachments/image.png", 1).unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        let fts_count: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'representation'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fts_count, 0);
        let vec_count: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vec_count, 1);
        let hit = s.chunk_for_hit(id).unwrap().unwrap();
        assert_eq!(hit.media_type, "image");
        assert_eq!(hit.mime_type.as_deref(), Some("image/png"));
    }

    #[test]
    fn media_vector_search_filters_text_and_preserves_dense_media_order() {
        let (_d, s) = open_temp();
        let text_id = s
            .upsert_chunk(&sample_chunk(0, "text", "text-hash"), "note.md", 1)
            .unwrap();
        let mut image = sample_chunk(0, "image", "image-hash");
        image.media_type = crate::media::MediaType::Image;
        let image_id = s.upsert_chunk(&image, "image.png", 1).unwrap();
        let mut pdf = sample_chunk(0, "pdf", "pdf-hash");
        pdf.media_type = crate::media::MediaType::Pdf;
        let pdf_id = s.upsert_chunk(&pdf, "paper.pdf", 1).unwrap();

        s.set_vector_for_chunk(text_id, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        s.set_vector_for_chunk(image_id, &[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        s.set_vector_for_chunk(pdf_id, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        let hits = s
            .search_media_vec(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10)
            .unwrap();
        assert_eq!(
            hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![image_id, pdf_id]
        );
    }

    /// Confirm that `search_media_vec` with a cap of `k` returns exactly the
    /// `k` closest media chunks and that eviction actually fires.
    ///
    /// Chunks are inserted in an order that interleaves near and far distances:
    /// the heap fills with two far entries first, then a closer entry arrives
    /// and evicts the worst, then another closer entry arrives and evicts again.
    /// A min-heap (the original broken implementation) keeps the two furthest
    /// instead of the two closest, so this test would fail against it.
    #[test]
    fn search_media_vec_top_k_is_bounded_and_ordered() {
        // Use a small cap so eviction is forced.
        const CAP: usize = 2;
        let (_d, s) = open_temp();

        // Insert one text chunk (must be excluded) and four media chunks.
        // Insertion order is far→far→near→near so eviction fires twice.
        let text_id = s
            .upsert_chunk(&sample_chunk(0, "text", "tx"), "t.md", 1)
            .unwrap();
        s.set_vector_for_chunk(text_id, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        let make_image = |content: &str, hash: &str| {
            let mut c = sample_chunk(0, content, hash);
            c.media_type = crate::media::MediaType::Image;
            c
        };

        // Insert in order: far first (dist=1.0), second-far (dist=0.5),
        // then close (dist≈0.025), then closest (dist≈0.0). The heap fills
        // after the first two, then evicts id_c when id_b arrives, and evicts
        // id_b when id_a arrives — two evictions total.
        let id_c = s.upsert_chunk(&make_image("c", "hc"), "c.png", 1).unwrap(); // dist=1.0
        let id_d = s.upsert_chunk(&make_image("d", "hd"), "d.png", 1).unwrap(); // dist=0.5
        let id_b = s.upsert_chunk(&make_image("b", "hb"), "b.png", 1).unwrap(); // dist≈0.025
        let id_a = s.upsert_chunk(&make_image("a", "ha"), "a.png", 1).unwrap(); // dist≈0.0

        // Embeddings (pre-normalised unit vectors; cosine dist = 1 − dot product):
        //   id_c → [-1, 0, …]  dist = 1.0  (inserted 1st — fills heap slot 1)
        //   id_d → [ 0, 1, …]  dist = 0.5  (inserted 2nd — fills heap slot 2)
        //   id_b → [√0.95, √0.05, …]  dist ≈ 0.025  (closer than id_d → evicts id_c)
        //   id_a → [ 1, 0, …]  dist ≈ 0.0  (closest → evicts id_d)
        s.set_vector_for_chunk(id_c, &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        s.set_vector_for_chunk(id_d, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let near_vec = {
            let x: f32 = (0.95f32).sqrt();
            let y: f32 = (0.05f32).sqrt();
            [x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        };
        s.set_vector_for_chunk(id_b, &near_vec).unwrap();
        s.set_vector_for_chunk(id_a, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        let query = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hits = s.search_media_vec(&query, CAP).unwrap();

        // Exactly CAP results returned.
        assert_eq!(
            hits.len(),
            CAP,
            "expected exactly {CAP} hits, got {}",
            hits.len()
        );

        // The two returned ids must be the two closest: id_a (dist≈0.0) and id_b (dist≈0.025).
        let ids: Vec<i64> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![id_a, id_b],
            "top-2 must be id_a then id_b (closest first)"
        );

        // Distances must be non-decreasing (ascending order).
        assert!(
            hits[0].1 <= hits[1].1,
            "hits must be ordered ascending by distance: {:?}",
            hits
        );

        // Text chunk and the two far chunks must never appear.
        let result_ids: std::collections::HashSet<i64> = hits.iter().map(|(id, _)| *id).collect();
        assert!(
            !result_ids.contains(&text_id),
            "text chunk must be excluded"
        );
        assert!(
            !result_ids.contains(&id_c),
            "far chunk id_c must be evicted"
        );
        assert!(
            !result_ids.contains(&id_d),
            "far chunk id_d must be evicted"
        );
    }

    /// Same correctness check as above but with chunks inserted in REVERSE
    /// distance order (furthest rowid first, closest rowid last). The table
    /// scan returns rows in rowid order, so the heap sees distances
    /// [1.0, 0.5, ≈0.05, ≈0.0] — furthest first, closest last.
    ///
    /// A min-heap (the original broken implementation) would evict the two
    /// closest entries as they arrive, leaving the two furthest in the result.
    /// A correct max-heap must still return the two closest.
    #[test]
    fn search_media_vec_top_k_correct_when_furthest_inserted_first() {
        const CAP: usize = 2;
        let (_d, s) = open_temp();

        let make_image = |content: &str, hash: &str| {
            let mut c = sample_chunk(0, content, hash);
            c.media_type = crate::media::MediaType::Image;
            c
        };

        // Insert in DESCENDING distance order so rowid order = furthest first.
        // id_far1 is farthest (dist=1.0), id_far2 is next (dist=0.5),
        // id_near2 is close (dist≈0.05), id_near1 is closest (dist≈0.0).
        let id_far1 = s
            .upsert_chunk(&make_image("far1", "hfar1"), "far1.png", 1)
            .unwrap();
        let id_far2 = s
            .upsert_chunk(&make_image("far2", "hfar2"), "far2.png", 1)
            .unwrap();
        let id_near2 = s
            .upsert_chunk(&make_image("near2", "hnear2"), "near2.png", 1)
            .unwrap();
        let id_near1 = s
            .upsert_chunk(&make_image("near1", "hnear1"), "near1.png", 1)
            .unwrap();

        let query = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Assign embeddings: further from query = higher rowid.
        s.set_vector_for_chunk(id_far1, &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap(); // dist = 1.0
        s.set_vector_for_chunk(id_far2, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap(); // dist = 0.5
        let near2_vec = {
            let x: f32 = (0.95f32).sqrt();
            let y: f32 = (0.05f32).sqrt();
            [x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        };
        s.set_vector_for_chunk(id_near2, &near2_vec).unwrap(); // dist ≈ 0.025
        s.set_vector_for_chunk(id_near1, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap(); // dist ≈ 0.0

        let hits = s.search_media_vec(&query, CAP).unwrap();

        assert_eq!(
            hits.len(),
            CAP,
            "expected exactly {CAP} hits, got {}",
            hits.len()
        );

        // Must return the two CLOSEST (id_near1 and id_near2), not the two furthest.
        let ids: Vec<i64> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![id_near1, id_near2],
            "top-2 must be the two closest chunks regardless of insertion order; \
             a min-heap bug returns the two furthest instead: got {ids:?}"
        );

        assert!(
            hits[0].1 <= hits[1].1,
            "result must be sorted ascending by distance: {:?}",
            hits
        );
    }

    /// Inserting many media chunks (more than SQLITE_MAX_VARIABLE_NUMBER, which
    /// is typically 999) must not fail with a "too many SQL variables" error.
    /// The old `IN (…)` implementation would blow up here.
    #[test]
    fn search_media_vec_handles_many_media_chunks_without_variable_overflow() {
        let (_d, s) = open_temp();
        const N: usize = 1_100; // exceeds the default SQLITE_MAX_VARIABLE_NUMBER (999)

        let make_image = |idx: usize| {
            let mut c = sample_chunk(idx, "img content", &format!("h{idx}"));
            c.media_type = crate::media::MediaType::Image;
            c
        };

        for i in 0..N {
            let id = s
                .upsert_chunk(&make_image(i), &format!("img{i}.png"), 1)
                .unwrap();
            // Use a simple unit vector (dimension 0 = 1/(sqrt(8)), rest zero).
            let mut v = [0.0f32; TEST_DIM];
            v[i % TEST_DIM] = 1.0;
            s.set_vector_for_chunk(id, &v).unwrap();
        }

        let query = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Must succeed without panicking or returning a SQLite error.
        let hits = s
            .search_media_vec(&query, 10)
            .expect("search must succeed with many media chunks");
        // The cap of 10 must be respected exactly. With 1100 chunks and a query
        // that aligns with dimension-0, there are more than 10 candidates, so
        // the result must be exactly 10.
        assert_eq!(
            hits.len(),
            10,
            "result count must equal the requested cap when candidates exceed it"
        );
    }

    #[test]
    fn text_media_fts_transitions_remove_and_restore_terms() {
        let (_d, s) = open_temp();
        let path = "Attachments/item.png";
        let id = s
            .upsert_chunk(&sample_chunk(0, "text_only_token", "text"), path, 1)
            .unwrap();
        let mut image = sample_chunk(0, "image_only_token", "image");
        image.media_type = crate::media::MediaType::Image;
        image.mime_type = Some("image/png".into());
        assert_eq!(s.upsert_chunk(&image, path, 2).unwrap(), id);
        let text_hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'text_only_token'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let image_hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'image_only_token'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(text_hits, 0);
        assert_eq!(image_hits, 0);
        assert_eq!(
            s.upsert_chunk(&sample_chunk(0, "restored_text_token", "restored"), path, 3)
                .unwrap(),
            id
        );
        let restored_hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'restored_text_token'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(restored_hits, 1);
    }

    #[test]
    fn upsert_chunk_inserts_and_indexes_fts() {
        let (_d, s) = open_temp();
        let c = Chunk {
            idx: 0,
            heading: "T".into(),
            heading_path: "T".into(),
            content: "# T\nhello world".into(),
            content_hash: "hash1".into(),
            tokens: 3,
            media_type: crate::media::MediaType::Text,
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        };
        let id = s.upsert_chunk(&c, "/vault/a.md", 42).unwrap();
        assert!(id > 0);
        let content: String = s
            .conn()
            .query_row("SELECT content FROM chunks WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(content, c.content);
        let hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'hello'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1);
    }

    #[test]
    fn upsert_chunk_updates_in_place() {
        let (_d, s) = open_temp();
        let id1 = s
            .upsert_chunk(&sample_chunk(0, "alpha", "h1"), "/v/a.md", 1)
            .unwrap();
        let id2 = s
            .upsert_chunk(&sample_chunk(0, "beta changed", "h2"), "/v/a.md", 2)
            .unwrap();
        assert_eq!(id1, id2, "rowid should be stable");
        let alpha_hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(alpha_hits, 0);
        let beta_hits: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'beta'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(beta_hits, 1);
    }

    #[test]
    fn remove_file_cleans_up_chunks_vectors_fts_and_file_state() {
        let (_d, s) = open_temp();
        let path = "/v/a.md";
        let id = s
            .upsert_chunk(&sample_chunk(0, "searchable token", "h"), path, 1)
            .unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        s.set_file_state(path, "file-hash", 1).unwrap();
        s.remove_file(path).unwrap();
        let n: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
        let n: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'searchable'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0);
        let n: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0);
        assert!(s.get_file_state(path).unwrap().is_none());
    }

    #[test]
    fn remove_file_rolls_back_when_file_state_deletion_fails() {
        let (_d, s) = open_temp();
        let path = "note.md";
        let id = s
            .upsert_chunk(&sample_chunk(0, "prior searchable token", "h"), path, 1)
            .unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        s.set_file_state(path, "file-hash", 1).unwrap();
        s.conn()
            .execute_batch(
                "CREATE TRIGGER fail_file_delete BEFORE DELETE ON files
                 BEGIN SELECT RAISE(ABORT, 'forced file state delete failure'); END;",
            )
            .unwrap();

        let err = s.remove_file(path).unwrap_err();
        assert!(
            matches!(err, StoreError::Sqlite(_)),
            "unexpected error: {err}"
        );
        assert_eq!(
            s.get_file_state(path).unwrap(),
            Some(("file-hash".into(), 1))
        );
        assert_eq!(
            s.chunks_for_path(path).unwrap(),
            vec![(id, "prior searchable token".into())]
        );
        assert_eq!(
            s.vectors_for_chunks(&[id]).unwrap(),
            vec![(id, vec![0.0; TEST_DIM])]
        );
        assert_eq!(s.search_fts("prior", 10).unwrap().len(), 1);
    }

    #[test]
    fn replace_file_writes_chunks_vectors_cache_and_file_state_together() {
        let (_d, s) = open_temp();
        let chunks = vec![sample_chunk(
            0,
            "replacement searchable token",
            "chunk-hash",
        )];
        let embeddings = vec![vec![0.25; TEST_DIM]];
        let cache_entries = [PreparedEmbeddingCacheEntry {
            content_hash: "chunk-hash",
            model: "test-model",
            task_type: "RETRIEVAL_DOCUMENT",
            dim: TEST_DIM,
            embedding: &embeddings[0],
        }];

        s.replace_file(FileReplacement {
            path: "note.md",
            content_hash: "file-hash",
            mtime_ns: 42,
            chunks: &chunks,
            embeddings: &embeddings,
            cache_entries: &cache_entries,
        })
        .unwrap();

        assert_eq!(s.count_chunks().unwrap(), 1);
        assert_eq!(
            s.get_file_state("note.md").unwrap(),
            Some(("file-hash".into(), 42))
        );
        assert_eq!(
            s.get_embedding_cache("chunk-hash").unwrap(),
            Some(embeddings[0].clone())
        );
        let (id, _) = s.chunks_for_path("note.md").unwrap().pop().unwrap();
        assert_eq!(
            s.vectors_for_chunks(&[id]).unwrap(),
            vec![(id, embeddings[0].clone())]
        );
        assert_eq!(s.search_fts("replacement", 10).unwrap().len(), 1);
    }

    #[test]
    fn replace_file_rolls_back_when_vector_is_invalid() {
        let (_d, s) = open_temp();
        let old_chunks = vec![sample_chunk(0, "prior searchable token", "old-chunk-hash")];
        let old_embeddings = vec![vec![0.5; TEST_DIM]];
        s.replace_file(FileReplacement {
            path: "note.md",
            content_hash: "old-file-hash",
            mtime_ns: 1,
            chunks: &old_chunks,
            embeddings: &old_embeddings,
            cache_entries: &[],
        })
        .unwrap();
        let (old_id, _) = s.chunks_for_path("note.md").unwrap().pop().unwrap();

        let new_chunks = vec![sample_chunk(0, "new searchable token", "new-chunk-hash")];
        let invalid_embeddings = vec![vec![0.0; TEST_DIM - 1]];
        let err = s
            .replace_file(FileReplacement {
                path: "note.md",
                content_hash: "new-file-hash",
                mtime_ns: 2,
                chunks: &new_chunks,
                embeddings: &invalid_embeddings,
                cache_entries: &[],
            })
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Sqlite(_)),
            "unexpected error: {err}"
        );

        assert_eq!(
            s.get_file_state("note.md").unwrap(),
            Some(("old-file-hash".into(), 1))
        );
        assert_eq!(
            s.chunks_for_path("note.md").unwrap(),
            vec![(old_id, "prior searchable token".into())]
        );
        assert_eq!(
            s.vectors_for_chunks(&[old_id]).unwrap(),
            vec![(old_id, old_embeddings[0].clone())]
        );
        assert_eq!(s.search_fts("prior", 10).unwrap().len(), 1);
        assert_eq!(s.search_fts("new", 10).unwrap().len(), 0);
    }

    /// A failure inside `put_embedding_cache_in_tx` (after vectors are
    /// written but before `set_file_state_in_tx`) must roll back the whole
    /// transaction: old file state, old chunk, and old vector must all be
    /// preserved, and the new cache entry must not appear.
    #[test]
    fn replace_file_rolls_back_when_cache_insert_fails() {
        let (_d, s) = open_temp();
        let old_chunks = vec![sample_chunk(0, "prior searchable token", "old-chunk-hash")];
        let old_embeddings = vec![vec![0.5; TEST_DIM]];
        s.replace_file(FileReplacement {
            path: "note.md",
            content_hash: "old-file-hash",
            mtime_ns: 1,
            chunks: &old_chunks,
            embeddings: &old_embeddings,
            cache_entries: &[],
        })
        .unwrap();
        let (old_id, _) = s.chunks_for_path("note.md").unwrap().pop().unwrap();

        // Install a trigger that aborts any INSERT into embedding_cache.
        // This fires after the vector INSERT succeeds but before file state
        // is written, exercising the cache-insert failure rollback path.
        s.conn()
            .execute_batch(
                "CREATE TRIGGER fail_cache_insert BEFORE INSERT ON embedding_cache
                 BEGIN SELECT RAISE(ABORT, 'forced cache insert failure'); END;",
            )
            .unwrap();

        let new_chunks = vec![sample_chunk(0, "new searchable token", "new-chunk-hash")];
        let new_embeddings = vec![vec![0.75; TEST_DIM]];
        let cache_entries = [PreparedEmbeddingCacheEntry {
            content_hash: "new-chunk-hash",
            model: "test-model",
            task_type: "RETRIEVAL_DOCUMENT",
            dim: TEST_DIM,
            embedding: &new_embeddings[0],
        }];
        let err = s
            .replace_file(FileReplacement {
                path: "note.md",
                content_hash: "new-file-hash",
                mtime_ns: 2,
                chunks: &new_chunks,
                embeddings: &new_embeddings,
                cache_entries: &cache_entries,
            })
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Sqlite(_)),
            "unexpected error: {err}"
        );

        // All prior state must be intact.
        assert_eq!(
            s.get_file_state("note.md").unwrap(),
            Some(("old-file-hash".into(), 1))
        );
        assert_eq!(
            s.chunks_for_path("note.md").unwrap(),
            vec![(old_id, "prior searchable token".into())]
        );
        assert_eq!(
            s.vectors_for_chunks(&[old_id]).unwrap(),
            vec![(old_id, old_embeddings[0].clone())]
        );
        assert_eq!(s.search_fts("prior", 10).unwrap().len(), 1);
        // The new cache entry must NOT have been committed.
        assert!(
            s.get_embedding_cache("new-chunk-hash").unwrap().is_none(),
            "cache entry must not be present after rollback"
        );
    }

    #[test]
    fn embedding_cache_roundtrip() {
        let (_d, s) = open_temp();
        let v = vec![0.1f32, -0.2, 0.3, 0.4];
        s.put_embedding_cache("h", "m", "RETRIEVAL_DOCUMENT", 4, &v)
            .unwrap();
        let got = s.get_embedding_cache("h").unwrap().expect("hit");
        assert_eq!(got, v);
        assert!(s.get_embedding_cache("missing").unwrap().is_none());
    }

    #[test]
    fn embedding_cache_dim_mismatch() {
        let (_d, s) = open_temp();
        let err = s
            .put_embedding_cache("h", "m", "t", 4, &[1.0, 2.0, 3.0])
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::CacheDimMismatch { got: 3, want: 4 }
        ));
    }

    #[test]
    fn set_vector_for_chunk_replaces() {
        let (_d, s) = open_temp();
        let id = s
            .upsert_chunk(&sample_chunk(0, "x", "h"), "/v/a.md", 1)
            .unwrap();
        let v: Vec<f32> = (0..TEST_DIM).map(|i| i as f32 / TEST_DIM as f32).collect();
        s.set_vector_for_chunk(id, &v).unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        let n: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn meta_roundtrip() {
        let (_d, s) = open_temp();
        s.set_meta("k", "v1").unwrap();
        s.set_meta("k", "v2").unwrap();
        assert_eq!(s.get_meta("k").unwrap().as_deref(), Some("v2"));
    }

    #[test]
    fn reopen_persists() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        let cached_vec: Vec<f32> = (0..TEST_DIM).map(|i| i as f32 / 10.0).collect();
        let id = {
            let s = Store::open(&path, TEST_DIM).unwrap();
            let id = s
                .upsert_chunk(&sample_chunk(0, "persist me", "h"), "/v/a.md", 1)
                .unwrap();
            s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
            s.put_embedding_cache("h", "m", "RETRIEVAL_DOCUMENT", TEST_DIM, &cached_vec)
                .unwrap();
            id
        };
        let s2 = Store::open(&path, TEST_DIM).unwrap();
        let content: String = s2
            .conn()
            .query_row("SELECT content FROM chunks WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(content, "persist me");
        let cached = s2.get_embedding_cache("h").unwrap().expect("hit");
        assert_eq!(cached.len(), TEST_DIM);
    }

    #[test]
    fn rejects_mixed_dim_on_reopen() {
        // Open at one dim, close, reopen at another — must refuse with the
        // SchemaDimMismatch error so the operator has to explicitly nuke the DB.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            let _s = Store::open(&path, 8).expect("first open");
        }
        let result = Store::open(&path, 16);
        assert!(
            matches!(
                result,
                Err(StoreError::SchemaDimMismatch {
                    stored: 8,
                    config: 16,
                })
            ),
            "expected SchemaDimMismatch{{ stored: 8, config: 16 }}, got {:?}",
            result.err()
        );
    }

    /// The `SchemaDimMismatch` error message must offer `--reembed` as the
    /// primary recovery path. Deleting the DB is a fallback, not the first
    /// instruction — operators who have data they want to preserve should
    /// reach for `--reembed` first.
    #[test]
    fn schema_dim_mismatch_message_names_reembed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            let _s = Store::open(&path, 8).expect("first open");
        }
        let err = match Store::open(&path, 16) {
            Err(e) => e,
            Ok(_) => panic!("expected SchemaDimMismatch error"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("--reembed"),
            "SchemaDimMismatch message must name --reembed; got: {msg}"
        );
    }

    #[test]
    fn reopen_at_same_dim_ok() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            let _s = Store::open(&path, 16).expect("first open");
        }
        let _s = Store::open(&path, 16).expect("second open at same dim");
    }

    #[test]
    fn embedding_cache_swept_on_dim_change() {
        // Any embedding_cache row whose stored dim doesn't match the
        // currently-open dim is deleted on open. We can't actually reopen
        // at a mismatched dim (SchemaDimMismatch would block that), so
        // simulate the state by poking a stale row with `dim = 99` into
        // the cache while open at dim=8, then closing and reopening.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            let s = Store::open(&path, 8).expect("first open");
            // Bypass the dim check inside put_embedding_cache.
            s.conn()
                .execute(
                    "INSERT INTO embedding_cache(content_hash, model, task_type, dim, embedding, created_at) \
                     VALUES ('stale', 'm', 'RETRIEVAL_DOCUMENT', 99, X'00', strftime('%s','now'))",
                    [],
                )
                .unwrap();
        }
        // Reopen at the same dim — sweep should delete the dim=99 row.
        let s = Store::open(&path, 8).expect("reopen");
        let n: i64 = s
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM embedding_cache WHERE content_hash = 'stale'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "stale cache row should be swept on open");
    }

    /// Seed a row with an absolute path into both chunks+files and vec, with
    /// an FTS entry, so the migration has a realistic surface to rewrite.
    fn seed_absolute_row(store: &Store, abs_path: &str, content: &str) -> i64 {
        let c = Chunk {
            idx: 0,
            heading: String::new(),
            heading_path: String::new(),
            content: content.to_string(),
            content_hash: format!("hash-{abs_path}"),
            tokens: content.split_whitespace().count(),
            media_type: crate::media::MediaType::Text,
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        };
        let id = store.upsert_chunk(&c, abs_path, 100).unwrap();
        store
            .set_vector_for_chunk(id, &[0.0_f32; TEST_DIM])
            .unwrap();
        store
            .set_file_state(abs_path, &c.content_hash, 100)
            .unwrap();
        id
    }

    #[test]
    fn migrate_rewrites_absolute_paths_to_relative() {
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        let canonical = vault.canonicalize().unwrap();
        let vault_str = canonical.to_string_lossy().into_owned();
        let db = dir.path().join("x.db");

        {
            let s = Store::open(&db, TEST_DIM).unwrap();
            seed_absolute_row(&s, &format!("{vault_str}/a.md"), "body a");
            seed_absolute_row(&s, &format!("{vault_str}/sub/b.md"), "body b");
        }

        let s = Store::open(&db, TEST_DIM).unwrap();
        let outcome = s.migrate_paths_to_relative(&canonical).unwrap();
        assert_eq!(
            outcome,
            MigrationOutcome::Migrated {
                chunks_rewritten: 2,
                files_rewritten: 2,
            }
        );

        let chunk_paths: Vec<String> = s
            .conn()
            .prepare("SELECT path FROM chunks ORDER BY path")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            chunk_paths,
            vec!["a.md".to_string(), "sub/b.md".to_string()]
        );
        let file_paths: Vec<String> = s
            .conn()
            .prepare("SELECT path FROM files ORDER BY path")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(file_paths, vec!["a.md".to_string(), "sub/b.md".to_string()]);
        assert_eq!(
            s.get_meta("path_schema_version").unwrap().as_deref(),
            Some(PATH_SCHEMA_VERSION)
        );
    }

    #[test]
    fn migrate_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        let canonical = vault.canonicalize().unwrap();
        let vault_str = canonical.to_string_lossy().into_owned();
        let db = dir.path().join("x.db");

        {
            let s = Store::open(&db, TEST_DIM).unwrap();
            seed_absolute_row(&s, &format!("{vault_str}/a.md"), "body a");
        }
        let s = Store::open(&db, TEST_DIM).unwrap();
        let first = s.migrate_paths_to_relative(&canonical).unwrap();
        assert!(matches!(first, MigrationOutcome::Migrated { .. }));
        let second = s.migrate_paths_to_relative(&canonical).unwrap();
        assert_eq!(second, MigrationOutcome::AlreadyCurrent);
        // Paths still relative after the second call.
        let p: String = s
            .conn()
            .query_row("SELECT path FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(p, "a.md");
    }

    #[test]
    fn migrate_refuses_when_rows_outside_vault() {
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        let canonical = vault.canonicalize().unwrap();
        let db = dir.path().join("x.db");

        {
            let s = Store::open(&db, TEST_DIM).unwrap();
            // Row from a different vault — the migration must refuse.
            seed_absolute_row(&s, "/somewhere/else/a.md", "body a");
        }
        let s = Store::open(&db, TEST_DIM).unwrap();
        let outcome = s.migrate_paths_to_relative(&canonical).unwrap();
        assert_eq!(outcome, MigrationOutcome::Refused { offending_rows: 2 });
        // meta.path_schema_version must NOT be set.
        assert_eq!(s.get_meta("path_schema_version").unwrap(), None);
        // Row still intact.
        let p: String = s
            .conn()
            .query_row("SELECT path FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(p, "/somewhere/else/a.md");
    }

    #[test]
    fn migrate_is_noop_on_empty_db() {
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        let canonical = vault.canonicalize().unwrap();
        let s = Store::open(dir.path().join("x.db"), TEST_DIM).unwrap();
        let outcome = s.migrate_paths_to_relative(&canonical).unwrap();
        assert_eq!(
            outcome,
            MigrationOutcome::Migrated {
                chunks_rewritten: 0,
                files_rewritten: 0,
            }
        );
        assert_eq!(
            s.get_meta("path_schema_version").unwrap().as_deref(),
            Some(PATH_SCHEMA_VERSION)
        );
    }

    // --- fingerprint guard -------------------------------------------

    #[test]
    fn fingerprint_fresh_on_new_db() {
        let (_d, s) = open_temp();
        let outcome = s
            .check_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        assert_eq!(outcome, FingerprintOutcome::Fresh);
    }

    #[test]
    fn fingerprint_matches_after_set() {
        let (_d, s) = open_temp();
        s.set_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        let outcome = s
            .check_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        assert_eq!(outcome, FingerprintOutcome::Match);
    }

    #[test]
    fn fingerprint_mismatch_provider_names_both_values() {
        let (_d, s) = open_temp();
        s.set_fingerprint("gemini", "gemini-embedding-001", 3072)
            .unwrap();
        let outcome = s.check_fingerprint("voyage", "voyage-4", 1024).unwrap();
        let msg = match outcome {
            FingerprintOutcome::Mismatch(m) => m,
            other => panic!("expected Mismatch, got {other:?}"),
        };
        assert!(msg.contains("provider=gemini"), "{msg}");
        assert!(msg.contains("provider=voyage"), "{msg}");
        assert!(msg.contains("model=gemini-embedding-001"), "{msg}");
        assert!(msg.contains("model=voyage-4"), "{msg}");
        assert!(msg.contains("dim=3072"), "{msg}");
        assert!(msg.contains("dim=1024"), "{msg}");
        assert!(msg.contains("--reembed"), "{msg}");
    }

    #[test]
    fn fingerprint_mismatch_model_only() {
        let (_d, s) = open_temp();
        s.set_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        let outcome = s
            .check_fingerprint("gemini", "some-other-model", TEST_DIM)
            .unwrap();
        assert!(matches!(outcome, FingerprintOutcome::Mismatch(_)));
    }

    #[test]
    fn fingerprint_mismatch_dim_only() {
        let (_d, s) = open_temp();
        s.set_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        let outcome = s
            .check_fingerprint("gemini", "gemini-embedding-001", TEST_DIM + 1)
            .unwrap();
        assert!(matches!(outcome, FingerprintOutcome::Mismatch(_)));
    }

    #[test]
    fn wipe_and_rebuild_clears_data_and_sets_fingerprint() {
        let (_d, s) = open_temp();
        let id = s
            .upsert_chunk(&sample_chunk(0, "alpha", "h1"), "/v/a.md", 1)
            .unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        s.set_file_state("/v/a.md", "h1", 1).unwrap();
        s.put_embedding_cache("h1", "m", "t", TEST_DIM, &[0.0; TEST_DIM])
            .unwrap();

        s.wipe_and_rebuild(16, "voyage", "voyage-4").unwrap();

        assert_eq!(s.count_chunks().unwrap(), 0);
        assert!(s.list_indexed_paths().unwrap().is_empty());
        assert!(s.get_embedding_cache("h1").unwrap().is_none());
        let outcome = s.check_fingerprint("voyage", "voyage-4", 16).unwrap();
        assert_eq!(outcome, FingerprintOutcome::Match);
        // chunks_vec must be usable at the new dim.
        s.conn()
            .execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (1, ?1)",
                params![encode_f32(&[0.0_f32; 16])],
            )
            .expect("insert at new dim");
    }

    /// All three fingerprint meta keys must be present immediately after
    /// `wipe_and_rebuild` and match what was requested. They are written
    /// inside the wipe transaction, so this also validates atomicity — there
    /// is no post-commit window where any key can be absent.
    #[test]
    fn wipe_and_rebuild_fingerprint_is_atomic() {
        let (_d, s) = open_temp();
        s.wipe_and_rebuild(32, "voyage", "voyage-4-lite").unwrap();

        assert_eq!(
            s.get_meta("embedding_provider").unwrap().as_deref(),
            Some("voyage"),
            "embedding_provider not written"
        );
        assert_eq!(
            s.get_meta("embedding_model").unwrap().as_deref(),
            Some("voyage-4-lite"),
            "embedding_model not written"
        );
        assert_eq!(
            s.get_meta("embedding_dim").unwrap().as_deref(),
            Some("32"),
            "embedding_dim not written"
        );
        let outcome = s.check_fingerprint("voyage", "voyage-4-lite", 32).unwrap();
        assert_eq!(
            outcome,
            FingerprintOutcome::Match,
            "fingerprint check should match after wipe_and_rebuild"
        );
    }

    // --- peek_fingerprint ------------------------------------------------

    /// `peek_fingerprint` on a non-existent path returns `None` immediately
    /// without creating the file.
    #[test]
    fn peek_fingerprint_none_for_nonexistent_path() {
        let dir = TempDir::new().unwrap();
        let result = Store::peek_fingerprint(dir.path().join("no-such.db")).unwrap();
        assert!(result.is_none(), "expected None for missing DB");
    }

    /// After `set_fingerprint` on an open store, `peek_fingerprint` on the
    /// same path (with the store dropped) returns the exact same triple.
    #[test]
    fn peek_fingerprint_returns_written_triple() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            let s = Store::open(&path, TEST_DIM).unwrap();
            s.set_fingerprint("voyage", "voyage-4", TEST_DIM).unwrap();
        }
        let result = Store::peek_fingerprint(&path).unwrap();
        assert_eq!(
            result,
            Some(("voyage".into(), "voyage-4".into(), TEST_DIM)),
            "peek_fingerprint should return the stored triple"
        );
    }

    /// When only 2 of the 3 fingerprint keys are present (`embedding_provider`
    /// and `embedding_model` but not `embedding_dim`), `peek_fingerprint` must
    /// return `None`. The partial-key branch is what `FingerprintOutcome::from_peek`
    /// maps to `Fresh`, so a wrong return here would let a corrupted DB slip
    /// through as a dim=0 Mismatch.
    #[test]
    fn peek_fingerprint_returns_none_for_partial_keys() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            // open_for_reembed skips reconcile_embedding_dim, so embedding_dim
            // is not written to meta. Write only provider and model manually.
            let s = Store::open_for_reembed(&path, TEST_DIM).unwrap();
            s.set_meta("embedding_provider", "gemini").unwrap();
            s.set_meta("embedding_model", "gemini-embedding-001")
                .unwrap();
        }
        let result = Store::peek_fingerprint(&path).unwrap();
        assert!(
            result.is_none(),
            "2-of-3 fingerprint keys must return None, got: {result:?}"
        );
    }

    /// When only 1 of the 3 fingerprint keys is present, `peek_fingerprint`
    /// must also return `None`.
    #[test]
    fn peek_fingerprint_returns_none_for_single_key() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        {
            // open_for_reembed skips reconcile_embedding_dim so embedding_dim
            // is absent; write only embedding_provider.
            let s = Store::open_for_reembed(&path, TEST_DIM).unwrap();
            s.set_meta("embedding_provider", "gemini").unwrap();
        }
        let result = Store::peek_fingerprint(&path).unwrap();
        assert!(
            result.is_none(),
            "1-of-3 fingerprint keys must return None, got: {result:?}"
        );
    }

    // --- provider-only mismatch ------------------------------------------

    /// `check_fingerprint` must detect a provider change even when model and
    /// dim are identical. This exercises the branch that was untested by the
    /// E2E test (which changed the dim, not just the provider).
    #[test]
    fn fingerprint_mismatch_provider_only() {
        let (_d, s) = open_temp();
        s.set_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        // Same model and dim as stored, only the provider differs.
        let outcome = s
            .check_fingerprint("voyage", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        assert!(
            matches!(outcome, FingerprintOutcome::Mismatch(_)),
            "provider-only change must produce Mismatch, got: {outcome:?}"
        );
        let msg = match outcome {
            FingerprintOutcome::Mismatch(m) => m,
            _ => unreachable!(),
        };
        assert!(msg.contains("provider=gemini"), "{msg}");
        assert!(msg.contains("provider=voyage"), "{msg}");
        assert!(msg.contains("--reembed"), "{msg}");
    }

    /// A DB with only `embedding_provider` written (partial fingerprint from
    /// a hypothetical crash mid-`set_fingerprint`) must be treated as
    /// `Fresh`, not `Mismatch`. This prevents confusing error messages with
    /// fabricated empty-string field values.
    #[test]
    fn partial_fingerprint_is_fresh() {
        let (_d, s) = open_temp();
        // Write only the first of the three fingerprint keys directly.
        s.set_meta("embedding_provider", "gemini").unwrap();
        let outcome = s
            .check_fingerprint("gemini", "gemini-embedding-001", TEST_DIM)
            .unwrap();
        assert_eq!(
            outcome,
            FingerprintOutcome::Fresh,
            "one key present but not all three must be treated as Fresh"
        );
    }

    /// All three fingerprint meta keys must be present immediately after
    /// `set_fingerprint`, with no window where any key can be absent.
    /// Mirrors `wipe_and_rebuild_fingerprint_is_atomic`.
    #[test]
    fn set_fingerprint_writes_all_three_keys() {
        let (_d, s) = open_temp();
        s.set_fingerprint("voyage", "voyage-4", 1024).unwrap();

        assert_eq!(
            s.get_meta("embedding_provider").unwrap().as_deref(),
            Some("voyage"),
            "embedding_provider not written"
        );
        assert_eq!(
            s.get_meta("embedding_model").unwrap().as_deref(),
            Some("voyage-4"),
            "embedding_model not written"
        );
        assert_eq!(
            s.get_meta("embedding_dim").unwrap().as_deref(),
            Some("1024"),
            "embedding_dim not written"
        );
        let outcome = s.check_fingerprint("voyage", "voyage-4", 1024).unwrap();
        assert_eq!(
            outcome,
            FingerprintOutcome::Match,
            "fingerprint check must Match after set_fingerprint"
        );
    }
}
