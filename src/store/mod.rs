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
pub const SCHEMA_VERSION: &str = "2";

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
    #[error(
        "store: embedding_dim on disk is {stored}, config says {config} — refusing to mix. Delete the index DB to reindex at the new dim."
    )]
    SchemaDimMismatch { stored: usize, config: usize },
}

/// Wraps a `rusqlite::Connection` with the docindex schema applied and
/// the `sqlite-vec` extension loaded.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (or create) the SQLite DB at `path`. Registers `sqlite-vec` as
    /// an auto-extension exactly once per process, applies the base schema,
    /// renders + applies the `chunks_vec` DDL with `embed_dim` baked into
    /// `FLOAT[...]`, and enforces that `meta.embedding_dim` matches
    /// `embed_dim` (refusing to start on mismatch).
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
        s.set_meta("schema_version", SCHEMA_VERSION)?;
        if !skip_dim_check {
            s.reconcile_embedding_dim(embed_dim)?;
        }
        Ok(s)
    }

    /// Drop and recreate `chunks_fts` and `chunks_vec`, delete every row of
    /// `chunks` / `files` / `embedding_cache`, and write the new embedding
    /// fingerprint. Used by `--reembed` after [`Store::open_for_reembed`] —
    /// the only supported way to change the embedding dim on an existing
    /// DB, since `chunks_vec`'s dim is baked into its DDL.
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
        tx.commit()?;
        self.set_meta("embedding_dim", &embed_dim.to_string())?;
        self.set_meta("embedding_provider", provider)?;
        self.set_meta("embedding_model", model)?;
        Ok(())
    }

    /// Compare the stored embedding fingerprint (`embedding_provider` /
    /// `embedding_model` / `embedding_dim` in `meta`) against the effective
    /// config. A DB with no fingerprint recorded yet (pre-existing
    /// deployments, or a genuinely empty index) is [`FingerprintOutcome::Fresh`]
    /// — the caller should adopt the current config as the fingerprint via
    /// [`Store::set_fingerprint`] rather than error, so upgrading an
    /// existing production DB never breaks on its own.
    pub fn check_fingerprint(
        &self,
        provider: &str,
        model: &str,
        dim: usize,
    ) -> Result<FingerprintOutcome, StoreError> {
        let stored_provider = self.get_meta("embedding_provider")?;
        let stored_model = self.get_meta("embedding_model")?;
        let (stored_provider, stored_model) = match (stored_provider, stored_model) {
            (None, None) => return Ok(FingerprintOutcome::Fresh),
            (p, m) => (p.unwrap_or_default(), m.unwrap_or_default()),
        };
        let stored_dim: usize = self
            .get_meta("embedding_dim")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(dim);
        if stored_provider == provider && stored_model == model && stored_dim == dim {
            return Ok(FingerprintOutcome::Match);
        }
        Ok(FingerprintOutcome::Mismatch(format!(
            "index built with provider={stored_provider} model={stored_model} dim={stored_dim}, \
             config says provider={provider} model={model} dim={dim}; re-embed required: run with --reembed"
        )))
    }

    /// Record the embedding fingerprint. Called once, on a genuinely fresh
    /// index (see [`FingerprintOutcome::Fresh`]) or immediately after
    /// [`Store::wipe_and_rebuild`] (which writes it directly).
    pub fn set_fingerprint(
        &self,
        provider: &str,
        model: &str,
        dim: usize,
    ) -> Result<(), StoreError> {
        self.set_meta("embedding_provider", provider)?;
        self.set_meta("embedding_model", model)?;
        self.set_meta("embedding_dim", &dim.to_string())?;
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

    /// Insert or update the chunk at (`path`, `chunk.idx`) and keep
    /// `chunks_fts` in sync. Returns the stable `chunks.id` rowid.
    pub fn upsert_chunk(&self, c: &Chunk, path: &str, mtime_ns: i64) -> Result<i64, StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        let existing: Option<(i64, String, Option<String>)> = tx
            .query_row(
                "SELECT id, content, heading_path FROM chunks WHERE path = ? AND chunk_idx = ?",
                params![path, c.idx as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()?;

        let id = match existing {
            None => {
                tx.execute(
                    "INSERT INTO chunks(path, chunk_idx, heading, heading_path, content, content_hash, mtime_ns, tokens)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        path,
                        c.idx as i64,
                        null_if_empty(&c.heading),
                        null_if_empty(&c.heading_path),
                        c.content,
                        c.content_hash,
                        mtime_ns,
                        c.tokens as i64,
                    ],
                )?;
                tx.last_insert_rowid()
            }
            Some((id, old_content, old_path)) => {
                // Delete old FTS row before updating chunks, then re-insert.
                tx.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?1, ?2, ?3)",
                    params![id, old_content, old_path.unwrap_or_default()],
                )?;
                tx.execute(
                    "UPDATE chunks SET heading=?1, heading_path=?2, content=?3, content_hash=?4, mtime_ns=?5, tokens=?6
                     WHERE id=?7",
                    params![
                        null_if_empty(&c.heading),
                        null_if_empty(&c.heading_path),
                        c.content,
                        c.content_hash,
                        mtime_ns,
                        c.tokens as i64,
                        id,
                    ],
                )?;
                id
            }
        };

        tx.execute(
            "INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (?1, ?2, ?3)",
            params![id, c.content, c.heading_path],
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Remove every chunk for `path`, including FTS and vec rows.
    pub fn delete_chunks_for_path(&self, path: &str) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        let rows: Vec<(i64, String, Option<String>)> = {
            let mut stmt =
                tx.prepare("SELECT id, content, heading_path FROM chunks WHERE path = ?")?;
            let mapped = stmt.query_map(params![path], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            let mut v = Vec::new();
            for r in mapped {
                v.push(r?);
            }
            v
        };
        for (id, content, heading_path) in &rows {
            tx.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?1, ?2, ?3)",
                params![id, content, heading_path.clone().unwrap_or_default()],
            )?;
            tx.execute("DELETE FROM chunks_vec WHERE rowid = ?1", params![id])?;
        }
        tx.execute("DELETE FROM chunks WHERE path = ?1", params![path])?;
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
                "SELECT id, path, heading, heading_path, content FROM chunks WHERE id = ?1",
                params![id],
                |r| {
                    Ok(HitRow {
                        id: r.get::<_, i64>(0)?,
                        path: r.get::<_, String>(1)?,
                        heading: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        heading_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        content: r.get::<_, String>(4)?,
                    })
                },
            )
            .optional()?)
    }

    /// All chunks for `path` (id, content). Used by /similar to build a
    /// pseudo-query from the bag-of-words + average of stored vectors.
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
/// effective config, from [`Store::check_fingerprint`].
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

/// Minimal row projection used to hydrate a search hit. Kept in the store
/// module so SQL column order stays colocated with the schema.
#[derive(Debug, Clone)]
pub struct HitRow {
    pub id: i64,
    pub path: String,
    pub heading: String,
    pub heading_path: String,
    pub content: String,
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
        }
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
    fn upsert_chunk_inserts_and_indexes_fts() {
        let (_d, s) = open_temp();
        let c = Chunk {
            idx: 0,
            heading: "T".into(),
            heading_path: "T".into(),
            content: "# T\nhello world".into(),
            content_hash: "hash1".into(),
            tokens: 3,
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
    fn delete_chunks_for_path_cleans_up() {
        let (_d, s) = open_temp();
        let id = s
            .upsert_chunk(&sample_chunk(0, "searchable token", "h"), "/v/a.md", 1)
            .unwrap();
        s.set_vector_for_chunk(id, &[0.0; TEST_DIM]).unwrap();
        s.delete_chunks_for_path("/v/a.md").unwrap();
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
}
