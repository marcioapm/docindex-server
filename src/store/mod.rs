//! SQLite + sqlite-vec + FTS5 store.

mod vec;

use std::path::Path;
use std::sync::OnceLock;

use rusqlite::{Connection, OptionalExtension, ffi::sqlite3_auto_extension, params};
use thiserror::Error;

use crate::chunk::Chunk;

pub use self::vec::{decode_f32, encode_f32};

const SCHEMA_SQL: &str = include_str!("schema.sql");

/// Current schema version, written to meta on open.
pub const SCHEMA_VERSION: &str = "1";

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store: {0}")]
    Msg(String),
    #[error("store: sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("store: sqlite-vec extension failed to load: {0}")]
    VecExtension(String),
    #[error("store: dim mismatch: got {got}, want {want}")]
    DimMismatch { got: usize, want: usize },
}

/// Wraps a `rusqlite::Connection` with the docindex schema applied and
/// the `sqlite-vec` extension loaded.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (or create) the SQLite DB at `path`. Registers `sqlite-vec` as
    /// an auto-extension exactly once per process, then applies the schema
    /// and writes `meta.schema_version`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        register_sqlite_vec()?;
        let conn = Connection::open(path.as_ref())?;
        init_pragmas(&conn)?;
        verify_vec_loaded(&conn)?;
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| StoreError::Msg(format!("apply schema: {e}")))?;
        let s = Self { conn };
        s.set_meta("schema_version", SCHEMA_VERSION)?;
        Ok(s)
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
            return Err(StoreError::DimMismatch {
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

    fn open_temp() -> (TempDir, Store) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("x.db");
        let store = Store::open(&path).expect("open");
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
                params![1i64, encode_f32(&vec![0.0_f32; 768])],
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
        s.set_vector_for_chunk(id, &vec![0.0; 768]).unwrap();
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
        assert!(matches!(err, StoreError::DimMismatch { got: 3, want: 4 }));
    }

    #[test]
    fn set_vector_for_chunk_replaces() {
        let (_d, s) = open_temp();
        let id = s
            .upsert_chunk(&sample_chunk(0, "x", "h"), "/v/a.md", 1)
            .unwrap();
        let v: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
        s.set_vector_for_chunk(id, &v).unwrap();
        s.set_vector_for_chunk(id, &vec![0.0; 768]).unwrap();
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
        let id = {
            let s = Store::open(&path).unwrap();
            let id = s
                .upsert_chunk(&sample_chunk(0, "persist me", "h"), "/v/a.md", 1)
                .unwrap();
            s.set_vector_for_chunk(id, &vec![0.0; 768]).unwrap();
            s.put_embedding_cache("h", "m", "RETRIEVAL_DOCUMENT", 4, &[1.0, 2.0, 3.0, 4.0])
                .unwrap();
            id
        };
        let s2 = Store::open(&path).unwrap();
        let content: String = s2
            .conn()
            .query_row("SELECT content FROM chunks WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(content, "persist me");
        let cached = s2.get_embedding_cache("h").unwrap().expect("hit");
        assert_eq!(cached.len(), 4);
    }
}
