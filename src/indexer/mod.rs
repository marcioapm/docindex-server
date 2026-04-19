//! Indexer: the single place where a path becomes chunks, vectors, and FTS
//! rows in the store. Both the startup walker and the live watcher feed
//! their dirty paths to `run` through the same channel, so there is exactly
//! one indexing pipeline.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, Ordering},
    },
    time::UNIX_EPOCH,
};

use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::{
    chunk::{self, Chunk},
    embed::{AnyEmbedder, EmbedError, TASK_RETRIEVAL_DOCUMENT},
    store::{Store, StoreError},
    walk,
};

#[derive(Debug, Error)]
pub enum IndexerError {
    #[error("indexer: {0}")]
    Msg(String),
    #[error("indexer: io: {0}")]
    Io(#[from] std::io::Error),
    #[error("indexer: walk: {0}")]
    Walk(#[from] walk::WalkError),
    #[error("indexer: store: {0}")]
    Store(#[from] StoreError),
    #[error("indexer: embed: {0}")]
    Embed(#[from] EmbedError),
    #[error("indexer: join: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Configuration shared between initial scan and the consumer loop.
#[derive(Clone)]
pub struct IndexerCtx {
    pub store: Arc<Mutex<Store>>,
    pub embedder: AnyEmbedder,
    pub vault_dir: PathBuf,
    pub embed_model: String,
    pub embed_dim: usize,
    pub last_reindex_ms: Arc<AtomicI64>,
}

/// Background task: drain `rx` and reindex each dirty path. Returns when
/// the channel is closed (i.e. every sender has been dropped).
pub async fn run(ctx: IndexerCtx, mut rx: mpsc::UnboundedReceiver<PathBuf>) {
    info!("indexer task started");
    while let Some(path) = rx.recv().await {
        // Deduplicate a burst of events for the same path that may arrive
        // from walker + watcher back-to-back.
        let mut batch: HashSet<PathBuf> = HashSet::new();
        batch.insert(path);
        while let Ok(extra) = rx.try_recv() {
            batch.insert(extra);
        }
        for p in batch {
            if let Err(e) = reindex_one(&ctx, &p).await {
                error!(path = %p.display(), error = %e, "reindex failed");
            }
        }
    }
    info!("indexer task stopped");
}

/// Run the startup full-tree diff and push every dirty path into `tx`.
/// Returns `(num_scanned, num_dirty, num_pruned)`.
pub async fn initial_scan(
    ctx: &IndexerCtx,
    tx: &mpsc::UnboundedSender<PathBuf>,
) -> Result<(usize, usize, usize), IndexerError> {
    let vault = ctx.vault_dir.clone();
    let files = tokio::task::spawn_blocking(move || walk::scan(&vault)).await??;

    let known: HashMap<String, String> = {
        let store = ctx.store.clone();
        tokio::task::spawn_blocking(move || -> Result<HashMap<String, String>, IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            let paths = guard.list_indexed_paths()?;
            let mut m = HashMap::with_capacity(paths.len());
            for p in paths {
                if let Some((h, _)) = guard.get_file_state(&p)? {
                    m.insert(p, h);
                }
            }
            Ok(m)
        })
        .await??
    };

    let seen: HashSet<String> = files
        .iter()
        .map(|f| f.path.to_string_lossy().into_owned())
        .collect();

    // Prune paths that vanished from disk.
    let to_prune: Vec<String> = known
        .keys()
        .filter(|p| !seen.contains(*p))
        .cloned()
        .collect();
    let pruned = to_prune.len();
    for path in to_prune {
        let store = ctx.store.clone();
        let p = path.clone();
        tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            guard.delete_chunks_for_path(&p)?;
            guard.delete_file_state(&p)?;
            Ok(())
        })
        .await??;
        info!(path = %path, "pruned missing file");
    }

    let scanned = files.len();
    let mut dirty = 0usize;
    for fs in files {
        let path_str = fs.path.to_string_lossy().into_owned();
        match known.get(&path_str) {
            Some(h) if h == &fs.content_hash => {
                debug!(path = %path_str, "unchanged; skipping");
                continue;
            }
            _ => {
                dirty += 1;
                if tx.send(fs.path.clone()).is_err() {
                    warn!("indexer channel closed during initial scan");
                    break;
                }
            }
        }
    }
    info!(scanned, dirty, pruned, "initial scan complete");
    Ok((scanned, dirty, pruned))
}

/// Reindex one path. Resolves to a no-op if the file vanished since the
/// event was queued (self-heals via the prune phase on next startup).
async fn reindex_one(ctx: &IndexerCtx, path: &Path) -> Result<(), IndexerError> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Deleted — drop it from the index.
            let path_str = path.to_string_lossy().into_owned();
            let store = ctx.store.clone();
            let p = path_str.clone();
            tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
                let guard = store
                    .lock()
                    .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
                guard.delete_chunks_for_path(&p)?;
                guard.delete_file_state(&p)?;
                Ok(())
            })
            .await??;
            info!(path = %path_str, "deleted from index");
            bump_reindex(ctx);
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    if !meta.is_file() {
        return Ok(());
    }
    let path_str = path.to_string_lossy().into_owned();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);

    let bytes = std::fs::read(path)?;
    let file_hash = {
        let mut h = Sha256::new();
        h.update(&bytes);
        hex::encode(h.finalize())
    };

    // Skip if the stored file hash matches — same bytes, same chunks.
    let stored = {
        let store = ctx.store.clone();
        let p = path_str.clone();
        tokio::task::spawn_blocking(move || -> Result<_, IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            Ok(guard.get_file_state(&p)?)
        })
        .await??
    };
    if let Some((h, _)) = stored
        && h == file_hash
    {
        debug!(path = %path_str, "unchanged on disk; skipping reindex");
        return Ok(());
    }

    let chunks = chunk::split(&bytes);
    if chunks.is_empty() {
        // Empty file: wipe any prior chunks and record the file_hash so we
        // don't keep re-processing it.
        let store = ctx.store.clone();
        let p = path_str.clone();
        let hash_c = file_hash.clone();
        tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            guard.delete_chunks_for_path(&p)?;
            guard.set_file_state(&p, &hash_c, mtime_ns)?;
            Ok(())
        })
        .await??;
        bump_reindex(ctx);
        return Ok(());
    }

    // Resolve embeddings (cache-first).
    let (embeddings, cached_count, embedded_count) = resolve_embeddings(ctx, &chunks).await?;

    // Persist: delete existing chunks for path, re-insert, set vectors,
    // update file_state, update last_reindex. Each store op is short; we
    // take the lock once per op via spawn_blocking.
    let store = ctx.store.clone();
    let path_c = path_str.clone();
    let hash_c = file_hash.clone();
    let chunks_c = chunks.clone();
    let embeddings_c = embeddings.clone();
    tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
        let guard = store
            .lock()
            .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
        guard.delete_chunks_for_path(&path_c)?;
        for (i, c) in chunks_c.iter().enumerate() {
            let id = guard.upsert_chunk(c, &path_c, mtime_ns)?;
            guard.set_vector_for_chunk(id, &embeddings_c[i])?;
        }
        guard.set_file_state(&path_c, &hash_c, mtime_ns)?;
        Ok(())
    })
    .await??;

    info!(
        path = %path_str,
        chunks = chunks.len(),
        embedded = embedded_count,
        cached = cached_count,
        "reindexed"
    );
    bump_reindex(ctx);
    Ok(())
}

/// Fetch embeddings for every chunk: cache hits come back instantly, cache
/// misses are batched into a single embedder call.
async fn resolve_embeddings(
    ctx: &IndexerCtx,
    chunks: &[Chunk],
) -> Result<(Vec<Vec<f32>>, usize, usize), IndexerError> {
    let hashes: Vec<String> = chunks.iter().map(|c| c.content_hash.clone()).collect();

    // Lookup cache.
    let cached: Vec<Option<Vec<f32>>> = {
        let store = ctx.store.clone();
        let hashes_c = hashes.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Option<Vec<f32>>>, IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            let mut out = Vec::with_capacity(hashes_c.len());
            for h in &hashes_c {
                out.push(guard.get_embedding_cache(h)?);
            }
            Ok(out)
        })
        .await??
    };

    let mut to_embed_texts: Vec<String> = Vec::new();
    let mut to_embed_indexes: Vec<usize> = Vec::new();
    for (i, hit) in cached.iter().enumerate() {
        if hit.is_none() {
            to_embed_texts.push(chunks[i].content.clone());
            to_embed_indexes.push(i);
        }
    }

    let embedded_count = to_embed_texts.len();
    let cached_count = chunks.len() - embedded_count;

    let fresh: Vec<Vec<f32>> = if to_embed_texts.is_empty() {
        Vec::new()
    } else {
        ctx.embedder.embed_documents(&to_embed_texts).await?
    };

    if fresh.len() != to_embed_texts.len() {
        return Err(IndexerError::Msg(format!(
            "embedder returned {} vectors for {} inputs",
            fresh.len(),
            to_embed_texts.len()
        )));
    }

    // Populate cache for fresh ones.
    if !fresh.is_empty() {
        let store = ctx.store.clone();
        let model = ctx.embed_model.clone();
        let dim = ctx.embed_dim;
        let fresh_pairs: Vec<(String, Vec<f32>)> = to_embed_indexes
            .iter()
            .zip(fresh.iter())
            .map(|(i, v)| (chunks[*i].content_hash.clone(), v.clone()))
            .collect();
        tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            for (hash, v) in fresh_pairs {
                guard.put_embedding_cache(&hash, &model, TASK_RETRIEVAL_DOCUMENT, dim, &v)?;
            }
            Ok(())
        })
        .await??;
    }

    // Assemble final vector list in chunk order.
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    let mut fresh_iter = fresh.into_iter();
    for (i, hit) in cached.into_iter().enumerate() {
        match hit {
            Some(v) => out.push(v),
            None => {
                let v = fresh_iter.next().ok_or_else(|| {
                    IndexerError::Msg(format!("missing fresh embedding for chunk {i}"))
                })?;
                out.push(v);
            }
        }
    }
    Ok((out, cached_count, embedded_count))
}

fn bump_reindex(ctx: &IndexerCtx) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    ctx.last_reindex_ms.store(now_ms, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::Fake;
    use tempfile::TempDir;

    fn mk_ctx(dir: &Path) -> IndexerCtx {
        const DIM: usize = 8;
        let store = Arc::new(Mutex::new(Store::open(dir.join("x.db"), DIM).unwrap()));
        IndexerCtx {
            store,
            embedder: AnyEmbedder::Fake(Arc::new(Fake::new(DIM))),
            vault_dir: dir.join("vault"),
            embed_model: "test".into(),
            embed_dim: DIM,
            last_reindex_ms: Arc::new(AtomicI64::new(0)),
        }
    }

    #[tokio::test]
    async fn end_to_end_indexes_a_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("vault")).unwrap();
        std::fs::write(
            dir.path().join("vault/hello.md"),
            b"# Hello\n\nworld body\n",
        )
        .unwrap();
        let ctx = mk_ctx(dir.path());
        let (tx, rx) = mpsc::unbounded_channel();
        let scan_tx = tx.clone();
        let (scanned, dirty, pruned) = initial_scan(&ctx, &scan_tx).await.unwrap();
        assert_eq!(scanned, 1);
        assert_eq!(dirty, 1);
        assert_eq!(pruned, 0);

        drop(tx);
        drop(scan_tx);
        let task_ctx = ctx.clone();
        run(task_ctx, rx).await;

        let n = {
            let g = ctx.store.lock().unwrap();
            g.count_chunks().unwrap()
        };
        assert!(n >= 1);
        assert!(ctx.last_reindex_ms.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn no_op_when_unchanged() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("vault")).unwrap();
        std::fs::write(
            dir.path().join("vault/hello.md"),
            b"# Hello\n\nworld body\n",
        )
        .unwrap();
        let ctx = mk_ctx(dir.path());
        // First pass
        let (tx, rx) = mpsc::unbounded_channel();
        let (_, dirty, _) = initial_scan(&ctx, &tx).await.unwrap();
        assert_eq!(dirty, 1);
        drop(tx);
        run(ctx.clone(), rx).await;
        let n1 = ctx.store.lock().unwrap().count_chunks().unwrap();
        // Second pass — should be a no-op.
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let (_, dirty2, _) = initial_scan(&ctx, &tx2).await.unwrap();
        assert_eq!(dirty2, 0, "unchanged file should not be dirty");
        let n2 = ctx.store.lock().unwrap().count_chunks().unwrap();
        assert_eq!(n1, n2);
    }

    #[tokio::test]
    async fn delete_prunes_from_index() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("vault")).unwrap();
        let p = dir.path().join("vault/gone.md");
        std::fs::write(&p, b"# Gone\n\nsoon\n").unwrap();
        let ctx = mk_ctx(dir.path());
        let (tx, rx) = mpsc::unbounded_channel();
        initial_scan(&ctx, &tx).await.unwrap();
        drop(tx);
        run(ctx.clone(), rx).await;
        assert!(ctx.store.lock().unwrap().count_chunks().unwrap() >= 1);

        std::fs::remove_file(&p).unwrap();
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let (_, _, pruned) = initial_scan(&ctx, &tx2).await.unwrap();
        assert_eq!(pruned, 1);
        assert_eq!(ctx.store.lock().unwrap().count_chunks().unwrap(), 0);
    }
}
