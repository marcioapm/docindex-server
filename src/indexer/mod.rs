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

use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::{
    chunk::{self, Chunk},
    embed::registry,
    embed::{AnyEmbedder, EmbedError, EmbedInput, MEDIA_DOCUMENT_TASK, TASK_RETRIEVAL_DOCUMENT},
    media::{MediaPolicy, MediaType},
    media_prepare::{MediaPrepareError, PrepareOptions, prepare_media},
    store::{FileReplacement, PreparedEmbeddingCacheEntry, Store, StoreError},
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
    #[error("indexer: media preparation: {0}")]
    MediaPrepare(#[from] MediaPrepareError),
    #[error("indexer: model registry: {0}")]
    Registry(#[from] registry::RegistryError),
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
    pub media_policy: MediaPolicy,
    pub last_reindex_ms: Arc<AtomicI64>,
}

/// Background task: drain `rx` and reindex each dirty path. Returns when
/// the channel is closed (i.e. every sender has been dropped).
///
/// Paths on the channel are **vault-relative** — both the walker and the
/// watcher strip `vault_dir` before sending.
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
    let policy = ctx.media_policy.clone();
    let files =
        tokio::task::spawn_blocking(move || walk::scan_with_policy(&vault, &policy)).await??;

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
        .map(|f| f.rel_path.to_string_lossy().into_owned())
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
            guard.remove_file(&p)?;
            Ok(())
        })
        .await??;
        info!(path = %path, "pruned missing file");
    }

    let scanned = files.len();
    let mut dirty = 0usize;
    for fs in files {
        let path_str = fs.rel_path.to_string_lossy().into_owned();
        match known.get(&path_str) {
            Some(h) if h == &fs.content_hash => {
                debug!(path = %path_str, "unchanged; skipping");
                continue;
            }
            _ => {
                dirty += 1;
                if tx.send(fs.rel_path.clone()).is_err() {
                    warn!("indexer channel closed during initial scan");
                    break;
                }
            }
        }
    }
    info!(scanned, dirty, pruned, "initial scan complete");
    Ok((scanned, dirty, pruned))
}

/// Reindex one path. `rel_path` is relative to `ctx.vault_dir`; the file is
/// read from `ctx.vault_dir.join(rel_path)` but stored in the DB under the
/// relative form. Resolves to a no-op if the file vanished since the event
/// was queued (self-heals via the prune phase on next startup).
async fn reindex_one(ctx: &IndexerCtx, rel_path: &Path) -> Result<(), IndexerError> {
    // Defensive: channel contract says relative, but if something ever hands
    // us an absolute path we'd silently end up storing absolute paths again.
    if rel_path.is_absolute() {
        return Err(IndexerError::Msg(format!(
            "reindex_one called with absolute path {}",
            rel_path.display()
        )));
    }
    let abs_path = ctx.vault_dir.join(rel_path);
    let meta = match std::fs::metadata(&abs_path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Deleted — drop it from the index.
            let path_str = rel_path.to_string_lossy().into_owned();
            let store = ctx.store.clone();
            let p = path_str.clone();
            tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
                let guard = store
                    .lock()
                    .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
                guard.remove_file(&p)?;
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
    let path_str = rel_path.to_string_lossy().into_owned();
    let Some(media_type) = ctx.media_policy.allows_existing_file(rel_path, meta.len()) else {
        let store = ctx.store.clone();
        let p = path_str.clone();
        tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            guard.remove_file(&p)?;
            Ok(())
        })
        .await??;
        bump_reindex(ctx);
        return Ok(());
    };
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or(0);
    // Read and prepare everything before making any store mutation. This keeps
    // a decode/embed failure from destroying the prior indexed representation.
    let bytes = std::fs::read(&abs_path)?;
    let file_hash = ctx.media_policy.effective_file_hash(&bytes, media_type);

    // Skip if the effective hash matches. For media this incorporates the
    // preparation policy, so policy changes correctly force a reindex.
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

    let (chunks, inputs) = match media_type {
        MediaType::Text => {
            let chunks = chunk::split(&bytes);
            let inputs = chunks
                .iter()
                .map(|chunk| EmbedInput::Text(chunk.content.clone()))
                .collect();
            (chunks, inputs)
        }
        _ => {
            let provider = match &ctx.embedder {
                AnyEmbedder::Gemini(_) => registry::EmbedProvider::Gemini,
                AnyEmbedder::Voyage(_) => registry::EmbedProvider::Voyage,
                AnyEmbedder::Fake(_) => registry::EmbedProvider::Fake,
            };
            let mut model = registry::lookup(provider, &ctx.embed_model)?;
            // The deterministic fake embedder accepts typed media inputs so
            // offline tests exercise the same preparation pipeline.
            if provider == registry::EmbedProvider::Fake {
                model.media_capable = true;
            }
            let prepared = prepare_media(
                rel_path,
                &bytes,
                &model,
                PrepareOptions {
                    pdf_pages_per_chunk: ctx.media_policy.pdf_pages_per_chunk,
                    pdf_dpi: ctx.media_policy.pdf_dpi,
                    ..PrepareOptions::default()
                },
            )?;
            let mut chunks = Vec::with_capacity(prepared.chunks.len());
            let mut inputs = Vec::with_capacity(prepared.chunks.len());
            for prepared_chunk in prepared.chunks {
                let metadata = prepared_chunk.metadata;
                chunks.push(Chunk {
                    idx: metadata.chunk_index,
                    heading: String::new(),
                    heading_path: String::new(),
                    content: String::new(),
                    content_hash: metadata.cache_key,
                    tokens: 0,
                    media_type: prepared.media_type,
                    mime_type: Some(metadata.mime_type),
                    media_start: metadata.page_range.map(|(start, _)| start as i64),
                    media_end: metadata.page_range.map(|(_, end)| end as i64),
                    media_unit: metadata.page_range.map(|_| "page".to_owned()),
                    truncated: metadata.truncated_animation,
                });
                inputs.push(prepared_chunk.input);
            }
            (chunks, inputs)
        }
    };

    // Resolve all cache hits and typed misses before atomically replacing the
    // file. Fresh cache entries are committed with the file replacement.
    let (embeddings, fresh_embeddings, cached_count, embedded_count) =
        resolve_embeddings(ctx, &chunks, &inputs).await?;
    let chunk_count = chunks.len();
    let cache_model = ctx.embed_model.clone();
    let cache_dim = ctx.embed_dim;
    let store = ctx.store.clone();
    let path_c = path_str.clone();
    let hash_c = file_hash.clone();
    tokio::task::spawn_blocking(move || -> Result<(), IndexerError> {
        let cache_entries: Vec<PreparedEmbeddingCacheEntry<'_>> = fresh_embeddings
            .iter()
            .map(|(index, embedding)| PreparedEmbeddingCacheEntry {
                content_hash: &chunks[*index].content_hash,
                model: &cache_model,
                task_type: if media_type == MediaType::Text {
                    TASK_RETRIEVAL_DOCUMENT
                } else {
                    MEDIA_DOCUMENT_TASK
                },
                dim: cache_dim,
                embedding,
            })
            .collect();
        let guard = store
            .lock()
            .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
        guard.replace_file(FileReplacement {
            path: &path_c,
            content_hash: &hash_c,
            mtime_ns,
            chunks: &chunks,
            embeddings: &embeddings,
            cache_entries: &cache_entries,
        })?;
        Ok(())
    })
    .await??;

    info!(
        path = %path_str,
        chunks = chunk_count,
        embedded = embedded_count,
        cached = cached_count,
        "reindexed"
    );
    bump_reindex(ctx);
    Ok(())
}

/// Fetch embeddings for typed document inputs. Cache misses are batched and
/// returned as prepared cache entries; this function never mutates the store.
async fn resolve_embeddings(
    ctx: &IndexerCtx,
    chunks: &[Chunk],
    inputs: &[EmbedInput],
) -> Result<(Vec<Vec<f32>>, Vec<(usize, Vec<f32>)>, usize, usize), IndexerError> {
    if chunks.len() != inputs.len() {
        return Err(IndexerError::Msg("chunk/input count mismatch".into()));
    }
    let hashes: Vec<String> = chunks
        .iter()
        .map(|chunk| chunk.content_hash.clone())
        .collect();
    let cached: Vec<Option<Vec<f32>>> = {
        let store = ctx.store.clone();
        let hashes = hashes.clone();
        tokio::task::spawn_blocking(move || -> Result<_, IndexerError> {
            let guard = store
                .lock()
                .map_err(|e| IndexerError::Msg(format!("store lock: {e}")))?;
            hashes
                .iter()
                .map(|hash| guard.get_embedding_cache(hash).map_err(IndexerError::from))
                .collect()
        })
        .await??
    };
    let missing: Vec<usize> = cached
        .iter()
        .enumerate()
        .filter_map(|(index, hit)| hit.is_none().then_some(index))
        .collect();
    let cached_count = chunks.len() - missing.len();
    let fresh = if missing.is_empty() {
        Vec::new()
    } else {
        let missing_inputs: Vec<EmbedInput> =
            missing.iter().map(|&index| inputs[index].clone()).collect();
        ctx.embedder.embed_documents(&missing_inputs).await?
    };
    if fresh.len() != missing.len() {
        return Err(IndexerError::Msg(format!(
            "embedder returned {} vectors for {} inputs",
            fresh.len(),
            missing.len()
        )));
    }
    let mut embeddings = Vec::with_capacity(chunks.len());
    let mut fresh_iter = fresh.iter();
    for hit in &cached {
        match hit {
            Some(embedding) => embeddings.push(embedding.clone()),
            None => embeddings.push(
                fresh_iter
                    .next()
                    .ok_or_else(|| IndexerError::Msg("missing fresh embedding".into()))?
                    .clone(),
            ),
        }
    }
    let embedded_count = missing.len();
    let fresh_embeddings = missing.into_iter().zip(fresh).collect();
    Ok((embeddings, fresh_embeddings, cached_count, embedded_count))
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
            media_policy: MediaPolicy::default(),
            last_reindex_ms: Arc::new(AtomicI64::new(0)),
        }
    }

    #[tokio::test]
    async fn fake_png_reindex_records_file_state_without_fts_rows() {
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        image::RgbaImage::new(1, 1)
            .save(vault.join("image.png"))
            .unwrap();
        let mut ctx = mk_ctx(dir.path());
        ctx.embed_model = "gemini-embedding-2".into();
        ctx.media_policy = MediaPolicy::new(true, &[], &[], &[], 20, 1, 150).unwrap();

        reindex_one(&ctx, Path::new("image.png")).await.unwrap();
        let store = ctx.store.lock().unwrap();
        assert!(store.get_file_state("image.png").unwrap().is_some());
        assert_eq!(store.count_chunks().unwrap(), 1);
        assert!(
            store.search_fts("image", 10).unwrap().is_empty(),
            "media chunks must not be added to FTS"
        );
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

    #[tokio::test]
    async fn stores_relative_paths_in_db() {
        // Indexing a nested file must persist the vault-relative path in
        // both `chunks.path` and `files.path` — never the absolute path.
        let dir = TempDir::new().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(vault.join("notes")).unwrap();
        std::fs::write(vault.join("notes/a.md"), b"# A\n\nalpha\n").unwrap();
        let ctx = mk_ctx(dir.path());
        let (tx, rx) = mpsc::unbounded_channel();
        initial_scan(&ctx, &tx).await.unwrap();
        drop(tx);
        run(ctx.clone(), rx).await;

        let guard = ctx.store.lock().unwrap();
        let chunk_paths: Vec<String> = guard
            .conn()
            .prepare("SELECT DISTINCT path FROM chunks")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunk_paths, vec!["notes/a.md".to_string()]);
        let file_paths = guard.list_indexed_paths().unwrap();
        assert_eq!(file_paths, vec!["notes/a.md".to_string()]);
        for p in chunk_paths.iter().chain(file_paths.iter()) {
            assert!(
                !p.starts_with('/'),
                "stored path must be relative, got {p:?}"
            );
        }
    }

    #[tokio::test]
    async fn rejects_absolute_paths_in_reindex() {
        // The channel contract says relative; if an absolute path ever slips
        // through we fail loudly instead of silently re-introducing the bug.
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("vault")).unwrap();
        let abs = dir.path().join("vault/anything.md");
        std::fs::write(&abs, b"x").unwrap();
        let ctx = mk_ctx(dir.path());
        let err = reindex_one(&ctx, &abs).await.unwrap_err();
        assert!(
            format!("{err}").contains("absolute"),
            "unexpected error: {err}"
        );
    }
}
