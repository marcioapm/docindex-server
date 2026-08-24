//! Wires config + store + embedder + indexer + watcher + HTTP server into
//! a single `run(config)` function. `main.rs` is a thin shell on top.

use std::{
    net::SocketAddr,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::{
    Config,
    api::{self, AppState},
    embed::{AnyEmbedder, Fake, Gemini, Voyage},
    indexer::{self, IndexerCtx},
    store::Store,
    watch as watcher,
};

/// Start the server. Returns when the listener is closed (either via
/// graceful shutdown on SIGINT/SIGTERM or a bind error).
pub async fn run(cfg: Config) -> Result<()> {
    let embedder = build_embedder(&cfg)?;
    let store = open_store_with_fingerprint_check(&cfg)?;
    // Rewrite any pre-0.2.0 absolute paths to vault-relative form. Idempotent
    // across restarts; a mismatch between the DB's paths and the configured
    // vault_dir is surfaced as a refusal (logged, not fatal) so operators can
    // reconcile before the indexer starts touching rows.
    store
        .migrate_paths_to_relative(&cfg.vault_dir)
        .context("migrate paths to relative")?;
    let store = Arc::new(Mutex::new(store));

    let last_reindex_ms = Arc::new(AtomicI64::new(now_ms().checked_sub(1).unwrap_or(0)));

    let state = AppState {
        store: store.clone(),
        embedder: embedder.clone(),
        bearer: Arc::new(cfg.bearer.clone()),
        embed_model: Arc::new(cfg.embed_model.clone()),
        embed_dim: cfg.embed_dim,
        last_reindex_ms: last_reindex_ms.clone(),
        display_scoring: crate::search::DisplayScoring {
            k: cfg.display_k,
            w_vec: cfg.weight_vec,
            w_bm25: cfg.weight_bm25,
        },
    };

    let (dirty_tx, dirty_rx) = mpsc::unbounded_channel::<std::path::PathBuf>();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let idx_ctx = IndexerCtx {
        store: store.clone(),
        embedder: embedder.clone(),
        vault_dir: cfg.vault_dir.clone(),
        embed_model: cfg.embed_model.clone(),
        embed_dim: cfg.embed_dim,
        last_reindex_ms: last_reindex_ms.clone(),
    };

    // Spawn indexer first so it's draining the channel before we push into it.
    let idx_handle = tokio::spawn({
        let ctx = idx_ctx.clone();
        async move { indexer::run(ctx, dirty_rx).await }
    });

    // Spawn watcher.
    let watch_handle = tokio::spawn({
        let tx = dirty_tx.clone();
        let vault = cfg.vault_dir.clone();
        let debounce = cfg.debounce;
        let cancel = shutdown_rx.clone();
        async move {
            if let Err(e) = watcher::run(vault, tx, debounce, cancel).await {
                warn!(error = %e, "watcher exited with error");
            }
        }
    });

    // Initial scan pushes dirty paths into the same channel.
    let scan_tx = dirty_tx.clone();
    let scan_ctx = idx_ctx.clone();
    tokio::spawn(async move {
        if let Err(e) = indexer::initial_scan(&scan_ctx, &scan_tx).await {
            warn!(error = %e, "initial scan failed");
        }
    });
    // The server-side sender is cloned into watcher + initial_scan; drop our
    // original so the indexer task terminates once both senders are done
    // *and* they've been dropped.
    drop(dirty_tx);

    // Bind the listener + serve.
    let addr = SocketAddr::from_str(&cfg.listen)
        .with_context(|| format!("parse DOCINDEX_LISTEN {:?}", cfg.listen))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let bound = listener.local_addr().context("local_addr")?;
    info!(
        listen = %bound,
        vault = %cfg.vault_dir.display(),
        db = %cfg.db_path.display(),
        embed_backend = %cfg.embed_provider,
        "docindex-server listening"
    );

    let router = api::build_router(state);
    let mut shutdown_listener = shutdown_rx.clone();
    let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
        wait_for_shutdown().await;
        let _ = shutdown_tx.send(true);
        // Give background tasks a beat to notice.
        let _ = shutdown_listener.changed().await;
    });

    let res = serve.await.context("axum serve");

    // Best-effort wait for background tasks.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let _ = watch_handle.await;
        let _ = idx_handle.await;
    })
    .await;
    res
}

/// Open the store, resolving the embedding fingerprint (provider/model/dim)
/// *before* the dim gets baked into `chunks_vec`'s DDL.
///
/// - No prior fingerprint (fresh DB, or pre-fingerprint upgrade): open
///   normally and adopt the current config as the fingerprint.
/// - Fingerprint matches: open normally.
/// - Fingerprint mismatches and `--reembed` is not set: refuse with a
///   message naming every changed field and both values.
/// - Fingerprint mismatches and `--reembed` is set: open in reembed mode
///   (skips the low-level dim refusal) and wipe + rebuild at the new dim.
///
/// The routing decision (normal open vs reembed open) and the mismatch
/// message are both derived from `FingerprintOutcome::from_peek`, which
/// applies the same comparison semantics as `Store::check_fingerprint`,
/// so there is a single implementation of the equality rule.
fn open_store_with_fingerprint_check(cfg: &Config) -> Result<Store> {
    let provider = cfg.embed_provider.as_str();
    let model = &cfg.embed_model;
    let dim = cfg.embed_dim;

    // Peek the stored fingerprint before the dim gets baked into chunks_vec's
    // DDL — a mismatched dim passed to Store::open would be rejected by the
    // low-level dim guard before we could surface the full fingerprint message.
    let stored = Store::peek_fingerprint(&cfg.db_path).context("peek embedding fingerprint")?;

    // Derive the routing decision from FingerprintOutcome so the comparison
    // semantics (Fresh / Match / Mismatch) are not duplicated inline.
    let peek_outcome = crate::store::FingerprintOutcome::from_peek(stored, provider, model, dim);
    let store = match peek_outcome {
        crate::store::FingerprintOutcome::Mismatch(_) => {
            // Open without the dim guard so check_fingerprint can read meta
            // before any reembed wipe.
            Store::open_for_reembed(&cfg.db_path, dim).context("open store")?
        }
        _ => Store::open(&cfg.db_path, dim).context("open store")?,
    };

    // check_fingerprint reads the live meta rows and produces the
    // authoritative outcome, including the canonical mismatch message.
    match store
        .check_fingerprint(provider, model, dim)
        .context("check embedding fingerprint")?
    {
        crate::store::FingerprintOutcome::Fresh => {
            store
                .set_fingerprint(provider, model, dim)
                .context("set embedding fingerprint")?;
            Ok(store)
        }
        crate::store::FingerprintOutcome::Match => Ok(store),
        crate::store::FingerprintOutcome::Mismatch(msg) if cfg.reembed => {
            warn!(reason = %msg, "fingerprint mismatch; --reembed set, wiping and rebuilding");
            store
                .wipe_and_rebuild(dim, provider, model)
                .context("wipe and rebuild index for --reembed")?;
            Ok(store)
        }
        crate::store::FingerprintOutcome::Mismatch(msg) => Err(anyhow!("{msg}")),
    }
}

fn build_embedder(cfg: &Config) -> Result<AnyEmbedder> {
    match cfg.embed_provider {
        crate::embed::registry::EmbedProvider::Gemini => {
            let mut g = Gemini::new(
                cfg.embed_api_key.clone(),
                cfg.embed_model.clone(),
                cfg.embed_dim,
                cfg.http_timeout,
            )
            .map_err(|e| anyhow!("build gemini client: {e}"))?;
            if let Some(base_url) = &cfg.embed_base_url {
                g.base_url = base_url.clone();
            }
            Ok(AnyEmbedder::Gemini(Arc::new(g)))
        }
        crate::embed::registry::EmbedProvider::Voyage => {
            let mut v = Voyage::new(
                cfg.embed_api_key.clone(),
                cfg.embed_model.clone(),
                cfg.embed_dim,
                cfg.http_timeout,
            )
            .map_err(|e| anyhow!("build voyage client: {e}"))?;
            if let Some(base_url) = &cfg.embed_base_url {
                v.base_url = base_url.clone();
            }
            Ok(AnyEmbedder::Voyage(Arc::new(v)))
        }
        crate::embed::registry::EmbedProvider::Fake => {
            Ok(AnyEmbedder::Fake(Arc::new(Fake::new(cfg.embed_dim))))
        }
    }
}

async fn wait_for_shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "failed to install SIGTERM handler; using ctrl_c only");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => info!("received SIGINT"),
            _ = term.recv() => info!("received SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        info!("received ctrl-c");
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[allow(dead_code)]
fn _type_assertions() {
    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}
    is_send::<AppState>();
    is_sync::<AppState>();
}

// Silence unused `Ordering` when the cfg paths don't use it.
#[allow(dead_code)]
const _ORDERING: Ordering = Ordering::Relaxed;

#[cfg(test)]
mod tests {
    use crate::store::{FingerprintOutcome, Store};
    use tempfile::TempDir;

    /// A first-boot open (fresh DB) must write the fingerprint so that a
    /// second open detects a Match rather than Fresh. If the write were
    /// dropped, the second open would also return Fresh and never catch a
    /// provider/model/dim change.
    #[test]
    fn fresh_db_boot_writes_fingerprint() {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("x.db");

        // Simulate first boot: no stored fingerprint.
        {
            let store = Store::open(&db, 8).expect("open");
            let outcome = store.check_fingerprint("fake", "fake", 8).expect("check");
            assert_eq!(outcome, FingerprintOutcome::Fresh, "new DB must be Fresh");
            store
                .set_fingerprint("fake", "fake", 8)
                .expect("set fingerprint");
        }

        // Second open: fingerprint must now Match, not Fresh.
        let store2 = Store::open(&db, 8).expect("reopen");
        let outcome2 = store2
            .check_fingerprint("fake", "fake", 8)
            .expect("check again");
        assert_eq!(
            outcome2,
            FingerprintOutcome::Match,
            "fingerprint written on first boot must be detected as Match on second open"
        );
    }
}
