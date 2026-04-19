//! Filesystem watcher: translates `notify` events into dirty-path messages
//! for the indexer.
//!
//! Debouncing is done in-house with a HashMap<PathBuf, Instant>: when an
//! event comes in we stamp the path with `now`, and a 500ms ticker sweeps
//! the map emitting every path whose stamp is older than `debounce`.
//! Entries younger than `debounce` stay for the next tick. This gives
//! coalesced-but-not-delayed behavior: a burst of writes to the same file
//! (an editor save + lockfile dance) collapses to a single emit.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use notify::{EventKind, RecursiveMode, Watcher};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::walk::is_indexable_extension;

#[derive(Debug, Error)]
pub enum WatchError {
    #[error("watch: {0}")]
    Msg(String),
    #[error("watch: notify: {0}")]
    Notify(#[from] notify::Error),
}

const SKIPPED_DIR_SEGMENTS: &[&str] = &[".git", ".obsidian", "node_modules"];
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Watch `vault_dir` recursively and push any indexable path that changes
/// into `tx`. Returns once `cancel` resolves.
///
/// The task owns the watcher for its lifetime — dropping the returned
/// future releases the underlying OS resources.
pub async fn run(
    vault_dir: PathBuf,
    tx: mpsc::UnboundedSender<PathBuf>,
    debounce: Duration,
    mut cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<(), WatchError> {
    let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<notify::Event>();

    // The notify callback runs on a notify-owned thread; it must not block.
    let notify_tx = raw_tx.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        match res {
            Ok(ev) => {
                if notify_tx.send(ev).is_err() {
                    // receiver dropped; shutting down.
                }
            }
            Err(e) => error!(error = %e, "notify error"),
        }
    })?;
    watcher.watch(&vault_dir, RecursiveMode::Recursive)?;
    info!(vault = %vault_dir.display(), "watcher started");
    drop(raw_tx);

    let mut pending: HashMap<PathBuf, Instant> = HashMap::new();
    let mut tick = tokio::time::interval(POLL_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            _ = cancel.changed() => {
                info!("watcher cancel signal received; stopping");
                drop(watcher);
                break;
            }
            ev = raw_rx.recv() => {
                match ev {
                    Some(ev) => record(&mut pending, &ev, &vault_dir),
                    None => {
                        warn!("notify sender closed; stopping watcher");
                        break;
                    }
                }
            }
            _ = tick.tick() => {
                flush(&mut pending, &tx, debounce);
            }
        }
    }
    Ok(())
}

fn record(pending: &mut HashMap<PathBuf, Instant>, ev: &notify::Event, root: &Path) {
    match ev.kind {
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {}
        _ => return,
    }
    let now = Instant::now();
    for p in &ev.paths {
        let abs = if p.is_absolute() {
            p.clone()
        } else {
            root.join(p)
        };
        if !is_relevant(&abs) {
            continue;
        }
        pending.insert(abs, now);
    }
}

fn flush(
    pending: &mut HashMap<PathBuf, Instant>,
    tx: &mpsc::UnboundedSender<PathBuf>,
    debounce: Duration,
) {
    let now = Instant::now();
    let mut to_emit = Vec::new();
    pending.retain(|path, stamped_at| {
        if now.duration_since(*stamped_at) >= debounce {
            to_emit.push(path.clone());
            false
        } else {
            true
        }
    });
    for path in to_emit {
        debug!(path = %path.display(), "debounced emit");
        if tx.send(path).is_err() {
            warn!("indexer channel closed; watcher draining remainder");
            pending.clear();
            return;
        }
    }
}

fn is_relevant(path: &Path) -> bool {
    // Filter: indexable extensions only, skip dot-files and known noisy dirs.
    let ext_ok = path
        .extension()
        .and_then(|s| s.to_str())
        .map(is_indexable_extension)
        .unwrap_or(false);
    if !ext_ok {
        return false;
    }
    for comp in path.components() {
        let s = comp.as_os_str().to_string_lossy();
        if SKIPPED_DIR_SEGMENTS.iter().any(|d| s == *d) {
            return false;
        }
        // Skip hidden files/dirs (but not the path root if absolute).
        if s.starts_with('.') && s != "." && s != ".." && !s.contains(':') {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn relevance_filters() {
        assert!(is_relevant(&PathBuf::from("/vault/note.md")));
        assert!(is_relevant(&PathBuf::from("/vault/sub/note.MD")));
        assert!(is_relevant(&PathBuf::from("/vault/plain.txt")));
        assert!(is_relevant(&PathBuf::from("/vault/sub/plain.TXT")));
        assert!(!is_relevant(&PathBuf::from("/vault/draft.rtf")));
        assert!(!is_relevant(&PathBuf::from("/vault/.hidden.md")));
        assert!(!is_relevant(&PathBuf::from("/vault/.git/a.md")));
        assert!(!is_relevant(&PathBuf::from("/vault/node_modules/a.md")));
        assert!(!is_relevant(&PathBuf::from(
            "/vault/.obsidian/workspace.md"
        )));
    }

    #[test]
    fn flush_respects_debounce() {
        let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
        let mut pending = HashMap::new();
        let now = Instant::now();
        pending.insert(PathBuf::from("/a.md"), now);
        // Not yet elapsed: nothing emitted.
        flush(&mut pending, &tx, Duration::from_secs(5));
        assert!(rx.try_recv().is_err());
        assert_eq!(pending.len(), 1);
        // Force expiry by rewinding the stamp.
        let old = now.checked_sub(Duration::from_secs(10)).unwrap_or(now);
        pending.insert(PathBuf::from("/a.md"), old);
        flush(&mut pending, &tx, Duration::from_secs(5));
        assert_eq!(rx.try_recv().unwrap(), PathBuf::from("/a.md"));
        assert!(pending.is_empty());
    }
}
