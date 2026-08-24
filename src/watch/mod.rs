//! Filesystem watcher: translates `notify` events into dirty-path messages
//! for the indexer.
//!
//! Debouncing is done in-house with a HashMap<PathBuf, Instant>: when an
//! event comes in we stamp the path with `now`, and a 500ms ticker sweeps
//! the map emitting every path whose stamp is older than `debounce`.
//! Entries younger than `debounce` stay for the next tick. This gives
//! coalesced-but-not-delayed behavior: a burst of writes to the same file
//! (an editor save + lockfile dance) collapses to a single emit.
//!
//! Paths pushed into the indexer channel are **vault-relative** — the raw
//! absolute paths from `notify` are stripped against the canonicalized
//! `vault_dir` before being stamped into the pending map.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use notify::{EventKind, RecursiveMode, Watcher};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::media::MediaPolicy;

#[derive(Debug, Error)]
pub enum WatchError {
    #[error("watch: {0}")]
    Msg(String),
    #[error("watch: notify: {0}")]
    Notify(#[from] notify::Error),
}

const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Watch `vault_dir` recursively and push any indexable path that changes
/// into `tx`, **relative to `vault_dir`**. Returns once `cancel` resolves.
///
/// The task owns the watcher for its lifetime — dropping the returned
/// future releases the underlying OS resources.
pub async fn run(
    vault_dir: PathBuf,
    tx: mpsc::UnboundedSender<PathBuf>,
    policy: MediaPolicy,
    debounce: Duration,
    mut cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<(), WatchError> {
    // Canonicalize once so strip_prefix works even when the operator passes a
    // vault dir with a trailing slash or via a symlink.
    let canonical_vault = vault_dir.canonicalize().unwrap_or_else(|e| {
        warn!(
            path = %vault_dir.display(),
            error = %e,
            "could not canonicalize vault_dir; falling back to raw path"
        );
        vault_dir.clone()
    });

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
    watcher.watch(&canonical_vault, RecursiveMode::Recursive)?;
    info!(vault = %canonical_vault.display(), "watcher started");
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
                    Some(ev) => record(&mut pending, &ev, &canonical_vault, &policy),
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

fn record(
    pending: &mut HashMap<PathBuf, Instant>,
    ev: &notify::Event,
    root: &Path,
    policy: &MediaPolicy,
) {
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
        let Some(rel) = strip_vault_prefix(root, &abs) else {
            debug!(
                path = %abs.display(),
                root = %root.display(),
                "watch event path outside vault; skipping"
            );
            continue;
        };
        let relevant = if matches!(ev.kind, EventKind::Remove(_)) {
            policy.allows_remove(&rel)
        } else {
            std::fs::metadata(&abs)
                .ok()
                .filter(|metadata| metadata.is_file())
                .is_some_and(|_| policy.classify_path(&rel).is_some())
        };
        if relevant {
            pending.insert(rel, now);
        }
    }
}

/// Return `abs` relative to `root`, or `None` if it's not under `root` or
/// would require traversing `..`. Falls back to canonicalizing `abs` when
/// the raw prefix check fails (handles symlinks to files inside the vault).
fn strip_vault_prefix(root: &Path, abs: &Path) -> Option<PathBuf> {
    if let Ok(rel) = abs.strip_prefix(root) {
        return relative_inside(rel);
    }
    let canon = abs.canonicalize().ok()?;
    let rel = canon.strip_prefix(root).ok()?;
    relative_inside(rel)
}

fn relative_inside(rel: &Path) -> Option<PathBuf> {
    if rel.as_os_str().is_empty() {
        return None;
    }
    if rel.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        return None;
    }
    Some(rel.to_path_buf())
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

#[cfg(test)]
fn is_relevant(path: &Path, policy: &MediaPolicy) -> bool {
    policy.classify_path(path).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn relevance_filters() {
        let policy = MediaPolicy::default();
        assert!(is_relevant(&PathBuf::from("note.md"), &policy));
        assert!(is_relevant(&PathBuf::from("sub/note.MD"), &policy));
        assert!(is_relevant(&PathBuf::from("plain.txt"), &policy));
        assert!(is_relevant(&PathBuf::from("sub/plain.TXT"), &policy));
        assert!(!is_relevant(&PathBuf::from("draft.rtf"), &policy));
        assert!(!is_relevant(&PathBuf::from(".hidden.md"), &policy));
        assert!(!is_relevant(&PathBuf::from(".git/a.md"), &policy));
        assert!(!is_relevant(&PathBuf::from("node_modules/a.md"), &policy));
        assert!(!is_relevant(
            &PathBuf::from(".obsidian/workspace.md"),
            &policy
        ));
    }

    #[test]
    fn scanner_and_watcher_share_policy_classification() {
        let policy = MediaPolicy::new(
            true,
            &["Attachments/**".into()],
            &["Attachments/Private/**".into()],
            &[],
            20,
            6,
            150,
        )
        .unwrap();
        for (path, want) in [
            ("note.md", true),
            ("Attachments/image.png", true),
            ("Attachments/Private/image.png", false),
            ("Attachments/paper.pdf", true),
            ("unsupported.bin", false),
            (".hidden/image.png", false),
        ] {
            assert_eq!(is_relevant(Path::new(path), &policy), want, "{path}");
        }
    }

    #[test]
    fn remove_event_for_oversize_media_is_recorded_without_stat() {
        let root = tempfile::TempDir::new().unwrap();
        let policy = MediaPolicy::new(true, &[], &[], &[], 1, 6, 150).unwrap();
        let event = notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::File))
            .add_path(root.path().join("image.png"));
        let mut pending = HashMap::new();
        record(&mut pending, &event, root.path(), &policy);
        assert!(pending.contains_key(&PathBuf::from("image.png")));
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

    #[test]
    fn strip_vault_prefix_strips_root() {
        let root = Path::new("/vault");
        assert_eq!(
            strip_vault_prefix(root, Path::new("/vault/note.md")),
            Some(PathBuf::from("note.md"))
        );
        assert_eq!(
            strip_vault_prefix(root, Path::new("/vault/sub/note.md")),
            Some(PathBuf::from("sub/note.md"))
        );
    }

    #[test]
    fn strip_vault_prefix_rejects_escape() {
        let root = Path::new("/vault");
        // Same prefix, different vault ("/vaults/..." is not under "/vault").
        assert_eq!(
            strip_vault_prefix(root, Path::new("/other/note.md")),
            None,
            "path outside vault must not be stripped"
        );
    }

    #[test]
    fn strip_vault_prefix_rejects_root_itself() {
        let root = Path::new("/vault");
        assert_eq!(
            strip_vault_prefix(root, Path::new("/vault")),
            None,
            "the vault root itself has no relative form"
        );
    }
}
