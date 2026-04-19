//! Initial full-tree scan of a markdown vault.
//!
//! Emits a deterministic, path-sorted slice of `FileState` entries that
//! downstream code diffs against stored content hashes to decide which
//! files need re-chunking and re-embedding.

use std::{
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use sha2::{Digest, Sha256};
use thiserror::Error;
use walkdir::WalkDir;

/// One markdown file's snapshot at scan time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    /// Path relative to the vault root. Always forward-slash, no leading `/`,
    /// never contains `..`.
    pub rel_path: PathBuf,
    /// File mtime in nanoseconds since the UNIX epoch.
    pub mtime_ns: i64,
    /// Hex sha256 of the file's bytes.
    pub content_hash: String,
}

#[derive(Debug, Error)]
pub enum WalkError {
    #[error("walk: {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("walk: {0}")]
    Other(String),
}

impl WalkError {
    fn io(ctx: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: ctx.into(),
            source,
        }
    }
}

/// Directory names we never descend into, regardless of depth.
const SKIPPED_DIRS: &[&str] = &[".git", ".obsidian", "node_modules"];

/// File extensions (lowercase, no leading dot) the indexer will pick up.
/// Matched case-insensitively. Keep this list small — anything listed here
/// must be sensible to feed straight into the chunker.
pub const INDEXABLE_EXTENSIONS: &[&str] = &["md", "txt"];

/// True iff `ext` matches one of [`INDEXABLE_EXTENSIONS`] case-insensitively.
pub fn is_indexable_extension(ext: &str) -> bool {
    INDEXABLE_EXTENSIONS
        .iter()
        .any(|known| ext.eq_ignore_ascii_case(known))
}

/// Recursively scan `root` and return one `FileState` per indexable file.
///
/// Returned paths are **relative to `root`** (canonicalized). They never
/// start with `/`, never contain `..`, and are sorted lexicographically for
/// determinism.
///
/// Rules:
/// * Only files whose extension is in [`INDEXABLE_EXTENSIONS`]
///   (case-insensitive) are returned.
/// * Directories named in [`SKIPPED_DIRS`] are skipped, as is any directory
///   whose name starts with `.`.
/// * Files whose names start with `.` are skipped.
/// * Files whose canonicalized path escapes the canonicalized `root`
///   (e.g. via a symlink pointing outside the vault) are logged and skipped.
pub fn scan(root: &Path) -> Result<Vec<FileState>, WalkError> {
    let abs_root = root
        .canonicalize()
        .map_err(|e| WalkError::io(format!("canonicalize({})", root.display()), e))?;
    let md = std::fs::metadata(&abs_root)
        .map_err(|e| WalkError::io(format!("stat({})", abs_root.display()), e))?;
    if !md.is_dir() {
        return Err(WalkError::Other(format!(
            "{} is not a directory",
            abs_root.display()
        )));
    }

    let mut out = Vec::new();
    let walker = WalkDir::new(&abs_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let is_root = e.depth() == 0;
            let name = e.file_name().to_string_lossy();
            if e.file_type().is_dir() {
                is_root || (!name.starts_with('.') && !SKIPPED_DIRS.contains(&name.as_ref()))
            } else {
                !name.starts_with('.')
            }
        });

    for entry in walker {
        let entry = entry.map_err(|e| {
            WalkError::Other(format!(
                "walkdir at {}: {e}",
                e.path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            ))
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !is_indexable_extension(ext) {
            continue;
        }
        let rel = match relativize_inside(&abs_root, path) {
            Some(r) => r,
            None => {
                tracing::warn!(
                    path = %path.display(),
                    root = %abs_root.display(),
                    "skipping file whose canonical path escapes the vault"
                );
                continue;
            }
        };
        out.push(hash_file(path, rel)?);
    }

    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}

/// Canonicalize `candidate`, ensure it lives under `canonical_root`, and
/// return the path relative to `canonical_root`. Returns `None` on any
/// canonicalize error or if the canonical path escapes the root.
fn relativize_inside(canonical_root: &Path, candidate: &Path) -> Option<PathBuf> {
    let canon = candidate.canonicalize().ok()?;
    let rel = canon.strip_prefix(canonical_root).ok()?;
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

fn hash_file(path: &Path, rel_path: PathBuf) -> Result<FileState, WalkError> {
    let md = std::fs::metadata(path)
        .map_err(|e| WalkError::io(format!("metadata({})", path.display()), e))?;
    let mtime_ns = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    let bytes =
        std::fs::read(path).map_err(|e| WalkError::io(format!("read({})", path.display()), e))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    Ok(FileState {
        rel_path,
        mtime_ns,
        content_hash: hex::encode(digest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &str) {
        let full = root.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(full, content).unwrap();
    }

    fn sha(s: &str) -> String {
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        hex::encode(h.finalize())
    }

    #[test]
    fn filters_and_sorts() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "b.md", "beta");
        write(root, "a.md", "alpha");
        write(root, "notes/c.md", "gamma");
        write(root, "image.png", "binary");
        write(root, "draft.rtf", "nope");
        write(root, ".secret.md", "nope");
        write(root, ".git/config.md", "nope");
        write(root, ".obsidian/plugins.md", "nope");
        write(root, "node_modules/lib.md", "nope");

        let got = scan(root).unwrap();
        assert_eq!(got.len(), 3, "{got:?}");
        assert!(got.windows(2).all(|w| w[0].rel_path <= w[1].rel_path));
        let rels: Vec<_> = got
            .iter()
            .map(|fs| fs.rel_path.to_string_lossy().into_owned())
            .collect();
        assert_eq!(rels, vec!["a.md", "b.md", "notes/c.md"]);
        for fs in &got {
            assert!(
                !fs.rel_path.is_absolute(),
                "rel_path must not be absolute: {}",
                fs.rel_path.display()
            );
        }
        let by_rel: std::collections::HashMap<_, _> = got
            .iter()
            .map(|fs| (fs.rel_path.to_string_lossy().into_owned(), fs))
            .collect();
        assert_eq!(by_rel["a.md"].content_hash, sha("alpha"));
        assert_eq!(by_rel["b.md"].content_hash, sha("beta"));
        assert_eq!(by_rel["notes/c.md"].content_hash, sha("gamma"));
        for fs in &got {
            assert!(fs.mtime_ns != 0, "mtime zero for {}", fs.rel_path.display());
        }
    }

    #[test]
    fn mixes_md_and_txt_sorted() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "readme.md", "md body");
        write(root, "notes.txt", "txt body");
        write(root, "a.TXT", "upper txt");
        write(root, "draft.rtf", "skip");
        write(root, "image.png", "skip");

        let got = scan(root).unwrap();
        let rels: Vec<_> = got
            .iter()
            .map(|fs| fs.rel_path.to_string_lossy().into_owned())
            .collect();
        // Path-sorted; .rtf and .png rejected; case-insensitive .TXT kept.
        assert_eq!(rels, vec!["a.TXT", "notes.txt", "readme.md"]);
    }

    #[test]
    fn non_directory_errors() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("not-a-dir");
        fs::write(&file, "x").unwrap();
        assert!(scan(&file).is_err());
    }

    #[test]
    fn missing_root_errors() {
        let dir = TempDir::new().unwrap();
        assert!(scan(&dir.path().join("nope")).is_err());
    }

    #[test]
    fn case_insensitive_extension() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "UPPER.MD", "x");
        let got = scan(dir.path()).unwrap();
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn is_indexable_extension_matches() {
        assert!(is_indexable_extension("md"));
        assert!(is_indexable_extension("MD"));
        assert!(is_indexable_extension("txt"));
        assert!(is_indexable_extension("TXT"));
        assert!(!is_indexable_extension("rtf"));
        assert!(!is_indexable_extension(""));
    }
}
