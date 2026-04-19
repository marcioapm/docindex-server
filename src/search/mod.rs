//! Hybrid BM25 + semantic search with Reciprocal Rank Fusion.
//!
//! The full pipeline:
//! 1. Embed the query with `RETRIEVAL_QUERY`.
//! 2. Pull top-`CAND` chunks by cosine distance from `chunks_vec`.
//! 3. Pull top-`CAND` chunks by BM25 from `chunks_fts`.
//! 4. Fuse the two ranked lists with RRF (k=60).
//! 5. Hydrate hits from `chunks` and return the top-`limit`.
//!
//! All SQL runs inside `spawn_blocking` — `rusqlite` is sync and we do not
//! want to block the async runtime's worker threads.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use thiserror::Error;

use crate::{
    embed::{AnyEmbedder, EmbedError},
    store::{HitRow, Store, StoreError},
};

/// Candidate pool size for each ranker before fusion.
pub const CANDIDATE_K: usize = 30;
/// RRF smoothing constant. 60 is the value from the original Cormack paper.
pub const RRF_K: f64 = 60.0;
/// Maximum snippet length in the hit payload.
pub const SNIPPET_MAX: usize = 240;
/// Hard ceiling on `limit` — keeps handlers from accidentally blowing up.
pub const LIMIT_MAX: usize = 50;

/// Default smoothing constant for the *display* normalization. Smaller than
/// `RRF_K` on purpose: ranking wants stability in the long tail, but display
/// wants the rank-1 doc to score ~1.0 and rank-10ish to score ~0.55 so a
/// single threshold is meaningful across queries.
pub const DEFAULT_DISPLAY_K: f64 = 10.0;
/// Default weight for the semantic branch in `score_normalized`.
pub const DEFAULT_WEIGHT_VEC: f64 = 0.55;
/// Default weight for the BM25 branch in `score_normalized`.
pub const DEFAULT_WEIGHT_BM25: f64 = 0.45;

/// Parameters for the query-independent display score.
///
/// Ranking still uses RRF with `k=60`; this struct only controls the 0..1
/// `score_normalized` field attached to each hit (used by the plugin for
/// threshold filtering + percentage display).
#[derive(Debug, Clone, Copy)]
pub struct DisplayScoring {
    pub k: f64,
    pub w_vec: f64,
    pub w_bm25: f64,
}

impl Default for DisplayScoring {
    fn default() -> Self {
        Self {
            k: DEFAULT_DISPLAY_K,
            w_vec: DEFAULT_WEIGHT_VEC,
            w_bm25: DEFAULT_WEIGHT_BM25,
        }
    }
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("search: {0}")]
    Msg(String),
    #[error("search: store: {0}")]
    Store(#[from] StoreError),
    #[error("search: embed: {0}")]
    Embed(#[from] EmbedError),
    #[error("search: join: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("search: dim mismatch: got {got}, want {want}")]
    DimMismatch { got: usize, want: usize },
    #[error("search: path not indexed: {0}")]
    PathNotIndexed(String),
}

/// Single search hit returned in API responses.
///
/// `score` is kept for back-compat; it is identical to `score_rrf` (the RRF
/// fusion score used for ranking). `score_normalized` is the 0..1
/// query-independent display score derived from per-branch ranks via
/// [`normalize_score`].
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Hit {
    pub path: String,
    pub title: String,
    pub heading_path: String,
    pub snippet: String,
    pub score: f64,
    pub score_rrf: f64,
    pub score_normalized: f64,
    pub chunk_id: i64,
}

/// Clamp a user-supplied limit into the safe [1, LIMIT_MAX] band.
pub fn clamp_limit(n: usize) -> usize {
    if n == 0 { 10 } else { n.min(LIMIT_MAX) }
}

/// Run a query against the index.
pub async fn search(
    store: Arc<Mutex<Store>>,
    embedder: &AnyEmbedder,
    embed_dim: usize,
    query: &str,
    limit: usize,
    display: DisplayScoring,
) -> Result<Vec<Hit>, SearchError> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let q_vec = embedder.embed_query(query).await?;
    if q_vec.len() != embed_dim {
        return Err(SearchError::DimMismatch {
            got: q_vec.len(),
            want: embed_dim,
        });
    }
    let fts_query = fts_query_from_user(query);
    let (vec_hits, fts_hits) = run_candidate_queries(store.clone(), q_vec, fts_query).await?;
    let fused = fuse_rrf_ranked(&rank_ids(&vec_hits), &rank_ids(&fts_hits), RRF_K);
    hydrate(store, &fused, clamp_limit(limit), None, display).await
}

/// Find chunks similar to the stored content at `path`.
///
/// Uses the average of the path's stored chunk vectors as a pseudo-query
/// vector for the semantic side, and the concatenated chunk content
/// (FTS-escaped) as a bag-of-words query for the lexical side.
pub async fn similar(
    store: Arc<Mutex<Store>>,
    embed_dim: usize,
    path: &str,
    limit: usize,
    display: DisplayScoring,
) -> Result<Vec<Hit>, SearchError> {
    let path_owned = path.to_string();
    let (q_vec, bag) = {
        let store_c = store.clone();
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, String), SearchError> {
            let guard = store_c
                .lock()
                .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
            let chunks = guard.chunks_for_path(&path_owned)?;
            if chunks.is_empty() {
                return Err(SearchError::PathNotIndexed(path_owned));
            }
            let ids: Vec<i64> = chunks.iter().map(|(id, _)| *id).collect();
            let vectors = guard.vectors_for_chunks(&ids)?;
            let mut avg = vec![0f64; embed_dim];
            let mut count = 0usize;
            for (_, v) in &vectors {
                if v.len() != embed_dim {
                    continue;
                }
                for (i, x) in v.iter().enumerate() {
                    avg[i] += f64::from(*x);
                }
                count += 1;
            }
            if count == 0 {
                return Err(SearchError::Msg(format!(
                    "no stored vectors for path {:?}",
                    chunks[0].0
                )));
            }
            for x in avg.iter_mut() {
                *x /= count as f64;
            }
            let norm: f64 = avg.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in avg.iter_mut() {
                    *x /= norm;
                }
            }
            let q: Vec<f32> = avg.iter().map(|x| *x as f32).collect();
            let mut bag = String::new();
            for (_, c) in chunks.iter().take(4) {
                bag.push_str(c);
                bag.push(' ');
            }
            Ok((q, bag))
        })
        .await??
    };

    let fts_query = fts_query_from_user(&bag);
    let (vec_hits, fts_hits) = run_candidate_queries(store.clone(), q_vec, fts_query).await?;
    let fused = fuse_rrf_ranked(&rank_ids(&vec_hits), &rank_ids(&fts_hits), RRF_K);
    hydrate(
        store,
        &fused,
        clamp_limit(limit),
        Some(path.to_string()),
        display,
    )
    .await
}

/// Reciprocal Rank Fusion over two ranked lists.
///
/// Each list is assumed to be in rank order (best first, rank 1). The score
/// for a document `d` is the sum over lists it appears in of `1/(k + rank)`.
/// The return value is sorted by descending score. Ties are broken by
/// id ascending for determinism.
pub fn fuse_rrf(a: &[i64], b: &[i64], k: f64) -> Vec<(i64, f64)> {
    fuse_rrf_ranked(a, b, k)
        .into_iter()
        .map(|f| (f.id, f.score_rrf))
        .collect()
}

/// Fusion result that also carries the per-branch rank of each doc.
///
/// `v_rank` / `b_rank` are 1-based and `None` when the doc was missing from
/// that branch. These ranks feed into [`normalize_score`] for the
/// query-independent display score.
#[derive(Debug, Clone, Copy)]
pub struct FusedHit {
    pub id: i64,
    pub score_rrf: f64,
    pub v_rank: Option<usize>,
    pub b_rank: Option<usize>,
}

/// Rank-preserving variant of [`fuse_rrf`].
///
/// Same ordering and scores as `fuse_rrf`, but each entry also carries the
/// 1-based rank it held in each branch (or `None` if absent). Callers that
/// only want `(id, score)` can use the thin `fuse_rrf` wrapper.
pub fn fuse_rrf_ranked(a: &[i64], b: &[i64], k: f64) -> Vec<FusedHit> {
    use std::collections::HashMap;
    let mut by_id: HashMap<i64, FusedHit> = HashMap::new();
    for (i, id) in a.iter().enumerate() {
        let rank = i + 1;
        let entry = by_id.entry(*id).or_insert(FusedHit {
            id: *id,
            score_rrf: 0.0,
            v_rank: None,
            b_rank: None,
        });
        entry.score_rrf += 1.0 / (k + rank as f64);
        entry.v_rank.get_or_insert(rank);
    }
    for (i, id) in b.iter().enumerate() {
        let rank = i + 1;
        let entry = by_id.entry(*id).or_insert(FusedHit {
            id: *id,
            score_rrf: 0.0,
            v_rank: None,
            b_rank: None,
        });
        entry.score_rrf += 1.0 / (k + rank as f64);
        entry.b_rank.get_or_insert(rank);
    }
    let mut out: Vec<FusedHit> = by_id.into_values().collect();
    out.sort_by(|x, y| {
        y.score_rrf
            .partial_cmp(&x.score_rrf)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.id.cmp(&y.id))
    });
    out
}

/// Per-branch normalized contribution: rank-1 → 1.0, then monotonically
/// decaying in `rank`. `None` (absent from this branch) → 0.0.
fn branch_norm(rank: Option<usize>, k: f64) -> f64 {
    match rank {
        Some(r) => (k + 1.0) / (k + r as f64),
        None => 0.0,
    }
}

/// Query-independent display score in `[0, 1]`.
///
/// `v_rank` / `b_rank` are the 1-based ranks in the vec + BM25 candidate
/// lists (or `None` if the doc didn't make that list). `k` is the display
/// smoothing constant (NOT the RRF constant — defaults to
/// [`DEFAULT_DISPLAY_K`] = 10). `w_vec + w_bm25` should sum to 1.0.
pub fn normalize_score(
    v_rank: Option<usize>,
    b_rank: Option<usize>,
    k: f64,
    w_vec: f64,
    w_bm25: f64,
) -> f64 {
    w_vec * branch_norm(v_rank, k) + w_bm25 * branch_norm(b_rank, k)
}

fn rank_ids<T>(hits: &[(i64, T)]) -> Vec<i64> {
    hits.iter().map(|(id, _)| *id).collect()
}

async fn run_candidate_queries(
    store: Arc<Mutex<Store>>,
    q_vec: Vec<f32>,
    fts_query: String,
) -> Result<(Vec<(i64, f32)>, Vec<(i64, f64)>), SearchError> {
    let store_vec = store.clone();
    let store_fts = store.clone();

    let vec_task = tokio::task::spawn_blocking(move || -> Result<_, SearchError> {
        let guard = store_vec
            .lock()
            .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
        Ok(guard.search_vec(&q_vec, CANDIDATE_K)?)
    });
    let fts_task = tokio::task::spawn_blocking(move || -> Result<_, SearchError> {
        let guard = store_fts
            .lock()
            .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
        if fts_query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // FTS MATCH will error on syntactically invalid queries — be
        // forgiving and fall back to empty so the semantic side still runs.
        match guard.search_fts(&fts_query, CANDIDATE_K) {
            Ok(v) => Ok(v),
            Err(StoreError::Sqlite(e)) => {
                tracing::debug!(error = %e, query = %fts_query, "fts query failed; using empty candidate list");
                Ok(Vec::new())
            }
            Err(e) => Err(SearchError::Store(e)),
        }
    });
    let (vec_res, fts_res) = tokio::join!(vec_task, fts_task);
    Ok((vec_res??, fts_res??))
}

async fn hydrate(
    store: Arc<Mutex<Store>>,
    fused: &[FusedHit],
    limit: usize,
    exclude_path: Option<String>,
    display: DisplayScoring,
) -> Result<Vec<Hit>, SearchError> {
    let fused = fused.to_vec();
    tokio::task::spawn_blocking(move || -> Result<Vec<Hit>, SearchError> {
        let guard = store
            .lock()
            .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
        let mut out = Vec::with_capacity(limit);
        for f in fused {
            if out.len() >= limit {
                break;
            }
            let Some(row) = guard.chunk_for_hit(f.id)? else {
                continue;
            };
            if let Some(p) = &exclude_path
                && &row.path == p
            {
                continue;
            }
            let norm =
                normalize_score(f.v_rank, f.b_rank, display.k, display.w_vec, display.w_bm25);
            out.push(to_hit(&row, f.score_rrf, norm));
        }
        Ok(out)
    })
    .await?
}

fn to_hit(row: &HitRow, score_rrf: f64, score_normalized: f64) -> Hit {
    let title = std::path::Path::new(&row.path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Hit {
        path: row.path.clone(),
        title,
        heading_path: row.heading_path.clone(),
        snippet: make_snippet(&row.content, SNIPPET_MAX),
        score: score_rrf,
        score_rrf,
        score_normalized,
        chunk_id: row.id,
    }
}

fn make_snippet(content: &str, max: usize) -> String {
    // Strip leading markdown heading line and collapse whitespace; the
    // heading is already returned in `heading_path`, so repeating it in the
    // snippet is noise.
    let body = content
        .strip_prefix('#')
        .map(|rest| rest.split_once('\n').map(|(_, tail)| tail).unwrap_or(rest))
        .unwrap_or(content);
    let body_chars = body.chars().count();
    let mut flat = String::with_capacity(max.min(body.len()));
    let mut last_space = true;
    let mut truncated = false;
    let mut written_chars = 0usize;
    let mut consumed = 0usize;
    for c in body.chars() {
        consumed += 1;
        if c.is_whitespace() {
            if !last_space {
                flat.push(' ');
                last_space = true;
                written_chars += 1;
            }
        } else {
            flat.push(c);
            last_space = false;
            written_chars += 1;
        }
        if written_chars >= max {
            truncated = consumed < body_chars;
            break;
        }
    }
    let trimmed = flat.trim().to_string();
    if truncated {
        let cap = max.saturating_sub(3);
        let mut s: String = trimmed.chars().take(cap).collect();
        s.push_str("...");
        s
    } else {
        trimmed
    }
}

/// Turn a user query string into a safe FTS5 MATCH query.
///
/// FTS5 MATCH syntax is picky about punctuation (parens, colons, quotes,
/// backslashes are all operators). We tokenize on Unicode alphanumerics +
/// underscore, wrap each token in double quotes, and AND them implicitly.
fn fts_query_from_user(q: &str) -> String {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in q.chars() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            cur.push(c);
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
        .into_iter()
        .filter(|t| t.len() > 1)
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_empty() {
        assert!(fuse_rrf(&[], &[], RRF_K).is_empty());
    }

    #[test]
    fn rrf_single_list() {
        let fused = fuse_rrf(&[1, 2, 3], &[], RRF_K);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].0, 1);
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_overlap_beats_disjoint() {
        // 7 appears in both -> bigger score than anything that appears once.
        let a = vec![7, 1, 2];
        let b = vec![7, 3, 4];
        let fused = fuse_rrf(&a, &b, 60.0);
        assert_eq!(fused[0].0, 7);
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_identical_lists() {
        let a = vec![1, 2, 3];
        let fused = fuse_rrf(&a, &a, 60.0);
        assert_eq!(fused[0].0, 1);
        assert_eq!(fused[1].0, 2);
        assert_eq!(fused[2].0, 3);
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_disjoint_merges() {
        let a = vec![1, 2];
        let b = vec![3, 4];
        let fused = fuse_rrf(&a, &b, 60.0);
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn fts_query_tokenizes_and_quotes() {
        assert_eq!(fts_query_from_user("hello world"), "\"hello\" \"world\"");
        assert_eq!(
            fts_query_from_user("rust-lang (sqlite)"),
            "\"rust-lang\" \"sqlite\""
        );
        assert_eq!(fts_query_from_user(""), "");
        assert_eq!(fts_query_from_user("a b c"), "");
        // Single-char tokens dropped.
    }

    #[test]
    fn clamp_limit_bounds() {
        assert_eq!(clamp_limit(0), 10);
        assert_eq!(clamp_limit(1), 1);
        assert_eq!(clamp_limit(100), LIMIT_MAX);
        assert_eq!(clamp_limit(10), 10);
    }

    #[test]
    fn snippet_strips_heading_line() {
        let s = make_snippet("# Title\n\nbody text here", 100);
        assert_eq!(s, "body text here");
    }

    #[test]
    fn snippet_truncates() {
        // Long content without word breaks at max boundary forces ellipsis.
        let s = make_snippet(&"x".repeat(500), 20);
        assert!(s.ends_with("..."), "got {s:?}");
        assert!(s.chars().count() <= 20);
    }

    #[test]
    fn snippet_bounded() {
        // Spec: hit snippet stays under the declared max even when words
        // happen to align at the boundary.
        let s = make_snippet(&"word ".repeat(100), 20);
        assert!(s.chars().count() <= 20, "got {s:?}");
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn normalize_score_rank1_both_is_one() {
        let s = normalize_score(Some(1), Some(1), 10.0, 0.55, 0.45);
        assert!(approx(s, 1.0), "got {s}");
    }

    #[test]
    fn normalize_score_only_vec_is_w_vec() {
        let s = normalize_score(Some(1), None, 10.0, 0.55, 0.45);
        assert!(approx(s, 0.55), "got {s}");
    }

    #[test]
    fn normalize_score_only_bm25_is_w_bm25() {
        let s = normalize_score(None, Some(1), 10.0, 0.55, 0.45);
        assert!(approx(s, 0.45), "got {s}");
    }

    #[test]
    fn normalize_score_rank10_both_mid() {
        // (10+1)/(10+10) = 0.55 per branch → total 0.55.
        let s = normalize_score(Some(10), Some(10), 10.0, 0.55, 0.45);
        assert!(approx(s, 0.55), "got {s}");
    }

    #[test]
    fn normalize_score_absent_is_zero() {
        let s = normalize_score(None, None, 10.0, 0.55, 0.45);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn normalize_score_rank20_both_roughly_0_367() {
        let s = normalize_score(Some(20), Some(20), 10.0, 0.55, 0.45);
        assert!((s - 0.3667).abs() < 0.01, "got {s}");
    }

    #[test]
    fn normalize_score_monotonic_in_rank() {
        let s1 = normalize_score(Some(1), Some(1), 10.0, 0.55, 0.45);
        let s5 = normalize_score(Some(5), Some(5), 10.0, 0.55, 0.45);
        let s20 = normalize_score(Some(20), Some(20), 10.0, 0.55, 0.45);
        assert!(s1 > s5 && s5 > s20, "{s1} {s5} {s20}");
    }

    #[test]
    fn fuse_rrf_ranked_tracks_ranks() {
        let vec_list = vec![10, 20, 30];
        let fts_list = vec![30, 10, 40];
        let fused = fuse_rrf_ranked(&vec_list, &fts_list, 60.0);
        let by_id: std::collections::HashMap<i64, FusedHit> =
            fused.iter().map(|f| (f.id, *f)).collect();
        assert_eq!(by_id[&10].v_rank, Some(1));
        assert_eq!(by_id[&10].b_rank, Some(2));
        assert_eq!(by_id[&20].v_rank, Some(2));
        assert_eq!(by_id[&20].b_rank, None);
        assert_eq!(by_id[&40].v_rank, None);
        assert_eq!(by_id[&40].b_rank, Some(3));
    }

    #[test]
    fn display_scoring_default_matches_constants() {
        let d = DisplayScoring::default();
        assert_eq!(d.k, DEFAULT_DISPLAY_K);
        assert_eq!(d.w_vec, DEFAULT_WEIGHT_VEC);
        assert_eq!(d.w_bm25, DEFAULT_WEIGHT_BM25);
        assert!(approx(d.w_vec + d.w_bm25, 1.0));
    }
}
