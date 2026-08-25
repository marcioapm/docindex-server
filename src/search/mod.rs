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

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    embed::{AnyEmbedder, EmbedError},
    media::MediaType,
    store::{HitRow, Store, StoreError},
};

/// Minimum candidate pool size for each ranker before fusion. The actual
/// pool passed to each branch is `CANDIDATE_K.max(clamp_limit(limit))` so a
/// request for more hits than this floor still gets a large enough pool to
/// fill its `limit`.
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
    pub media_lane: MediaLaneScoring,
}

impl Default for DisplayScoring {
    fn default() -> Self {
        Self {
            k: DEFAULT_DISPLAY_K,
            w_vec: DEFAULT_WEIGHT_VEC,
            w_bm25: DEFAULT_WEIGHT_BM25,
            media_lane: MediaLaneScoring::default(),
        }
    }
}

/// Blended-search media admission and distance-derived display score settings.
#[derive(Debug, Clone, Copy)]
pub struct MediaLaneScoring {
    pub enabled: bool,
    pub fraction: f64,
    pub gate_image: f64,
    pub gate_pdf: f64,
    pub display_image_best: f64,
    pub display_image_worst: f64,
    pub display_pdf_best: f64,
    pub display_pdf_worst: f64,
}

impl Default for MediaLaneScoring {
    fn default() -> Self {
        Self {
            enabled: false,
            fraction: 0.25,
            gate_image: 0.40,
            gate_pdf: 0.45,
            display_image_best: 0.25,
            display_image_worst: 0.50,
            display_pdf_best: 0.35,
            display_pdf_worst: 0.60,
        }
    }
}

impl MediaLaneScoring {
    pub fn validate(self) -> Result<(), String> {
        if !self.fraction.is_finite() || !(0.0..=1.0).contains(&self.fraction) {
            return Err(format!(
                "search.media_lane_fraction {}: must be finite and in [0, 1]",
                self.fraction
            ));
        }
        for (name, value) in [
            ("search.media_gate_image", self.gate_image),
            ("search.media_gate_pdf", self.gate_pdf),
            ("search.media_display_image_best", self.display_image_best),
            ("search.media_display_image_worst", self.display_image_worst),
            ("search.media_display_pdf_best", self.display_pdf_best),
            ("search.media_display_pdf_worst", self.display_pdf_worst),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} {value}: must be finite and > 0"));
            }
        }
        if self.display_image_best >= self.display_image_worst {
            return Err(
                "search.media_display_image_best must be < search.media_display_image_worst".into(),
            );
        }
        if self.display_pdf_best >= self.display_pdf_worst {
            return Err(
                "search.media_display_pdf_best must be < search.media_display_pdf_worst".into(),
            );
        }
        Ok(())
    }

    fn gate_for(self, media_type: &str) -> Option<f64> {
        match media_type {
            "image" => Some(self.gate_image),
            "pdf" => Some(self.gate_pdf),
            // Audio and video are uncalibrated until corpus measurements exist.
            "audio" | "video" => Some(self.gate_image),
            _ => None,
        }
    }

    fn display_endpoints_for(self, media_type: &str) -> Option<(f64, f64)> {
        match media_type {
            "image" => Some((self.display_image_best, self.display_image_worst)),
            "pdf" => Some((self.display_pdf_best, self.display_pdf_worst)),
            // Audio and video are uncalibrated until corpus measurements exist.
            "audio" | "video" => Some((self.display_image_best, self.display_image_worst)),
            _ => None,
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hit {
    pub path: String,
    pub title: String,
    pub heading_path: String,
    pub snippet: String,
    pub score: f64,
    pub score_rrf: f64,
    pub score_normalized: f64,
    pub chunk_id: i64,
    #[serde(default)]
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_start: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_end: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_unit: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Clamp a user-supplied limit into the safe [1, LIMIT_MAX] band.
pub fn clamp_limit(n: usize) -> usize {
    if n == 0 { 10 } else { n.min(LIMIT_MAX) }
}

/// Parameters controlling which corpus a search considers.
///
/// The default preserves hybrid text-and-media search. Set `media_only` to
/// search only non-text chunks with the vector ranker; media is intentionally
/// absent from FTS.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub media_only: bool,
    pub media_types: Vec<MediaType>,
}

/// Run a query against the index using the default hybrid corpus.
pub async fn search(
    store: Arc<Mutex<Store>>,
    embedder: &AnyEmbedder,
    embed_dim: usize,
    query: &str,
    limit: usize,
    display: DisplayScoring,
) -> Result<Vec<Hit>, SearchError> {
    search_with_options(
        store,
        embedder,
        embed_dim,
        query,
        limit,
        display,
        SearchOptions::default(),
    )
    .await
}

/// Run a query against the index with corpus-selection options.
pub async fn search_with_options(
    store: Arc<Mutex<Store>>,
    embedder: &AnyEmbedder,
    embed_dim: usize,
    query: &str,
    limit: usize,
    display: DisplayScoring,
    options: SearchOptions,
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

    let clamped_limit = clamp_limit(limit);
    let candidate_k = CANDIDATE_K.max(clamped_limit);

    if options.media_only {
        let media_hits =
            run_media_candidate_query(store.clone(), q_vec, candidate_k, options.media_types)
                .await?;
        let fused = fuse_rrf_ranked(&rank_ids(&media_hits), &[], RRF_K);
        let media_distances = display
            .media_lane
            .enabled
            .then(|| media_distance_map(&media_hits));
        return hydrate(store, &fused, clamped_limit, None, display, media_distances).await;
    }

    let fts_query = fts_query_from_user(query);
    let (vec_hits, fts_hits, media_hits) = if display.media_lane.enabled {
        let (text, media) = tokio::join!(
            run_candidate_queries(store.clone(), q_vec.clone(), fts_query, candidate_k),
            run_media_candidate_query(store.clone(), q_vec, candidate_k, Vec::new())
        );
        let (vec_hits, fts_hits) = text?;
        (vec_hits, fts_hits, Some(media?))
    } else {
        let (vec_hits, fts_hits) =
            run_candidate_queries(store.clone(), q_vec, fts_query, candidate_k).await?;
        (vec_hits, fts_hits, None)
    };
    let fused = fuse_rrf_ranked(&rank_ids(&vec_hits), &rank_ids(&fts_hits), RRF_K);
    let media_distances = media_hits.as_ref().map(|hits| media_distance_map(hits));
    let mut hydrated = hydrate(
        store.clone(),
        &fused,
        clamped_limit,
        None,
        display,
        media_distances.clone(),
    )
    .await?;
    if let Some(media_hits) = media_hits {
        let media_fused = fuse_rrf_ranked(&rank_ids(&media_hits), &[], RRF_K);
        let media_hydrated = hydrate(
            store,
            &media_fused,
            candidate_k,
            None,
            display,
            media_distances.clone(),
        )
        .await?;
        insert_media_lane(
            &mut hydrated,
            media_hydrated,
            clamped_limit,
            display.media_lane,
            media_distances.as_ref(),
        );
    }
    Ok(hydrated)
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
    let source = {
        let store_c = store.clone();
        tokio::task::spawn_blocking(move || -> Result<Option<(Vec<f32>, String)>, SearchError> {
            let guard = store_c
                .lock()
                .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
            let chunks = guard.chunks_for_similar(&path_owned)?;
            if chunks.is_empty() {
                return match guard.get_file_state(&path_owned)? {
                    Some(_) => Ok(None),
                    None => Err(SearchError::PathNotIndexed(path_owned)),
                };
            }
            let ids: Vec<i64> = chunks.iter().map(|chunk| chunk.id).collect();
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
                    chunks[0].id
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
            for chunk in chunks
                .iter()
                .filter(|chunk| chunk.media_type == "text")
                .take(4)
            {
                bag.push_str(&chunk.content);
                bag.push(' ');
            }
            Ok(Some((q, bag)))
        })
        .await??
    };
    let Some((q_vec, bag)) = source else {
        return Ok(Vec::new());
    };

    let clamped_limit = clamp_limit(limit);
    let candidate_k = CANDIDATE_K.max(clamped_limit);
    let fts_query = fts_query_from_user(&bag);
    let (vec_hits, fts_hits) =
        run_candidate_queries(store.clone(), q_vec, fts_query, candidate_k).await?;
    let fused = fuse_rrf_ranked(&rank_ids(&vec_hits), &rank_ids(&fts_hits), RRF_K);
    hydrate(
        store,
        &fused,
        clamped_limit,
        Some(path.to_string()),
        display,
        None,
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

pub fn vector_only_media_score(v_rank: Option<usize>, k: f64) -> f64 {
    branch_norm(v_rank, k)
}

fn rank_ids<T>(hits: &[(i64, T)]) -> Vec<i64> {
    hits.iter().map(|(id, _)| *id).collect()
}

fn media_distance_map(hits: &[(i64, f32)]) -> HashMap<i64, f32> {
    hits.iter().copied().collect()
}

fn media_display_score(distance: f32, media_type: &str, lane: MediaLaneScoring) -> Option<f64> {
    let (best, worst) = lane.display_endpoints_for(media_type)?;
    Some(((worst - f64::from(distance)) / (worst - best)).clamp(0.0, 1.0))
}

fn insert_media_lane(
    results: &mut Vec<Hit>,
    candidates: Vec<Hit>,
    limit: usize,
    lane: MediaLaneScoring,
    distances: Option<&HashMap<i64, f32>>,
) {
    let slots = (limit as f64 * lane.fraction).floor() as usize;
    if slots == 0 {
        return;
    }
    let present: std::collections::HashSet<i64> = results.iter().map(|hit| hit.chunk_id).collect();
    let mut inserted: Vec<Hit> = candidates
        .into_iter()
        .filter(|hit| {
            lane.gate_for(&hit.media_type)
                .zip(
                    distances
                        .and_then(|distances| distances.get(&hit.chunk_id))
                        .copied(),
                )
                .is_some_and(|(gate, distance)| distance <= gate as f32)
                && !present.contains(&hit.chunk_id)
        })
        .take(slots)
        .collect();
    if inserted.is_empty() {
        return;
    }
    let stride = limit.div_ceil(inserted.len());
    for (i, hit) in inserted.drain(..).enumerate() {
        if results.len() >= limit
            && let Some(index) = results
                .iter()
                .rposition(|result| result.media_type == "text")
        {
            results.remove(index);
        }
        results.insert((stride * (i + 1) - 1).min(results.len()), hit);
    }
    results.truncate(limit);
}

async fn run_media_candidate_query(
    store: Arc<Mutex<Store>>,
    q_vec: Vec<f32>,
    candidate_k: usize,
    media_types: Vec<MediaType>,
) -> Result<Vec<(i64, f32)>, SearchError> {
    tokio::task::spawn_blocking(move || -> Result<_, SearchError> {
        let guard = store
            .lock()
            .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
        Ok(guard.search_media_vec(&q_vec, candidate_k, &media_types)?)
    })
    .await?
}

async fn run_candidate_queries(
    store: Arc<Mutex<Store>>,
    q_vec: Vec<f32>,
    fts_query: String,
    candidate_k: usize,
) -> Result<(Vec<(i64, f32)>, Vec<(i64, f64)>), SearchError> {
    let store_vec = store.clone();
    let store_fts = store.clone();

    let vec_task = tokio::task::spawn_blocking(move || -> Result<_, SearchError> {
        let guard = store_vec
            .lock()
            .map_err(|e| SearchError::Msg(format!("store lock: {e}")))?;
        Ok(guard.search_vec(&q_vec, candidate_k)?)
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
        match guard.search_fts(&fts_query, candidate_k) {
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
    media_distances: Option<HashMap<i64, f32>>,
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
            let norm = if row.media_type == "text" {
                normalize_score(f.v_rank, f.b_rank, display.k, display.w_vec, display.w_bm25)
            } else {
                media_distances
                    .as_ref()
                    .and_then(|distances| distances.get(&f.id))
                    .and_then(|distance| {
                        media_display_score(*distance, &row.media_type, display.media_lane)
                    })
                    .unwrap_or_else(|| vector_only_media_score(f.v_rank, display.k))
            };
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
        snippet: media_snippet(row).unwrap_or_else(|| make_snippet(&row.content, SNIPPET_MAX)),
        score: score_rrf,
        score_rrf,
        score_normalized,
        chunk_id: row.id,
        media_type: row.media_type.clone(),
        mime_type: row.mime_type.clone(),
        media_start: row.media_start,
        media_end: row.media_end,
        media_unit: row.media_unit.clone(),
        truncated: row.truncated,
    }
}

fn media_snippet(row: &HitRow) -> Option<String> {
    match row.media_type.as_str() {
        "image" => Some("Image".into()),
        "pdf" => match (row.media_start, row.media_end) {
            (Some(start), Some(end)) if end == start + 1 => Some(format!("PDF page {}", start + 1)),
            (Some(start), Some(end)) if end > start + 1 => {
                Some(format!("PDF pages {}–{}", start + 1, end))
            }
            _ => Some("PDF".into()),
        },
        _ => None,
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
    fn hit_json_shapes_distinguish_text_image_and_pdf() {
        let text = Hit {
            path: "note.md".into(),
            title: "note".into(),
            heading_path: String::new(),
            snippet: "text".into(),
            score: 1.0,
            score_rrf: 1.0,
            score_normalized: 1.0,
            chunk_id: 1,
            media_type: "text".into(),
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        };
        let text_json = serde_json::to_value(text).unwrap();
        assert_eq!(text_json["media_type"], "text");
        assert!(text_json.get("mime_type").is_none());
        assert!(text_json.get("truncated").is_none());
        let image = Hit {
            media_type: "image".into(),
            mime_type: Some("image/png".into()),
            ..Hit {
                path: "image.png".into(),
                title: "image".into(),
                heading_path: String::new(),
                snippet: String::new(),
                score: 1.0,
                score_rrf: 1.0,
                score_normalized: 1.0,
                chunk_id: 2,
                media_type: String::new(),
                mime_type: None,
                media_start: None,
                media_end: None,
                media_unit: None,
                truncated: false,
            }
        };
        assert_eq!(
            serde_json::to_value(image).unwrap()["mime_type"],
            "image/png"
        );
        let pdf = Hit {
            media_type: "pdf".into(),
            mime_type: Some("application/pdf".into()),
            media_start: Some(0),
            media_end: Some(6),
            media_unit: Some("page".into()),
            truncated: true,
            ..Hit {
                path: "paper.pdf".into(),
                title: "paper".into(),
                heading_path: String::new(),
                snippet: String::new(),
                score: 1.0,
                score_rrf: 1.0,
                score_normalized: 1.0,
                chunk_id: 3,
                media_type: String::new(),
                mime_type: None,
                media_start: None,
                media_end: None,
                media_unit: None,
                truncated: false,
            }
        };
        let pdf_json = serde_json::to_value(pdf).unwrap();
        assert_eq!(pdf_json["media_start"], 0);
        assert_eq!(pdf_json["media_end"], 6);
        assert_eq!(pdf_json["media_unit"], "page");
        assert_eq!(pdf_json["truncated"], true);
    }

    #[test]
    fn media_snippets_describe_images_and_pdf_page_ranges() {
        let mut image = test_hit_row("image");
        image.media_type = "image".into();
        assert_eq!(media_snippet(&image).as_deref(), Some("Image"));

        let mut single_page = test_hit_row("pdf");
        single_page.media_type = "pdf".into();
        single_page.media_start = Some(0);
        single_page.media_end = Some(1);
        assert_eq!(media_snippet(&single_page).as_deref(), Some("PDF page 1"));

        let mut pages = test_hit_row("pdf");
        pages.media_type = "pdf".into();
        pages.media_start = Some(2);
        pages.media_end = Some(5);
        assert_eq!(media_snippet(&pages).as_deref(), Some("PDF pages 3–5"));
    }

    fn test_hit_row(media_type: &str) -> HitRow {
        HitRow {
            id: 1,
            path: "item".into(),
            heading: String::new(),
            heading_path: String::new(),
            content: "internal representation".into(),
            media_type: media_type.into(),
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        }
    }

    #[test]
    fn vector_only_media_scores_are_unweighted() {
        assert!(approx(vector_only_media_score(Some(1), 10.0), 1.0));
        assert!(approx(vector_only_media_score(Some(10), 10.0), 0.55));
        assert!(approx(
            normalize_score(Some(1), None, 10.0, 0.55, 0.45),
            0.55
        ));
    }

    #[test]
    fn display_scoring_default_matches_constants() {
        let d = DisplayScoring::default();
        assert_eq!(d.k, DEFAULT_DISPLAY_K);
        assert_eq!(d.w_vec, DEFAULT_WEIGHT_VEC);
        assert_eq!(d.w_bm25, DEFAULT_WEIGHT_BM25);
        assert!(approx(d.w_vec + d.w_bm25, 1.0));
    }

    const CANDIDATE_TEST_DIM: usize = 8;
    const ABOVE_CANDIDATE_K: usize = 40;

    fn candidate_test_store() -> (tempfile::TempDir, Arc<Mutex<Store>>) {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path().join("x.db"), CANDIDATE_TEST_DIM).unwrap();
        (dir, Arc::new(Mutex::new(store)))
    }

    async fn candidate_test_fixture(
        query: &str,
    ) -> (
        tempfile::TempDir,
        Arc<Mutex<Store>>,
        crate::embed::AnyEmbedder,
        Vec<f32>,
        Vec<f32>,
    ) {
        let (dir, store) = candidate_test_store();
        let embedder =
            crate::embed::AnyEmbedder::Fake(Arc::new(crate::embed::Fake::new(CANDIDATE_TEST_DIM)));
        let q_vec = embedder.embed_query(query).await.unwrap();
        let orth = orthogonal_unit(&q_vec);
        (dir, store, embedder, q_vec, orth)
    }

    #[tokio::test]
    async fn media_only_search_above_candidate_k_reaches_requested_limit() {
        let (_dir, store, embedder, q_vec, _) = candidate_test_fixture("photo").await;

        {
            let guard = store.lock().unwrap();
            for i in 0..ABOVE_CANDIDATE_K {
                let chunk = crate::chunk::Chunk {
                    idx: 0,
                    heading: String::new(),
                    heading_path: String::new(),
                    content: format!("media chunk {i}"),
                    content_hash: format!("media-hash-{i}"),
                    tokens: 3,
                    media_type: MediaType::Image,
                    mime_type: Some("image/png".into()),
                    media_start: None,
                    media_end: None,
                    media_unit: None,
                    truncated: false,
                };
                let id = guard
                    .upsert_chunk(&chunk, &format!("img{i}.png"), 1)
                    .unwrap();
                guard.set_vector_for_chunk(id, &q_vec).unwrap();
            }
        }

        let hits = search_with_options(
            store,
            &embedder,
            CANDIDATE_TEST_DIM,
            "photo",
            50,
            DisplayScoring::default(),
            SearchOptions {
                media_only: true,
                media_types: vec![],
            },
        )
        .await
        .unwrap();

        assert_eq!(
            hits.len(),
            ABOVE_CANDIDATE_K,
            "limit=50 with {ABOVE_CANDIDATE_K} eligible media chunks must not be capped at CANDIDATE_K"
        );
    }

    fn orthogonal_unit(q_vec: &[f32]) -> Vec<f32> {
        let mut raw = vec![0f32; q_vec.len()];
        if q_vec[0].abs() > 0.9 {
            raw[1] = 1.0;
        } else {
            raw[0] = 1.0;
        }
        let dot: f32 = raw.iter().zip(q_vec).map(|(a, b)| a * b).sum();
        let mut orth: Vec<f32> = raw.iter().zip(q_vec).map(|(a, b)| a - dot * b).collect();
        let norm: f32 = orth.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in orth.iter_mut() {
            *x /= norm;
        }
        orth
    }

    fn seed_text_chunk(
        guard: &Store,
        q_vec: &[f32],
        orth: &[f32],
        path: &str,
        content: String,
        tokens: usize,
        theta: f32,
    ) {
        let chunk = crate::chunk::Chunk {
            idx: 0,
            heading: String::new(),
            heading_path: String::new(),
            content,
            content_hash: format!("hash-{path}"),
            tokens,
            media_type: MediaType::Text,
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        };
        let id = guard.upsert_chunk(&chunk, path, 1).unwrap();
        let vector: Vec<f32> = q_vec
            .iter()
            .zip(orth)
            .map(|(q, o)| theta.cos() * q + theta.sin() * o)
            .collect();
        guard.set_vector_for_chunk(id, &vector).unwrap();
    }

    #[tokio::test]
    async fn hybrid_search_reaches_requested_limit_when_only_the_vector_branch_supplies_the_tail() {
        let (_dir, store, embedder, q_vec, orth) = candidate_test_fixture("widget").await;

        // Only the first CANDIDATE_K vector-ranked chunks match FTS.
        {
            let guard = store.lock().unwrap();
            for i in 0..ABOVE_CANDIDATE_K {
                let word = if i < CANDIDATE_K { "widget" } else { "gadget" };
                let repeats = ABOVE_CANDIDATE_K - i;
                let content = std::iter::repeat_n(word, repeats)
                    .collect::<Vec<_>>()
                    .join(" ");
                seed_text_chunk(
                    &guard,
                    &q_vec,
                    &orth,
                    &format!("note{i}.md"),
                    content,
                    repeats,
                    (i as f32 + 1.0) * 0.01,
                );
            }
        }

        let hits = search(
            store,
            &embedder,
            CANDIDATE_TEST_DIM,
            "widget",
            50,
            DisplayScoring::default(),
        )
        .await
        .unwrap();

        let paths: std::collections::HashSet<&str> =
            hits.iter().map(|hit| hit.path.as_str()).collect();
        for i in CANDIDATE_K..ABOVE_CANDIDATE_K {
            let path = format!("note{i}.md");
            assert!(
                paths.contains(path.as_str()),
                "{path} is reachable only from the vector branch and must survive a limit=50 search"
            );
        }
    }

    #[tokio::test]
    async fn hybrid_search_reaches_requested_limit_when_only_the_fts_branch_supplies_the_tail() {
        let (_dir, store, embedder, q_vec, orth) = candidate_test_fixture("widget").await;

        // The 20 decoys at angles 0.31..0.50 push the FTS tail outside the
        // 50-entry vector window.
        {
            let guard = store.lock().unwrap();
            for i in 0..ABOVE_CANDIDATE_K {
                let repeats = ABOVE_CANDIDATE_K - i;
                let content = std::iter::repeat_n("widget", repeats)
                    .collect::<Vec<_>>()
                    .join(" ");
                let theta = if i < CANDIDATE_K {
                    (i as f32 + 1.0) * 0.01
                } else {
                    0.61 + (i - CANDIDATE_K) as f32 * 0.01
                };
                seed_text_chunk(
                    &guard,
                    &q_vec,
                    &orth,
                    &format!("note{i}.md"),
                    content,
                    repeats,
                    theta,
                );
            }
            for j in 0..20 {
                seed_text_chunk(
                    &guard,
                    &q_vec,
                    &orth,
                    &format!("decoy{j}.md"),
                    "gadget gadget gadget".to_string(),
                    3,
                    0.31 + j as f32 * 0.01,
                );
            }
        }

        let hits = search(
            store,
            &embedder,
            CANDIDATE_TEST_DIM,
            "widget",
            50,
            DisplayScoring::default(),
        )
        .await
        .unwrap();

        let paths: std::collections::HashSet<&str> =
            hits.iter().map(|hit| hit.path.as_str()).collect();
        for i in CANDIDATE_K..ABOVE_CANDIDATE_K {
            let path = format!("note{i}.md");
            assert!(
                paths.contains(path.as_str()),
                "{path} is reachable only from the FTS branch and must survive a limit=50 search"
            );
        }
    }

    #[tokio::test]
    async fn hybrid_search_with_full_overlap_above_candidate_k_reaches_requested_limit() {
        let (_dir, store, embedder, q_vec, orth) = candidate_test_fixture("widget").await;

        {
            let guard = store.lock().unwrap();
            for i in 0..ABOVE_CANDIDATE_K {
                let repeats = ABOVE_CANDIDATE_K - i;
                let content = std::iter::repeat_n("widget", repeats)
                    .collect::<Vec<_>>()
                    .join(" ");
                seed_text_chunk(
                    &guard,
                    &q_vec,
                    &orth,
                    &format!("note{i}.md"),
                    content,
                    repeats,
                    (i as f32 + 1.0) * 0.01,
                );
            }
        }

        let hits = search(
            store,
            &embedder,
            CANDIDATE_TEST_DIM,
            "widget",
            50,
            DisplayScoring::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            hits.len(),
            ABOVE_CANDIDATE_K,
            "limit=50 with {ABOVE_CANDIDATE_K} identically-ranked hybrid hits must not be capped at CANDIDATE_K"
        );
    }

    fn lane_hit(id: i64, media_type: &str) -> Hit {
        Hit {
            path: format!("{id}"),
            title: String::new(),
            heading_path: String::new(),
            snippet: String::new(),
            score: 0.0,
            score_rrf: 0.0,
            score_normalized: 0.0,
            chunk_id: id,
            media_type: media_type.into(),
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        }
    }

    #[test]
    fn media_gate_includes_threshold_and_rejects_past_it_per_type() {
        // Mutation: changing <= to < or removing per-type gates admits/rejects a boundary candidate.
        let lane = MediaLaneScoring::default();
        for (media_type, threshold) in [("image", 0.40), ("pdf", 0.45)] {
            let base: Vec<_> = (1..=4).map(|id| lane_hit(id, "text")).collect();
            let mut admitted = base.clone();
            let mut rejected = base;
            insert_media_lane(
                &mut admitted,
                vec![lane_hit(99, media_type)],
                4,
                lane,
                Some(&HashMap::from([(99, threshold)])),
            );
            insert_media_lane(
                &mut rejected,
                vec![lane_hit(99, media_type)],
                4,
                lane,
                Some(&HashMap::from([(99, threshold + 0.000_001)])),
            );
            assert!(
                admitted.iter().any(|hit| hit.chunk_id == 99),
                "{media_type}"
            );
            assert!(
                !rejected.iter().any(|hit| hit.chunk_id == 99),
                "{media_type}"
            );
        }
    }

    #[test]
    fn media_lane_adds_candidate_outside_text_window() {
        // Mutation: omitting the independent media candidate query leaves id 99 absent.
        let lane = MediaLaneScoring::default();
        let mut results = (1..=20).map(|id| lane_hit(id, "text")).collect();
        let media = vec![lane_hit(99, "image")];
        let distances = HashMap::from([(99, 0.40)]);
        insert_media_lane(&mut results, media, 20, lane, Some(&distances));
        assert!(results.iter().any(|hit| hit.chunk_id == 99));
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn media_lane_does_not_duplicate_already_ranked_media() {
        // Mutation: removing the present-id filter duplicates id 4.
        let lane = MediaLaneScoring::default();
        let mut results = vec![lane_hit(1, "text"), lane_hit(4, "image")];
        results.extend(
            (2..=20)
                .filter(|id| *id != 4)
                .map(|id| lane_hit(id, "text")),
        );
        let distances = HashMap::from([(4, 0.20)]);
        insert_media_lane(
            &mut results,
            vec![lane_hit(4, "image")],
            20,
            lane,
            Some(&distances),
        );
        assert_eq!(results.iter().filter(|hit| hit.chunk_id == 4).count(), 1);
        assert_eq!(results[1].chunk_id, 4);
    }

    #[test]
    fn media_lane_borrows_slots_when_no_candidate_passes_gate() {
        // Mutation: inserting gate-failing media changes this byte-for-byte baseline result.
        let lane = MediaLaneScoring::default();
        let mut results: Vec<_> = (1..=20).map(|id| lane_hit(id, "text")).collect();
        let baseline = results.clone();
        let distances = HashMap::from([(99, 0.400_001)]);
        insert_media_lane(
            &mut results,
            vec![lane_hit(99, "image")],
            20,
            lane,
            Some(&distances),
        );
        assert_eq!(results, baseline);
    }

    #[test]
    fn media_lane_borrows_unfilled_reserved_slots() {
        // Mutation: padding the reservation with media changes text ids or result length.
        let lane = MediaLaneScoring::default();
        let mut results: Vec<_> = (1..=8).map(|id| lane_hit(id, "text")).collect();
        let distances = HashMap::from([(99, 0.20)]);
        insert_media_lane(
            &mut results,
            vec![lane_hit(99, "image")],
            8,
            lane,
            Some(&distances),
        );
        assert_eq!(results.len(), 8);
        assert_eq!(
            results
                .iter()
                .filter(|hit| hit.media_type == "image")
                .count(),
            1
        );
        assert!(results.iter().any(|hit| hit.chunk_id == 7));
    }

    #[test]
    fn media_lane_uses_evenly_spaced_insert_positions() {
        // Mutation: changing stride or the first insertion index moves ids 91 and 92.
        let lane = MediaLaneScoring {
            fraction: 0.5,
            ..MediaLaneScoring::default()
        };
        let mut results: Vec<_> = (1..=8).map(|id| lane_hit(id, "text")).collect();
        let distances = HashMap::from([(91, 0.20), (92, 0.20)]);
        insert_media_lane(
            &mut results,
            vec![lane_hit(91, "image"), lane_hit(92, "image")],
            8,
            lane,
            Some(&distances),
        );
        assert_eq!(
            results.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>(),
            [1, 2, 3, 91, 4, 5, 6, 92]
        );
    }

    #[test]
    fn media_display_score_clamps_decreases_and_uses_pdf_map() {
        // Mutation: using the image map for PDFs makes the invoice score fall below 0.4.
        let lane = MediaLaneScoring::default();
        assert_eq!(media_display_score(0.20, "image", lane), Some(1.0));
        assert_eq!(media_display_score(0.60, "image", lane), Some(0.0));
        let near = media_display_score(0.30, "image", lane).unwrap();
        let far = media_display_score(0.40, "image", lane).unwrap();
        assert!(near > far);
        let pdf = media_display_score(0.4211, "pdf", lane).unwrap();
        let image = media_display_score(0.4211, "image", lane).unwrap();
        assert!(pdf > 0.4, "{pdf}");
        assert!(image < 0.4, "{image}");
    }
}
