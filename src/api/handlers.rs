//! HTTP handlers for `/health`, `/search`, `/similar`.

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::search::{self, Hit};

use super::{AppState, error::ApiError};

#[derive(Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub indexed_chunks: i64,
    pub last_reindex_ms: i64,
    pub embedding_model: String,
    pub dim: usize,
}

pub async fn health(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let store = state.store.clone();
    let indexed_chunks = tokio::task::spawn_blocking(move || -> Result<i64, ApiError> {
        let guard = store
            .lock()
            .map_err(|e| ApiError::Internal(format!("store lock: {e}")))?;
        guard
            .count_chunks()
            .map_err(|e| ApiError::Internal(format!("count chunks: {e}")))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("join: {e}")))??;
    Ok(Json(HealthResponse {
        ok: true,
        indexed_chunks,
        last_reindex_ms: state
            .last_reindex_ms
            .load(std::sync::atomic::Ordering::Relaxed),
        embedding_model: state.embed_model.to_string(),
        dim: state.embed_dim,
    }))
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub hits: Vec<Hit>,
}

pub async fn search(
    State(state): State<AppState>,
    body: Result<Json<SearchRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<SearchResponse>, ApiError> {
    let Json(req) = body?;
    if req.query.trim().is_empty() {
        return Err(ApiError::BadRequest("query must not be empty".into()));
    }
    let hits = search::search(
        state.store.clone(),
        &state.embedder,
        state.embed_dim,
        &req.query,
        req.limit,
    )
    .await?;
    Ok(Json(SearchResponse { hits }))
}

#[derive(Deserialize)]
pub struct SimilarRequest {
    pub path: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

pub async fn similar(
    State(state): State<AppState>,
    body: Result<Json<SimilarRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<SearchResponse>, ApiError> {
    let Json(req) = body?;
    if req.path.trim().is_empty() {
        return Err(ApiError::BadRequest("path must not be empty".into()));
    }
    let hits = search::similar(state.store.clone(), state.embed_dim, &req.path, req.limit).await?;
    Ok(Json(SearchResponse { hits }))
}
