//! HTTP handlers for `/health`, `/search`, `/similar`.

use axum::{Json, extract::State, http::HeaderMap};
use serde::{Deserialize, Serialize};

use crate::search::{self, Hit};

use super::{AppState, auth, error::ApiError};

#[derive(Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub authenticated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_chunks: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_reindex_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim: Option<usize>,
}

pub async fn health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<HealthResponse>, ApiError> {
    if !auth::has_valid_bearer(&headers, &state.bearer) {
        return Ok(Json(HealthResponse {
            ok: true,
            authenticated: false,
            indexed_chunks: None,
            last_reindex_ms: None,
            embedding_model: None,
            dim: None,
        }));
    }

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
        authenticated: true,
        indexed_chunks: Some(indexed_chunks),
        last_reindex_ms: Some(
            state
                .last_reindex_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        ),
        embedding_model: Some(state.embed_model.to_string()),
        dim: Some(state.embed_dim),
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
        state.display_scoring,
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
    let hits = search::similar(
        state.store.clone(),
        state.embed_dim,
        &req.path,
        req.limit,
        state.display_scoring,
    )
    .await?;
    Ok(Json(SearchResponse { hits }))
}
