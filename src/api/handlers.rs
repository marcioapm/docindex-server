//! HTTP handlers for `/health`, `/search`, `/similar`.

use axum::{Json, extract::State, http::HeaderMap};
use serde::{Deserialize, Serialize};

use crate::{
    media::MediaType,
    search::{self, Hit, SearchOptions},
};

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
    #[serde(default)]
    pub media_only: bool,
    #[serde(default)]
    pub media_types: Vec<String>,
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
    // Surrounding whitespace changes the embedding and therefore hit order, so
    // trim once here and use the trimmed value for both the guard and the query.
    let query = req.query.trim();
    if query.is_empty() {
        return Err(ApiError::BadRequest("query must not be empty".into()));
    }
    let media_types = media_types_for_request(req.media_only, &req.media_types)?;
    let hits = search::search_with_options(
        state.store.clone(),
        &state.embedder,
        state.embed_dim,
        query,
        req.limit,
        state.display_scoring,
        SearchOptions {
            media_only: req.media_only,
            media_types,
        },
    )
    .await?;
    Ok(Json(SearchResponse { hits }))
}

fn media_types_for_request(
    media_only: bool,
    values: &[String],
) -> Result<Vec<MediaType>, ApiError> {
    if !media_only && !values.is_empty() {
        return Err(ApiError::BadRequest(
            "media_types requires media_only".into(),
        ));
    }

    let mut media_types = Vec::with_capacity(values.len());
    for value in values {
        let Some(media_type) = MediaType::from_exclude_value(value) else {
            return Err(ApiError::BadRequest(format!(
                "media_types: unknown value {value:?}; valid: {}",
                MediaType::EXCLUDE_VALUES.join(", ")
            )));
        };
        if !media_types.contains(&media_type) {
            media_types.push(media_type);
        }
    }
    Ok(media_types)
}

#[cfg(test)]
mod tests {
    use crate::media::MediaType;

    use super::{SearchRequest, media_types_for_request};

    #[test]
    fn search_request_defaults_media_only_to_false() {
        let request: SearchRequest = serde_json::from_str(r#"{"query":"cats"}"#).unwrap();

        assert_eq!(request.query, "cats");
        assert_eq!(request.limit, 10);
        assert!(!request.media_only);
        assert!(request.media_types.is_empty());
    }

    #[test]
    fn search_request_deserializes_media_only() {
        let request: SearchRequest =
            serde_json::from_str(r#"{"query":"cats","limit":4,"media_only":true}"#).unwrap();

        assert_eq!(request.limit, 4);
        assert!(request.media_only);
    }

    #[test]
    fn search_request_defaults_media_types_to_empty() {
        let request: SearchRequest = serde_json::from_str(r#"{"query":"cats"}"#).unwrap();

        assert!(request.media_types.is_empty());
    }

    #[test]
    fn media_types_rejects_hybrid_search() {
        let error = media_types_for_request(false, &["pdf".into()]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "bad request: media_types requires media_only"
        );
    }

    #[test]
    fn media_types_validates_and_deduplicates() {
        let types =
            media_types_for_request(true, &["pdf".into(), "image".into(), "pdf".into()]).unwrap();

        assert_eq!(types, [MediaType::Pdf, MediaType::Image]);
    }

    #[test]
    fn media_types_names_invalid_value() {
        let error = media_types_for_request(true, &["jpeg".into()]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "bad request: media_types: unknown value \"jpeg\"; valid: image, pdf, audio, video"
        );
    }
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
    // Path lookup is exact-match, so surrounding whitespace would turn a valid
    // vault path into a not-found.
    let path = req.path.trim();
    if path.is_empty() {
        return Err(ApiError::BadRequest("path must not be empty".into()));
    }
    let hits = search::similar(
        state.store.clone(),
        state.embed_dim,
        path,
        req.limit,
        state.display_scoring,
    )
    .await?;
    Ok(Json(SearchResponse { hits }))
}
