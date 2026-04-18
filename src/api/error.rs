//! Structured API error. Every handler returns `Result<Json<T>, ApiError>`
//! so a single `IntoResponse` impl picks the right status + JSON shape.

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use thiserror::Error;

use crate::search::SearchError;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized")]
    Unauthorized,
    #[error("not found")]
    NotFound(String),
    #[error("internal error")]
    Internal(String),
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
    code: &'a str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, code, msg): (StatusCode, &'static str, String) = match &self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "bad_request", m.clone()),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or invalid bearer".into(),
            ),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "not_found", m.clone()),
            ApiError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "internal error".into(),
            ),
        };
        if status.is_server_error() {
            // Log the full message server-side; do not leak internals.
            tracing::error!(error = %self, "request failed");
        } else {
            tracing::warn!(error = %self, "request rejected");
        }
        (
            status,
            Json(ErrorBody {
                error: msg.as_str(),
                code,
            }),
        )
            .into_response()
    }
}

impl From<SearchError> for ApiError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::PathNotIndexed(p) => ApiError::NotFound(format!("path not indexed: {p}")),
            SearchError::DimMismatch { got, want } => {
                ApiError::BadRequest(format!("embedding dim {got} != expected {want}"))
            }
            other => ApiError::Internal(other.to_string()),
        }
    }
}

impl From<axum::extract::rejection::JsonRejection> for ApiError {
    fn from(r: axum::extract::rejection::JsonRejection) -> Self {
        ApiError::BadRequest(r.to_string())
    }
}
