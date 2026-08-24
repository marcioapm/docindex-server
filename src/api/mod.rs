//! HTTP API surface.
//!
//! - [`AppState`] is the shared handle passed to every handler.
//! - [`build_router`] wires `/health` (public), `/search` and `/similar`
//!   (bearer-protected) into a single `axum::Router`.

pub mod auth;
pub mod error;
pub mod handlers;

use std::sync::{Arc, Mutex, atomic::AtomicI64};

use axum::{Router, middleware, routing::post};

use crate::{embed::AnyEmbedder, search::DisplayScoring, store::Store};

/// State shared by every handler. Clone is cheap (reference counted).
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<Store>>,
    pub embedder: AnyEmbedder,
    pub bearer: Arc<String>,
    pub embed_model: Arc<String>,
    pub embed_dim: usize,
    pub last_reindex_ms: Arc<AtomicI64>,
    pub display_scoring: DisplayScoring,
}

/// Compose the full HTTP router. `/health` is public; `/search` and
/// `/similar` sit behind [`auth::require_bearer`].
pub fn build_router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/search", post(handlers::search))
        .route("/similar", post(handlers::similar))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_bearer,
        ));

    Router::new()
        .route("/health", axum::routing::get(handlers::health))
        .merge(protected)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, atomic::AtomicI64};

    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use tempfile::TempDir;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        embed::{AnyEmbedder, Fake},
        store::Store,
    };

    fn test_state() -> (TempDir, AppState) {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path().join("index.db"), 8).unwrap();
        let state = AppState {
            store: Arc::new(Mutex::new(store)),
            embedder: AnyEmbedder::Fake(Arc::new(Fake::new(8))),
            bearer: Arc::new("right-token".into()),
            embed_model: Arc::new("fake-model".into()),
            embed_dim: 8,
            last_reindex_ms: Arc::new(AtomicI64::new(123)),
            display_scoring: DisplayScoring::default(),
        };
        (dir, state)
    }

    async fn response_body(response: axum::response::Response) -> Vec<u8> {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    #[tokio::test]
    async fn health_without_bearer_is_minimal_liveness() {
        let (_dir, state) = test_state();
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response_body(response).await).unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["authenticated"], false);
        for key in [
            "indexed_chunks",
            "last_reindex_ms",
            "embedding_model",
            "dim",
        ] {
            assert!(body.get(key).is_none(), "unexpected {key}: {body}");
        }
    }

    #[tokio::test]
    async fn health_with_wrong_bearer_matches_missing_bearer() {
        let (_dir, state) = test_state();
        let router = build_router(state);
        let missing = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let wrong = router
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("authorization", "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(missing.status(), StatusCode::OK);
        assert_eq!(wrong.status(), StatusCode::OK);
        assert_eq!(response_body(missing).await, response_body(wrong).await);
    }

    #[tokio::test]
    async fn health_with_valid_bearer_includes_details() {
        let (_dir, state) = test_state();
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("authorization", "Bearer right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response_body(response).await).unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["authenticated"], true);
        assert_eq!(body["indexed_chunks"], 0);
        assert_eq!(body["last_reindex_ms"], 123);
        assert_eq!(body["embedding_model"], "fake-model");
        assert_eq!(body["dim"], 8);
    }

    #[tokio::test]
    async fn search_accepts_media_only_request() {
        let (_dir, state) = test_state();
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/search")
                    .header("authorization", "Bearer right-token")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"image","media_only":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response_body(response).await).unwrap();
        assert_eq!(body, serde_json::json!({ "hits": [] }));
    }

    #[tokio::test]
    async fn similar_known_empty_file_returns_empty_hits() {
        let (_dir, state) = test_state();
        state
            .store
            .lock()
            .unwrap()
            .set_file_state("empty.md", "hash", 123)
            .unwrap();
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/similar")
                    .header("authorization", "Bearer right-token")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"path":"empty.md"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response_body(response).await).unwrap();
        assert_eq!(body, serde_json::json!({ "hits": [] }));
    }

    #[tokio::test]
    async fn similar_unknown_path_returns_not_found() {
        let (_dir, state) = test_state();
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/similar")
                    .header("authorization", "Bearer right-token")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"path":"missing.md"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body: serde_json::Value =
            serde_json::from_slice(&response_body(response).await).unwrap();
        assert_eq!(body["code"], "not_found");
        assert_eq!(body["error"], "path not indexed: missing.md");
    }
    #[tokio::test]
    async fn protected_routes_reject_bad_bearer() {
        let (_dir, state) = test_state();
        let router = build_router(state);
        for uri in ["/search", "/similar"] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("authorization", "Bearer wrong-token")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{uri}");
        }
    }
}
