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
