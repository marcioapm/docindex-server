//! Embedder trait and implementations.
//!
//! Gemini uses task-asymmetric embeddings: documents with
//! `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY`. Getting this wrong
//! silently degrades ranking quality — the vectors still parse, they are
//! just miscalibrated. Output is Matryoshka-truncated to the configured
//! dim (`DOCINDEX_EMBED_DIM`, default 3072 — the model's native size; 768
//! is a smaller Matryoshka truncation that trades a little quality for
//! disk/ANN cost at tiny scale).

pub mod fake;
pub mod gemini;
pub mod registry;
pub mod voyage;

use std::future::Future;
use std::sync::Arc;

use thiserror::Error;

pub use fake::Fake;
pub use gemini::Gemini;
pub use voyage::Voyage;

/// Used when embedding chunks for indexing.
pub const TASK_RETRIEVAL_DOCUMENT: &str = "RETRIEVAL_DOCUMENT";
/// Used when embedding a user query for search.
pub const TASK_RETRIEVAL_QUERY: &str = "RETRIEVAL_QUERY";

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embed: config: {0}")]
    Config(String),
    #[error("embed: http: {0}")]
    Http(String),
    #[error("embed: api: status {status}: {message}")]
    Api { status: u16, message: String },
    #[error("embed: decode: {0}")]
    Decode(String),
    #[error("embed: response dim mismatch: got {got}, want {want}")]
    DimMismatch { got: usize, want: usize },
    #[error("embed: retries exhausted: {0}")]
    RetriesExhausted(String),
}

/// Produce float32 vectors for document chunks or user queries.
///
/// Uses native `async fn` in trait; callers keep the returned futures on the
/// local runtime and do not need a Send bound for Phase 1 wiring.
pub trait Embedder: Send + Sync {
    fn embed_documents(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send;

    fn embed_query(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbedError>> + Send;
}

/// Erased embedder for runtime selection. Native `async fn` in traits is not
/// dyn-compatible, so we enumerate the known implementations and static-
/// dispatch in a match — callers get `Clone + Send + Sync` without pulling
/// in `async_trait`.
#[derive(Clone)]
pub enum AnyEmbedder {
    Gemini(Arc<Gemini>),
    Voyage(Arc<Voyage>),
    Fake(Arc<Fake>),
}

impl AnyEmbedder {
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        match self {
            Self::Gemini(g) => g.embed_documents(texts).await,
            Self::Voyage(v) => v.embed_documents(texts).await,
            Self::Fake(f) => f.embed_documents(texts).await,
        }
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        match self {
            Self::Gemini(g) => g.embed_query(text).await,
            Self::Voyage(v) => v.embed_query(text).await,
            Self::Fake(f) => f.embed_query(text).await,
        }
    }
}
