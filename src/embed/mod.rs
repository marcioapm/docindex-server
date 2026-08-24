//! Embedder trait and implementations.

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

pub const TASK_RETRIEVAL_DOCUMENT: &str = "RETRIEVAL_DOCUMENT";
pub const TASK_RETRIEVAL_QUERY: &str = "RETRIEVAL_QUERY";
pub const MEDIA_DOCUMENT_TASK: &str = "document";

/// A provider-neutral document embedding input. Media bytes are deliberately
/// not `Debug` so they cannot accidentally be emitted in logs or errors.
#[derive(Clone)]
pub enum EmbedInput {
    Text(String),
    Media(Vec<MediaPart>),
}

#[derive(Clone)]
pub struct MediaPart {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

impl EmbedInput {
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }
}

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

pub trait Embedder: Send + Sync {
    fn embed_documents(
        &self,
        inputs: &[EmbedInput],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send;

    fn embed_query(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbedError>> + Send;
}

#[derive(Clone)]
pub enum AnyEmbedder {
    Gemini(Arc<Gemini>),
    Voyage(Arc<Voyage>),
    Fake(Arc<Fake>),
}

impl AnyEmbedder {
    pub async fn embed_documents(
        &self,
        inputs: &[EmbedInput],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        match self {
            Self::Gemini(g) => g.embed_documents(inputs).await,
            Self::Voyage(v) => v.embed_documents(inputs).await,
            Self::Fake(f) => f.embed_documents(inputs).await,
        }
    }

    pub async fn embed_text_documents(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let inputs: Vec<_> = texts.iter().cloned().map(EmbedInput::Text).collect();
        self.embed_documents(&inputs).await
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        match self {
            Self::Gemini(g) => g.embed_query(text).await,
            Self::Voyage(v) => v.embed_query(text).await,
            Self::Fake(f) => f.embed_query(text).await,
        }
    }
}
