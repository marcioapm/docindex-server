//! Static provider/model registry.
//!
//! All dim/task-label lookups for building an embedder go through this
//! module so validation logic lives in one place.

use std::fmt;

/// Embedding provider selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedProvider {
    Gemini,
    Voyage,
    Fake,
}

impl EmbedProvider {
    pub const ALL: [EmbedProvider; 3] = [
        EmbedProvider::Gemini,
        EmbedProvider::Voyage,
        EmbedProvider::Fake,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedProvider::Gemini => "gemini",
            EmbedProvider::Voyage => "voyage",
            EmbedProvider::Fake => "fake",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gemini" => Some(EmbedProvider::Gemini),
            "voyage" => Some(EmbedProvider::Voyage),
            "fake" => Some(EmbedProvider::Fake),
            _ => None,
        }
    }

    /// Default model id for this provider when none is configured.
    pub fn default_model(&self) -> &'static str {
        match self {
            EmbedProvider::Gemini => "gemini-embedding-2",
            EmbedProvider::Voyage => "voyage-4",
            EmbedProvider::Fake => "fake",
        }
    }

    /// Env var to consult for the API key when none is set in config.
    /// `Fake` requires no key.
    pub fn key_env_var(&self) -> Option<&'static str> {
        match self {
            EmbedProvider::Gemini => Some("GEMINI_API_KEY"),
            EmbedProvider::Voyage => Some("VOYAGE_API_KEY"),
            EmbedProvider::Fake => None,
        }
    }
}

impl fmt::Display for EmbedProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// How a model represents PDF input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfMode {
    None,
    Native,
    Raster,
}

/// One registered model: its provider, native/allowed dims, and the
/// provider-specific task labels used for document vs. query embedding.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: EmbedProvider,
    pub model: &'static str,
    pub native_dim: usize,
    pub allowed_dims: &'static [usize],
    pub doc_task: &'static str,
    pub query_task: &'static str,
    pub media_capable: bool,
    pub pdf_mode: PdfMode,
}

/// Gemini's Matryoshka-truncatable dims for `gemini-embedding-001`.
const GEMINI_DIMS: &[usize] = &[768, 1536, 3072];
/// Voyage-4 generation's output dims (all models below share this set).
const VOYAGE_DIMS: &[usize] = &[256, 512, 1024, 2048];

/// The full set of known models. `Fake` has no fixed entries — any model
/// name / dim is accepted, matched dynamically in [`lookup`].
static MODELS: &[ModelInfo] = &[
    ModelInfo {
        provider: EmbedProvider::Gemini,
        model: "gemini-embedding-001",
        native_dim: 3072,
        allowed_dims: GEMINI_DIMS,
        doc_task: "RETRIEVAL_DOCUMENT",
        query_task: "RETRIEVAL_QUERY",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Gemini,
        model: "gemini-embedding-2",
        native_dim: 3072,
        allowed_dims: GEMINI_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: true,
        pdf_mode: PdfMode::Native,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-4",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-4-lite",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-4-large",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-context-4",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-code-3",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: false,
        pdf_mode: PdfMode::None,
    },
    ModelInfo {
        provider: EmbedProvider::Voyage,
        model: "voyage-multimodal-3.5",
        native_dim: 1024,
        allowed_dims: VOYAGE_DIMS,
        doc_task: "document",
        query_task: "query",
        media_capable: true,
        pdf_mode: PdfMode::Raster,
    },
];

/// `Fake`'s synthetic model entry — accepts any dim, used only for its task
/// labels and native_dim default (128, an arbitrary small test-friendly
/// value; callers always pass an explicit dim in practice).
fn fake_model(model: &'static str) -> ModelInfo {
    ModelInfo {
        provider: EmbedProvider::Fake,
        model,
        native_dim: 128,
        allowed_dims: &[],
        doc_task: "document",
        query_task: "query",
        media_capable: true,
        pdf_mode: PdfMode::Raster,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    UnknownProvider {
        got: String,
        valid: Vec<&'static str>,
    },
    UnknownModel {
        provider: EmbedProvider,
        got: String,
        valid: Vec<&'static str>,
    },
    BadDim {
        provider: EmbedProvider,
        model: String,
        got: usize,
        allowed: Vec<usize>,
    },
    MissingKey {
        provider: EmbedProvider,
        env_var: &'static str,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::UnknownProvider { got, valid } => write!(
                f,
                "unknown embed provider {got:?}; valid providers: {}",
                valid.join(", ")
            ),
            RegistryError::UnknownModel {
                provider,
                got,
                valid,
            } => write!(
                f,
                "unknown model {got:?} for provider {provider}; valid models: {}",
                valid.join(", ")
            ),
            RegistryError::BadDim {
                provider,
                model,
                got,
                allowed,
            } => write!(
                f,
                "dim {got} not allowed for {provider}/{model}; allowed dims: {}",
                allowed
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RegistryError::MissingKey { provider, env_var } => write!(
                f,
                "provider {provider} requires an API key; set [embed].api_key, [embed].api_key_env, or ${env_var}"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Every model id registered for `provider`, in table order.
pub fn models_for(provider: EmbedProvider) -> Vec<&'static str> {
    if provider == EmbedProvider::Fake {
        return vec!["fake"];
    }
    MODELS
        .iter()
        .filter(|m| m.provider == provider)
        .map(|m| m.model)
        .collect()
}

/// Look up a `(provider, model)` pair. `Fake` matches any model name.
pub fn lookup(provider: EmbedProvider, model: &str) -> Result<ModelInfo, RegistryError> {
    if provider == EmbedProvider::Fake {
        return Ok(fake_model("fake"));
    }
    MODELS
        .iter()
        .find(|m| m.provider == provider && m.model == model)
        .cloned()
        .ok_or_else(|| RegistryError::UnknownModel {
            provider,
            got: model.to_string(),
            valid: models_for(provider),
        })
}

/// Validate a resolved `(provider, model, dim)` triple. `Fake` accepts any
/// dim > 0 — that check is left to the caller (config already enforces it
/// generically).
pub fn validate_dim(info: &ModelInfo, dim: usize) -> Result<(), RegistryError> {
    if info.provider == EmbedProvider::Fake {
        return Ok(());
    }
    if info.allowed_dims.contains(&dim) {
        Ok(())
    } else {
        Err(RegistryError::BadDim {
            provider: info.provider,
            model: info.model.to_string(),
            got: dim,
            allowed: info.allowed_dims.to_vec(),
        })
    }
}

/// Parse a provider string, producing a listing error naming valid values.
pub fn parse_provider(s: &str) -> Result<EmbedProvider, RegistryError> {
    EmbedProvider::parse(s).ok_or_else(|| RegistryError::UnknownProvider {
        got: s.to_string(),
        valid: EmbedProvider::ALL.iter().map(|p| p.as_str()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_provider_known_values() {
        assert_eq!(parse_provider("gemini").unwrap(), EmbedProvider::Gemini);
        assert_eq!(parse_provider("voyage").unwrap(), EmbedProvider::Voyage);
        assert_eq!(parse_provider("fake").unwrap(), EmbedProvider::Fake);
    }

    #[test]
    fn parse_provider_unknown_lists_valid() {
        let err = parse_provider("bogus").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus"));
        assert!(msg.contains("gemini"));
        assert!(msg.contains("voyage"));
        assert!(msg.contains("fake"));
    }

    #[test]
    fn lookup_gemini_default_model() {
        let info = lookup(EmbedProvider::Gemini, "gemini-embedding-001").unwrap();
        assert_eq!(info.native_dim, 3072);
        assert_eq!(info.doc_task, "RETRIEVAL_DOCUMENT");
        assert_eq!(info.query_task, "RETRIEVAL_QUERY");
    }

    #[test]
    fn lookup_unknown_model_lists_valid_for_provider() {
        let err = lookup(EmbedProvider::Voyage, "voyage-nope").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("voyage-nope"));
        assert!(msg.contains("voyage-4"));
        assert!(!msg.contains("gemini-embedding-001"));
    }

    #[test]
    fn validate_dim_gemini_allowed() {
        let info = lookup(EmbedProvider::Gemini, "gemini-embedding-001").unwrap();
        assert!(validate_dim(&info, 768).is_ok());
        assert!(validate_dim(&info, 1536).is_ok());
        assert!(validate_dim(&info, 3072).is_ok());
        assert!(validate_dim(&info, 512).is_err());
    }

    #[test]
    fn validate_dim_voyage_allowed() {
        let info = lookup(EmbedProvider::Voyage, "voyage-4").unwrap();
        for d in [256, 512, 1024, 2048] {
            assert!(validate_dim(&info, d).is_ok(), "dim {d} should be allowed");
        }
        assert!(validate_dim(&info, 3072).is_err());
    }

    #[test]
    fn validate_dim_error_lists_allowed() {
        let info = lookup(EmbedProvider::Voyage, "voyage-4").unwrap();
        let err = validate_dim(&info, 99).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("512"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("2048"));
    }

    #[test]
    fn fake_accepts_any_model_and_dim() {
        let info = lookup(EmbedProvider::Fake, "anything").unwrap();
        assert!(validate_dim(&info, 7).is_ok());
        assert!(validate_dim(&info, 99999).is_ok());
    }

    #[test]
    fn default_models_per_provider() {
        assert_eq!(EmbedProvider::Gemini.default_model(), "gemini-embedding-2");
        assert_eq!(EmbedProvider::Voyage.default_model(), "voyage-4");
    }

    #[test]
    fn key_env_vars() {
        assert_eq!(EmbedProvider::Gemini.key_env_var(), Some("GEMINI_API_KEY"));
        assert_eq!(EmbedProvider::Voyage.key_env_var(), Some("VOYAGE_API_KEY"));
        assert_eq!(EmbedProvider::Fake.key_env_var(), None);
    }

    #[test]
    fn all_voyage_models_registered() {
        let models = models_for(EmbedProvider::Voyage);
        for m in [
            "voyage-4",
            "voyage-4-lite",
            "voyage-4-large",
            "voyage-context-4",
            "voyage-code-3",
            "voyage-multimodal-3.5",
        ] {
            assert!(models.contains(&m), "missing {m} in {models:?}");
        }
    }
}
