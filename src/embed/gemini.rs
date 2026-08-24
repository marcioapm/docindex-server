//! Gemini embeddings client.

use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::{
    EmbedError, EmbedInput, Embedder, MediaPart, TASK_RETRIEVAL_DOCUMENT, TASK_RETRIEVAL_QUERY,
};

const GEMINI_EMBEDDING_2: &str = "gemini-embedding-2";
// 16 inputs per batch is a pragmatic bound on base64-encoded request body
// size for typical document chunks, not a documented API limit.
const EMBEDDING_2_BATCH_SIZE: usize = 16;

/// Embedder backed by Google's Generative Language REST API.
pub struct Gemini {
    pub api_key: String,
    pub model: String,
    pub dim: usize,
    pub base_url: String,
    pub client: reqwest::Client,
    pub max_retries: u32,
    pub base_delay: Duration,
}

impl Gemini {
    /// Construct with sensible defaults.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
        timeout: Duration,
    ) -> Result<Self, EmbedError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| EmbedError::Config(format!("build reqwest client: {e}")))?;
        Ok(Self {
            api_key: api_key.into(),
            model: model.into(),
            dim,
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            client,
            max_retries: 3,
            base_delay: Duration::from_millis(200),
        })
    }

    fn validate(&self) -> Result<(), EmbedError> {
        if self.api_key.is_empty() {
            return Err(EmbedError::Config("api_key is empty".into()));
        }
        if self.model.is_empty() {
            return Err(EmbedError::Config("model is empty".into()));
        }
        if self.dim == 0 {
            return Err(EmbedError::Config("dim must be > 0".into()));
        }
        Ok(())
    }

    fn is_embedding_2(&self) -> bool {
        self.model == GEMINI_EMBEDDING_2
    }

    async fn post_json<T: Serialize>(&self, url: &str, body: &T) -> Result<Vec<u8>, EmbedError> {
        self.validate()?;
        let attempts = self.max_retries.saturating_add(1);
        let mut delay = self.base_delay;
        let mut last_err = EmbedError::Http("no attempts".into());

        for attempt in 0..attempts {
            if attempt > 0 {
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
            let resp = match self
                .client
                .post(url)
                .header("x-goog-api-key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(body)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    last_err = EmbedError::Http(error.to_string());
                    continue;
                }
            };
            let status = resp.status();
            let raw = resp
                .bytes()
                .await
                .map_err(|e| EmbedError::Http(format!("read body: {e}")))?;

            if status.is_success() {
                return Ok(raw.to_vec());
            }

            last_err = EmbedError::Api {
                status: status.as_u16(),
                message: parse_api_error(&raw)
                    .unwrap_or_else(|| "non-JSON API error response".into()),
            };
            if !is_retryable(status) {
                return Err(last_err);
            }
        }
        Err(EmbedError::RetriesExhausted(format!("{last_err}")))
    }

    /// Legacy `gemini-embedding-001` request. Its endpoint and taskType
    /// behavior intentionally remain unchanged.
    async fn embed_legacy(&self, text: &str, task_type: &str) -> Result<Vec<f32>, EmbedError> {
        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.base_url, self.model
        );
        let body = LegacyEmbedRequest {
            content: Content {
                parts: vec![Part::text(text)],
            },
            task_type: task_type.to_string(),
            output_dimensionality: self.dim,
        };
        let raw = self.post_json(&url, &body).await?;
        let parsed: SingleEmbedResponse =
            serde_json::from_slice(&raw).map_err(|e| EmbedError::Decode(format!("decode: {e}")))?;
        self.validate_embedding(parsed.embedding.values)
    }

    async fn embed_embedding_2_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.base_url, self.model
        );
        let body = Embedding2SingleRequest {
            model: model_name(&self.model),
            content: Content {
                parts: vec![Part::text(format!("task: search result | query: {text}"))],
            },
            output_dimensionality: self.dim,
        };
        let raw = self.post_json(&url, &body).await?;
        let parsed: SingleEmbedResponse =
            serde_json::from_slice(&raw).map_err(|e| EmbedError::Decode(format!("decode: {e}")))?;
        self.validate_embedding(parsed.embedding.values)
    }

    async fn embed_embedding_2_documents(
        &self,
        inputs: &[EmbedInput],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents",
            self.base_url, self.model
        );
        let mut embeddings = Vec::with_capacity(inputs.len());

        for (batch_index, batch) in inputs.chunks(EMBEDDING_2_BATCH_SIZE).enumerate() {
            let requests = batch
                .iter()
                .map(|input| Embedding2BatchRequest {
                    model: model_name(&self.model),
                    content: embedding_2_document_content(input),
                    output_dimensionality: self.dim,
                })
                .collect();
            let raw = self
                .post_json(&url, &Embedding2BatchEmbedRequest { requests })
                .await?;
            let parsed: BatchEmbedResponse = serde_json::from_slice(&raw)
                .map_err(|e| EmbedError::Decode(format!("decode batch {batch_index}: {e}")))?;
            if parsed.embeddings.len() != batch.len() {
                return Err(EmbedError::Decode(format!(
                    "batch {batch_index} returned {} embeddings for {} inputs",
                    parsed.embeddings.len(),
                    batch.len()
                )));
            }
            for embedding in parsed.embeddings {
                embeddings.push(self.validate_embedding(embedding.values)?);
            }
        }
        Ok(embeddings)
    }

    fn validate_embedding(&self, values: Vec<f32>) -> Result<Vec<f32>, EmbedError> {
        if values.is_empty() {
            return Err(EmbedError::Decode("empty embedding values".into()));
        }
        if values.len() != self.dim {
            return Err(EmbedError::DimMismatch {
                got: values.len(),
                want: self.dim,
            });
        }
        Ok(values)
    }
}

impl Embedder for Gemini {
    async fn embed_documents(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if self.is_embedding_2() {
            return self.embed_embedding_2_documents(inputs).await;
        }

        let mut out = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let EmbedInput::Text(text) = input else {
                return Err(EmbedError::Config(format!(
                    "model {} does not support media inputs (input {index})",
                    self.model
                )));
            };
            let vector = self
                .embed_legacy(text, TASK_RETRIEVAL_DOCUMENT)
                .await
                .map_err(|error| match error {
                    EmbedError::Api { status, message } => EmbedError::Api {
                        status,
                        message: format!("doc[{index}]: {message}"),
                    },
                    other => other,
                })?;
            out.push(vector);
        }
        Ok(out)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if self.is_embedding_2() {
            self.embed_embedding_2_query(text).await
        } else {
            self.embed_legacy(text, TASK_RETRIEVAL_QUERY).await
        }
    }
}

fn model_name(model: &str) -> String {
    format!("models/{model}")
}

fn embedding_2_document_content(input: &EmbedInput) -> Content {
    match input {
        EmbedInput::Text(text) => Content {
            parts: vec![Part::text(format!("title: none | text: {text}"))],
        },
        EmbedInput::Media(parts) => Content {
            parts: parts.iter().map(Part::media).collect(),
        },
    }
}

fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn parse_api_error(raw: &[u8]) -> Option<String> {
    let value: ApiError = serde_json::from_slice(raw).ok()?;
    (!value.error.message.is_empty()).then_some(value.error.message)
}

#[derive(Serialize)]
struct LegacyEmbedRequest {
    content: Content,
    #[serde(rename = "taskType")]
    task_type: String,
    #[serde(rename = "outputDimensionality")]
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct Embedding2SingleRequest {
    model: String,
    content: Content,
    #[serde(rename = "outputDimensionality")]
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct Embedding2BatchEmbedRequest {
    requests: Vec<Embedding2BatchRequest>,
}

#[derive(Serialize)]
struct Embedding2BatchRequest {
    model: String,
    content: Content,
    #[serde(rename = "outputDimensionality")]
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    Media {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
}

impl Part {
    fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    fn media(part: &MediaPart) -> Self {
        Self::Media {
            inline_data: InlineData {
                mime_type: part.mime_type.clone(),
                data: STANDARD.encode(&part.bytes),
            },
        }
    }
}

#[derive(Serialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Deserialize)]
struct SingleEmbedResponse {
    embedding: EmbedValues,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbedValues>,
}

#[derive(Deserialize)]
struct EmbedValues {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct ApiError {
    #[serde(default)]
    error: ApiErrorBody,
}

#[derive(Deserialize, Default)]
struct ApiErrorBody {
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn test_gemini(server: &MockServer, model: &str, max_retries: u32) -> Gemini {
        Gemini {
            api_key: "test-key".into(),
            model: model.into(),
            dim: 4,
            base_url: server.uri(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            max_retries,
            base_delay: Duration::from_millis(1),
        }
    }

    fn vector(value: f32) -> serde_json::Value {
        json!({ "values": [value, 0.2, 0.3, 0.4] })
    }

    #[tokio::test]
    async fn legacy_documents_keep_embed_content_task_type_and_format() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/v1beta/models/gemini-embedding-001:embedContent$",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .and(header("content-type", "application/json"))
            .and(body_json(json!({
                "content": { "parts": [{ "text": "hello" }] },
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 4,
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embedding": vector(0.1) })),
            )
            .mount(&server)
            .await;

        let gemini = test_gemini(&server, "gemini-embedding-001", 0).await;
        let output = gemini
            .embed_documents(&[EmbedInput::text("hello")])
            .await
            .unwrap();
        assert_eq!(output, vec![vec![0.1, 0.2, 0.3, 0.4]]);
    }

    #[tokio::test]
    async fn legacy_query_keeps_task_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_json(json!({
                "content": { "parts": [{ "text": "q" }] },
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": 4,
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embedding": vector(1.0) })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, "gemini-embedding-001", 0).await;
        assert_eq!(gemini.embed_query("q").await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn legacy_model_rejects_media_without_request() {
        let server = MockServer::start().await;
        let gemini = test_gemini(&server, "gemini-embedding-001", 0).await;
        let error = gemini
            .embed_documents(&[EmbedInput::Media(vec![MediaPart {
                mime_type: "image/png".into(),
                bytes: vec![1, 2, 3],
            }])])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("does not support media"));
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn embedding_2_query_uses_prefixed_text_and_model_name() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/v1beta/models/gemini-embedding-2:embedContent$",
            ))
            .and(body_json(json!({
                "model": "models/gemini-embedding-2",
                "content": { "parts": [{ "text": "task: search result | query: cats" }] },
                "outputDimensionality": 4,
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embedding": vector(1.0) })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 0).await;
        assert_eq!(gemini.embed_query("cats").await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn embedding_2_batches_text_and_media_in_input_order() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/v1beta/models/gemini-embedding-2:batchEmbedContents$"))
            .and(body_json(json!({ "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": { "parts": [{ "text": "title: none | text: note" }] },
                    "outputDimensionality": 4
                },
                {
                    "model": "models/gemini-embedding-2",
                    "content": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": "AQID" } }] },
                    "outputDimensionality": 4
                }
            ]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": [vector(0.1), vector(0.9)]
            })))
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 0).await;
        let output = gemini
            .embed_documents(&[
                EmbedInput::text("note"),
                EmbedInput::Media(vec![MediaPart {
                    mime_type: "image/png".into(),
                    bytes: vec![1, 2, 3],
                }]),
            ])
            .await
            .unwrap();
        assert_eq!(output[0][0], 0.1);
        assert_eq!(output[1][0], 0.9);
    }

    #[tokio::test]
    async fn embedding_2_splits_document_batches_at_sixteen() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/v1beta/models/gemini-embedding-2:batchEmbedContents$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": (0..16).map(|_| vector(1.0)).collect::<Vec<_>>()
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/v1beta/models/gemini-embedding-2:batchEmbedContents$",
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embeddings": [vector(2.0)] })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 0).await;
        let inputs = (0..17)
            .map(|index| EmbedInput::text(index.to_string()))
            .collect::<Vec<_>>();
        let output = gemini.embed_documents(&inputs).await.unwrap();
        assert_eq!(output.len(), 17);
        // Two batches must have been sent: one of 16, one of 1.
        let reqs = server.received_requests().await.unwrap();
        assert_eq!(
            reqs.len(),
            2,
            "expected exactly 2 batch requests, got {}",
            reqs.len()
        );
        // Batch sizes: first request has 16 items, second has 1.
        let first_count: usize = serde_json::from_slice::<serde_json::Value>(&reqs[0].body)
            .unwrap()["requests"]
            .as_array()
            .unwrap()
            .len();
        let second_count: usize = serde_json::from_slice::<serde_json::Value>(&reqs[1].body)
            .unwrap()["requests"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(first_count, 16, "first batch must have 16 items");
        assert_eq!(second_count, 1, "second batch must have 1 item");
    }

    #[tokio::test]
    async fn embedding_2_retries_a_batch_and_validates_response_count_and_dimension() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(json!({ "error": { "message": "rate limited" } })),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embeddings": [vector(1.0)] })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 1).await;
        assert_eq!(
            gemini
                .embed_documents(&[EmbedInput::text("a")])
                .await
                .unwrap()
                .len(),
            1
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "embeddings": [] })))
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 0).await;
        assert!(
            gemini
                .embed_documents(&[EmbedInput::text("a")])
                .await
                .unwrap_err()
                .to_string()
                .contains("returned 0 embeddings")
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": [{ "values": [1.0, 2.0] }]
            })))
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, GEMINI_EMBEDDING_2, 0).await;
        assert!(matches!(
            gemini.embed_documents(&[EmbedInput::text("a")]).await,
            Err(EmbedError::DimMismatch { got: 2, want: 4 })
        ));
    }

    /// Legacy model (gemini-embedding-001) must retry on 429 and succeed on
    /// the subsequent attempt.
    #[tokio::test]
    async fn legacy_retry_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(json!({ "error": { "message": "rate limited" } })),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embedding": vector(0.5) })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, "gemini-embedding-001", 1).await;
        let out = gemini
            .embed_documents(&[EmbedInput::text("hello")])
            .await
            .expect("should succeed after one 429");
        assert_eq!(out.len(), 1);
    }

    /// Legacy model must retry on 503 and succeed on the subsequent attempt.
    #[tokio::test]
    async fn legacy_retry_on_503() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503)
                    .set_body_json(json!({ "error": { "message": "service unavailable" } })),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embedding": vector(0.5) })),
            )
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, "gemini-embedding-001", 1).await;
        let out = gemini
            .embed_documents(&[EmbedInput::text("hello")])
            .await
            .expect("should succeed after one 503");
        assert_eq!(out.len(), 1);
    }

    /// Legacy model must fail fast on a 4xx other than 429 — no retries.
    #[tokio::test]
    async fn legacy_fail_fast_on_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(json!({ "error": { "message": "bad request" } })),
            )
            .expect(1) // exactly one request — no retries
            .mount(&server)
            .await;
        let gemini = test_gemini(&server, "gemini-embedding-001", 3).await;
        let err = gemini
            .embed_documents(&[EmbedInput::text("hello")])
            .await
            .expect_err("400 must not be retried");
        assert!(
            matches!(err, EmbedError::Api { status: 400, .. }),
            "expected Api{{status:400}}, got: {err:?}"
        );
    }

    /// Config validation (empty api_key, empty model, zero dim) must be
    /// caught before any HTTP request is sent.
    #[tokio::test]
    async fn legacy_config_validation_errors() {
        let server = MockServer::start().await;

        let empty_key = Gemini {
            api_key: String::new(),
            model: "gemini-embedding-001".into(),
            dim: 4,
            base_url: server.uri(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(
            matches!(
                empty_key.embed_documents(&[EmbedInput::text("x")]).await,
                Err(EmbedError::Config(_))
            ),
            "empty api_key must produce Config error"
        );

        let empty_model = Gemini {
            api_key: "k".into(),
            model: String::new(),
            dim: 4,
            base_url: server.uri(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(
            matches!(
                empty_model.embed_documents(&[EmbedInput::text("x")]).await,
                Err(EmbedError::Config(_))
            ),
            "empty model must produce Config error"
        );

        let zero_dim = Gemini {
            api_key: "k".into(),
            model: "gemini-embedding-001".into(),
            dim: 0,
            base_url: server.uri(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(
            matches!(
                zero_dim.embed_documents(&[EmbedInput::text("x")]).await,
                Err(EmbedError::Config(_))
            ),
            "dim=0 must produce Config error"
        );

        // No HTTP requests must have been sent for any of the above.
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            0,
            "config errors must be caught before any HTTP request"
        );
    }
}
