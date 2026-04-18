//! Gemini embeddings client.

use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::{EmbedError, Embedder, TASK_RETRIEVAL_DOCUMENT, TASK_RETRIEVAL_QUERY};

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

    async fn embed(&self, text: &str, task_type: &str) -> Result<Vec<f32>, EmbedError> {
        if self.api_key.is_empty() {
            return Err(EmbedError::Config("api_key is empty".into()));
        }
        if self.model.is_empty() {
            return Err(EmbedError::Config("model is empty".into()));
        }
        if self.dim == 0 {
            return Err(EmbedError::Config("dim must be > 0".into()));
        }

        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.base_url, self.model
        );
        let body = EmbedRequest {
            content: EmbedContent {
                parts: vec![EmbedPart {
                    text: text.to_string(),
                }],
            },
            task_type: task_type.to_string(),
            output_dimensionality: self.dim,
        };

        let attempts = self.max_retries.saturating_add(1);
        let mut delay = self.base_delay;
        let mut last_err: EmbedError = EmbedError::Http("no attempts".into());

        for attempt in 0..attempts {
            if attempt > 0 {
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
            let resp = match self
                .client
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = EmbedError::Http(e.to_string());
                    continue;
                }
            };
            let status = resp.status();
            let raw = resp
                .bytes()
                .await
                .map_err(|e| EmbedError::Http(format!("read body: {e}")))?;

            if status.is_success() {
                let parsed: EmbedResponse = serde_json::from_slice(&raw)
                    .map_err(|e| EmbedError::Decode(format!("decode: {e}")))?;
                if parsed.embedding.values.is_empty() {
                    return Err(EmbedError::Decode("empty embedding values".into()));
                }
                return Ok(parsed.embedding.values);
            }

            let message = parse_api_error(&raw).unwrap_or_else(|| {
                String::from_utf8(raw.to_vec()).unwrap_or_else(|_| "<binary body>".into())
            });
            last_err = EmbedError::Api {
                status: status.as_u16(),
                message,
            };
            if !is_retryable(status) {
                return Err(last_err);
            }
        }
        Err(EmbedError::RetriesExhausted(format!("{last_err}")))
    }
}

impl Embedder for Gemini {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for (i, t) in texts.iter().enumerate() {
            let v = self
                .embed(t, TASK_RETRIEVAL_DOCUMENT)
                .await
                .map_err(|e| match e {
                    EmbedError::Api { status, message } => EmbedError::Api {
                        status,
                        message: format!("doc[{i}]: {message}"),
                    },
                    other => other,
                })?;
            out.push(v);
        }
        Ok(out)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed(text, TASK_RETRIEVAL_QUERY).await
    }
}

fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn parse_api_error(raw: &[u8]) -> Option<String> {
    let v: ApiError = serde_json::from_slice(raw).ok()?;
    let msg = v.error.message;
    if msg.is_empty() { None } else { Some(msg) }
}

#[derive(Serialize)]
struct EmbedRequest {
    content: EmbedContent,
    #[serde(rename = "taskType")]
    task_type: String,
    #[serde(rename = "outputDimensionality")]
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<EmbedPart>,
}

#[derive(Serialize)]
struct EmbedPart {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbedValues,
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

    async fn test_gemini(server: &MockServer, max_retries: u32) -> Gemini {
        Gemini {
            api_key: "test-key".into(),
            model: "gemini-embedding-001".into(),
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

    #[tokio::test]
    async fn embed_documents_url_headers_body() {
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
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embedding": { "values": [0.1, 0.2, 0.3, 0.4] }
            })))
            .mount(&server)
            .await;
        let g = test_gemini(&server, 0).await;
        let out = g.embed_documents(&["hello".to_string()]).await.expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![0.1_f32, 0.2, 0.3, 0.4]);
    }

    #[tokio::test]
    async fn embed_query_task_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_json(json!({
                "content": { "parts": [{ "text": "q" }] },
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embedding": { "values": [1.0, 0.0, 0.0, 0.0] }
            })))
            .mount(&server)
            .await;
        let g = test_gemini(&server, 0).await;
        let v = g.embed_query("q").await.unwrap();
        assert_eq!(v.len(), 4);
    }

    #[tokio::test]
    async fn retry_on_429() {
        let server = MockServer::start().await;
        // Two 429s, then a 200.
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": { "code": 429, "message": "rate limited", "status": "ERROR" }
            })))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embedding": { "values": [0.5, 0.5, 0.5, 0.5] }
            })))
            .mount(&server)
            .await;
        let g = test_gemini(&server, 3).await;
        let v = g.embed_query("q").await.expect("success after retry");
        assert_eq!(v.len(), 4);
    }

    #[tokio::test]
    async fn retry_on_503() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embedding": { "values": [1.0, 2.0, 3.0, 4.0] }
            })))
            .mount(&server)
            .await;
        let g = test_gemini(&server, 2).await;
        assert!(g.embed_query("q").await.is_ok());
    }

    #[tokio::test]
    async fn fail_fast_on_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": { "code": 400, "message": "bad", "status": "INVALID_ARGUMENT" }
            })))
            .expect(1)
            .mount(&server)
            .await;
        let g = test_gemini(&server, 5).await;
        let err = g.embed_query("q").await.expect_err("should fail");
        matches!(err, EmbedError::Api { status: 400, .. });
    }

    #[tokio::test]
    async fn validation_errors() {
        let g = Gemini {
            api_key: String::new(),
            model: "m".into(),
            dim: 4,
            base_url: "http://x".into(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(matches!(
            g.embed_query("q").await,
            Err(EmbedError::Config(_))
        ));

        let g = Gemini {
            api_key: "k".into(),
            model: String::new(),
            dim: 4,
            base_url: "http://x".into(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(matches!(
            g.embed_query("q").await,
            Err(EmbedError::Config(_))
        ));

        let g = Gemini {
            api_key: "k".into(),
            model: "m".into(),
            dim: 0,
            base_url: "http://x".into(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(matches!(
            g.embed_query("q").await,
            Err(EmbedError::Config(_))
        ));
    }
}
