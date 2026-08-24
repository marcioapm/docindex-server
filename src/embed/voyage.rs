//! Voyage AI embeddings client.
//!
//! Mirrors the Gemini client's retry/backoff and `EmbedError` mapping.
//! Voyage's free-tier limits are per-minute; `Retry-After` (when present)
//! is honored verbatim, otherwise backoff starts at 1s (not the Gemini
//! client's 200ms) so a burst of 429s doesn't exhaust retries before the
//! window resets.

use std::time::Duration;

use base64::Engine;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::{EmbedError, EmbedInput, Embedder, MediaPart};

/// Max inputs per Voyage text embeddings request. Larger batches are chunked.
pub const MAX_BATCH: usize = 128;
/// Max content items per Voyage multimodal embeddings request.
const MAX_MULTIMODAL_BATCH: usize = 4;
const VOYAGE_MULTIMODAL_MODEL: &str = "voyage-multimodal-3.5";

/// Upper bound on the `Retry-After` sleep duration. Prevents a hostile or
/// misconfigured endpoint from stalling indexing indefinitely via a crafted
/// header.
const RETRY_AFTER_MAX: Duration = Duration::from_secs(60);

/// Embedder backed by Voyage AI's REST API.
pub struct Voyage {
    pub api_key: String,
    pub model: String,
    pub dim: usize,
    pub base_url: String,
    pub client: reqwest::Client,
    pub max_retries: u32,
    pub base_delay: Duration,
}

impl Voyage {
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
            base_url: "https://api.voyageai.com".to_string(),
            client,
            max_retries: 5,
            base_delay: Duration::from_secs(1),
        })
    }

    async fn embed_text_batch(
        &self,
        texts: &[String],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
            input_type: input_type.to_string(),
            output_dimension: self.dim,
        };
        self.send_request(&url, &body, texts.len()).await
    }

    async fn embed_text(
        &self,
        texts: &[String],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.len() <= MAX_BATCH {
            return self.embed_text_batch(texts, input_type).await;
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            out.extend(self.embed_text_batch(chunk, input_type).await?);
        }
        Ok(out)
    }

    async fn embed_multimodal_batch(
        &self,
        inputs: &[EmbedInput],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/multimodalembeddings", self.base_url);
        let body = MultimodalEmbedRequest {
            model: self.model.clone(),
            inputs: inputs.iter().map(multimodal_input).collect(),
            input_type: input_type.to_string(),
            truncation: false,
            output_dimension: self.dim,
        };
        self.send_request(&url, &body, inputs.len()).await
    }

    async fn embed_multimodal(
        &self,
        inputs: &[EmbedInput],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(MAX_MULTIMODAL_BATCH) {
            out.extend(self.embed_multimodal_batch(batch, input_type).await?);
        }
        Ok(out)
    }

    async fn send_request<T: Serialize>(
        &self,
        url: &str,
        body: &T,
        input_count: usize,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
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
                .post(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(body)
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
            let retry_after = retry_after_delay(&resp);
            let raw = resp
                .bytes()
                .await
                .map_err(|e| EmbedError::Http(format!("read body: {e}")))?;

            if status.is_success() {
                return decode_embeddings(&raw, input_count, self.dim);
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
            if status == StatusCode::TOO_MANY_REQUESTS {
                delay = effective_retry_delay(retry_after, delay);
            }
        }
        Err(EmbedError::RetriesExhausted(format!("{last_err}")))
    }

    /// Validate struct-level invariants once per logical request. Cheaper
    /// than checking inside the inner per-batch loop, and ensures the error
    /// is surfaced before any HTTP traffic occurs.
    fn validate_config(&self) -> Result<(), EmbedError> {
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

    pub async fn embed_documents(
        &self,
        inputs: &[EmbedInput],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.validate_config()?;
        if self.is_multimodal_model() {
            self.embed_multimodal(inputs, "document").await
        } else {
            self.embed_text(&text_inputs(inputs)?, "document").await
        }
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.validate_config()?;
        let out = if self.is_multimodal_model() {
            self.embed_multimodal(&[EmbedInput::text(text)], "query")
                .await?
        } else {
            self.embed_text(&[text.to_string()], "query").await?
        };
        out.into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("empty response for single query".into()))
    }

    fn is_multimodal_model(&self) -> bool {
        self.model == VOYAGE_MULTIMODAL_MODEL
    }
}

impl Embedder for Voyage {
    async fn embed_documents(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Voyage::embed_documents(self, inputs).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Voyage::embed_query(self, text).await
    }
}

fn text_inputs(inputs: &[EmbedInput]) -> Result<Vec<String>, EmbedError> {
    inputs
        .iter()
        .map(|input| match input {
            EmbedInput::Text(text) => Ok(text.clone()),
            EmbedInput::Media(_) => Err(EmbedError::Config(
                "Voyage text embedding models do not support media inputs".into(),
            )),
        })
        .collect()
}

fn multimodal_input(input: &EmbedInput) -> MultimodalInput {
    let content = match input {
        EmbedInput::Text(text) => vec![MultimodalContent::Text { text: text.clone() }],
        EmbedInput::Media(parts) => parts.iter().map(media_content).collect(),
    };
    MultimodalInput { content }
}

fn media_content(part: &MediaPart) -> MultimodalContent {
    let data = base64::engine::general_purpose::STANDARD.encode(&part.bytes);
    MultimodalContent::ImageBase64 {
        image_base64: format!("data:{};base64,{data}", part.mime_type),
    }
}

fn decode_embeddings(
    raw: &[u8],
    input_count: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let parsed: EmbedResponse =
        serde_json::from_slice(raw).map_err(|e| EmbedError::Decode(format!("decode: {e}")))?;
    let mut data = parsed.data;
    data.sort_by_key(|d| d.index);
    if data.len() != input_count {
        return Err(EmbedError::Decode(format!(
            "response has {} embeddings for {input_count} inputs",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len());
    for datum in data {
        if datum.embedding.len() != dim {
            return Err(EmbedError::DimMismatch {
                got: datum.embedding.len(),
                want: dim,
            });
        }
        out.push(datum.embedding);
    }
    Ok(out)
}

/// Extract `Retry-After` from a response, if present and a valid integer
/// number of seconds (the only form Voyage documents).
fn retry_after_delay(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Choose the sleep duration for the next retry attempt.
///
/// When the server supplies a `Retry-After` header (`retry_after = Some(ra)`),
/// use `ra` clamped to `RETRY_AFTER_MAX` so a hostile header can't stall
/// indexing indefinitely. When no header is present, use the exponential
/// `backoff` accumulated by the caller.
pub(crate) fn effective_retry_delay(retry_after: Option<Duration>, backoff: Duration) -> Duration {
    match retry_after {
        Some(ra) => ra.min(RETRY_AFTER_MAX),
        None => backoff,
    }
}

fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn parse_api_error(raw: &[u8]) -> Option<String> {
    let v: ApiError = serde_json::from_slice(raw).ok()?;
    let msg = v.detail.or(v.error);
    match msg {
        Some(m) if !m.is_empty() => Some(m),
        _ => None,
    }
}

#[derive(Serialize, Deserialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
    input_type: String,
    output_dimension: usize,
}

#[derive(Serialize)]
struct MultimodalEmbedRequest {
    model: String,
    inputs: Vec<MultimodalInput>,
    input_type: String,
    truncation: bool,
    output_dimension: usize,
}

#[derive(Serialize)]
struct MultimodalInput {
    content: Vec<MultimodalContent>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MultimodalContent {
    Text { text: String },
    ImageBase64 { image_base64: String },
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize, Default)]
struct ApiError {
    #[serde(default)]
    detail: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::MediaPart;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_voyage(server: &MockServer, max_retries: u32, base_delay: Duration) -> Voyage {
        Voyage {
            api_key: "test-key".into(),
            model: "voyage-4".into(),
            dim: 4,
            base_url: server.uri(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            max_retries,
            base_delay,
        }
    }

    #[tokio::test]
    async fn embed_documents_request_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key"))
            .and(header("content-type", "application/json"))
            .and(body_json(json!({
                "model": "voyage-4",
                "input": ["hello"],
                "input_type": "document",
                "output_dimension": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [0.1, 0.2, 0.3, 0.4], "index": 0 }]
            })))
            .mount(&server)
            .await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let out = v
            .embed_documents(&[EmbedInput::text("hello")])
            .await
            .expect("ok");
        assert_eq!(out, vec![vec![0.1_f32, 0.2, 0.3, 0.4]]);
    }

    #[tokio::test]
    async fn embed_query_uses_query_input_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_json(json!({
                "model": "voyage-4",
                "input": ["q"],
                "input_type": "query",
                "output_dimension": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0], "index": 0 }]
            })))
            .mount(&server)
            .await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let out = v.embed_query("q").await.unwrap();
        assert_eq!(out, vec![1.0_f32, 0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn multimodal_documents_serialize_text_and_media() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/multimodalembeddings"))
            .and(header("Authorization", "Bearer test-key"))
            .and(body_json(json!({
                "model": "voyage-multimodal-3.5",
                "inputs": [
                    { "content": [{ "type": "text", "text": "hello" }] },
                    { "content": [{ "type": "image_base64", "image_base64": "data:image/png;base64,AQI=" }] },
                ],
                "input_type": "document",
                "truncation": false,
                "output_dimension": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "embedding": [2.0, 0.0, 0.0, 0.0], "index": 1 },
                    { "embedding": [1.0, 0.0, 0.0, 0.0], "index": 0 },
                ]
            })))
            .mount(&server)
            .await;
        let mut v = test_voyage(&server, 0, Duration::from_millis(1));
        v.model = VOYAGE_MULTIMODAL_MODEL.into();
        let out = v
            .embed_documents(&[
                EmbedInput::text("hello"),
                EmbedInput::Media(vec![MediaPart {
                    mime_type: "image/png".into(),
                    bytes: vec![1, 2],
                }]),
            ])
            .await
            .expect("multimodal document embeddings");
        assert_eq!(out[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert_eq!(out[1], vec![2.0_f32, 0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn multimodal_query_uses_text_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/multimodalembeddings"))
            .and(body_json(json!({
                "model": "voyage-multimodal-3.5",
                "inputs": [{ "content": [{ "type": "text", "text": "q" }] }],
                "input_type": "query",
                "truncation": false,
                "output_dimension": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0], "index": 0 }]
            })))
            .mount(&server)
            .await;
        let mut v = test_voyage(&server, 0, Duration::from_millis(1));
        v.model = VOYAGE_MULTIMODAL_MODEL.into();
        assert_eq!(
            v.embed_query("q").await.unwrap(),
            vec![1.0_f32, 0.0, 0.0, 0.0]
        );
    }

    #[tokio::test]
    async fn multimodal_documents_batch_sequentially_in_groups_of_four() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/multimodalembeddings"))
            .and(body_multimodal_input_len(4))
            .respond_with(multimodal_response(1.0))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/multimodalembeddings"))
            .and(body_multimodal_input_len(1))
            .respond_with(multimodal_response(2.0))
            .expect(1)
            .mount(&server)
            .await;
        let mut v = test_voyage(&server, 0, Duration::from_millis(1));
        v.model = VOYAGE_MULTIMODAL_MODEL.into();
        let inputs: Vec<_> = (0..5)
            .map(|i| EmbedInput::text(format!("text-{i}")))
            .collect();
        let out = v.embed_documents(&inputs).await.unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert_eq!(out[4], vec![2.0_f32, 0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn text_models_reject_media_without_http_request() {
        let server = MockServer::start().await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let err = v
            .embed_documents(&[EmbedInput::Media(vec![MediaPart {
                mime_type: "image/png".into(),
                bytes: vec![1, 2],
            }])])
            .await
            .expect_err("media must be rejected by text-only Voyage models");
        assert!(matches!(err, EmbedError::Config(_)));
    }

    fn multimodal_response(value: f32) -> impl wiremock::Respond {
        move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            let count = body["inputs"].as_array().unwrap().len();
            let data: Vec<_> = (0..count)
                .map(|index| json!({ "embedding": [value, 0.0, 0.0, 0.0], "index": index }))
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({ "data": data }))
        }
    }

    fn body_multimodal_input_len(n: usize) -> impl wiremock::Match {
        struct LenMatch(usize);
        impl wiremock::Match for LenMatch {
            fn matches(&self, req: &wiremock::Request) -> bool {
                serde_json::from_slice::<serde_json::Value>(&req.body)
                    .ok()
                    .and_then(|body| body["inputs"].as_array().map(Vec::len))
                    == Some(self.0)
            }
        }
        LenMatch(n)
    }
    #[tokio::test]
    async fn out_of_order_index_is_reordered() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "embedding": [2.0, 0.0, 0.0, 0.0], "index": 1 },
                    { "embedding": [1.0, 0.0, 0.0, 0.0], "index": 0 },
                ]
            })))
            .mount(&server)
            .await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let out = v
            .embed_documents(&[EmbedInput::text("a"), EmbedInput::text("b")])
            .await
            .unwrap();
        assert_eq!(out[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert_eq!(out[1], vec![2.0_f32, 0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn retry_429_honours_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "0")
                    .set_body_json(json!({"error": "rate limited"})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [0.5, 0.5, 0.5, 0.5], "index": 0 }]
            })))
            .mount(&server)
            .await;
        // Large base_delay proves Retry-After (0s) was used instead — the
        // request must still complete quickly.
        let v = test_voyage(&server, 2, Duration::from_secs(30));
        let start = std::time::Instant::now();
        let out = v.embed_query("q").await.expect("success after retry");
        assert_eq!(out.len(), 4);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "Retry-After should have short-circuited the 30s base delay"
        );
    }

    #[tokio::test]
    async fn retry_429_without_retry_after_uses_backoff() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(json!({"error": "rate limited"})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [0.5, 0.5, 0.5, 0.5], "index": 0 }]
            })))
            .mount(&server)
            .await;
        let v = test_voyage(&server, 2, Duration::from_millis(5));
        let out = v.embed_query("q").await.expect("success after retry");
        assert_eq!(out.len(), 4);
    }

    #[tokio::test]
    async fn batch_chunking_over_128_inputs() {
        let server = MockServer::start().await;
        // First batch of 128, then remainder of 5.
        Mock::given(method("POST"))
            .and(body_json_partial_len(128))
            .respond_with(move |req: &wiremock::Request| {
                let body: EmbedRequest = serde_json::from_slice(&req.body).unwrap();
                let data: Vec<_> = body
                    .input
                    .iter()
                    .enumerate()
                    .map(|(i, _)| json!({ "embedding": [1.0, 0.0, 0.0, 0.0], "index": i }))
                    .collect();
                ResponseTemplate::new(200).set_body_json(json!({ "data": data }))
            })
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_json_partial_len(5))
            .respond_with(move |req: &wiremock::Request| {
                let body: EmbedRequest = serde_json::from_slice(&req.body).unwrap();
                let data: Vec<_> = body
                    .input
                    .iter()
                    .enumerate()
                    .map(|(i, _)| json!({ "embedding": [2.0, 0.0, 0.0, 0.0], "index": i }))
                    .collect();
                ResponseTemplate::new(200).set_body_json(json!({ "data": data }))
            })
            .mount(&server)
            .await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let inputs: Vec<_> = (0..133)
            .map(|i| EmbedInput::text(format!("text-{i}")))
            .collect();
        let out = v.embed_documents(&inputs).await.expect("ok");
        assert_eq!(out.len(), 133);
        assert_eq!(out[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert_eq!(out[132], vec![2.0_f32, 0.0, 0.0, 0.0]);
    }

    fn body_json_partial_len(n: usize) -> impl wiremock::Match {
        struct LenMatch(usize);
        impl wiremock::Match for LenMatch {
            fn matches(&self, req: &wiremock::Request) -> bool {
                serde_json::from_slice::<EmbedRequest>(&req.body)
                    .map(|b| b.input.len() == self.0)
                    .unwrap_or(false)
            }
        }
        LenMatch(n)
    }

    #[tokio::test]
    async fn fail_fast_on_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": "bad request"})))
            .expect(1)
            .mount(&server)
            .await;
        let v = test_voyage(&server, 5, Duration::from_millis(1));
        let err = v.embed_query("q").await.expect_err("should fail");
        assert!(matches!(err, EmbedError::Api { status: 400, .. }));
    }

    /// `effective_retry_delay` must clamp a large `Retry-After` to
    /// `RETRY_AFTER_MAX`. This test proves the clamp is load-bearing:
    /// deleting the `.min(RETRY_AFTER_MAX)` inside `effective_retry_delay`
    /// would return `Duration::from_secs(100_000)` instead of
    /// `RETRY_AFTER_MAX`, failing the assertion below.
    #[test]
    fn retry_after_capped_at_max_by_pure_fn() {
        // Value far exceeding RETRY_AFTER_MAX — without the clamp the fn
        // returns this verbatim, breaking the assertion.
        let large = Duration::from_secs(100_000);
        assert_eq!(
            effective_retry_delay(Some(large), Duration::from_secs(1)),
            RETRY_AFTER_MAX,
            "retry_after > RETRY_AFTER_MAX must be clamped to RETRY_AFTER_MAX"
        );
        // Below-cap value is passed through unchanged.
        assert_eq!(
            effective_retry_delay(Some(Duration::from_secs(5)), Duration::from_secs(1)),
            Duration::from_secs(5),
        );
        // No Retry-After header: falls back to the supplied backoff delay.
        assert_eq!(
            effective_retry_delay(None, Duration::from_secs(2)),
            Duration::from_secs(2),
        );
    }

    /// Constant-sanity check: `RETRY_AFTER_MAX` is spec'd at 60 s.
    #[test]
    fn retry_after_max_constant_is_60s() {
        assert_eq!(RETRY_AFTER_MAX, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn validation_errors() {
        let v = Voyage {
            api_key: String::new(),
            model: "m".into(),
            dim: 4,
            base_url: "http://x".into(),
            client: reqwest::Client::new(),
            max_retries: 0,
            base_delay: Duration::from_millis(1),
        };
        assert!(matches!(
            v.embed_query("q").await,
            Err(EmbedError::Config(_))
        ));
    }

    /// A response whose embedding length differs from the configured dim must
    /// produce `EmbedError::DimMismatch`.
    #[tokio::test]
    async fn dim_mismatch_error_on_wrong_embedding_length() {
        let server = MockServer::start().await;
        // Client dim = 4, but the response returns a 3-element vector.
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [0.1, 0.2, 0.3], "index": 0 }]
            })))
            .mount(&server)
            .await;
        let v = test_voyage(&server, 0, Duration::from_millis(1));
        let err = v
            .embed_query("q")
            .await
            .expect_err("should fail on dim mismatch");
        assert!(
            matches!(err, EmbedError::DimMismatch { got: 3, want: 4 }),
            "expected DimMismatch{{got:3, want:4}}, got: {err:?}"
        );
    }

    /// When every attempt returns 429, `embed_query` must exhaust all retries
    /// and return `EmbedError::RetriesExhausted`.
    #[tokio::test]
    async fn retries_exhausted_on_persistent_429() {
        let server = MockServer::start().await;
        // No `up_to_n_times` — replies 429 to every request.
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(json!({"error": "rate limited"})),
            )
            .mount(&server)
            .await;
        // max_retries = 1 → 2 total attempts, both fail.
        let v = test_voyage(&server, 1, Duration::from_millis(0));
        let err = v
            .embed_query("q")
            .await
            .expect_err("should exhaust retries");
        assert!(
            matches!(err, EmbedError::RetriesExhausted(_)),
            "expected RetriesExhausted, got: {err:?}"
        );
    }

    /// A single 503 followed by 200 must succeed: `is_retryable` must return
    /// true for server errors, and the retry must use exponential backoff
    /// (not the Retry-After branch, which is 429-only).
    #[tokio::test]
    async fn retry_5xx_uses_backoff() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503).set_body_json(json!({"detail": "service unavailable"})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": [0.1, 0.2, 0.3, 0.4], "index": 0 }]
            })))
            .mount(&server)
            .await;
        // max_retries=1, tiny base_delay — must succeed on the second attempt.
        let v = test_voyage(&server, 1, Duration::from_millis(1));
        let out = v
            .embed_query("q")
            .await
            .expect("should succeed after one 503");
        assert_eq!(out.len(), 4, "expected 4-element embedding");
    }

    /// When every attempt returns 503, all retries are exhausted and
    /// `EmbedError::RetriesExhausted` is returned. Proves `is_retryable`
    /// accepts 5xx and that the retry loop does not give up early.
    #[tokio::test]
    async fn retries_exhausted_on_persistent_5xx() {
        let server = MockServer::start().await;
        // No `up_to_n_times` — replies 503 to every request.
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503).set_body_json(json!({"detail": "service unavailable"})),
            )
            .mount(&server)
            .await;
        // max_retries=1 → 2 total attempts, both fail.
        let v = test_voyage(&server, 1, Duration::from_millis(0));
        let err = v
            .embed_query("q")
            .await
            .expect_err("should exhaust retries on persistent 503");
        assert!(
            matches!(err, EmbedError::RetriesExhausted(_)),
            "expected RetriesExhausted, got: {err:?}"
        );
    }
}
