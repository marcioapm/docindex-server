//! HTTP client for `docindex-search` against a running docindex server.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::search::Hit;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("network: {0}")]
    Network(String),
    #[error("auth: {0}")]
    Auth(String),
    #[error("server: {status}: {message}")]
    Server { status: u16, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub authenticated: bool,
    #[serde(default)]
    pub indexed_chunks: Option<i64>,
    #[serde(default)]
    pub last_reindex_ms: Option<i64>,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<Hit>,
}

pub struct Client {
    base_url: String,
    token: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ClientError::Network(e.to_string()))?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token: token.into(),
            http,
        })
    }

    pub async fn health(&self) -> Result<HealthResponse, ClientError> {
        let resp = self
            .http
            .get(format!("{}/health", self.base_url))
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|e| ClientError::Network(e.to_string()))?;
        handle_response(resp).await
    }

    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        media_only: bool,
        media_types: &[String],
    ) -> Result<SearchResponse, ClientError> {
        let mut body = serde_json::json!({
            "query": query,
            "limit": limit,
            "media_only": media_only
        });
        if !media_types.is_empty() {
            body["media_types"] = serde_json::json!(media_types);
        }
        let resp = self
            .http
            .post(format!("{}/search", self.base_url))
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .map_err(|e| ClientError::Network(e.to_string()))?;
        handle_response(resp).await
    }

    pub async fn similar(&self, path: &str, limit: usize) -> Result<SearchResponse, ClientError> {
        let resp = self
            .http
            .post(format!("{}/similar", self.base_url))
            .bearer_auth(&self.token)
            .json(&serde_json::json!({ "path": path, "limit": limit }))
            .send()
            .await
            .map_err(|e| ClientError::Network(e.to_string()))?;
        handle_response(resp).await
    }
}

async fn handle_response<T: for<'de> Deserialize<'de>>(
    resp: reqwest::Response,
) -> Result<T, ClientError> {
    let status = resp.status();
    if status.is_success() {
        resp.json::<T>()
            .await
            .map_err(|e| ClientError::Network(format!("decode response: {e}")))
    } else if status.as_u16() == 401 || status.as_u16() == 403 {
        let message = error_message(resp).await;
        Err(ClientError::Auth(message))
    } else {
        let message = error_message(resp).await;
        Err(ClientError::Server {
            status: status.as_u16(),
            message,
        })
    }
}

async fn error_message(resp: reqwest::Response) -> String {
    #[derive(Deserialize)]
    struct Body {
        #[serde(default)]
        error: String,
    }
    match resp.json::<Body>().await {
        Ok(b) if !b.error.is_empty() => b.error,
        _ => "request failed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{Json, Router, routing::post};
    use serde_json::{Value, json};
    use tokio::net::TcpListener;

    use super::Client;

    #[tokio::test]
    async fn search_sends_media_only_in_request_body() {
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new().route(
            "/search",
            post({
                let bodies = Arc::clone(&bodies);
                move |Json(body): Json<Value>| {
                    let bodies = Arc::clone(&bodies);
                    async move {
                        bodies.lock().unwrap().push(body);
                        Json(json!({ "hits": [] }))
                    }
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = Client::new(format!("http://{address}"), "token").unwrap();
        let media_response = client
            .search("sunset", 7, true, &["pdf".into()])
            .await
            .unwrap();
        let default_response = client.search("notes", 3, false, &[]).await.unwrap();

        assert!(media_response.hits.is_empty());
        assert!(default_response.hits.is_empty());
        assert_eq!(
            bodies.lock().unwrap().as_slice(),
            [
                json!({ "query": "sunset", "limit": 7, "media_only": true, "media_types": ["pdf"] }),
                json!({ "query": "notes", "limit": 3, "media_only": false }),
            ]
        );
        server.abort();
    }
}
