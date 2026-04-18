//! Bearer-auth middleware for `/search` and `/similar`.
//!
//! `/health` is deliberately public — liveness probes and the Obsidian
//! plugin's "is the server up?" check should not require a secret. The
//! middleware is added only to the protected router in [`super::build_router`].

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::{AppState, error::ApiError};

/// Tower middleware that 401s any request without a valid bearer token.
pub async fn require_bearer(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    match check(&req, &state.bearer) {
        Ok(()) => next.run(req).await,
        Err(_) => ApiError::Unauthorized.into_response(),
    }
}

fn check(req: &Request<Body>, expected: &str) -> Result<(), StatusCode> {
    let header: &HeaderValue = req
        .headers()
        .get(header::AUTHORIZATION)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    let raw = header.to_str().map_err(|_| StatusCode::UNAUTHORIZED)?;
    let token = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;
    if constant_time_eq(token.as_bytes(), expected.as_bytes()) {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_basic() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }
}
