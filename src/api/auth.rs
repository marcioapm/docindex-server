//! Bearer-auth middleware and validation helpers.
//!
//! The middleware protects `/search` and `/similar`; `/health` evaluates the
//! same bearer credentials optionally so liveness probes remain public.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, header},
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
    if has_valid_bearer(req.headers(), &state.bearer) {
        next.run(req).await
    } else {
        ApiError::Unauthorized.into_response()
    }
}

/// Return whether the request carries the configured bearer token.
pub fn has_valid_bearer(headers: &HeaderMap, expected: &str) -> bool {
    let Some(header) = headers.get(header::AUTHORIZATION) else {
        return false;
    };
    let Ok(raw) = header.to_str() else {
        return false;
    };
    let Some(token) = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
    else {
        return false;
    };
    constant_time_eq(token.as_bytes(), expected.as_bytes())
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
