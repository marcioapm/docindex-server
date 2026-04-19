//! Deterministic, offline embedder for tests.
//!
//! Vectors are derived from sha256(text) and L2-normalized so cosine
//! distance behaves. Doc and query vectors for the same text differ so
//! tests can detect task-type mismatches.

use sha2::{Digest, Sha256};

use super::{EmbedError, Embedder};

pub struct Fake {
    pub dim: usize,
}

impl Fake {
    pub fn new(dim: usize) -> Self {
        // Caller is responsible for dim > 0 (config enforces it).
        Self { dim }
    }

    fn vector(&self, seed: &str) -> Vec<f32> {
        let mut out = vec![0f32; self.dim];
        // Fill by hashing (seed, counter) blocks.
        let mut i = 0;
        while i < self.dim {
            let chunk = format!("{seed}:{i}");
            let mut hasher = Sha256::new();
            hasher.update(chunk.as_bytes());
            let h = hasher.finalize();
            for j in 0..8 {
                if i + j >= self.dim {
                    break;
                }
                // Two bytes per float -> i16 -> map to [-1, 1].
                let u = (u16::from(h[2 * j]) << 8) | u16::from(h[2 * j + 1]);
                out[i + j] = f32::from(u as i16) / 32768.0;
            }
            i += 8;
        }
        l2_normalize(&mut out);
        out
    }
}

fn l2_normalize(v: &mut [f32]) {
    let sum: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    if sum == 0.0 {
        return;
    }
    let norm = sum.sqrt() as f32;
    for x in v.iter_mut() {
        *x /= norm;
    }
}

impl Embedder for Fake {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|t| self.vector(&format!("{t}|doc")))
            .collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.vector(&format!("{text}|query")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic() {
        let f = Fake::new(8);
        let a = f.embed_query("hello").await.unwrap();
        let b = f.embed_query("hello").await.unwrap();
        assert_eq!(a.len(), 8);
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn doc_vs_query_differ() {
        let f = Fake::new(16);
        let q = f.embed_query("x").await.unwrap();
        let d = f.embed_documents(&["x".to_string()]).await.unwrap();
        assert_ne!(q, d[0]);
    }

    #[tokio::test]
    async fn normalized() {
        let f = Fake::new(32);
        let v = f.embed_query("abc").await.unwrap();
        let sum: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        assert!((0.99..=1.01).contains(&sum), "||v||^2 = {sum}");
    }
}
