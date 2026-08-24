//! Deterministic, offline embedder for tests.

use sha2::{Digest, Sha256};

use super::{EmbedError, EmbedInput, Embedder};

pub struct Fake {
    pub dim: usize,
}

impl Fake {
    pub fn new(dim: usize) -> Self {
        debug_assert!(dim > 0, "Fake embedder dim must be positive");
        Self { dim }
    }

    fn vector(&self, seed: &[u8]) -> Vec<f32> {
        let mut out = vec![0f32; self.dim];
        let mut i = 0;
        while i < self.dim {
            let mut hasher = Sha256::new();
            hasher.update(seed);
            hasher.update(i.to_le_bytes());
            let h = hasher.finalize();
            for j in 0..8 {
                if i + j >= self.dim {
                    break;
                }
                let u = (u16::from(h[2 * j]) << 8) | u16::from(h[2 * j + 1]);
                out[i + j] = f32::from(u as i16) / 32768.0;
            }
            i += 8;
        }
        l2_normalize(&mut out);
        out
    }

    fn document_seed(input: &EmbedInput) -> Vec<u8> {
        let mut seed = b"document\0".to_vec();
        match input {
            EmbedInput::Text(text) => {
                seed.extend_from_slice(b"text\0");
                seed.extend_from_slice(text.as_bytes());
            }
            EmbedInput::Media(parts) => {
                seed.extend_from_slice(b"media\0");
                for part in parts {
                    seed.extend_from_slice(&(part.mime_type.len() as u64).to_le_bytes());
                    seed.extend_from_slice(part.mime_type.as_bytes());
                    seed.extend_from_slice(&(part.bytes.len() as u64).to_le_bytes());
                    seed.extend_from_slice(&part.bytes);
                }
            }
        }
        seed
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
    async fn embed_documents(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(inputs
            .iter()
            .map(|input| self.vector(&Self::document_seed(input)))
            .collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.vector(format!("query\0{text}").as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::MediaPart;

    #[tokio::test]
    async fn deterministic_typed_inputs() {
        let f = Fake::new(8);
        let input = EmbedInput::Media(vec![MediaPart {
            mime_type: "image/png".into(),
            bytes: vec![1, 2],
        }]);
        let first = f
            .embed_documents(std::slice::from_ref(&input))
            .await
            .unwrap();
        let second = f.embed_documents(&[input]).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn doc_vs_query_differ() {
        let f = Fake::new(16);
        let q = f.embed_query("x").await.unwrap();
        let d = f.embed_documents(&[EmbedInput::text("x")]).await.unwrap();
        assert_ne!(q, d[0]);
    }

    fn norm_sq(v: &[f32]) -> f64 {
        v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum()
    }

    /// All outputs must be L2-normalised: ||v||² ≈ 1.0.
    #[tokio::test]
    async fn outputs_are_l2_normalised() {
        let f = Fake::new(16);

        let doc_vec = &f
            .embed_documents(&[EmbedInput::text("hello world")])
            .await
            .unwrap()[0];
        assert!(
            (norm_sq(doc_vec) - 1.0).abs() < 1e-5,
            "document embedding must be L2-normalised: ||v||² = {}",
            norm_sq(doc_vec)
        );

        let media_vec = &f
            .embed_documents(&[EmbedInput::Media(vec![MediaPart {
                mime_type: "image/png".into(),
                bytes: vec![1, 2, 3],
            }])])
            .await
            .unwrap()[0];
        assert!(
            (norm_sq(media_vec) - 1.0).abs() < 1e-5,
            "media embedding must be L2-normalised: ||v||² = {}",
            norm_sq(media_vec)
        );

        let query_vec = f.embed_query("test query").await.unwrap();
        assert!(
            (norm_sq(&query_vec) - 1.0).abs() < 1e-5,
            "query embedding must be L2-normalised: ||v||² = {}",
            norm_sq(&query_vec)
        );
    }
}
