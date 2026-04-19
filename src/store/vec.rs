//! Little-endian float32 (de)serialization.
//!
//! This is the exact wire format `sqlite-vec` consumes for `vec_distance_*`
//! and the `vec0` virtual table. Keep these helpers together so the round
//! trip is provable from a single file.

/// Pack a slice of f32 into little-endian bytes.
pub fn encode_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a little-endian float32 blob. Length must be divisible by 4.
pub fn decode_f32(b: &[u8]) -> Result<Vec<f32>, String> {
    if !b.len().is_multiple_of(4) {
        return Err(format!(
            "float32 blob length {} not divisible by 4",
            b.len()
        ));
    }
    let n = b.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = &b[i * 4..(i + 1) * 4];
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Render the DDL for the `chunks_vec` virtual table with `embed_dim` baked
/// in. `vec0` requires the dimension as a SQL literal, so we can't template
/// it from a bound parameter — we have to splice it into the text. Wrapped
/// in `IF NOT EXISTS` so opening an existing DB is idempotent; the
/// schema/config match is enforced separately via `meta.embedding_dim`.
pub fn vec_schema_ddl(embed_dim: usize) -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding FLOAT[{embed_dim}] distance_metric=cosine)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let v = vec![0.0_f32, 1.5, -3.25, f32::MIN_POSITIVE, f32::MAX];
        let b = encode_f32(&v);
        assert_eq!(b.len(), v.len() * 4);
        assert_eq!(decode_f32(&b).unwrap(), v);
    }

    #[test]
    fn bad_length() {
        assert!(decode_f32(&[0u8; 3]).is_err());
    }

    #[test]
    fn vec_schema_ddl_embeds_dim() {
        let ddl = vec_schema_ddl(3072);
        assert!(
            ddl.contains("FLOAT[3072]"),
            "expected FLOAT[3072] literal in DDL: {ddl}"
        );
        assert!(ddl.contains("distance_metric=cosine"));
        assert!(ddl.contains("IF NOT EXISTS"));
    }

    #[test]
    fn vec_schema_ddl_tracks_dim() {
        assert!(vec_schema_ddl(8).contains("FLOAT[8]"));
        assert!(vec_schema_ddl(768).contains("FLOAT[768]"));
    }
}
