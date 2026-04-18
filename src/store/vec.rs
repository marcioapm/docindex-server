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
}
