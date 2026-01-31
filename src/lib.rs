//! Rust port of ggwave's AUDIBLE_FAST codec.
//!
//! Encode text to 48 kHz f32 audio and decode f32 audio back to text.
//!
//! # Example
//!
//! ```
//! let audio = ggwave::encode(b"hello", 25).unwrap();
//! let mut decoder = ggwave::Decoder::new();
//! let payload = decoder.decode(&audio).unwrap().unwrap();
//! assert_eq!(&payload, b"hello");
//! ```

pub mod protocol;
pub mod reed_solomon;
pub mod dss;
pub mod fft;
pub mod encoder;
pub mod decoder;

pub use decoder::Decoder;

/// Encode a payload into f32 audio samples using the AUDIBLE_FAST protocol.
///
/// `payload`: the bytes to transmit (max 140 bytes).
/// `volume`: output volume, 0..=100 (typically 10-50).
///
/// Returns a `Vec<f32>` of audio samples at 48 kHz.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, Error> {
    encoder::encode(payload, volume)
}

/// Errors returned by encode/decode operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("payload is empty")]
    EmptyPayload,

    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("invalid volume: {0} (must be 0..=100)")]
    InvalidVolume(u8),

    #[error("decode failed: could not extract valid payload from audio")]
    DecodeFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Parameter validation (ported from test-ggwave.cpp lines 180-201) ---

    #[test]
    fn test_encode_empty_payload() {
        assert!(matches!(encode(b"", 25), Err(Error::EmptyPayload)));
    }

    #[test]
    fn test_encode_too_large() {
        let big = vec![0u8; 141];
        assert!(matches!(
            encode(&big, 25),
            Err(Error::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn test_encode_volume_bounds() {
        // volume 0 should succeed
        assert!(encode(b"hello", 0).is_ok());
        // volume 100 should succeed
        assert!(encode(b"hello", 100).is_ok());
        // volume 101 should fail
        assert!(matches!(
            encode(b"hello", 101),
            Err(Error::InvalidVolume(101))
        ));
        // volume 50 should succeed
        assert!(encode(b"hello", 50).is_ok());
    }

    #[test]
    fn test_encode_various_sizes() {
        // Valid sizes 1..=3 with small payload (mirrors C++ init checks)
        for size in 1..=3 {
            let payload = &b"asd"[..size];
            assert!(encode(payload, 25).is_ok(), "encode failed for size {size}");
        }
    }

    // --- Encode size prediction (ported from test-ggwave.cpp lines 275-278) ---

    #[test]
    fn test_encode_size_prediction() {
        // The number of output samples should match the formula:
        // (N_MARKER_FRAMES + total_data_frames + N_MARKER_FRAMES) * SAMPLES_PER_FRAME
        let payload = b"a0Z5kR2g";
        let audio = encode(payload, 25).unwrap();

        let data_length = payload.len();
        let n_ecc = protocol::ecc_bytes_for_length(data_length);
        let total_bytes = protocol::ENCODED_DATA_OFFSET + data_length + n_ecc;
        let total_data_frames = protocol::EXTRA
            * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
            * protocol::FRAMES_PER_TX;
        let expected_samples = (protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES)
            * protocol::SAMPLES_PER_FRAME;

        assert_eq!(
            audio.len(),
            expected_samples,
            "encode size mismatch: got {} expected {}",
            audio.len(),
            expected_samples
        );
    }

    // --- Variable-length round-trip for every length 1..8 ---
    // (ported from test-ggwave.cpp lines 257-289)

    #[test]
    fn test_roundtrip_incremental_lengths() {
        let payload_full = b"a0Z5kR2g";
        for length in 1..=payload_full.len() {
            let payload = &payload_full[..length];
            let audio = encode(payload, 25).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "round-trip failed for length {length}"
            );
        }
    }

    // --- Noise robustness (ported from test-ggwave.cpp lines 280-281) ---

    #[test]
    fn test_roundtrip_with_noise() {
        let payload = b"a0Z5kR2g";
        let mut audio = encode(payload, 25).unwrap();

        // Add noise at 0.02 level (matching C++ variable-length test)
        let mut rng_state: u32 = 12345;
        for sample in audio.iter_mut() {
            // Simple LCG PRNG for deterministic noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let frand = (rng_state >> 16) as f32 / 65535.0;
            *sample += (frand - 0.5) * 0.02;
            *sample = sample.clamp(-1.0, 1.0);
        }

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Single byte (edge case: smallest payload) ---

    #[test]
    fn test_roundtrip_single_byte() {
        let payload = b"A";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- "hello" basic round-trip ---

    #[test]
    fn test_roundtrip_hello() {
        let payload = b"hello";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Medium payload ---

    #[test]
    fn test_roundtrip_medium() {
        let payload = b"The quick brown fox jumps over the lazy dog!12345";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Max length (140 bytes) ---

    #[test]
    fn test_roundtrip_max_length() {
        let payload: Vec<u8> = (0..140).map(|i| (i % 256) as u8).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Binary payload (non-ASCII, all byte values) ---

    #[test]
    fn test_roundtrip_binary_payload() {
        // Payload with bytes 0x00..0x1F — exercises low nibble values
        let payload: Vec<u8> = (0..32).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Streaming decode (feed in sub-frame chunks) ---

    #[test]
    fn test_roundtrip_streaming() {
        let payload = b"stream test";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 512; // half a frame
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Streaming with very small chunks ---

    #[test]
    fn test_roundtrip_streaming_tiny_chunks() {
        let payload = b"tiny";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 100; // much less than a frame
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decoder reset ---

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Decoder::new();
        let payload = b"test";
        let audio = encode(payload, 50).unwrap();
        let _ = decoder.decode(&audio);
        decoder.reset();
        // After reset, should be able to decode again
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Volume 0 produces silent but decodable output ---

    #[test]
    fn test_encode_volume_zero_produces_silence() {
        let audio = encode(b"hi", 0).unwrap();
        // All samples should be exactly 0.0
        assert!(audio.iter().all(|&s| s == 0.0));
    }

    // --- ECC bytes calculation matches C++ ---

    #[test]
    fn test_ecc_bytes_for_length() {
        // len < 4 => 2
        assert_eq!(protocol::ecc_bytes_for_length(1), 2);
        assert_eq!(protocol::ecc_bytes_for_length(2), 2);
        assert_eq!(protocol::ecc_bytes_for_length(3), 2);
        // len = 4 => max(4, 2*(4/5)) = max(4, 0) = 4
        assert_eq!(protocol::ecc_bytes_for_length(4), 4);
        // len = 5 => max(4, 2*(5/5)) = max(4, 2) = 4
        assert_eq!(protocol::ecc_bytes_for_length(5), 4);
        // len = 10 => max(4, 2*(10/5)) = max(4, 4) = 4
        assert_eq!(protocol::ecc_bytes_for_length(10), 4);
        // len = 15 => max(4, 2*(15/5)) = max(4, 6) = 6
        assert_eq!(protocol::ecc_bytes_for_length(15), 6);
        // len = 50 => max(4, 2*(50/5)) = max(4, 20) = 20
        assert_eq!(protocol::ecc_bytes_for_length(50), 20);
        // len = 140 => max(4, 2*(140/5)) = max(4, 56) = 56
        assert_eq!(protocol::ecc_bytes_for_length(140), 56);
    }

    // --- Noise robustness at higher noise level ---

    #[test]
    fn test_roundtrip_with_moderate_noise() {
        // Test with each length 1..4 at slightly higher noise to stress RS correction
        let payload_full = b"a0Z5";
        for length in 1..=payload_full.len() {
            let payload = &payload_full[..length];
            let mut audio = encode(payload, 50).unwrap();

            let mut rng_state: u32 = 42 + length as u32;
            for sample in audio.iter_mut() {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let frand = (rng_state >> 16) as f32 / 65535.0;
                *sample += (frand - 0.5) * 0.01;
                *sample = sample.clamp(-1.0, 1.0);
            }

            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "noisy round-trip failed for length {length}"
            );
        }
    }

    // --- ECC boundary lengths (stress RS code word boundaries) ---

    #[test]
    fn test_roundtrip_ecc_boundaries() {
        // These lengths trigger ECC byte count transitions:
        // len=3 -> 2 ECC, len=4 -> 4 ECC, len=5 -> 4 ECC,
        // len=10 -> 4 ECC, len=15 -> 6 ECC, len=25 -> 10 ECC
        let test_lengths = [3, 4, 5, 10, 15, 25, 50, 100, 140];
        for &len in &test_lengths {
            let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + 13) % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "ECC boundary round-trip failed for length {len}"
            );
        }
    }

    // --- All same byte values (degenerate payloads) ---

    #[test]
    fn test_roundtrip_all_zeros() {
        let payload = vec![0u8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_all_ones() {
        let payload = vec![0xFFu8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decoder handles silence without panicking ---

    #[test]
    fn test_decode_silence_returns_none() {
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        let mut decoder = Decoder::new();
        let result = decoder.decode(&silence).unwrap();
        assert_eq!(result, None, "silence should produce None, not a decode");
    }

    // --- Decoder handles random noise without panicking ---

    #[test]
    fn test_decode_random_noise_no_panic() {
        let mut rng_state: u32 = 99999;
        let noise: Vec<f32> = (0..48000)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65535.0;
                (f - 0.5) * 2.0
            })
            .collect();
        let mut decoder = Decoder::new();
        // Should not panic — either returns None or an error
        let _ = decoder.decode(&noise);
    }

    // --- Multiple sequential decodes on one decoder instance ---

    #[test]
    fn test_sequential_decodes() {
        let payloads: &[&[u8]] = &[b"first", b"second", b"third"];
        let mut decoder = Decoder::new();
        for payload in payloads {
            let audio = encode(payload, 50).unwrap();
            decoder.reset();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(*payload),
                "sequential decode failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Volume sweep: ensure round-trip works at various volumes ---

    #[test]
    fn test_roundtrip_volume_sweep() {
        let payload = b"volume test";
        for volume in [1, 5, 10, 25, 50, 75, 100] {
            let audio = encode(payload, volume).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "round-trip failed at volume {volume}"
            );
        }
    }

    // --- Encode determinism: same input → same output ---

    #[test]
    fn test_encode_deterministic() {
        let payload = b"deterministic";
        let audio1 = encode(payload, 50).unwrap();
        let audio2 = encode(payload, 50).unwrap();
        assert_eq!(audio1.len(), audio2.len());
        for (i, (a, b)) in audio1.iter().zip(audio2.iter()).enumerate() {
            assert_eq!(a, b, "sample {i} differs between two encodes");
        }
    }

    // --- Decode audio with trailing garbage after end marker ---

    #[test]
    fn test_roundtrip_with_trailing_noise() {
        let payload = b"trail";
        let mut audio = encode(payload, 50).unwrap();

        // Append 1 second of random noise after the signal
        let mut rng_state: u32 = 77777;
        let noise: Vec<f32> = (0..48000)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65535.0;
                (f - 0.5) * 0.5
            })
            .collect();
        audio.extend_from_slice(&noise);

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decode with both leading silence and trailing noise ---

    #[test]
    fn test_roundtrip_padded_both_sides() {
        let payload = b"padded";
        let audio = encode(payload, 50).unwrap();

        let mut padded = vec![0.0f32; 24000]; // 0.5s leading silence
        padded.extend_from_slice(&audio);
        // 0.5s trailing silence
        padded.extend(vec![0.0f32; 24000]);

        let mut decoder = Decoder::new();
        let result = decoder.decode(&padded).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Back-to-back messages without reset (decoder should find first) ---

    #[test]
    fn test_back_to_back_messages() {
        let payload1 = b"msg1";
        let payload2 = b"msg2";
        let audio1 = encode(payload1, 50).unwrap();
        let audio2 = encode(payload2, 50).unwrap();

        let mut combined = audio1;
        // Small gap between messages
        combined.extend(vec![0.0f32; 4096]);
        combined.extend_from_slice(&audio2);

        let mut decoder = Decoder::new();
        let result1 = decoder.decode(&combined).unwrap();
        assert_eq!(
            result1.as_deref(),
            Some(&payload1[..]),
            "first message not decoded"
        );
    }

    // --- Payload at every byte value in a single byte ---

    #[test]
    fn test_roundtrip_every_single_byte_value() {
        // Stress-test all 256 possible single-byte payloads
        for b in 0..=255u8 {
            let payload = [b];
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "single-byte round-trip failed for byte 0x{b:02X}"
            );
        }
    }

    // --- Sample count matches for boundary payload sizes ---

    #[test]
    fn test_encode_sample_counts_consistent() {
        // Verify the output length formula holds for various sizes
        for size in [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 140] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();

            let n_ecc = protocol::ecc_bytes_for_length(size);
            let total_bytes = protocol::ENCODED_DATA_OFFSET + size + n_ecc;
            let total_data_frames = protocol::EXTRA
                * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
                * protocol::FRAMES_PER_TX;
            let expected = (protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES)
                * protocol::SAMPLES_PER_FRAME;

            assert_eq!(
                audio.len(),
                expected,
                "sample count mismatch for payload size {size}: got {} expected {}",
                audio.len(),
                expected
            );
        }
    }
}
