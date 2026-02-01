//! Encoder: payload → f32 audio samples (2-channel orthogonal vocal modem).

use crate::formant::{self, PREAMBLE_END, PREAMBLE_START};
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Encode a payload into f32 audio samples.
///
/// `volume` is 0..=100 (typically 10-50).
/// Returns a Vec<f32> containing the complete waveform at 48 kHz.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    let data_length = payload.len();
    if data_length == 0 {
        return Err(crate::Error::EmptyPayload);
    }
    if data_length > MAX_LENGTH_VARIABLE {
        return Err(crate::Error::PayloadTooLarge {
            size: data_length,
            max: MAX_LENGTH_VARIABLE,
        });
    }
    if volume > 100 {
        return Err(crate::Error::InvalidVolume(volume));
    }

    let send_volume = volume as f64 / 100.0;

    // --- Step 1: Prepare tx_data = [length_byte, payload_bytes...] ---
    let mut tx_data = vec![0u8; data_length + 1];
    tx_data[0] = data_length as u8;
    tx_data[1..=data_length].copy_from_slice(payload);

    // --- Step 2: Reed-Solomon encode ---
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let encoded_length = rs_length.encode(&tx_data[0..1]);

    let n_ecc = ecc_bytes_for_length(data_length);
    let rs_data = ReedSolomon::new(data_length, n_ecc);
    let encoded_data = rs_data.encode(&tx_data[1..=data_length]);

    let mut data_encoded = Vec::with_capacity(ENCODED_DATA_OFFSET + data_length + n_ecc);
    data_encoded.extend_from_slice(&encoded_length);
    data_encoded.extend_from_slice(&encoded_data);

    // --- Step 3: Convert bytes to 4-bit symbol sequence ---
    let data_symbols = formant::bytes_to_symbols(&data_encoded);

    // --- Step 4: Build full symbol sequence: preamble_start + data + preamble_end ---
    let total_symbols = PREAMBLE_LEN + data_symbols.len() + PREAMBLE_LEN;
    let total_samples = total_symbols * SYMBOL_TOTAL_SAMPLES;

    let mut output = Vec::with_capacity(total_samples);

    // Synthesize and append a symbol (voiced samples + guard silence)
    let mut emit_symbol = |sym_idx: usize| {
        let voiced = formant::synthesize_symbol(sym_idx, send_volume);
        for &s in &voiced {
            output.push(s as f32);
        }
        // Guard silence
        output.extend(std::iter::repeat(0.0f32).take(GUARD_SAMPLES));
    };

    // Start preamble
    for &sym in &PREAMBLE_START {
        emit_symbol(sym);
    }

    // Data symbols
    for &sym in &data_symbols {
        emit_symbol(sym);
    }

    // End preamble
    for &sym in &PREAMBLE_END {
        emit_symbol(sym);
    }

    Ok(output)
}
