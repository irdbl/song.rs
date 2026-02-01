/// Sample rate in Hz.
pub const SAMPLE_RATE: f64 = 48000.0;

/// Low fundamental frequency (harmonics at 420, 630, 840 Hz — within phone band).
pub const F0_LOW: f64 = 210.0;

/// High fundamental frequency (harmonics at 540, 810, 1080 Hz — within phone band).
pub const F0_HIGH: f64 = 270.0;

/// Pitch threshold: below = low, above = high (midpoint of 210/270).
pub const PITCH_THRESHOLD: f64 = 240.0;

/// Number of harmonics in the synthesis model (H1=F0 .. H16=16*F0).
pub const NUM_HARMONICS: usize = 16;

/// Samples per symbol (50 ms at 48 kHz).
pub const SAMPLES_PER_SYMBOL: usize = 2400;

/// Guard silence after each symbol (10 ms).
pub const GUARD_SAMPLES: usize = 480;

/// Total samples per symbol slot (symbol + guard).
pub const SYMBOL_TOTAL_SAMPLES: usize = SAMPLES_PER_SYMBOL + GUARD_SAMPLES; // 2880

/// Number of vowel shapes in the alphabet.
pub const NUM_VOWELS: usize = 8;

/// Number of pitch classes.
pub const NUM_PITCHES: usize = 2;

/// Total symbol alphabet size (8 * 2 = 16).
pub const NUM_SYMBOLS: usize = NUM_VOWELS * NUM_PITCHES;

/// Bits per symbol (log2(16) = 4).
pub const BITS_PER_SYMBOL: usize = 4;

/// Preamble length in symbols.
pub const PREAMBLE_LEN: usize = 4;

/// FFT size for spectral analysis (same as symbol length).
pub const FFT_SIZE: usize = SAMPLES_PER_SYMBOL;

/// Hz per FFT bin = SAMPLE_RATE / FFT_SIZE.
pub const HZ_PER_BIN: f64 = SAMPLE_RATE / FFT_SIZE as f64;

/// Formant bandwidth for F1 (Hz) — Gaussian sigma.
pub const BW1: f64 = 80.0;

/// Formant bandwidth for F2 (Hz) — Gaussian sigma.
pub const BW2: f64 = 120.0;

/// F1 band lower bound (Hz) — above 300 Hz phone cutoff.
pub const F1_LO: f64 = 300.0;

/// F1 band upper bound (Hz).
pub const F1_HI: f64 = 850.0;

/// F2 band lower bound (Hz).
pub const F2_LO: f64 = 850.0;

/// F2 band upper bound (Hz).
pub const F2_HI: f64 = 2500.0;

/// Candidate F0 values for multi-F0 detection in decoder (3 per pitch cluster).
pub const F0_CANDIDATES: [f64; 6] = [200.0, 210.0, 220.0, 260.0, 270.0, 280.0];

/// Encoded data offset (1 byte length + 2 bytes ECC for length).
pub const ENCODED_DATA_OFFSET: usize = 3;

/// Maximum variable-length payload size.
pub const MAX_LENGTH_VARIABLE: usize = 140;

/// Maximum total encoded data size.
pub const MAX_DATA_SIZE: usize = 256;

/// Compute ECC byte count for a given payload length.
///
/// Matches C++: `len < 4 ? 2 : max(4, 2*(len/5))`
pub fn ecc_bytes_for_length(len: usize) -> usize {
    if len < 4 {
        2
    } else {
        4usize.max(2 * (len / 5))
    }
}
