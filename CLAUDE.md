# ggwave-voice

Voice-formant acoustic modem. Encodes data as vowel-like audio that survives phone codecs and noise cancellation.

## Build & Test

```bash
cargo build
cargo test                   # unit + integration tests
cargo test --test robustness # channel simulation tests (phone, AM radio, etc.)
cargo bench                  # criterion benchmarks (encode/decode/roundtrip)
```

## Architecture

| Module | Purpose |
|---|---|
| `src/lib.rs` | Public API (`encode`, `Decoder`), top-level tests |
| `src/protocol.rs` | Constants (sample rate, symbol timing, FFT size, formant bands) |
| `src/formant.rs` | 2-channel symbol system: vowel alphabet, synthesis, detection, 4-bit packing |
| `src/encoder.rs` | Payload → RS-encode → symbols → synthesized audio |
| `src/decoder.rs` | Streaming state machine (Listening → Receiving → Analyzing) |
| `src/fft.rs` | FFT wrapper (Hann-windowed and raw power spectrum) |
| `src/reed_solomon.rs` | RS codec over GF(2^8) with erasure support, ported from ggwave C++ |
| `src/phy.rs` | PHY layer: stop-and-wait ARQ, fragmentation (up to 2,224 bytes), ACK/NAK, timeout/retransmit |
| `src/dss.rs` | Direct Sequence Spread (XOR whitening, currently unused) |

## Key Design

- **2-channel orthogonal encoding**: each symbol carries 4 bits via vowel (3 bits, F1/F2 formants) + pitch (1 bit, F0 = 210 or 270 Hz)
- **16 symbols** from 8 machine-optimized vowels × 2 pitch classes
- **Symbol timing**: 50 ms voiced + 10 ms guard = 60 ms per symbol (2880 samples at 48 kHz)
- **Preamble**: 4 symbols start ([0, 14, 0, 14]) and end ([14, 0, 14, 0]), using max-distance vowel pair
- **RS ECC**: adaptive — `len < 4 ? 2 : max(4, 2*(len/5))` parity bytes
- **Decoder**: streaming, processes symbol-sized windows. History ring buffer for alignment recovery. Multi-offset analysis as fallback. RS erasure decoding via symbol confidence as fallback when hard-decision fails.
- **PHY layer**: stop-and-wait ARQ over the acoustic channel. Fragments messages into 139-byte frames, ACK/NAK per fragment, timeout/retransmit (configurable). Test harness in `tests/phy.rs` with loopback, noise, lost-packet, and fragmentation scenarios.

## Conventions

- Audio is `f32` samples at 48 kHz (mono)
- Internal synthesis uses `f64` for precision, casts to `f32` at output
- Volume is `u8` 0..=100
- Max payload 140 bytes
- Tests use deterministic LCG PRNG (no randomness)
