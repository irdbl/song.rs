//! Sweep symbol duration to find the optimal throughput-robustness tradeoff.
//!
//! Tests each (symbol_ms, guard_ms) configuration across multiple channel
//! simulations, reporting both per-symbol classification accuracy and
//! full encode→decode success.
//!
//! Usage: cargo run --example sweep_duration --release

use song_rs::formant::{
    self, classify_vowel, detect_pitch, harmonic_amplitudes, params_to_symbol, symbol_to_params,
    symbol_vowel, PREAMBLE_END, PREAMBLE_START, VOWELS,
};
use song_rs::protocol::*;
use song_rs::reed_solomon::ReedSolomon;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

const SR: f64 = SAMPLE_RATE;

// ── Parameterized synthesis ─────────────────────────────────────────

fn synthesize_sym(sym_idx: usize, volume: f64, n_voiced: usize) -> Vec<f64> {
    let (vowel_idx, pitch_class) = symbol_to_params(sym_idx);
    let vowel = &VOWELS[vowel_idx];
    let f0 = pitch_class.f0();
    let amps = harmonic_amplitudes(vowel.f1, vowel.f2, f0);

    let fade_samples = (SR * 0.005) as usize;
    let mut out = vec![0.0f64; n_voiced];

    for i in 0..n_voiced {
        let t = i as f64 / SR;
        let mut sample = 0.0f64;
        for h in 0..NUM_HARMONICS {
            let freq = f0 * (h + 1) as f64;
            sample += amps[h] * (2.0 * PI * freq * t).sin();
        }
        let fade = if i < fade_samples {
            i as f64 / fade_samples as f64
        } else if i >= n_voiced.saturating_sub(fade_samples) {
            (n_voiced - 1 - i) as f64 / fade_samples as f64
        } else {
            1.0
        };
        out[i] = sample * volume * fade;
    }
    out
}

// ── Parameterized classification ────────────────────────────────────

fn power_spectrum_hann(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
            Complex::new(s * w as f32, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut buffer);

    let mut spec = vec![0.0f32; n];
    for i in 0..n {
        spec[i] = buffer[i].norm_sqr();
    }
    for i in 1..n / 2 {
        spec[i] += spec[n - i];
    }
    spec
}

fn detect_f0_flex(spectrum: &[f32], hz_per_bin: f64) -> f64 {
    let mut best_f0 = F0_LOW;
    let mut best_score = 0.0f64;

    for &f0 in &F0_CANDIDATES {
        let mut score = 0.0f64;
        for h in 1..=NUM_HARMONICS {
            let freq = f0 * h as f64;
            let bin = (freq / hz_per_bin).round() as usize;
            if bin >= spectrum.len() {
                break;
            }
            for offset in 0..=2usize {
                if offset == 0 {
                    score += spectrum[bin] as f64;
                } else {
                    if bin >= offset {
                        score += spectrum[bin - offset] as f64 * 0.5;
                    }
                    if bin + offset < spectrum.len() {
                        score += spectrum[bin + offset] as f64 * 0.5;
                    }
                }
            }
        }
        if score > best_score {
            best_score = score;
            best_f0 = f0;
        }
    }
    best_f0
}

fn detect_formants_flex(spectrum: &[f32], f0: f64, hz_per_bin: f64) -> (f64, f64) {
    let mut best_f1_power = 0.0f64;
    let mut best_f1_freq = F1_LO;
    let mut best_f2_power = 0.0f64;
    let mut best_f2_freq = 1500.0;

    for h in 1..=NUM_HARMONICS {
        let freq = f0 * h as f64;
        let bin = (freq / hz_per_bin).round() as usize;
        if bin >= spectrum.len() {
            break;
        }

        let mut power = 0.0f64;
        for offset in 0..=2usize {
            if offset == 0 {
                power += spectrum[bin] as f64;
            } else {
                if bin >= offset {
                    power += spectrum[bin - offset] as f64;
                }
                if bin + offset < spectrum.len() {
                    power += spectrum[bin + offset] as f64;
                }
            }
        }

        if freq >= F1_LO && freq <= F1_HI && power > best_f1_power {
            best_f1_power = power;
            best_f1_freq = freq;
        }
        if freq >= F2_LO && freq <= F2_HI && power > best_f2_power {
            best_f2_power = power;
            best_f2_freq = freq;
        }
    }

    (best_f1_freq, best_f2_freq)
}

fn classify_sym_flex(voiced: &[f32]) -> (usize, f64) {
    let n = voiced.len();
    let hz_per_bin = SR / n as f64;
    let spec = power_spectrum_hann(voiced);

    let f0 = detect_f0_flex(&spec, hz_per_bin);
    let (f1, f2) = detect_formants_flex(&spec, f0, hz_per_bin);
    let (vowel_idx, vowel_conf) = classify_vowel(f1, f2);
    let (pitch_class, pitch_conf) = detect_pitch(f0);

    let sym = params_to_symbol(vowel_idx, pitch_class as usize);
    (sym, vowel_conf * 0.7 + pitch_conf * 0.3)
}

// ── Parameterized encode/decode ─────────────────────────────────────

fn encode_flex(payload: &[u8], volume: f64, n_voiced: usize, n_guard: usize) -> Vec<f32> {
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let encoded_length = rs_length.encode(&[payload.len() as u8]);

    let n_ecc = ecc_bytes_for_length(payload.len());
    let rs_data = ReedSolomon::new(payload.len(), n_ecc);
    let encoded_data = rs_data.encode(payload);

    let mut data_encoded = Vec::new();
    data_encoded.extend_from_slice(&encoded_length);
    data_encoded.extend_from_slice(&encoded_data);

    let data_symbols = formant::bytes_to_symbols(&data_encoded);

    let mut output = Vec::new();
    let mut emit = |sym: usize| {
        let voiced = synthesize_sym(sym, volume, n_voiced);
        for &s in &voiced {
            output.push(s as f32);
        }
        output.extend(std::iter::repeat(0.0f32).take(n_guard));
    };

    for &s in &PREAMBLE_START {
        emit(s);
    }
    for &s in &data_symbols {
        emit(s);
    }
    for &s in &PREAMBLE_END {
        emit(s);
    }

    output
}

fn decode_flex(audio: &[f32], n_voiced: usize, n_guard: usize) -> Option<Vec<u8>> {
    let sym_total = n_voiced + n_guard;
    let n_windows = audio.len() / sym_total;

    let mut symbols = Vec::new();
    for i in 0..n_windows {
        let start = i * sym_total;
        if start + n_voiced > audio.len() {
            break;
        }
        let voiced = &audio[start..start + n_voiced];
        let (sym, _) = classify_sym_flex(voiced);
        symbols.push(sym);
    }

    if symbols.len() < PREAMBLE_LEN * 2 {
        return None;
    }

    // Find start preamble
    let preamble_start_vowels: Vec<usize> = PREAMBLE_START.iter().map(|&s| symbol_vowel(s)).collect();
    let preamble_end_vowels: Vec<usize> = PREAMBLE_END.iter().map(|&s| symbol_vowel(s)).collect();

    let min_data_syms = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);

    let mut data_start = PREAMBLE_LEN; // default: assume preamble is at the start
    for pos in 0..symbols.len().saturating_sub(PREAMBLE_LEN + min_data_syms) {
        let vowels: Vec<usize> = symbols[pos..pos + PREAMBLE_LEN]
            .iter()
            .map(|&s| symbol_vowel(s))
            .collect();
        if vowels == preamble_start_vowels {
            data_start = pos + PREAMBLE_LEN;
            break;
        }
    }

    // Find end preamble (search from end)
    let mut data_end = symbols.len().saturating_sub(PREAMBLE_LEN);
    for pos in (data_start + min_data_syms..=symbols.len().saturating_sub(PREAMBLE_LEN)).rev() {
        if pos + PREAMBLE_LEN > symbols.len() {
            continue;
        }
        // Try full symbol match first
        if symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_END[..] {
            data_end = pos;
            break;
        }
        let vowels: Vec<usize> = symbols[pos..pos + PREAMBLE_LEN]
            .iter()
            .map(|&s| symbol_vowel(s))
            .collect();
        if vowels == preamble_end_vowels {
            data_end = pos;
            break;
        }
    }

    if data_end <= data_start || data_end - data_start < min_data_syms {
        return None;
    }

    let data_symbols = &symbols[data_start..data_end];
    let max_bytes = (data_symbols.len() * BITS_PER_SYMBOL) / 8;
    let data_encoded = formant::symbols_to_bytes(data_symbols, max_bytes);

    if data_encoded.len() < ENCODED_DATA_OFFSET {
        return None;
    }

    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let decoded_len = rs_length.decode(&data_encoded[..ENCODED_DATA_OFFSET])?;
    let len = decoded_len[0] as usize;

    if len == 0 || len > MAX_LENGTH_VARIABLE {
        return None;
    }

    let n_ecc = ecc_bytes_for_length(len);
    let expected_total = ENCODED_DATA_OFFSET + len + n_ecc;
    if data_encoded.len() < expected_total {
        return None;
    }

    let rs_data = ReedSolomon::new(len, n_ecc);
    rs_data.decode(&data_encoded[ENCODED_DATA_OFFSET..ENCODED_DATA_OFFSET + len + n_ecc])
}

// ── DSP helpers (channel simulations) ───────────────────────────────

fn bandpass(samples: &[f32], lo_hz: f64, hi_hz: f64) -> Vec<f32> {
    let n = samples.len();
    let hz_per_bin = SR / n as f64;
    let mut buf: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, c) in buf.iter_mut().enumerate() {
        let freq = if i <= n / 2 {
            i as f64 * hz_per_bin
        } else {
            (n - i) as f64 * hz_per_bin
        };
        if freq < lo_hz || freq > hi_hz {
            *c = Complex::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f32).collect()
}

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    (*state >> 16) as f32 / 65535.0
}

fn normal(state: &mut u32) -> f32 {
    let u1 = lcg(state).max(1e-10);
    let u2 = lcg(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn add_noise(samples: &[f32], snr_db: f64, seed: u32) -> Vec<f32> {
    let sig_power: f64 =
        samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    let noise_rms = sig_power.sqrt() / 10.0f64.powf(snr_db / 20.0);
    let mut st = seed;
    samples
        .iter()
        .map(|&s| s + normal(&mut st) * noise_rms as f32)
        .collect()
}

fn mulaw_compress(x: f32) -> f32 {
    let mu: f32 = 255.0;
    x.signum() * (1.0 + mu * x.abs()).ln() / (1.0 + mu).ln()
}

fn mulaw_expand(y: f32) -> f32 {
    let mu: f32 = 255.0;
    y.signum() * (1.0 / mu) * ((1.0 + mu).powf(y.abs()) - 1.0)
}

fn mulaw_codec(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| {
            let c = mulaw_compress(s.clamp(-1.0, 1.0));
            let q = (c * 127.0).round() / 127.0;
            mulaw_expand(q)
        })
        .collect()
}

fn add_echo(samples: &[f32], delay_ms: f64, amplitude: f32) -> Vec<f32> {
    let delay = (delay_ms * SR / 1000.0) as usize;
    let mut out = samples.to_vec();
    for i in delay..out.len() {
        out[i] += samples[i - delay] * amplitude;
    }
    out
}

fn hard_clip(samples: &[f32], threshold: f32) -> Vec<f32> {
    samples.iter().map(|&s| s.clamp(-threshold, threshold)).collect()
}

fn reverb(samples: &[f32], echoes: &[(f64, f32)]) -> Vec<f32> {
    let mut out = samples.to_vec();
    for &(delay_ms, amp) in echoes {
        let delay = (delay_ms * SR / 1000.0) as usize;
        for i in delay..out.len() {
            out[i] += samples[i - delay] * amp;
        }
    }
    out
}

fn resample_8k(samples: &[f32]) -> Vec<f32> {
    let filtered = bandpass(samples, 0.0, 3400.0);
    let decimated: Vec<f32> = filtered.iter().step_by(6).copied().collect();
    let mut up = vec![0.0f32; samples.len()];
    for (i, &s) in decimated.iter().enumerate() {
        let idx = i * 6;
        if idx < up.len() {
            up[idx] = s * 6.0;
        }
    }
    bandpass(&up, 0.0, 3400.0)
}

fn agc(samples: &[f32], target_rms: f32) -> Vec<f32> {
    let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 1e-10 {
        return samples.to_vec();
    }
    let gain = target_rms / rms;
    samples
        .iter()
        .map(|&s| (s * gain).clamp(-1.0, 1.0))
        .collect()
}

// ── Channel definitions ─────────────────────────────────────────────

type ChannelFn = fn(&[f32]) -> Vec<f32>;

fn ch_clean(a: &[f32]) -> Vec<f32> {
    a.to_vec()
}
fn ch_phone(a: &[f32]) -> Vec<f32> {
    mulaw_codec(&bandpass(a, 300.0, 3400.0))
}
fn ch_phone_noisy(a: &[f32]) -> Vec<f32> {
    add_noise(&mulaw_codec(&bandpass(a, 300.0, 3400.0)), 15.0, 42)
}
fn ch_am(a: &[f32]) -> Vec<f32> {
    add_echo(&bandpass(a, 300.0, 2500.0), 30.0, 0.15)
}
fn ch_am_noisy(a: &[f32]) -> Vec<f32> {
    add_noise(
        &add_echo(&bandpass(a, 300.0, 2500.0), 50.0, 0.2),
        12.0,
        99,
    )
}
fn ch_noise_10db(a: &[f32]) -> Vec<f32> {
    add_noise(a, 10.0, 77)
}
fn ch_walkie(a: &[f32]) -> Vec<f32> {
    add_noise(
        &add_echo(&hard_clip(&bandpass(a, 300.0, 3000.0), 0.5), 10.0, 0.1),
        10.0,
        55,
    )
}
fn ch_nightmare(a: &[f32]) -> Vec<f32> {
    agc(
        &add_noise(
            &bandpass(
                &mulaw_codec(&resample_8k(&hard_clip(
                    &reverb(a, &[(15.0, 0.2), (35.0, 0.1)]),
                    0.5,
                ))),
                300.0,
                3400.0,
            ),
            10.0,
            13579,
        ),
        0.3,
    )
}
fn ch_hell(a: &[f32]) -> Vec<f32> {
    // Everything at once, cranked up
    agc(
        &add_noise(
            &bandpass(
                &mulaw_codec(&resample_8k(&hard_clip(
                    &reverb(a, &[(10.0, 0.3), (25.0, 0.2), (50.0, 0.1)]),
                    0.35,
                ))),
                300.0,
                3400.0,
            ),
            5.0,
            24680,
        ),
        0.3,
    )
}
fn ch_noise_3db(a: &[f32]) -> Vec<f32> {
    add_noise(a, 3.0, 77)
}
fn ch_noise_0db(a: &[f32]) -> Vec<f32> {
    add_noise(a, 0.0, 77)
}
fn ch_noise_neg3db(a: &[f32]) -> Vec<f32> {
    add_noise(a, -3.0, 77)
}

const CHANNELS: &[(&str, ChannelFn)] = &[
    ("Clean", ch_clean),
    ("Phone", ch_phone),
    ("Ph+15dB", ch_phone_noisy),
    ("AM+12dB", ch_am_noisy),
    ("Walkie", ch_walkie),
    ("Night", ch_nightmare),
    ("Hell", ch_hell),
    ("N 3dB", ch_noise_3db),
    ("N 0dB", ch_noise_0db),
    ("N-3dB", ch_noise_neg3db),
];

// ── Main sweep ──────────────────────────────────────────────────────

fn main() {
    let configs: &[(u32, u32)] = &[
        (50, 10),
        (75, 15),
        (100, 15),
        (100, 30),
        (125, 20),
        (150, 30), // current
    ];

    // Use max-length payload to stress-test accumulated errors
    let payload_max: Vec<u8> = (0..140).map(|i| ((i * 7 + 13) % 256) as u8).collect();
    // Medium payload
    let payload_med = b"The quick brown fox jumps over the lazy dog! 1234567890";
    // Short payload
    let payload_short = b"test payload data!";

    eprintln!("=== Symbol Duration Sweep ===\n");

    // ── Part 1: Full encode → channel → decode (3 payload sizes) ────

    for (label, payload) in [
        ("SHORT (18B)", &payload_short[..]),
        ("MEDIUM (54B)", &payload_med[..]),
        ("MAX (140B)", &payload_max[..]),
    ] {
        eprintln!("── Full decode: {} payload ──\n", label);

        let ch_header: String = CHANNELS.iter().map(|(n, _)| format!("{:>8}", n)).collect();
        eprintln!(
            "{:>14} {:>7} {:>6}  {}",
            "Config", "bit/s", "airtime", ch_header
        );
        eprintln!(
            "{}",
            "-".repeat(14 + 7 + 1 + 6 + 2 + CHANNELS.len() * 8 + 2)
        );

        for &(sym_ms, guard_ms) in configs {
            let n_voiced = (SR * sym_ms as f64 / 1000.0) as usize;
            let n_guard = (SR * guard_ms as f64 / 1000.0) as usize;
            let sym_total_ms = sym_ms + guard_ms;
            let bps = BITS_PER_SYMBOL as f64 * 1000.0 / sym_total_ms as f64;

            let n_ecc = ecc_bytes_for_length(payload.len());
            let total_bytes = ENCODED_DATA_OFFSET + payload.len() + n_ecc;
            let n_data_syms = formant::symbols_for_bytes(total_bytes);
            let total_syms = PREAMBLE_LEN + n_data_syms + PREAMBLE_LEN;
            let airtime_s = total_syms as f64 * sym_total_ms as f64 / 1000.0;

            let config_name = format!("{}+{}ms", sym_ms, guard_ms);
            let mut row = format!("{:>14} {:>7.1} {:>5.1}s", config_name, bps, airtime_s);

            let audio = encode_flex(payload, 0.5, n_voiced, n_guard);

            for &(_ch_name, ch_fn) in CHANNELS {
                let processed = ch_fn(&audio);
                match decode_flex(&processed, n_voiced, n_guard) {
                    Some(ref d) if d == payload => row.push_str("      OK"),
                    Some(_) => row.push_str("   WRONG"),
                    None => row.push_str("    FAIL"),
                }
            }

            eprintln!("{}", row);
        }
        eprintln!();
    }

    // ── Part 2: SNR floor sweep (push to negative) ──────────────────

    eprintln!("── SNR floor: raw noise (short payload) ──\n");
    eprintln!("{:>14}  {:>6}", "Config", "Floor");
    eprintln!("{}", "-".repeat(24));

    let snr_steps: &[f64] = &[
        10.0, 5.0, 0.0, -3.0, -5.0, -8.0, -10.0, -12.0, -15.0, -18.0, -20.0, -25.0, -30.0,
    ];

    for &(sym_ms, guard_ms) in configs {
        let n_voiced = (SR * sym_ms as f64 / 1000.0) as usize;
        let n_guard = (SR * guard_ms as f64 / 1000.0) as usize;

        let audio = encode_flex(payload_short, 0.5, n_voiced, n_guard);

        let mut floor = f64::NAN;
        for &snr in snr_steps {
            let noisy = add_noise(&audio, snr, 42);
            if let Some(ref d) = decode_flex(&noisy, n_voiced, n_guard) {
                if d == &payload_short[..] {
                    floor = snr;
                }
            }
        }

        let config_name = format!("{}+{}ms", sym_ms, guard_ms);
        if floor.is_nan() {
            eprintln!("{:>14}  {:>6}", config_name, ">20 dB");
        } else {
            eprintln!("{:>14}  {:>4.0} dB", config_name, floor);
        }
    }

    // ── Part 3: SNR floor with max payload ──────────────────────────

    eprintln!("\n── SNR floor: raw noise (MAX 140B payload) ──\n");
    eprintln!("{:>14}  {:>6}", "Config", "Floor");
    eprintln!("{}", "-".repeat(24));

    for &(sym_ms, guard_ms) in configs {
        let n_voiced = (SR * sym_ms as f64 / 1000.0) as usize;
        let n_guard = (SR * guard_ms as f64 / 1000.0) as usize;

        let audio = encode_flex(&payload_max, 0.5, n_voiced, n_guard);

        let mut floor = f64::NAN;
        for &snr in snr_steps {
            let noisy = add_noise(&audio, snr, 42);
            if let Some(ref d) = decode_flex(&noisy, n_voiced, n_guard) {
                if d == &payload_max[..] {
                    floor = snr;
                }
            }
        }

        let config_name = format!("{}+{}ms", sym_ms, guard_ms);
        if floor.is_nan() {
            eprintln!("{:>14}  {:>6}", config_name, ">20 dB");
        } else {
            eprintln!("{:>14}  {:>4.0} dB", config_name, floor);
        }
    }

    // ── Part 4: Phone channel SNR floor ─────────────────────────────

    eprintln!("\n── Phone SNR floor (bp + μ-law + noise, MAX 140B) ──\n");
    eprintln!("{:>14}  {:>6}", "Config", "Floor");
    eprintln!("{}", "-".repeat(24));

    for &(sym_ms, guard_ms) in configs {
        let n_voiced = (SR * sym_ms as f64 / 1000.0) as usize;
        let n_guard = (SR * guard_ms as f64 / 1000.0) as usize;

        let audio = encode_flex(&payload_max, 0.5, n_voiced, n_guard);
        let phone_base = mulaw_codec(&bandpass(&audio, 300.0, 3400.0));

        let mut floor = f64::NAN;
        for &snr in snr_steps {
            let noisy = add_noise(&phone_base, snr, 42);
            if let Some(ref d) = decode_flex(&noisy, n_voiced, n_guard) {
                if d == &payload_max[..] {
                    floor = snr;
                }
            }
        }

        let config_name = format!("{}+{}ms", sym_ms, guard_ms);
        if floor.is_nan() {
            eprintln!("{:>14}  {:>6}", config_name, ">20 dB");
        } else {
            eprintln!("{:>14}  {:>4.0} dB", config_name, floor);
        }
    }

    // ── Part 5: Nightmare channel SNR floor ─────────────────────────

    eprintln!("\n── Nightmare SNR floor (reverb+clip+8k+μ-law+bp + noise, short) ──\n");
    eprintln!("{:>14}  {:>6}", "Config", "Floor");
    eprintln!("{}", "-".repeat(24));

    for &(sym_ms, guard_ms) in configs {
        let n_voiced = (SR * sym_ms as f64 / 1000.0) as usize;
        let n_guard = (SR * guard_ms as f64 / 1000.0) as usize;

        let audio = encode_flex(payload_short, 0.5, n_voiced, n_guard);
        // Nightmare without the noise (we add our own)
        let nightmare_base = agc(
            &bandpass(
                &mulaw_codec(&resample_8k(&hard_clip(
                    &reverb(&audio, &[(15.0, 0.2), (35.0, 0.1)]),
                    0.5,
                ))),
                300.0,
                3400.0,
            ),
            0.3,
        );

        let mut floor = f64::NAN;
        for &snr in snr_steps {
            let noisy = add_noise(&nightmare_base, snr, 42);
            if let Some(ref d) = decode_flex(&noisy, n_voiced, n_guard) {
                if d == &payload_short[..] {
                    floor = snr;
                }
            }
        }

        let config_name = format!("{}+{}ms", sym_ms, guard_ms);
        if floor.is_nan() {
            eprintln!("{:>14}  {:>6}", config_name, ">20 dB");
        } else {
            eprintln!("{:>14}  {:>4.0} dB", config_name, floor);
        }
    }
}
