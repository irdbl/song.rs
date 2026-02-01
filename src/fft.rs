//! FFT wrapper around `rustfft` for computing power spectrum.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Compute the Hann-windowed power spectrum of a real-valued audio frame.
///
/// Input: `samples` of arbitrary length N.
/// Output: `spectrum` of length >= N containing power (magnitude squared) at each bin.
///
/// Applies a Hann window before FFT to reduce spectral leakage,
/// then folds negative frequencies: spectrum[i] += spectrum[N - i] for i in 1..N/2.
pub fn power_spectrum(samples: &[f32], spectrum: &mut [f32]) {
    let n = samples.len();
    assert!(spectrum.len() >= n);

    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
            Complex::new(s * w as f32, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    for i in 0..n {
        spectrum[i] = buffer[i].norm_sqr();
    }

    for i in 1..n / 2 {
        spectrum[i] += spectrum[n - i];
    }
}

/// Compute power spectrum without windowing (rectangular window).
///
/// More robust for preamble detection with misaligned windows,
/// since all samples contribute equally regardless of position.
pub fn power_spectrum_raw(samples: &[f32], spectrum: &mut [f32]) {
    let n = samples.len();
    assert!(spectrum.len() >= n);

    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    for i in 0..n {
        spectrum[i] = buffer[i].norm_sqr();
    }

    for i in 1..n / 2 {
        spectrum[i] += spectrum[n - i];
    }
}
