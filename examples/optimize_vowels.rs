//! Vowel alphabet optimizer for the 2-channel vocal modem.
//!
//! Finds the optimal set of 8 vowels (F1/F2 formant pairs) that
//! maximizes classification margin across both F0 values (210/270 Hz).
//!
//! The key constraint: the detector snaps formant frequencies to the
//! nearest harmonic of the detected F0. Two vowels are indistinguishable
//! if they snap to the same (F1, F2) harmonic pair at any F0 value.
//!
//! Run: cargo run --example optimize_vowels --release

use ggwave_voice::formant;
use ggwave_voice::protocol::*;
use std::collections::HashSet;

// ── Data types ───────────────────────────────────────────────────────

/// Detected formant pair at a specific F0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DetectedPair {
    f1: i32, // Hz as integer for reliable equality
    f2: i32,
}

/// A feasible vowel slot: a target (F1, F2) that maps to consistent
/// detected harmonics at both F0 values.
#[derive(Debug, Clone)]
struct Slot {
    f1: f64,
    f2: f64,
    det_low: DetectedPair,
    det_high: DetectedPair,
    f1_range: (f64, f64),
    f2_range: (f64, f64),
}

// ── Helpers ──────────────────────────────────────────────────────────

/// All harmonics of f0 within [lo, hi].
fn harmonics_in_band(f0: f64, lo: f64, hi: f64) -> Vec<f64> {
    (1..=NUM_HARMONICS as u32)
        .map(|h| h as f64 * f0)
        .filter(|&f| f >= lo && f <= hi)
        .collect()
}

/// Voronoi cells: each harmonic owns the interval to the midpoints
/// of its neighbors, clamped to [band_lo, band_hi].
fn voronoi_cells(harmonics: &[f64], band_lo: f64, band_hi: f64) -> Vec<(f64, f64)> {
    let n = harmonics.len();
    (0..n)
        .map(|i| {
            let lo = if i == 0 {
                band_lo
            } else {
                (harmonics[i - 1] + harmonics[i]) / 2.0
            };
            let hi = if i == n - 1 {
                band_hi
            } else {
                (harmonics[i] + harmonics[i + 1]) / 2.0
            };
            (lo, hi)
        })
        .collect()
}

/// Classification distance (same metric as classify_vowel).
fn dist(f1_a: f64, f2_a: f64, f1_b: f64, f2_b: f64) -> f64 {
    let d1 = (f1_a - f1_b) / 100.0;
    let d2 = (f2_a - f2_b) / 200.0;
    (d1 * d1 + d2 * d2).sqrt()
}

/// Classification margin: distance_to_nearest_other - distance_to_own.
/// Positive means correct classification.
fn margin(
    det_f1: f64,
    det_f2: f64,
    targets: &[(f64, f64)],
    own_idx: usize,
) -> f64 {
    let self_dist = dist(det_f1, det_f2, targets[own_idx].0, targets[own_idx].1);
    let min_other = targets
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != own_idx)
        .map(|(_, &(f1, f2))| dist(det_f1, det_f2, f1, f2))
        .fold(f64::MAX, f64::min);
    min_other - self_dist
}

/// Check if two slots collide (same detected pair at either F0).
fn conflicts(a: &Slot, b: &Slot) -> bool {
    a.det_low == b.det_low || a.det_high == b.det_high
}

/// Min pairwise target distance for a selected set.
fn min_pairwise_dist(slots: &[Slot], selected: &[usize]) -> f64 {
    let mut min_d = f64::MAX;
    for i in 0..selected.len() {
        for j in i + 1..selected.len() {
            let a = &slots[selected[i]];
            let b = &slots[selected[j]];
            min_d = min_d.min(dist(a.f1, a.f2, b.f1, b.f2));
        }
    }
    min_d
}

// ── Phase 1: Enumerate all feasible slots ────────────────────────────

fn enumerate_slots() -> Vec<Slot> {
    let f1_harms_low = harmonics_in_band(F0_LOW, F1_LO, F1_HI);
    let f1_harms_high = harmonics_in_band(F0_HIGH, F1_LO, F1_HI);
    let f2_harms_low = harmonics_in_band(F0_LOW, F2_LO, F2_HI);
    let f2_harms_high = harmonics_in_band(F0_HIGH, F2_LO, F2_HI);

    println!("Harmonic grid:");
    println!("  F1 @ F0_LOW  ({:.0}Hz): {:?}", F0_LOW, f1_harms_low);
    println!("  F1 @ F0_HIGH ({:.0}Hz): {:?}", F0_HIGH, f1_harms_high);
    println!("  F2 @ F0_LOW  ({:.0}Hz): {:?}", F0_LOW, f2_harms_low);
    println!("  F2 @ F0_HIGH ({:.0}Hz): {:?}", F0_HIGH, f2_harms_high);

    let f1_cells_low = voronoi_cells(&f1_harms_low, F1_LO, F1_HI);
    let f1_cells_high = voronoi_cells(&f1_harms_high, F1_LO, F1_HI);
    let f2_cells_low = voronoi_cells(&f2_harms_low, F2_LO, F2_HI);
    let f2_cells_high = voronoi_cells(&f2_harms_high, F2_LO, F2_HI);

    println!("\nF1 Voronoi cells at F0_LOW:  {:?}", f1_cells_low);
    println!("F1 Voronoi cells at F0_HIGH: {:?}", f1_cells_high);

    let mut slots = Vec::new();

    for (il, cl) in f1_cells_low.iter().enumerate() {
        for (ih, ch) in f1_cells_high.iter().enumerate() {
            let f1_lo = cl.0.max(ch.0);
            let f1_hi = cl.1.min(ch.1);
            if f1_lo >= f1_hi {
                continue;
            }

            for (jl, dl) in f2_cells_low.iter().enumerate() {
                for (jh, dh) in f2_cells_high.iter().enumerate() {
                    let f2_lo = dl.0.max(dh.0);
                    let f2_hi = dl.1.min(dh.1);
                    if f2_lo >= f2_hi {
                        continue;
                    }

                    let f1_det_lo = f1_harms_low[il];
                    let f1_det_hi = f1_harms_high[ih];
                    let f2_det_lo = f2_harms_low[jl];
                    let f2_det_hi = f2_harms_high[jh];

                    // Target = midpoint of detected values at both F0s
                    // (minimizes worst-case self-distance in classify metric)
                    // Clamp to stay within feasible region with margin
                    let f1 = ((f1_det_lo + f1_det_hi) / 2.0).clamp(f1_lo + 0.5, f1_hi - 0.5);
                    let f2 = ((f2_det_lo + f2_det_hi) / 2.0).clamp(f2_lo + 0.5, f2_hi - 0.5);

                    slots.push(Slot {
                        f1,
                        f2,
                        det_low: DetectedPair {
                            f1: f1_det_lo as i32,
                            f2: f2_det_lo as i32,
                        },
                        det_high: DetectedPair {
                            f1: f1_det_hi as i32,
                            f2: f2_det_hi as i32,
                        },
                        f1_range: (f1_lo, f1_hi),
                        f2_range: (f2_lo, f2_hi),
                    });
                }
            }
        }
    }

    let f1_combos: HashSet<_> = slots
        .iter()
        .map(|s| (s.det_low.f1, s.det_high.f1))
        .collect();
    let f2_combos: HashSet<_> = slots
        .iter()
        .map(|s| (s.det_low.f2, s.det_high.f2))
        .collect();

    println!("\nFeasible slots: {} total", slots.len());
    println!("  {} F1 groups, {} F2 groups", f1_combos.len(), f2_combos.len());
    println!();

    slots
}

// ── Phase 2: Greedy selection ────────────────────────────────────────

fn greedy_select(slots: &[Slot], count: usize) -> Vec<usize> {
    let n = slots.len();

    // Find best starting pair (max distance, no conflicts)
    let mut best_pair = (0, 1);
    let mut best_d = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            if conflicts(&slots[i], &slots[j]) {
                continue;
            }
            let d = dist(slots[i].f1, slots[i].f2, slots[j].f1, slots[j].f2);
            if d > best_d {
                best_d = d;
                best_pair = (i, j);
            }
        }
    }

    let mut selected = vec![best_pair.0, best_pair.1];

    while selected.len() < count {
        let mut best_candidate = None;
        let mut best_min = 0.0;

        'cand: for i in 0..n {
            if selected.contains(&i) {
                continue;
            }
            for &j in &selected {
                if conflicts(&slots[i], &slots[j]) {
                    continue 'cand;
                }
            }

            let min_d = selected
                .iter()
                .map(|&j| dist(slots[i].f1, slots[i].f2, slots[j].f1, slots[j].f2))
                .fold(f64::MAX, f64::min);

            if min_d > best_min {
                best_min = min_d;
                best_candidate = Some(i);
            }
        }

        match best_candidate {
            Some(idx) => selected.push(idx),
            None => {
                eprintln!("ERROR: could not find {} non-conflicting vowels", count);
                break;
            }
        }
    }

    selected
}

// ── Phase 3: Local search refinement ─────────────────────────────────

/// Worst-case classification margin across all vowels and both F0 values.
fn worst_case_margin(slots: &[Slot], selected: &[usize]) -> f64 {
    let targets: Vec<(f64, f64)> = selected.iter().map(|&i| (slots[i].f1, slots[i].f2)).collect();
    let mut worst = f64::MAX;
    for (i, &idx) in selected.iter().enumerate() {
        let s = &slots[idx];
        let m_lo = margin(s.det_low.f1 as f64, s.det_low.f2 as f64, &targets, i);
        let m_hi = margin(s.det_high.f1 as f64, s.det_high.f2 as f64, &targets, i);
        worst = worst.min(m_lo).min(m_hi);
    }
    worst
}

fn swap_refine(slots: &[Slot], selected: &mut Vec<usize>) {
    let n = slots.len();

    // First pass: maximize min pairwise target distance
    let mut current_min = min_pairwise_dist(slots, selected);
    let mut improved = true;
    while improved {
        improved = false;
        for sel_pos in 0..selected.len() {
            for cand in 0..n {
                if selected.contains(&cand) {
                    continue;
                }
                let mut valid = true;
                for (k, &j) in selected.iter().enumerate() {
                    if k == sel_pos {
                        continue;
                    }
                    if conflicts(&slots[cand], &slots[j]) {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    continue;
                }

                let old = selected[sel_pos];
                selected[sel_pos] = cand;
                let new_min = min_pairwise_dist(slots, selected);
                if new_min > current_min {
                    current_min = new_min;
                    improved = true;
                } else {
                    selected[sel_pos] = old;
                }
            }
        }
    }

    // Second pass: among solutions with same min pairwise distance,
    // prefer the one with better worst-case margin
    let mut current_margin = worst_case_margin(slots, selected);
    improved = true;
    while improved {
        improved = false;
        for sel_pos in 0..selected.len() {
            for cand in 0..n {
                if selected.contains(&cand) {
                    continue;
                }
                let mut valid = true;
                for (k, &j) in selected.iter().enumerate() {
                    if k == sel_pos {
                        continue;
                    }
                    if conflicts(&slots[cand], &slots[j]) {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    continue;
                }

                let old = selected[sel_pos];
                selected[sel_pos] = cand;
                let new_dist = min_pairwise_dist(slots, selected);
                let new_margin = worst_case_margin(slots, selected);
                if new_dist >= current_min && new_margin > current_margin {
                    current_min = new_dist;
                    current_margin = new_margin;
                    improved = true;
                } else {
                    selected[sel_pos] = old;
                }
            }
        }
    }
}

// ── Phase 4: Verification ────────────────────────────────────────────

/// Verify using the actual harmonic_amplitudes model: for each vowel at
/// each F0, find the strongest harmonic in each band and check it matches
/// the expected detected pair.
fn verify_analytic(slots: &[Slot], selected: &[usize]) -> bool {
    let mut all_ok = true;

    for &idx in selected {
        let s = &slots[idx];
        for &(f0, label) in &[(F0_LOW, "LOW"), (F0_HIGH, "HIGH")] {
            let amps = formant::harmonic_amplitudes(s.f1, s.f2, f0);
            let mut best_f1 = (0.0, 0.0f64);
            let mut best_f2 = (0.0, 0.0f64);

            for h in 0..NUM_HARMONICS {
                let freq = f0 * (h + 1) as f64;
                if freq >= F1_LO && freq <= F1_HI && amps[h] > best_f1.1 {
                    best_f1 = (freq, amps[h]);
                }
                if freq >= F2_LO && freq <= F2_HI && amps[h] > best_f2.1 {
                    best_f2 = (freq, amps[h]);
                }
            }

            let expected = if f0 == F0_LOW { s.det_low } else { s.det_high };

            if best_f1.0 as i32 != expected.f1 || best_f2.0 as i32 != expected.f2 {
                eprintln!(
                    "  MISMATCH vowel ({:.1},{:.1}) @ F0_{}: \
                     expected ({},{}), got ({:.0},{:.0})",
                    s.f1, s.f2, label, expected.f1, expected.f2, best_f1.0, best_f2.0
                );
                all_ok = false;
            }
        }
    }

    all_ok
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    println!("=== Vowel Alphabet Optimizer ===");
    println!(
        "=== {v} vowels × {p} pitches = {s} symbols, {b} bits/symbol ===\n",
        v = NUM_VOWELS,
        p = NUM_PITCHES,
        s = NUM_SYMBOLS,
        b = BITS_PER_SYMBOL
    );

    // Phase 1
    let slots = enumerate_slots();

    // Phase 2
    let mut selected = greedy_select(&slots, NUM_VOWELS);
    println!("Greedy min distance: {:.4}", min_pairwise_dist(&slots, &selected));

    // Phase 3
    swap_refine(&slots, &mut selected);
    println!(
        "After swap refinement: {:.4}\n",
        min_pairwise_dist(&slots, &selected)
    );

    // Sort by F1 then F2
    selected.sort_by(|&a, &b| {
        slots[a]
            .f1
            .partial_cmp(&slots[b].f1)
            .unwrap()
            .then(slots[a].f2.partial_cmp(&slots[b].f2).unwrap())
    });

    // Build targets for margin computation
    let targets: Vec<(f64, f64)> = selected.iter().map(|&i| (slots[i].f1, slots[i].f2)).collect();

    // Phase 4: Report
    println!(
        "{:>3} {:>8} {:>8} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} | {:>7} {:>7}",
        "#", "F1_tgt", "F2_tgt", "F1_rng", "F2_rng", "dLo_F1", "dLo_F2", "dHi_F1", "dHi_F2", "marg_lo", "marg_hi"
    );
    println!("{}", "-".repeat(100));

    let mut worst_margin = f64::MAX;
    for (i, &idx) in selected.iter().enumerate() {
        let s = &slots[idx];
        let m_lo = margin(s.det_low.f1 as f64, s.det_low.f2 as f64, &targets, i);
        let m_hi = margin(s.det_high.f1 as f64, s.det_high.f2 as f64, &targets, i);
        worst_margin = worst_margin.min(m_lo).min(m_hi);
        println!(
            "{:>3} {:>8.1} {:>8.1} {:>6.0} {:>6.0} | {:>6.0} {:>6.0} {:>6.0} {:>6.0} | {:>7.3} {:>7.3}",
            i,
            s.f1,
            s.f2,
            s.f1_range.1 - s.f1_range.0,
            s.f2_range.1 - s.f2_range.0,
            s.det_low.f1,
            s.det_low.f2,
            s.det_high.f1,
            s.det_high.f2,
            m_lo,
            m_hi,
        );
    }

    println!("\nMin pairwise target distance: {:.4}", min_pairwise_dist(&slots, &selected));
    println!("Worst-case classification margin: {:.4}", worst_margin);

    // Analytic verification
    print!("Harmonic model verification: ");
    if verify_analytic(&slots, &selected) {
        println!("PASS");
    } else {
        println!("FAIL (see mismatches above)");
    }

    // Compare with current vowels
    println!("\n--- Comparison with current VOWELS ---");
    let current = &formant::VOWELS;
    let cur_min = (0..current.len())
        .flat_map(|i| {
            (i + 1..current.len()).map(move |j| {
                dist(current[i].f1, current[i].f2, current[j].f1, current[j].f2)
            })
        })
        .fold(f64::MAX, f64::min);
    println!("Current min pairwise distance: {:.4}", cur_min);
    println!(
        "Optimized min pairwise distance: {:.4}",
        min_pairwise_dist(&slots, &selected)
    );

    // Suggest preamble pair (max distance)
    let mut best_pre = (0usize, 1usize);
    let mut best_pre_d = 0.0;
    for i in 0..selected.len() {
        for j in i + 1..selected.len() {
            let d = dist(
                slots[selected[i]].f1,
                slots[selected[i]].f2,
                slots[selected[j]].f1,
                slots[selected[j]].f2,
            );
            if d > best_pre_d {
                best_pre_d = d;
                best_pre = (i, j);
            }
        }
    }

    // Output
    println!("\n=== Optimized VOWELS (paste into formant.rs) ===\n");
    println!("pub const VOWELS: [VowelParams; NUM_VOWELS] = [");
    for (i, &idx) in selected.iter().enumerate() {
        let s = &slots[idx];
        let f1_label = if s.f1 < 350.0 {
            "low-F1"
        } else if s.f1 < 500.0 {
            "mid-F1"
        } else {
            "high-F1"
        };
        println!(
            "    VowelParams {{ f1: {:>6.1}, f2: {:>7.1} }}, // {i} — {f1}",
            s.f1,
            s.f2,
            i = i,
            f1 = f1_label,
        );
    }
    println!("];");

    let a = best_pre.0;
    let b = best_pre.1;
    println!(
        "\nPreamble vowels: {} and {} (distance {:.4})",
        a, b, best_pre_d
    );
    println!(
        "pub const PREAMBLE_START: [usize; PREAMBLE_LEN] = [{}, {}, {}, {}];",
        a * 2,
        b * 2,
        a * 2,
        b * 2
    );
    println!(
        "pub const PREAMBLE_END:   [usize; PREAMBLE_LEN] = [{}, {}, {}, {}];",
        b * 2,
        a * 2,
        b * 2,
        a * 2
    );
}
