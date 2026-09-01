//! Edge-case tests: degenerate inputs, boundary frame counts, extreme values,
//! special signals, unusual sample rates.

use kaldi_fbank::{FbankOptions, OnlineFbank, compute_fbank, num_frames};

fn base_opts() -> FbankOptions {
    let mut o = FbankOptions::default();
    o.frame_opts.dither = 0.0;
    o.mel_opts.num_bins = 80;
    o
}

fn extract_all(opts: &FbankOptions, sig: &[f32]) -> Vec<Vec<f32>> {
    compute_fbank(opts, sig)
}

// --- degenerate inputs ------------------------------------------------------

#[test]
fn zero_length_input() {
    let opts = base_opts();
    let frames = extract_all(&opts, &[]);
    assert!(frames.is_empty());
}

#[test]
fn one_sample_input() {
    let opts = base_opts();
    let frames = extract_all(&opts, &[0.5]);
    assert!(frames.is_empty());
}

#[test]
fn exactly_one_frame_length() {
    let opts = base_opts();
    // 400 samples = exactly one 25 ms frame at 16 kHz
    let sig = vec![0.1f32; 400];
    let frames = extract_all(&opts, &sig);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 80);
}

#[test]
fn one_frame_plus_one_sample() {
    let opts = base_opts();
    let sig = vec![0.1f32; 401];
    let frames = extract_all(&opts, &sig);
    assert_eq!(frames.len(), 1); // still 1 frame
}

#[test]
fn just_below_frame_length() {
    let opts = base_opts();
    let sig = vec![0.1f32; 399];
    let frames = extract_all(&opts, &sig);
    assert!(frames.is_empty());
}

// --- special signals ---------------------------------------------------------

#[test]
fn all_zero_signal_gives_floor() {
    let opts = base_opts();
    let sig = vec![0.0f32; 16000];
    let frames = extract_all(&opts, &sig);
    assert!(!frames.is_empty());
    let floor = f32::EPSILON.ln();
    for frame in &frames {
        for v in frame {
            assert!((v - floor).abs() < 1e-3, "v={v} floor={floor}");
        }
    }
}

#[test]
fn constant_signal_dc_removed() {
    // After DC removal a constant signal is all zeros -> floor values.
    let opts = base_opts();
    let sig = vec![0.3f32; 16000];
    let frames = extract_all(&opts, &sig);
    let floor = f32::EPSILON.ln();
    for frame in &frames {
        for v in frame {
            assert!((v - floor).abs() < 1e-3);
        }
    }
}

#[test]
fn constant_signal_no_dc_removal_has_energy() {
    let mut opts = base_opts();
    opts.frame_opts.remove_dc_offset = false;
    let sig = vec![0.3f32; 16000];
    let frames = extract_all(&opts, &sig);
    // DC energy dominates the lowest mel bin (preemphasis scales DC by 0.03,
    // so the value is still negative in log domain, but it must be the max)
    let f0 = &frames[0];
    let max = f0.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        f0[0] >= max - 1e-3,
        "DC bin should be the strongest: {f0:?}"
    );
    assert!(f0.iter().all(|v| v.is_finite()));
    // and a remove_dc=true run of the same signal must differ (all floor)
    opts.frame_opts.remove_dc_offset = true;
    let frames2 = extract_all(&opts, &sig);
    assert_ne!(f0, &frames2[0][..]);
}

#[test]
fn pure_tone_signal() {
    // 440 Hz sine: energy concentrated in low bins; log-fbank shape sane.
    let opts = base_opts();
    let sig: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let frames = extract_all(&opts, &sig);
    let f0 = &frames[0];
    assert!(f0.iter().all(|v| v.is_finite()));
    let max = f0.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        max > -2.0,
        "440 Hz should produce strong mel energy, max={max}"
    );
    // high bins (above 4 kHz) should be quiet for a pure tone
    assert!(
        f0[60..].iter().all(|&v| v < max - 3.0 || v < 0.0),
        "high bins too strong"
    );
}

#[test]
fn extreme_value_signal() {
    let opts = base_opts();
    let sig = vec![1e10f32; 1600];
    let frames = extract_all(&opts, &sig);
    assert!(
        frames.iter().flatten().all(|v| v.is_finite()),
        "extreme input must stay finite"
    );
    let tiny = vec![-1e10f32; 1600];
    let frames2 = extract_all(&opts, &tiny);
    assert!(frames2.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn mixed_amplitude_signal() {
    // Alternating loud/quiet: frames must reflect that.
    let opts = base_opts();
    let mut sig = Vec::with_capacity(32000);
    for i in 0..32000 {
        let amp = if (i / 16000) % 2 == 0 { 1.0 } else { 0.001 };
        sig.push(amp * (i as f32 * 0.01).sin());
    }
    let frames = extract_all(&opts, &sig);
    assert!(frames.len() >= 100);
    let loud: f32 = frames[20].iter().sum();
    let quiet: f32 = frames[120].iter().sum();
    assert!(loud > quiet, "loud={loud} quiet={quiet}");
}

// --- frame counting cross-check ----------------------------------------------

#[test]
fn frame_count_matches_formula_snip_true() {
    let opts = base_opts();
    for n in [0usize, 1, 399, 400, 560, 1600, 16000, 160000] {
        let frames = extract_all(&opts, &vec![0.0f32; n]);
        let expected = if n < 400 { 0 } else { 1 + (n - 400) / 160 };
        assert_eq!(frames.len(), expected, "n={n}");
    }
}

#[test]
fn frame_count_matches_formula_snip_false() {
    let mut opts = base_opts();
    opts.frame_opts.snip_edges = false;
    for n in [0usize, 1, 80, 160, 400, 560, 1600, 16000] {
        let frames = extract_all(&opts, &vec![0.0f32; n]);
        let expected = (n + 80) / 160;
        assert_eq!(frames.len(), expected, "n={n}");
    }
}

#[test]
fn num_frames_consistent_with_online() {
    let opts = base_opts();
    for n in [400usize, 1000, 1600, 12345] {
        let mut f = OnlineFbank::new(opts.clone());
        f.accept_waveform(16000.0, &vec![0.0f32; n]);
        f.input_finished();
        assert_eq!(
            f.num_frames_ready() as usize,
            num_frames(n as i64, &opts.frame_opts, true) as usize,
            "n={n}"
        );
    }
}

// --- unusual sample rates -----------------------------------------------------

#[test]
fn eight_khz_input() {
    let mut opts = base_opts();
    opts.frame_opts.samp_freq = 8000.0;
    let sig: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
    let frames = extract_all(&opts, &sig);
    assert!(!frames.is_empty());
    assert!(frames.iter().flatten().all(|v| v.is_finite()));
    // 2s at 8 kHz, 200-sample frames, 80-sample shift, snip: 1+(16000-200)/80
    assert_eq!(frames.len(), 1 + (16000 - 200) / 80);
}

#[test]
fn forty_four_khz_input() {
    let mut opts = base_opts();
    opts.frame_opts.samp_freq = 44100.0;
    let sig: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin()).collect();
    let frames = extract_all(&opts, &sig);
    assert!(!frames.is_empty());
    assert!(frames.iter().flatten().all(|v| v.is_finite()));
    // 1s at 44.1 kHz: window 1102, shift 441, snip: 1 + (44100-1102)/441
    assert_eq!(frames.len(), 1 + (44100 - 1102) / 441);
}

// --- non-power-of-two window ---------------------------------------------------

#[test]
fn non_pow2_window_size() {
    let mut opts = base_opts();
    opts.frame_opts.round_to_power_of_two = false;
    let sig: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
    let frames = extract_all(&opts, &sig);
    assert!(!frames.is_empty());
    assert!(frames.iter().flatten().all(|v| v.is_finite()));
    // 400-sample frames (not padded): same count as padded
    let padded = base_opts();
    let frames_padded = extract_all(&padded, &sig);
    assert_eq!(frames.len(), frames_padded.len());
}

// --- snip_edges=false boundary reflections --------------------------------------

#[test]
fn snip_false_first_and_last_frames() {
    let mut opts = base_opts();
    opts.frame_opts.snip_edges = false;
    let sig: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
    let frames = extract_all(&opts, &sig);
    assert_eq!(frames.len(), (16000 + 80) / 160);
    assert!(frames.iter().flatten().all(|v| v.is_finite()));
    // first frame uses reflection (starts at -120); must still be finite
    assert!(frames[0].iter().all(|v| v.is_finite()));
}

// --- use_energy combinations ----------------------------------------------------

#[test]
fn energy_dimension_sizes() {
    for (energy, htk, expect) in [(false, false, 80), (true, false, 81), (true, true, 81)] {
        let mut opts = base_opts();
        opts.use_energy = energy;
        opts.htk_compat = htk;
        let sig = vec![0.1f32; 16000];
        let frames = extract_all(&opts, &sig);
        assert_eq!(frames[0].len(), expect, "energy={energy} htk={htk}");
    }
}

// --- determinism -----------------------------------------------------------------

#[test]
fn repeated_calls_identical() {
    let opts = base_opts();
    let sig: Vec<f32> = (0..32000)
        .map(|i| ((i * 37) % 1000) as f32 / 1000.0)
        .collect();
    let a = extract_all(&opts, &sig);
    let b = extract_all(&opts, &sig);
    assert_eq!(a, b);
}
