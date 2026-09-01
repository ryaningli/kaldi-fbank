//! Golden-value tests: compares kaldi-fbank output against reference outputs
//! produced by the stock kaldi-native-fbank C++ library (see
//! `ref/fbank_cli.cc` and `scripts/gen_golden.sh`).
//!
//! The golden files cover 33 configurations x 3 signals (16k / 8k / 44.1k),
//! all generated deterministically.

use std::path::PathBuf;

use kaldi_fbank::{FbankOptions, KaldiRand, compute_fbank};

/// Loads a raw f32 file as a flat Vec<f32>.
fn load_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden file");
    assert_eq!(
        bytes.len() % 4,
        0,
        "file size not a multiple of 4: {path:?}"
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Compares two flat f32 buffers (same length) and asserts the max abs diff is
/// within the given tolerance, reporting the worst location.
fn assert_close(rs: &[f32], ref_: &[f32], atol: f32, ctx: &str) {
    assert_eq!(
        rs.len(),
        ref_.len(),
        "{ctx}: length mismatch: rust={} ref={}",
        rs.len(),
        ref_.len()
    );
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for (i, (a, b)) in rs.iter().zip(ref_.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        let rel = d / b.abs().max(1e-12);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_abs <= atol,
        "{ctx}: max abs diff {max_abs} > {atol} at index {worst}: rust={} ref={} (max_rel={max_rel})",
        rs[worst],
        ref_[worst]
    );
    println!("{ctx}: ok, max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}");
}

// ---------------------------------------------------------------------------
// Table-driven golden comparison
// ---------------------------------------------------------------------------

type OptsFn = fn(&mut FbankOptions);

struct GoldenCase {
    name: &'static str,
    signal: &'static str,
    sr: f32,
    opts: OptsFn,
}

fn base(o: &mut FbankOptions) {
    o.frame_opts.dither = 0.0;
}

fn d23(o: &mut FbankOptions) {
    base(o);
    o.mel_opts.num_bins = 23;
}
fn d40(o: &mut FbankOptions) {
    base(o);
    o.mel_opts.num_bins = 40;
}
fn d80(o: &mut FbankOptions) {
    base(o);
    o.mel_opts.num_bins = 80;
}
fn d80_snip0(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.snip_edges = false;
}
fn d40_snip0(o: &mut FbankOptions) {
    d40(o);
    o.frame_opts.snip_edges = false;
}
fn d80_energy(o: &mut FbankOptions) {
    d80(o);
    o.use_energy = true;
}
fn d80_energy_raw0(o: &mut FbankOptions) {
    d80_energy(o);
    o.raw_energy = false;
}
fn d80_energy_htk(o: &mut FbankOptions) {
    d80_energy(o);
    o.htk_compat = true;
}
fn d80_energy_floor(o: &mut FbankOptions) {
    d80_energy(o);
    o.energy_floor = 1.0;
}
fn d80_magnitude(o: &mut FbankOptions) {
    d80(o);
    o.use_power = false;
}
fn d80_linear(o: &mut FbankOptions) {
    d80(o);
    o.use_log_fbank = false;
}
fn d80_nopow2(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.round_to_power_of_two = false;
}
fn d80_shift5(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.frame_shift_ms = 5.0;
}
fn d80_shift20(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.frame_shift_ms = 20.0;
}
fn d80_len20(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.frame_length_ms = 20.0;
}
fn d80_len32(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.frame_length_ms = 32.0;
}
fn d80_preemph0(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.preemph_coeff = 0.0;
}
fn d80_removedc0(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.remove_dc_offset = false;
}
fn d80_lowfreq0(o: &mut FbankOptions) {
    d80(o);
    o.mel_opts.low_freq = 0.0;
}
fn d80_highfreq_neg(o: &mut FbankOptions) {
    d80(o);
    o.mel_opts.high_freq = -2000.0;
}
fn d80_librosa(o: &mut FbankOptions) {
    d80(o);
    o.mel_opts.is_librosa = true;
    o.mel_opts.low_freq = 0.0;
    o.mel_opts.norm = "slaney".into();
}
fn d80_librosa_nonorm(o: &mut FbankOptions) {
    d80_librosa(o);
    o.mel_opts.norm = String::new();
}
fn d80_librosa_noslaney(o: &mut FbankOptions) {
    d80_librosa(o);
    o.mel_opts.norm = String::new();
    o.mel_opts.use_slaney_mel_scale = false;
}
fn d80_librosa_floor(o: &mut FbankOptions) {
    d80_librosa(o);
    o.mel_opts.floor_to_int_bin = true;
}
fn d80_8k(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.samp_freq = 8000.0;
}
fn d40_snip0_8k(o: &mut FbankOptions) {
    d40_snip0(o);
    o.frame_opts.samp_freq = 8000.0;
}
fn d80_44k(o: &mut FbankOptions) {
    d80(o);
    o.frame_opts.samp_freq = 44100.0;
}
fn d80_snip0_44k(o: &mut FbankOptions) {
    d80_snip0(o);
    o.frame_opts.samp_freq = 44100.0;
}

const GOLDEN_CASES: &[GoldenCase] = &[
    // bins
    GoldenCase {
        name: "bins80",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80,
    },
    GoldenCase {
        name: "bins23",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d23,
    },
    GoldenCase {
        name: "bins40",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d40,
    },
    GoldenCase {
        name: "bins80_snip0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_snip0,
    },
    GoldenCase {
        name: "bins40_snip0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d40_snip0,
    },
    // windows
    GoldenCase {
        name: "hamming",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Hamming;
        },
    },
    GoldenCase {
        name: "hanning",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Hanning;
        },
    },
    GoldenCase {
        name: "hann",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Hann;
        },
    },
    GoldenCase {
        name: "sine",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Sine;
        },
    },
    GoldenCase {
        name: "rectangular",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Rectangular;
        },
    },
    GoldenCase {
        name: "blackman",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: |o| {
            d80(o);
            o.frame_opts.window_type = kaldi_fbank::WindowType::Blackman;
        },
    },
    // energy variants
    GoldenCase {
        name: "energy",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_energy,
    },
    GoldenCase {
        name: "energy_raw0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_energy_raw0,
    },
    GoldenCase {
        name: "energy_htk",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_energy_htk,
    },
    GoldenCase {
        name: "energy_floor",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_energy_floor,
    },
    // spectrum variants
    GoldenCase {
        name: "magnitude",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_magnitude,
    },
    GoldenCase {
        name: "linear",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_linear,
    },
    // geometry
    GoldenCase {
        name: "nopow2",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_nopow2,
    },
    GoldenCase {
        name: "shift5ms",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_shift5,
    },
    GoldenCase {
        name: "shift20ms",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_shift20,
    },
    GoldenCase {
        name: "len20ms",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_len20,
    },
    GoldenCase {
        name: "len32ms",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_len32,
    },
    // preprocessing
    GoldenCase {
        name: "preemph0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_preemph0,
    },
    GoldenCase {
        name: "removedc0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_removedc0,
    },
    // freq bounds
    GoldenCase {
        name: "lowfreq0",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_lowfreq0,
    },
    GoldenCase {
        name: "highfreq_neg",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_highfreq_neg,
    },
    // librosa
    GoldenCase {
        name: "librosa",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_librosa,
    },
    GoldenCase {
        name: "librosa_nonorm",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_librosa_nonorm,
    },
    GoldenCase {
        name: "librosa_noslaney",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_librosa_noslaney,
    },
    GoldenCase {
        name: "librosa_floor",
        signal: "pspeech_2s.f32",
        sr: 16000.0,
        opts: d80_librosa_floor,
    },
    // sample rates
    GoldenCase {
        name: "bins80_8k",
        signal: "pspeech_2s_8k.f32",
        sr: 8000.0,
        opts: d80_8k,
    },
    GoldenCase {
        name: "bins40_snip0_8k",
        signal: "pspeech_2s_8k.f32",
        sr: 8000.0,
        opts: d40_snip0_8k,
    },
    GoldenCase {
        name: "bins80_44k",
        signal: "pspeech_2s_44k.f32",
        sr: 44100.0,
        opts: d80_44k,
    },
    GoldenCase {
        name: "bins80_snip0_44k",
        signal: "pspeech_2s_44k.f32",
        sr: 44100.0,
        opts: d80_snip0_44k,
    },
];

#[test]
fn golden_table_all_configs() {
    for case in GOLDEN_CASES {
        let sig = load_f32(&golden_dir().join(case.signal));
        let mut opts = FbankOptions::default();
        (case.opts)(&mut opts);
        let out = compute_fbank(&opts, &sig).concat();
        let ref_ = load_f32(&golden_dir().join(format!("pspeech_{}.fbank", case.name)));
        // tolerance 2e-3: blackman-windowed bins near the eps floor amplify FFT
        // rounding differences slightly above 1e-3; still far below signal level
        assert_close(
            &out,
            &ref_,
            2e-3,
            &format!("pspeech {} (sr={})", case.name, case.sr),
        );
    }
}

#[test]
fn golden_table_chunked_streaming_matches() {
    // Same golden, but fed in small chunks (verifies the online path, not just
    // the batch path).
    for case in GOLDEN_CASES.iter().step_by(4) {
        let sig = load_f32(&golden_dir().join(case.signal));
        let mut opts = FbankOptions::default();
        (case.opts)(&mut opts);
        let mut fbank = kaldi_fbank::OnlineFbank::new(opts);
        for chunk in sig.chunks(997) {
            fbank.accept_waveform(case.sr, chunk);
        }
        fbank.input_finished();
        let n = fbank.num_frames_ready();
        let out: Vec<f32> = (0..n).flat_map(|i| fbank.get_frame(i).to_vec()).collect();
        let ref_ = load_f32(&golden_dir().join(format!("pspeech_{}.fbank", case.name)));
        assert_close(&out, &ref_, 2e-3, &format!("pspeech {} chunked", case.name));
    }
}

// ---------------------------------------------------------------------------
// Streaming / RNG / misc
// ---------------------------------------------------------------------------

#[test]
fn online_streaming_matches_batch() {
    let sig = load_f32(&golden_dir().join("pspeech_2s.f32"));
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.mel_opts.num_bins = 80;
    opts.frame_opts.snip_edges = false;

    let batch = compute_fbank(&opts, &sig);

    let mut fbank = kaldi_fbank::OnlineFbank::new(opts);
    for chunk in sig.chunks(100) {
        fbank.accept_waveform(16000.0, chunk);
    }
    fbank.input_finished();
    assert_eq!(fbank.num_frames_ready() as usize, batch.len());
    for (i, frame) in batch.iter().enumerate() {
        assert_eq!(fbank.get_frame(i as i32), frame.as_slice(), "frame {i}");
    }
}

#[test]
fn kaldi_rand_r_matches_glibc() {
    // Actual glibc rand_r values for seed 27437 (verified with a C program).
    let mut rng = KaldiRand::new(27437);
    let expected = [
        708_046_589u32,
        1_247_335_583,
        202_211_993,
        1_168_944_080,
        19_599_450,
    ];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(rng.rand(), e, "rand_r value {i}");
    }
}

#[test]
fn recycling_vector_pop_preserves_indices() {
    use kaldi_fbank::RecyclingVector;
    let mut v = RecyclingVector::new(3); // hold at most 3 items
    for i in 0..5 {
        v.push_back(vec![i as f32]);
    }
    assert_eq!(v.len(), 5);
    // Old frames are dropped but indices stay stable.
    assert_eq!(v.at(4), &[4.0]);
    v.pop(2);
    assert_eq!(v.len(), 5); // logical size unchanged
    assert_eq!(v.at(4), &[4.0]);
    v.pop(10); // popping more than available must be safe
    assert_eq!(v.len(), 5);
}

#[test]
fn dim_and_frame_shift() {
    let opts = FbankOptions::default();
    let fbank = kaldi_fbank::OnlineFbank::new(opts);
    assert_eq!(fbank.dim(), 23); // FbankOptions default num_bins
    assert!((fbank.frame_shift_in_seconds() - 0.01).abs() < 1e-6);
}

#[test]
fn dither_deterministic_and_distributed() {
    let sig = load_f32(&golden_dir().join("pspeech_2s.f32"));
    let mut opts = FbankOptions::default();
    opts.mel_opts.num_bins = 80;
    opts.frame_opts.dither = 0.0;

    let plain = compute_fbank(&opts, &sig);

    opts.frame_opts.dither = 0.00003;
    let d1 = compute_fbank(&opts, &sig);
    let d2 = compute_fbank(&opts, &sig);
    assert_eq!(d1, d2, "dither must be deterministic for a fixed seed");
    assert_ne!(d1, plain, "dither must change the output");
}
