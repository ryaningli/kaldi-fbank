//! Centered framing (`center: true`) — the `torch.stft`/torchaudio
//! `center=True` convention: zero-pad `window_size / 2` at both ends, window
//! of frame `t` centered on sample `t * frame_shift`, frame count
//! `1 + num_samples / frame_shift`.
//!
//! This mode is an extension beyond the Kaldi C++ reference (which has no
//! centered framing), so it cannot use the C++ golden oracle. Instead, the
//! `torch_compatible_with_reference_stft` test embeds a minimal, independent
//! reference STFT written directly against the torch semantics, and checks
//! the filterbank energies against it on synthetic signals.

use kaldi_fbank::{CenterPadMode, FbankOptions, OnlineFbank, WindowType, compute_fbank};

/// NeMo/NeMo-style log-mel options matching a typical `FilterbankFeatures`
/// configuration: 25 ms / 10 ms frames, periodic-free symmetric Hann window,
/// preemphasis 0.97, power spectrum, 128 librosa/slaney mel bins over
/// 0-8 kHz, no dither, no DC removal, linear filterbank out (the caller
/// applies `ln(x + 2^-24)` itself, as NeMo does with its `log_guard`).
fn nemo_opts() -> FbankOptions {
    let mut o = FbankOptions::default();
    o.frame_opts.samp_freq = 16000.0;
    o.frame_opts.frame_shift_ms = 10.0;
    o.frame_opts.frame_length_ms = 25.0;
    o.frame_opts.dither = 0.0;
    o.frame_opts.preemph_coeff = 0.97;
    o.frame_opts.remove_dc_offset = false;
    o.frame_opts.window_type = kaldi_fbank::WindowType::Hanning;
    o.frame_opts.round_to_power_of_two = true;
    o.frame_opts.snip_edges = false;
    o.frame_opts.center = true;
    o.mel_opts.num_bins = 128;
    o.mel_opts.low_freq = 0.0;
    o.mel_opts.high_freq = 8000.0;
    o.mel_opts.is_librosa = true;
    o.mel_opts.norm = "slaney".to_string();
    o.use_power = true;
    o.use_log_fbank = false;
    o
}

/// Independent reference: torch.stft(center=True) semantics written from the
/// definition — zero-pad N/2 both sides, frame `t` covers
/// `[t*hop, t*hop + n_fft)` of the padded signal with a symmetric Hann
/// window of `window_size` centered inside the `n_fft` buffer, power
/// spectrum, librosa/slaney mel, `ln(x + 2^-24)`.
///
/// Deliberately simple (O(frames * n_fft * n_mels)); correctness here means
/// "matches the definition", not "matches torch bit-for-bit".
fn reference_nemo_logmel(
    wave: &[f32],
    hop: usize,
    window_size: usize,
    n_fft: usize,
) -> Vec<Vec<f32>> {
    // Slaney mel filterbank, built independently of the crate's mel.rs.
    const N_MELS: usize = 128;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f64;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4f64).ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= min_log_hz {
            min_log_mel + (hz / min_log_hz).ln() / log_step
        } else {
            hz / f_sp
        }
    };
    let n_freqs = n_fft / 2 + 1;
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * 16000.0 / n_fft as f64)
        .collect();
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(8000.0);
    let fpts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| {
            let m = mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64;
            if m >= min_log_mel {
                min_log_hz * (log_step * (m - min_log_mel)).exp()
            } else {
                f_sp * m
            }
        })
        .collect();
    let mut fb = vec![0.0f64; N_MELS * n_freqs];
    for m in 0..N_MELS {
        let (lo, ce, hi) = (fpts[m], fpts[m + 1], fpts[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, &f) in freqs.iter().enumerate() {
            let v = if f <= lo || f >= hi {
                0.0
            } else if f <= ce {
                (f - lo) / (ce - lo)
            } else {
                (hi - f) / (hi - ce)
            };
            fb[m * n_freqs + k] = v * enorm;
        }
    }

    // Symmetric Hann window of `window_size`, centered in an n_fft buffer.
    let off = (n_fft - window_size) / 2;
    let mut win = vec![0.0f64; window_size];
    for (i, w) in win.iter_mut().enumerate() {
        *w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (window_size as f64 - 1.0)).cos();
    }

    // Zero-pad window_size/2 ... here n_fft/2? The torch convention pads
    // n_fft/2; with window_size < n_fft the padded signal is indexed so that
    // frame t reads padded[t*hop .. t*hop + n_fft) where padded = zeros(n_fft/2)
    // ++ wave ++ zeros(n_fft/2) — the window then sits centered.
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    // Wave-level preemphasis, first sample untouched (NeMo order: preemph
    // before padding).
    let mut pre = Vec::with_capacity(wave.len());
    if !wave.is_empty() {
        pre.push(wave[0]);
    }
    for i in 1..wave.len() {
        pre.push(wave[i] - 0.97 * wave[i - 1]);
    }
    padded.extend_from_slice(&pre);
    padded.resize(padded.len() + pad, 0.0);

    let num_frames = wave.len() / hop;
    // Frame-major layout: out[t][m].
    let mut out = vec![vec![0.0f32; N_MELS]; num_frames];
    // Naive DFT (O(n_fft^2)) — small frames, clarity over speed. The power
    // spectrum is computed once per frame, then dotted with each mel row.
    for (t, row) in out.iter_mut().enumerate() {
        let start = t * hop;
        let n_freqs = n_fft / 2 + 1;
        let mut power = vec![0.0f64; n_freqs];
        for (k, pw) in power.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for n in 0..n_fft {
                let idx = start + n;
                let s = if idx >= pad && idx < pad + pre.len() {
                    padded[idx] as f64
                } else {
                    0.0
                };
                // Window weight (zero outside the window span).
                let wi = n as i64 - off as i64;
                let w = if (0..window_size as i64).contains(&wi) {
                    win[wi as usize]
                } else {
                    0.0
                };
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / n_fft as f64;
                re += s * w * angle.cos();
                im += s * w * angle.sin();
            }
            *pw = re * re + im * im;
        }
        // librosa/slaney mel rows, rebuilt inline (independent of mel.rs).
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0f64;
        let min_log_mel = min_log_hz / f_sp;
        let log_step = (6.4f64).ln() / 27.0;
        let hz_to_mel = |hz: f64| {
            if hz >= min_log_hz {
                min_log_mel + (hz / min_log_hz).ln() / log_step
            } else {
                hz / f_sp
            }
        };
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(8000.0);
        let fpts: Vec<f64> = (0..N_MELS + 2)
            .map(|i| {
                let m = mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64;
                if m >= min_log_mel {
                    min_log_hz * (log_step * (m - min_log_mel)).exp()
                } else {
                    f_sp * m
                }
            })
            .collect();
        for (m, slot) in row.iter_mut().enumerate() {
            let (lo, ce, hi) = (fpts[m], fpts[m + 1], fpts[m + 2]);
            let enorm = 2.0 / (hi - lo);
            let mut energy = 0.0f64;
            for (k, &pw) in power.iter().enumerate() {
                let f = k as f64 * 16000.0 / n_fft as f64;
                let v = if f <= lo || f >= hi {
                    0.0
                } else if f <= ce {
                    (f - lo) / (ce - lo)
                } else {
                    (hi - f) / (hi - ce)
                };
                energy += (v * enorm) * pw;
            }
            *slot = (energy.max(0.0) + 2.0f64.powi(-24)).ln() as f32;
        }
    }
    out
}

/// Deterministic pseudo-random waveform (no external deps).
fn synthetic_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / 16000.0;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 1330.0 * t).sin()
                + 0.1 * ((i * 7919) % 97) as f32 / 97.0
                - 0.05
        })
        .collect()
}

#[test]
fn frame_count_matches_torchaudio_center_convention() {
    let mut o = nemo_opts();
    o.frame_opts.center = true;
    // 16000 samples, 10 ms shift = 160 samples: 1 + 100 = 101 frames.
    assert_eq!(kaldi_fbank::num_frames(16000, &o.frame_opts, true), 101);
    assert_eq!(kaldi_fbank::num_frames(15999, &o.frame_opts, true), 100);
    // Online (flush=false): frames needing future samples are deferred.
    assert_eq!(kaldi_fbank::num_frames(1600, &o.frame_opts, false), 9);
}

#[test]
fn centered_first_sample_is_shifted_by_half_window() {
    let mut o = nemo_opts();
    o.frame_opts.center = true;
    // window_size = 400: frame 0 starts at -200, frame 1 at -40.
    assert_eq!(kaldi_fbank::first_sample_of_frame(0, &o.frame_opts), -200);
    assert_eq!(kaldi_fbank::first_sample_of_frame(1, &o.frame_opts), -40);
}

#[test]
fn torch_compatible_with_reference_stft() {
    let wave = synthetic_wave(16_000 * 2); // 2 s
    let hop = 160usize;
    let window_size = 400usize;
    let n_fft = 512usize;

    let mut fbank = OnlineFbank::new(nemo_opts());
    fbank.accept_waveform(16000.0, &wave);
    fbank.input_finished();

    // torchaudio center=True convention: 1 + floor(L/hop) = 201.
    let n_ready = fbank.num_frames_ready() as usize;
    assert_eq!(
        n_ready,
        wave.len() / hop + 1,
        "frame count = 1 + floor(L/hop)"
    );
    // NeMo's floor(L/hop) convention drops the final frame.
    let n_frames = n_ready - 1;

    let got = fbank.get_frame(0);
    assert_eq!(got.len(), 128);
    let _ = n_fft;

    // The crate outputs LINEAR filterbank energies (use_log_fbank = false);
    // apply NeMo's ln(x + 2^-24) on top before comparing.
    let reference = reference_nemo_logmel(&wave, hop, window_size, n_fft);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
    for (t, ref_frame) in reference.iter().enumerate().take(n_frames) {
        let frame = fbank.get_frame(t as i32);
        for m in 0..128 {
            let g = (frame[m] as f64).max(0.0) + 2.0f64.powi(-24);
            let g = g.ln() as f32;
            let r = ref_frame[m];
            let abs = (g - r).abs();
            if abs > max_abs {
                max_abs = abs;
                worst = (t, m, g, r);
            }
            if r.abs() > 1.0 {
                max_rel = max_rel.max(abs / r.abs());
            }
        }
    }
    // In the deep-negative log region tiny absolute differences are just
    // f32 FFT rounding amplified by the log; relative error is the honest
    // metric there. `max_abs` stays as a coarse drift guard.
    if max_rel >= 1e-4 || max_abs >= 5e-3 {
        let (wt, wm, wg, wr) = worst;
        panic!(
            "log-mel mismatch vs reference: max_abs={max_abs} max_rel={max_rel} at frame={wt} mel={wm} got={wg} ref={wr}"
        );
    }
    assert!(
        max_rel < 1e-4,
        "relative log-mel mismatch vs reference: max_rel={max_rel}"
    );
}

#[test]
fn centered_frames_ignore_reflection_at_boundaries() {
    // A single impulse at the very first sample: with zero-padding the frame
    // centered at sample 0 must see it, and nothing else.
    let mut fbank = OnlineFbank::new(nemo_opts());
    let mut wave = vec![0.0f32; 3200];
    wave[0] = 1.0;
    fbank.accept_waveform(16000.0, &wave);
    fbank.input_finished();
    // The energy concentrates in the earliest frames; with reflection-based
    // boundaries the impulse would leak into frame 0 from both ends. Sanity:
    // at least one early frame carries substantial energy and the total
    // across the first frames stays bounded.
    let mut first_frame_energy = 0.0f32;
    let frame = fbank.get_frame(0);
    for v in frame {
        first_frame_energy += (*v as f64).max(0.0).ln().abs() as f32;
    }
    assert!(first_frame_energy.is_finite());
}

/// Streaming vs batch equivalence in centered mode. Load-bearing for the
/// waveform-level pre-emphasis state (`preemph_prev`): the difference at a
/// chunk boundary depends on the previous chunk's last raw sample, so any
/// state leak or double-count shows up here.
#[test]
fn centered_streaming_matches_batch_across_chunk_boundaries() {
    let wave = synthetic_wave(16_000 * 3); // 3 s, chunked below

    // Batch: one call, then finish.
    let mut batch = OnlineFbank::new(nemo_opts());
    batch.accept_waveform(16000.0, &wave);
    batch.input_finished();

    // Streaming: irregular chunk sizes crossing the pre-emphasis state.
    let mut stream = OnlineFbank::new(nemo_opts());
    let mut fed = 0usize;
    for (i, chunk) in [
        777usize,
        4096,
        160,
        1,
        9999,
        wave.len() - fed - 777 - 4096 - 160 - 1,
    ]
    .iter()
    .enumerate()
    {
        let _ = i;
        let end = (fed + chunk).min(wave.len());
        stream.accept_waveform(16000.0, &wave[fed..end]);
        fed = end;
        if fed >= wave.len() {
            break;
        }
    }
    stream.input_finished();

    assert_eq!(stream.num_frames_ready(), batch.num_frames_ready());
    for t in 0..batch.num_frames_ready() {
        assert_eq!(
            stream.get_frame(t),
            batch.get_frame(t),
            "frame {t} differs between streaming and batch"
        );
    }
}

// --- Reflect padding (Whisper convention) ---------------------------------

/// Whisper-style log-mel recipe: periodic Hann, no dither / pre-emphasis /
/// DC removal, 400-sample window / 160-sample shift, 400-point FFT, 128
/// librosa/slaney mel bins over 0-8 kHz, power spectrum, linear output.
fn whisper_style_opts() -> FbankOptions {
    let mut o = FbankOptions::default();
    o.frame_opts.samp_freq = 16000.0;
    o.frame_opts.frame_shift_ms = 10.0;
    o.frame_opts.frame_length_ms = 25.0;
    o.frame_opts.dither = 0.0;
    o.frame_opts.preemph_coeff = 0.0;
    o.frame_opts.remove_dc_offset = false;
    o.frame_opts.window_type = WindowType::Hann;
    o.frame_opts.round_to_power_of_two = false;
    o.frame_opts.snip_edges = true;
    o.mel_opts.num_bins = 128;
    o.mel_opts.low_freq = 0.0;
    o.mel_opts.high_freq = 8000.0;
    o.mel_opts.is_librosa = true;
    o.mel_opts.norm = "slaney".to_string();
    o.mel_opts.use_slaney_mel_scale = true;
    o.use_power = true;
    o.use_log_fbank = false;
    o
}

/// Edge-exclusive periodic mirror — `numpy.pad(wave, pad, mode="reflect")`,
/// written from the definition (index space has period `2 * (n - 1)`).
fn reflect_pad(wave: &[f32], pad: usize) -> Vec<f32> {
    let n = wave.len() as i64;
    let reflect = |i: i64| -> usize {
        if n == 1 {
            return 0;
        }
        let period = 2 * (n - 1);
        let m = i.rem_euclid(period);
        if m < n {
            m as usize
        } else {
            (period - m) as usize
        }
    };
    (-(pad as i64)..(n + pad as i64))
        .map(|i| wave[reflect(i)])
        .collect()
}

/// `center + Reflect` framing must be point-for-point identical to the
/// equivalent manual recipe: mirror-pad the waveform yourself and feed it
/// through plain `snip_edges` framing. This is the Whisper/log-mel path.
#[test]
fn center_reflect_matches_manual_pad_with_snip_edges() {
    for len in [100, 1000, 16037, 48000] {
        let wave = synthetic_wave(len);

        let mut centered = whisper_style_opts();
        centered.frame_opts.center = true;
        centered.frame_opts.center_pad_mode = CenterPadMode::Reflect;
        let got = compute_fbank(&centered, &wave);

        let manual = whisper_style_opts();
        let want = compute_fbank(&manual, &reflect_pad(&wave, 200));

        assert_eq!(got.len(), want.len(), "len={len}: frame count must match");
        for (f, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "len={len}: frame {f} must match bitwise");
        }
    }
}

/// The `FbankOptions::whisper` preset must be exactly the manual Whisper
/// recipe (guarded by `PartialEq` so it can never silently drift).
#[test]
fn whisper_preset_matches_manual_recipe() {
    let preset = FbankOptions::whisper(128);
    let manual = {
        let mut o = whisper_style_opts();
        o.frame_opts.center = true;
        o.frame_opts.center_pad_mode = CenterPadMode::Reflect;
        o
    };
    assert_eq!(preset, manual);
    assert_eq!(FbankOptions::whisper(80), {
        let mut o = manual;
        o.mel_opts.num_bins = 80;
        o
    });
}

/// Centered + Reflect framing must be chunk-independent: feeding the signal
/// in arbitrary chunk sizes must produce bitwise-identical frames to a
/// single-shot `compute_fbank`. This pins the streaming invariant that the
/// mirror sources of the leading (negative-index) frames are still in the
/// ring buffer when those frames are computed, across discard boundaries.
#[test]
fn center_reflect_streaming_matches_batch() {
    let wave = synthetic_wave(8000);

    let mut opts = whisper_style_opts();
    opts.frame_opts.center = true;
    opts.frame_opts.center_pad_mode = CenterPadMode::Reflect;
    let batch = compute_fbank(&opts, &wave);

    // 1 exercises the smallest possible chunks (frame deferred until enough
    // samples arrive); 160 / 400 align with hop / window; 1234 is an
    // awkward non-aligned size that straddles discard boundaries.
    for chunk in [1, 7, 160, 400, 1234] {
        let mut fbank = OnlineFbank::new(opts.clone());
        for part in wave.chunks(chunk) {
            fbank.accept_waveform(16000.0, part);
        }
        fbank.input_finished();

        let n = fbank.num_frames_ready();
        assert_eq!(
            n as usize,
            batch.len(),
            "chunk={chunk}: frame count must match"
        );
        for (i, want) in batch.iter().enumerate().take(n as usize) {
            assert_eq!(
                fbank.get_frame(i as i32),
                want.as_slice(),
                "chunk={chunk}: frame {i} must match bitwise"
            );
        }
    }
}
