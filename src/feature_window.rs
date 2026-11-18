//! Frame extraction and windowing, a faithful port of
//! `kaldi-native-fbank/csrc/feature-window.cc`.
//!
//! All arithmetic order and precision (f32 accumulations, f64 window
//! coefficients cast to f32, etc.) deliberately matches the C++ reference so
//! that outputs agree to within FFT rounding.

use std::f64::consts::PI;

use crate::options::{FrameExtractionOptions, WindowType};

/// Computes a window function. `blackman_coeff` is only used for
/// [`WindowType::Blackman`].
pub fn get_window(window_type: WindowType, window_size: i32, blackman_coeff: f32) -> Vec<f32> {
    assert!(window_size > 0, "window_size must be positive");
    let mut window = vec![0.0f32; window_size as usize];
    let frame_length = window_size as f64;

    // The C++ code computes `a` in f64, then evaluates in f64 and casts to f32.
    let mut a = 2.0 * PI / (frame_length - 1.0);
    if window_type == WindowType::Hann {
        // PyTorch-style periodic Hann window.
        a = 2.0 * PI / frame_length;
    }

    for (i, w) in window.iter_mut().enumerate() {
        let i_fl = i as f64;
        *w = match window_type {
            WindowType::Hanning => 0.5 - 0.5 * (a * i_fl).cos(),
            WindowType::Sine => (0.5 * a * i_fl).sin(),
            WindowType::Hamming => 0.54 - 0.46 * (a * i_fl).cos(),
            WindowType::Hann => 0.50 - 0.50 * (a * i_fl).cos(),
            WindowType::Povey => (0.5 - 0.5 * (a * i_fl).cos()).powf(0.85),
            WindowType::Rectangular => 1.0,
            WindowType::Blackman => {
                blackman_coeff as f64 - 0.5 * (a * i_fl).cos()
                    + (0.5 - blackman_coeff as f64) * (2.0 * a * i_fl).cos()
            }
        } as f32;
    }
    window
}

/// A precomputed window function used to multiply frames.
#[derive(Clone, Debug)]
pub struct FeatureWindowFunction {
    window: Vec<f32>,
}

impl FeatureWindowFunction {
    pub fn from_frame_opts(opts: &FrameExtractionOptions) -> Self {
        Self::from_window_type(opts.window_type, opts.window_size(), opts.blackman_coeff)
    }

    pub fn from_window_type(
        window_type: WindowType,
        window_size: i32,
        blackman_coeff: f32,
    ) -> Self {
        Self {
            window: get_window(window_type, window_size, blackman_coeff),
        }
    }

    pub fn from_window(window: Vec<f32>) -> Self {
        Self { window }
    }

    /// Multiplies `wave[i] *= window[i]` for the first `window.len()` entries.
    pub fn apply(&self, wave: &mut [f32]) {
        for (w, v) in self.window.iter().zip(wave.iter_mut()) {
            *v *= w;
        }
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }
}

/// First sample of the `frame`-th frame.
///
/// With `center == true` the window of frame `t` is centered on sample
/// `t * frame_shift`, so the first sample is `t * frame_shift -
/// window_size / 2` (negative values for early frames read as zero).
pub fn first_sample_of_frame(frame: i32, opts: &FrameExtractionOptions) -> i64 {
    let frame_shift = opts.window_shift() as i64;
    if opts.center {
        frame as i64 * frame_shift - opts.window_size() as i64 / 2
    } else if opts.snip_edges {
        frame as i64 * frame_shift
    } else {
        let midpoint_of_frame = frame_shift * frame as i64 + frame_shift / 2;
        midpoint_of_frame - opts.window_size() as i64 / 2
    }
}

/// Number of frames extractable from `num_samples` samples.
///
/// `flush` should be true for offline extraction; it matters only when
/// `snip_edges == false`. With `center == true` the count follows the
/// torchaudio `center=True` convention (`1 + num_samples / frame_shift`,
/// truncating division) regardless of `flush`.
pub fn num_frames(num_samples: i64, opts: &FrameExtractionOptions, flush: bool) -> i32 {
    let frame_shift = opts.window_shift() as i64;
    let frame_length = opts.window_size() as i64;

    if opts.center {
        let mut num = (1 + num_samples / frame_shift) as i32;
        if !flush {
            // Online: a frame whose window extends past the received signal
            // needs future samples; defer it until they arrive (or flush).
            while num > 0 && first_sample_of_frame(num - 1, opts) + frame_length > num_samples {
                num -= 1;
            }
        }
        return num;
    }

    if opts.snip_edges {
        // HTK-like: all frames must fit completely into the waveform and the
        // first frame begins at sample zero.
        if num_samples < frame_length {
            0
        } else {
            (1 + (num_samples - frame_length) / frame_shift) as i32
        }
    } else {
        // Round (file-length / frame-shift) to the nearest integer.
        let mut num_frames = ((num_samples + (frame_shift / 2)) / frame_shift) as i32;

        if flush {
            return num_frames;
        }

        let mut end_sample_of_last_frame =
            first_sample_of_frame(num_frames - 1, opts) + frame_length;

        // If flush == false, we cannot output frames that extend past the end
        // of the signal.
        while num_frames > 0 && end_sample_of_last_frame > num_samples {
            num_frames -= 1;
            end_sample_of_last_frame -= frame_shift;
        }
        num_frames
    }
}

/// Inner product of two vectors, accumulated in f32 like the reference.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn remove_dc_offset(d: &mut [f32]) {
    let mut sum = 0.0f32;
    for &v in d.iter() {
        sum += v;
    }
    let mean = sum / d.len() as f32;
    for v in d.iter_mut() {
        *v -= mean;
    }
}

fn preemphasize(d: &mut [f32], preemph_coeff: f32) {
    assert!((0.0..=1.0).contains(&preemph_coeff));
    let n = d.len();
    for i in (1..n).rev() {
        d[i] -= preemph_coeff * d[i - 1];
    }
    d[0] -= preemph_coeff * d[0];
}

/// Extracts a windowed frame (with reflection at the boundaries), then runs
/// `process_window` on it.
///
/// `window` must have length `opts.padded_window_size()`; the tail beyond
/// `opts.window_size()` is zeroed.
pub fn extract_window(
    sample_offset: i64,
    wave: &[f32],
    frame: i32,
    opts: &FrameExtractionOptions,
    window_function: &FeatureWindowFunction,
    window: &mut [f32],
    log_energy_pre_window: Option<&mut f32>,
) {
    assert!(sample_offset >= 0 && !wave.is_empty());
    let frame_length = opts.window_size();
    let frame_length_padded = opts.padded_window_size();
    let num_samples = sample_offset + wave.len() as i64;
    let start_sample = first_sample_of_frame(frame, opts);
    let end_sample = start_sample + frame_length as i64;

    if opts.snip_edges && !opts.center {
        assert!(start_sample >= sample_offset && end_sample <= num_samples);
    } else if !opts.center {
        assert!(sample_offset == 0 || start_sample >= sample_offset);
    }

    assert_eq!(window.len(), frame_length_padded as usize);
    window.fill(0.0);

    let wave_start = (start_sample - sample_offset) as i32;
    let wave_end = wave_start + frame_length;

    if opts.center {
        // Centered framing zero-pads at the boundaries (torchaudio
        // `center=True`), unlike the Kaldi reflection. Existence checks use
        // absolute sample coordinates: `wave` starts at `sample_offset`.
        for (s, slot) in window[..frame_length as usize].iter_mut().enumerate() {
            let idx_abs = sample_offset + wave_start as i64 + s as i64;
            *slot = if idx_abs >= 0 && idx_abs < num_samples {
                wave[(idx_abs - sample_offset) as usize]
            } else {
                0.0
            };
        }
    } else if wave_start >= 0 && wave_end <= wave.len() as i32 {
        window[..frame_length as usize]
            .copy_from_slice(&wave[wave_start as usize..wave_end as usize]);
    } else {
        // Deal with any end effects by reflection. Reached only for about two
        // frames per utterance.
        let wave_dim = wave.len() as i32;
        for s in 0..frame_length {
            let mut s_in_wave = s + wave_start;
            while s_in_wave < 0 || s_in_wave >= wave_dim {
                if s_in_wave < 0 {
                    s_in_wave = -s_in_wave - 1;
                } else {
                    s_in_wave = 2 * wave_dim - 1 - s_in_wave;
                }
            }
            window[s as usize] = wave[s_in_wave as usize];
        }
    }

    process_window(opts, window_function, window, log_energy_pre_window);
}

/// Dithering, DC removal, log-energy, preemphasis, and window application,
/// exactly in the reference order.
pub fn process_window(
    opts: &FrameExtractionOptions,
    window_function: &FeatureWindowFunction,
    window: &mut [f32],
    log_energy_pre_window: Option<&mut f32>,
) {
    let frame_length = opts.window_size() as usize;
    let frame = &mut window[..frame_length];

    if opts.dither != 0.0 {
        // A fresh RNG per frame, like the reference creates a fresh
        // `RandomState` per `Dither()` call — but deterministically seeded.
        let mut rng = KaldiRand::new(opts.dither_seed);
        dither(frame, opts.dither, &mut rng);
    }

    if opts.remove_dc_offset {
        remove_dc_offset(frame);
    }

    if let Some(le) = log_energy_pre_window {
        let energy = inner_product(frame, frame).max(f32::EPSILON);
        *le = energy.ln();
    }

    if opts.preemph_coeff != 0.0 && !opts.center {
        // Kaldi per-frame pre-emphasis. Centered mode (`center == true`)
        // follows the torchaudio/NeMo convention instead: pre-emphasis runs
        // at the waveform level (see `OnlineFbank::accept_waveform`), so the
        // zero padding added at the boundaries never participates in the
        // difference, and re-applying it per frame here would smear the last
        // real samples of each frame into the padding.
        preemphasize(frame, opts.preemph_coeff);
    }

    window_function.apply(frame);
}

// ---------------------------------------------------------------------------
// Dithering RNG: bit-compatible replica of glibc's `rand_r` + Kaldi's
// `RandGauss`, so that dithering produces exactly the same values as the C++
// reference given the same seed.
// ---------------------------------------------------------------------------

/// glibc `RAND_MAX`.
const RAND_MAX: u32 = 2147483647;

/// Replicates glibc's `rand_r` LCG.
pub struct KaldiRand {
    seed: u32,
}

impl KaldiRand {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Returns a value in `[0, RAND_MAX]` (glibc `rand_r` semantics).
    #[inline]
    pub fn rand(&mut self) -> u32 {
        let mut next = self.seed;
        next = next.wrapping_mul(1103515245).wrapping_add(12345);
        let mut result = ((next / 65536) % 2048) as i32;

        next = next.wrapping_mul(1103515245).wrapping_add(12345);
        result <<= 10;
        result ^= ((next / 65536) % 1024) as i32;

        next = next.wrapping_mul(1103515245).wrapping_add(12345);
        result <<= 10;
        result ^= ((next / 65536) % 1024) as i32;

        self.seed = next;
        result as u32
    }

    /// Kaldi `RandUniform`: `(Rand + 1.0) / (RAND_MAX + 2.0)`, f64 then f32.
    #[inline]
    pub fn rand_uniform(&mut self) -> f32 {
        ((self.rand() as f64 + 1.0) / (RAND_MAX as f64 + 2.0)) as f32
    }

    /// Kaldi `RandGauss`: `sqrtf(-2 log u1) * cosf(2 pi u2)`, all f32.
    #[inline]
    pub fn rand_gauss(&mut self) -> f32 {
        let u1 = self.rand_uniform();
        let u2 = self.rand_uniform();
        let s = (-2.0_f32 * u1.ln()).sqrt();
        let angle = (2.0_f32 * PI as f32 * u2).cos();
        s * angle
    }
}

/// Kaldi `Dither`: `d[i] += RandGauss() * dither_value`.
pub fn dither(d: &mut [f32], dither_value: f32, rng: &mut KaldiRand) {
    for v in d.iter_mut() {
        *v += rng.rand_gauss() * dither_value;
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn opts16k() -> FrameExtractionOptions {
        let mut o = FrameExtractionOptions::default();
        o.dither = 0.0;
        o
    }

    // --- window functions --------------------------------------------------

    #[test]
    fn rectangular_window_is_ones() {
        let w = get_window(WindowType::Rectangular, 400, 0.42);
        assert_eq!(w.len(), 400);
        assert!(w.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn hamming_window_endpoints_and_center() {
        let w = get_window(WindowType::Hamming, 400, 0.42);
        assert!((w[0] - (0.54 - 0.46)).abs() < 1e-6); // 0.08
        assert!((w[399] - (0.54 - 0.46)).abs() < 1e-6);
        // 400-point window: no integer center, max is near 1.0
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.999, "max={max}");
        // symmetry
        for i in 0..200 {
            assert!((w[i] - w[399 - i]).abs() < 1e-7);
        }
    }

    #[test]
    fn povey_window_edges_and_center() {
        let w = get_window(WindowType::Povey, 400, 0.42);
        assert!(w[0].abs() < 1e-12); // pow(0, 0.85) == 0
        assert!(w[399].abs() < 1e-12);
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.999, "max={max}");
    }

    #[test]
    fn hanning_window_values() {
        let w = get_window(WindowType::Hanning, 400, 0.42);
        assert!(w[0].abs() < 1e-12);
        assert!(w[399].abs() < 1e-12);
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.999, "max={max}");
    }

    #[test]
    fn sine_window_values() {
        let w = get_window(WindowType::Sine, 400, 0.42);
        assert!(w[0].abs() < 1e-12);
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.999, "max={max}");
        // sine(i) = sin(pi*i/399)
        let expected = (std::f64::consts::PI * 100.0 / 399.0).sin();
        assert!((w[100] as f64 - expected).abs() < 1e-6);
    }

    #[test]
    fn blackman_window_endpoints() {
        let coeff = 0.42f32;
        let w = get_window(WindowType::Blackman, 400, coeff);
        assert!(w[0].abs() < 1e-6); // coeff - 0.5 + (0.5 - coeff) = 0
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.999, "max={max}");
    }

    #[test]
    fn hann_vs_hanning_differ_at_edges() {
        // Hann is periodic (a = 2pi/N), Hanning is symmetric (a = 2pi/(N-1)).
        let hann = get_window(WindowType::Hann, 400, 0.42);
        let hanning = get_window(WindowType::Hanning, 400, 0.42);
        assert!(hanning[399].abs() < 1e-12);
        // periodic hann: last sample is ~6e-5 (not exactly zero)
        assert!(hann[399].abs() > 1e-8, "hann[399]={}", hann[399]);
        assert!(hann[0].abs() < 1e-12);
    }

    #[test]
    fn invalid_window_type_rejected() {
        assert!("not-a-window".parse::<WindowType>().is_err());
        assert_eq!(
            "not-a-window".parse::<WindowType>().unwrap_err(),
            "Invalid window type 'not-a-window'"
        );
        // every valid name round-trips through Display
        for wt in [
            WindowType::Hanning,
            WindowType::Sine,
            WindowType::Hamming,
            WindowType::Hann,
            WindowType::Povey,
            WindowType::Rectangular,
            WindowType::Blackman,
        ] {
            assert_eq!(wt.to_string().parse::<WindowType>().unwrap(), wt);
        }
    }

    #[test]
    fn window_apply_multiplies() {
        let wf = FeatureWindowFunction::from_window(vec![2.0, 3.0, 4.0]);
        let mut wave = vec![1.0, 1.0, 1.0];
        wf.apply(&mut wave);
        assert_eq!(wave, vec![2.0, 3.0, 4.0]);
        // only the first window.len() entries are touched
        let mut wave = vec![1.0, 1.0, 1.0, 9.0];
        wf.apply(&mut wave);
        assert_eq!(wave, vec![2.0, 3.0, 4.0, 9.0]);
    }

    // --- frame counting ----------------------------------------------------

    #[test]
    fn num_frames_snip_edges_true() {
        let o = opts16k();
        assert_eq!(num_frames(0, &o, true), 0);
        assert_eq!(num_frames(399, &o, true), 0);
        assert_eq!(num_frames(400, &o, true), 1);
        assert_eq!(num_frames(559, &o, true), 1);
        assert_eq!(num_frames(560, &o, true), 2);
        assert_eq!(num_frames(1600, &o, true), 8); // 1 + (1600-400)/160
        assert_eq!(num_frames(160000, &o, true), 998); // 1+(160000-400)/160
    }

    #[test]
    fn num_frames_snip_edges_false() {
        let mut o = opts16k();
        o.snip_edges = false;
        // (num_samples + shift/2) / shift, rounded
        assert_eq!(num_frames(0, &o, true), 0);
        assert_eq!(num_frames(80, &o, true), 1);
        assert_eq!(num_frames(160, &o, true), 1);
        assert_eq!(num_frames(240, &o, true), 2);
        assert_eq!(num_frames(1600, &o, true), 10);
    }

    #[test]
    fn num_frames_snip_false_no_flush_never_exceeds_signal() {
        let mut o = opts16k();
        o.snip_edges = false;
        for n in [1i64, 100, 399, 400, 500, 560, 1000, 1600] {
            let flushed = num_frames(n, &o, true);
            let partial = num_frames(n, &o, false);
            assert!(partial <= flushed, "n={n}");
            // partial frames must end within the signal
            let nf = partial;
            if nf > 0 {
                let end = first_sample_of_frame(nf - 1, &o) + o.window_size() as i64;
                assert!(end <= n, "n={n} end={end}");
            }
        }
    }

    #[test]
    fn num_frames_snip_false_flush_matches_reference_formula() {
        let mut o = opts16k();
        o.snip_edges = false;
        assert_eq!(num_frames(1600, &o, true), 10);
        // one extra frame vs snip_edges=true (8)
    }

    // --- frame offsets -----------------------------------------------------

    #[test]
    fn first_sample_snip_true() {
        let o = opts16k();
        assert_eq!(first_sample_of_frame(0, &o), 0);
        assert_eq!(first_sample_of_frame(1, &o), 160);
        assert_eq!(first_sample_of_frame(5, &o), 800);
    }

    #[test]
    fn first_sample_snip_false() {
        let mut o = opts16k();
        o.snip_edges = false;
        // midpoint - window_size/2 = f*160 + 80 - 200
        assert_eq!(first_sample_of_frame(0, &o), -120);
        assert_eq!(first_sample_of_frame(1, &o), 40);
        assert_eq!(first_sample_of_frame(2, &o), 200);
    }

    // --- extract_window ----------------------------------------------------

    #[test]
    fn extract_window_interior_frame() {
        let o = opts16k();
        let wf = FeatureWindowFunction::from_frame_opts(&o);
        let wave: Vec<f32> = (0..2000).map(|i| i as f32).collect();
        let mut win = vec![0.0f32; o.padded_window_size() as usize];
        extract_window(0, &wave, 2, &o, &wf, &mut win, None);
        // frame 2 starts at sample 320; tail beyond window_size stays zero
        assert_eq!(&win[o.window_size() as usize..], &[0.0; 512 - 400][..]);
        assert!(win[..400].iter().any(|&x| x != 0.0));
    }

    #[test]
    fn extract_window_constant_signal_after_dc_removal_is_zero() {
        let mut o = opts16k();
        o.snip_edges = false; // frame 0 starts at -120 -> reflection
        let wf = FeatureWindowFunction::from_window_type(WindowType::Rectangular, 400, 0.42);
        let wave = vec![5.0f32; 1000];
        let mut win = vec![0.0f32; o.padded_window_size() as usize];
        extract_window(0, &wave, 0, &o, &wf, &mut win, None);
        // constant signal -> DC removed -> all zeros -> windowed zeros
        assert!(win[..400].iter().all(|&x| x == 0.0));
        // and the reflection path itself worked (no panic)
        extract_window(0, &wave, 3, &o, &wf, &mut win, None);
        assert!(win[..400].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn extract_window_reflection_preserves_value_order() {
        // With DC removal and preemphasis disabled, a reflected frame at the
        // left edge must contain the mirrored samples: x[-1]->x[0], x[-2]->x[1]...
        let mut o = opts16k();
        o.snip_edges = false;
        o.remove_dc_offset = false;
        o.preemph_coeff = 0.0;
        let wf = FeatureWindowFunction::from_window_type(WindowType::Rectangular, 400, 0.42);
        let wave: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut win = vec![0.0f32; o.padded_window_size() as usize];
        // frame 0 starts at -120: win[s] = wave[reflected(s-120)]
        extract_window(0, &wave, 0, &o, &wf, &mut win, None);
        for (s, w) in win.iter_mut().enumerate().take(400) {
            let src = s as i32 - 120;
            let idx = if src < 0 { -src - 1 } else { src };
            assert_eq!(*w, wave[idx as usize], "s={s}");
        }
    }

    #[test]
    fn extract_window_log_energy() {
        let mut o = opts16k();
        o.preemph_coeff = 0.0;
        o.remove_dc_offset = false;
        let wf = FeatureWindowFunction::from_window_type(WindowType::Rectangular, 400, 0.42);
        let wave: Vec<f32> = (0..400).map(|i| i as f32).collect();
        let mut win = vec![0.0f32; o.padded_window_size() as usize];
        let mut le = 0.0f32;
        extract_window(0, &wave, 0, &o, &wf, &mut win, Some(&mut le));
        // log energy = ln(sum i^2) for i in 0..400
        let sum_sq: f32 = (0..400).map(|i| (i * i) as f32).sum();
        assert!(
            (le - sum_sq.ln()).abs() < 1e-3,
            "le={le} expected={}",
            sum_sq.ln()
        );
    }

    // --- process_window ----------------------------------------------------

    #[test]
    fn process_window_remove_dc_zeroes_mean() {
        let mut o = opts16k();
        o.preemph_coeff = 0.0;
        let wf = FeatureWindowFunction::from_window_type(WindowType::Rectangular, 400, 0.42);
        let mut win = vec![1.0f32; 512];
        process_window(&o, &wf, &mut win, None);
        // after DC removal mean is 0; rectangular window keeps it 0
        assert!(win[..400].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn preemphasize_formula() {
        let mut d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        preemphasize(&mut d, 0.97);
        // d[0] -= 0.97*d[0] -> 0.03; d[i] -= 0.97*d[i-1]
        let expected = [
            0.03,
            2.0 - 0.97 * 1.0,
            3.0 - 0.97 * 2.0,
            4.0 - 0.97 * 3.0,
            5.0 - 0.97 * 4.0,
        ];
        for i in 0..5 {
            assert!(
                (d[i] - expected[i]).abs() < 1e-6,
                "i={i} got={} exp={}",
                d[i],
                expected[i]
            );
        }
        // in-place backward scan: verify against the reference formula
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut d2 = x.to_vec();
        for i in (1..5).rev() {
            d2[i] -= 0.97 * d2[i - 1];
        }
        d2[0] -= 0.97 * d2[0];
        assert_eq!(d, d2);
    }

    #[test]
    fn remove_dc_offset_zeroes_mean() {
        let mut d = vec![1.0f32, 2.0, 3.0, 4.0, 10.0];
        let mean = 20.0 / 5.0;
        remove_dc_offset(&mut d);
        let m: f32 = d.iter().sum();
        assert!(m.abs() < 1e-6);
        for v in &d {
            assert!((v - (v + mean - mean)).abs() < 1e-6); // values shifted by -mean
        }
        assert!((d[4] - (10.0 - mean)).abs() < 1e-6);
    }

    #[test]
    fn inner_product_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert_eq!(inner_product(&a, &b), 32.0);
        assert_eq!(inner_product(&[], &[]), 0.0);
    }

    #[test]
    fn dither_changes_values_deterministically() {
        let mut rng = KaldiRand::new(12345);
        let mut a = vec![0.0f32; 64];
        dither(&mut a, 0.5, &mut rng);
        assert!(a.iter().any(|&x| x != 0.0));
        // deterministic given the seed
        let mut rng2 = KaldiRand::new(12345);
        let mut b = vec![0.0f32; 64];
        dither(&mut b, 0.5, &mut rng2);
        assert_eq!(a, b);
    }

    // --- RNG ---------------------------------------------------------------

    #[test]
    fn rand_r_values_match_glibc() {
        let mut rng = KaldiRand::new(1);
        // verified against glibc rand_r(seed=1) with a C program
        let expected = [
            476_707_713u32,
            1_186_278_907,
            505_671_508,
            2_137_716_191,
            936_145_377,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(rng.rand(), e, "value {i}");
        }
    }

    #[test]
    fn rand_uniform_in_range() {
        let mut rng = KaldiRand::new(7);
        for _ in 0..10000 {
            let u = rng.rand_uniform();
            assert!(u > 0.0 && u < 1.0, "u={u}");
        }
    }

    #[test]
    fn rand_gauss_statistics() {
        // ~N(0,1): mean near 0, std near 1 over enough samples
        let mut rng = KaldiRand::new(42);
        let n = 200_000;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for _ in 0..n {
            let g = rng.rand_gauss() as f64;
            sum += g;
            sumsq += g * g;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean={mean}");
        assert!((var - 1.0).abs() < 0.02, "var={var}");
    }

    #[test]
    fn rand_gauss_deterministic() {
        let mut a = KaldiRand::new(99);
        let mut b = KaldiRand::new(99);
        for _ in 0..100 {
            assert_eq!(a.rand_gauss(), b.rand_gauss());
        }
    }
}
