//! Option structs mirroring `kaldi-native-fbank` (which in turn mirrors Kaldi).

use std::fmt;

/// Rounds `n` (must be > 0) up to the nearest power of two.
pub fn round_up_to_nearest_power_of_two(mut n: i32) -> i32 {
    debug_assert!(n > 0, "n must be positive");
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n + 1
}

/// Padding mode for centered framing (`center == true`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CenterPadMode {
    /// Zero-pad the signal boundaries — the torchaudio `center=True` /
    /// torch.stft / NeMo convention.
    #[default]
    Constant,
    /// Mirror the signal at the boundaries without repeating the edge
    /// sample — the `numpy.pad(..., mode="reflect")` / librosa / Whisper
    /// convention. Degenerate signals shorter than two samples read as
    /// their single sample.
    Reflect,
}

/// Options for frame extraction: sample rate, frame shift / length, dithering,
/// DC removal, pre-emphasis, window type, edge sniping.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameExtractionOptions {
    pub samp_freq: f32,
    /// In milliseconds.
    pub frame_shift_ms: f32,
    /// In milliseconds.
    pub frame_length_ms: f32,
    /// Amount of dithering, 0.0 means no dither.
    /// Value 0.00003 is equivalent to 1.0 in Kaldi.
    pub dither: f32,
    /// Preemphasis coefficient.
    /// Preemphasis coefficient (0.0 disables it).
    ///
    /// The application convention follows the framing mode: with
    /// `snip_edges` framing the reference applies it per frame (the first
    /// sample of each frame is scaled by `1 - coeff`), while
    /// `center == true` follows the torchaudio/NeMo convention of applying
    /// it to the whole waveform before padding (see [`Self::center`]).
    pub preemph_coeff: f32,
    /// Subtract mean of wave before FFT.
    pub remove_dc_offset: bool,
    pub window_type: WindowType,
    pub round_to_power_of_two: bool,
    pub blackman_coeff: f32,
    pub snip_edges: bool,
    /// Torch-style centered framing (the `center=True` convention of
    /// `torch.stft` / torchaudio / NeMo / ESPnet): zero-pad
    /// `window_size / 2` samples at both ends so that the window of frame
    /// `t` is centered on sample `t * frame_shift`.
    ///
    /// When enabled this supersedes `snip_edges` (its Kaldi framing and its
    /// reflect-at-the-boundary behavior do not apply); boundary samples come
    /// from zero-padding, and the frame count is `1 + num_samples /
    /// frame_shift` after `input_finished()` — matching torchaudio's
    /// `center=True` convention. Callers that follow NeMo's
    /// `floor(samples / frame_shift)` convention simply drop the final frame.
    ///
    /// Defaults to `false` (plain Kaldi framing). This is an extension
    /// beyond the C++ reference, which has no centered mode.
    pub center: bool,
    /// Boundary padding used by centered framing (ignored when
    /// `center == false`, where plain Kaldi framing applies its own
    /// edge-inclusive reflection).
    ///
    /// [`CenterPadMode::Constant`] zero-pads (torchaudio / NeMo);
    /// [`CenterPadMode::Reflect`] mirrors the signal edge-exclusively
    /// (`numpy.pad(..., mode="reflect")`, librosa / Whisper). Defaults to
    /// [`CenterPadMode::Constant`].
    pub center_pad_mode: CenterPadMode,
    /// Seed for the dithering RNG.
    ///
    /// Rust extension: the C++ reference seeds its dithering RNG
    /// nondeterministically from `rand()`, so dithering is inherently
    /// stochastic there. We expose a fixed seed so that Rust runs are
    /// reproducible. `27437` is the prime Kaldi adds to `rand()`.
    pub dither_seed: u32,
}

impl Default for FrameExtractionOptions {
    fn default() -> Self {
        Self {
            samp_freq: 16000.0,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.00003,
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: WindowType::Povey,
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: true,
            center: false,
            center_pad_mode: CenterPadMode::Constant,
            dither_seed: 27437,
        }
    }
}

/// Window function type, matching kaldi-native-fbank's `window_type` string
/// values ("povey", "hamming", ...). `Display` yields the exact reference
/// strings so CLI flags and configuration stay source-compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum WindowType {
    Hanning,
    Sine,
    Hamming,
    /// PyTorch-style periodic Hann window.
    Hann,
    /// Like Hamming but goes to zero at the edges; the Kaldi default.
    #[default]
    Povey,
    Rectangular,
    Blackman,
}

impl std::fmt::Display for WindowType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WindowType::Hanning => "hanning",
            WindowType::Sine => "sine",
            WindowType::Hamming => "hamming",
            WindowType::Hann => "hann",
            WindowType::Povey => "povey",
            WindowType::Rectangular => "rectangular",
            WindowType::Blackman => "blackman",
        })
    }
}

impl std::str::FromStr for WindowType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hanning" => Ok(WindowType::Hanning),
            "sine" => Ok(WindowType::Sine),
            "hamming" => Ok(WindowType::Hamming),
            "hann" => Ok(WindowType::Hann),
            "povey" => Ok(WindowType::Povey),
            "rectangular" => Ok(WindowType::Rectangular),
            "blackman" => Ok(WindowType::Blackman),
            other => Err(format!("Invalid window type '{other}'")),
        }
    }
}

impl FrameExtractionOptions {
    /// Frame shift in samples (truncated to int, like Kaldi).
    pub fn window_shift(&self) -> i32 {
        (self.samp_freq * 0.001_f32 * self.frame_shift_ms) as i32
    }

    /// Frame length in samples (truncated to int, like Kaldi).
    pub fn window_size(&self) -> i32 {
        (self.samp_freq * 0.001_f32 * self.frame_length_ms) as i32
    }

    /// Padded window size: rounded up to a power of two if requested.
    pub fn padded_window_size(&self) -> i32 {
        if self.round_to_power_of_two {
            round_up_to_nearest_power_of_two(self.window_size())
        } else {
            self.window_size()
        }
    }
}

impl fmt::Display for FrameExtractionOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "samp_freq: {}\nframe_shift_ms: {}\nframe_length_ms: {}\ndither: {}\n\
             preemph_coeff: {}\nremove_dc_offset: {}\nwindow_type: {}\n\
             round_to_power_of_two: {}\nblackman_coeff: {}\nsnip_edges: {}",
            self.samp_freq,
            self.frame_shift_ms,
            self.frame_length_ms,
            self.dither,
            self.preemph_coeff,
            self.remove_dc_offset,
            self.window_type,
            self.round_to_power_of_two,
            self.blackman_coeff,
            self.snip_edges
        )
    }
}

/// Options for the mel filterbank.
#[derive(Clone, Debug, PartialEq)]
pub struct MelBanksOptions {
    /// Number of triangular mel bins, e.g. 23 / 80.
    pub num_bins: i32,
    /// Lower frequency cutoff, e.g. 20.
    pub low_freq: f32,
    /// Upper frequency cutoff; 0 -> no cutoff, negative -> added to Nyquist.
    pub high_freq: f32,
    /// VTLN lower cutoff of warping function.
    pub vtln_low: f32,
    /// VTLN upper cutoff of warping function; if negative, added to Nyquist.
    pub vtln_high: f32,
    pub debug_mel: bool,
    /// HTK compatibility mode: mel-energy flooring and an HTK bug replica.
    pub htk_mode: bool,
    /// If true, build the librosa-style mel filterbank instead of the Kaldi one.
    pub is_librosa: bool,
    /// Only used when `is_librosa` is true. Possible values: "", "slaney".
    pub norm: String,
    /// Only used when `is_librosa` is true.
    pub use_slaney_mel_scale: bool,
    /// Only used when `is_librosa` is true.
    pub floor_to_int_bin: bool,
}

impl Default for MelBanksOptions {
    fn default() -> Self {
        Self {
            num_bins: 25,
            low_freq: 20.0,
            high_freq: 0.0,
            vtln_low: 100.0,
            vtln_high: -500.0,
            debug_mel: false,
            htk_mode: false,
            is_librosa: false,
            norm: "slaney".into(),
            use_slaney_mel_scale: true,
            floor_to_int_bin: false,
        }
    }
}

impl fmt::Display for MelBanksOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "num_bins: {}\nlow_freq: {}\nhigh_freq: {}\nvtln_low: {}\nvtln_high: {}\n\
             debug_mel: {}\nhtk_mode: {}\nis_librosa: {}\nnorm: {}\n\
             use_slaney_mel_scale: {}\nfloor_to_int_bin: {}",
            self.num_bins,
            self.low_freq,
            self.high_freq,
            self.vtln_low,
            self.vtln_high,
            self.debug_mel,
            self.htk_mode,
            self.is_librosa,
            self.norm,
            self.use_slaney_mel_scale,
            self.floor_to_int_bin
        )
    }
}

/// Options for the FBank feature extractor.
#[derive(Clone, Debug, PartialEq)]
pub struct FbankOptions {
    pub frame_opts: FrameExtractionOptions,
    pub mel_opts: MelBanksOptions,
    /// Append an extra dimension with energy to the filterbanks.
    pub use_energy: bool,
    /// Active iff `use_energy` is true.
    pub energy_floor: f32,
    /// If true, compute log_energy before preemphasis and windowing; else after.
    pub raw_energy: bool,
    /// If true, put energy last (if using energy); else put energy first.
    pub htk_compat: bool,
    /// If true (default), produce log-filterbank, else linear.
    pub use_log_fbank: bool,
    /// If true (default), use power in filterbank analysis, else magnitude.
    pub use_power: bool,
}

impl Default for FbankOptions {
    fn default() -> Self {
        Self {
            frame_opts: FrameExtractionOptions::default(),
            mel_opts: MelBanksOptions {
                num_bins: 23,
                ..MelBanksOptions::default()
            },
            use_energy: false,
            energy_floor: 0.0,
            raw_energy: true,
            htk_compat: false,
            use_log_fbank: true,
            use_power: true,
        }
    }
}

impl FbankOptions {
    /// Whisper-style log-mel options — the feature definition shared by
    /// Whisper and Qwen3-ASR: 400-sample window / 160-sample shift
    /// (25 ms / 10 ms @ 16 kHz), periodic Hann window, power spectrum,
    /// `n_mels` librosa/slaney mel bins over 0-8 kHz, no dither /
    /// pre-emphasis / DC removal, centered framing with reflect padding
    /// (`numpy.pad(..., mode="reflect")`), and linear (un-logged)
    /// filterbank output.
    ///
    /// The Whisper post-processing (`log10(max(x, 1e-10))`, dynamic
    /// `max - 8` floor, `(x + 4) / 4` scaling) is model-recipe logic and is
    /// intentionally left to the caller.
    ///
    /// `n_mels` is 80 for Whisper small and below, 128 for large / turbo /
    /// Qwen3-ASR.
    ///
    /// # Panics
    ///
    /// Panics if `n_mels` is zero or exceeds `i32::MAX`.
    pub fn whisper(n_mels: usize) -> Self {
        assert!(
            n_mels > 0 && n_mels <= i32::MAX as usize,
            "n_mels must be in 1..={}",
            i32::MAX
        );
        let mut opts = Self {
            mel_opts: MelBanksOptions {
                num_bins: n_mels as i32,
                ..MelBanksOptions::default()
            },
            ..Self::default()
        };
        let f = &mut opts.frame_opts;
        f.samp_freq = 16000.0;
        f.frame_shift_ms = 10.0;
        f.frame_length_ms = 25.0; // 400 samples
        f.dither = 0.0;
        f.preemph_coeff = 0.0;
        f.remove_dc_offset = false;
        f.window_type = WindowType::Hann; // periodic (2*pi*n / N)
        f.round_to_power_of_two = false; // 400-point FFT, not rounded to 512
        f.snip_edges = true; // superseded by `center`, set for consistency
        f.center = true;
        f.center_pad_mode = CenterPadMode::Reflect;
        opts.mel_opts.low_freq = 0.0;
        opts.mel_opts.high_freq = 8000.0;
        opts.mel_opts.is_librosa = true;
        opts.mel_opts.norm = "slaney".to_string();
        opts.mel_opts.use_slaney_mel_scale = true;
        opts.use_power = true;
        opts.use_log_fbank = false;
        opts
    }
}

impl fmt::Display for FbankOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frame_opts:\n{}\n\nmel_opts:\n{}\n\nuse_energy: {}\nenergy_floor: {}\n\
             raw_energy: {}\nhtk_compat: {}\nuse_log_fbank: {}\nuse_power: {}",
            self.frame_opts,
            self.mel_opts,
            self.use_energy,
            self.energy_floor,
            self.raw_energy,
            self.htk_compat,
            self.use_log_fbank,
            self.use_power
        )
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn round_up_to_nearest_power_of_two_values() {
        let cases = [
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (400, 512),
            (512, 512),
            (513, 1024),
            (1600, 2048),
            (1102, 2048),
            (2048, 2048),
        ];
        for (n, expected) in cases {
            assert_eq!(round_up_to_nearest_power_of_two(n), expected, "n={n}");
        }
    }

    #[test]
    fn window_geometry_16k() {
        let opts = FrameExtractionOptions::default(); // 16 kHz
        assert_eq!(opts.window_shift(), 160); // 10 ms
        assert_eq!(opts.window_size(), 400); // 25 ms
        assert_eq!(opts.padded_window_size(), 512); // next pow2
    }

    #[test]
    fn window_geometry_8k() {
        let mut opts = FrameExtractionOptions::default();
        opts.samp_freq = 8000.0;
        assert_eq!(opts.window_shift(), 80);
        assert_eq!(opts.window_size(), 200);
        assert_eq!(opts.padded_window_size(), 256);
    }

    #[test]
    fn window_geometry_44k1() {
        let mut opts = FrameExtractionOptions::default();
        opts.samp_freq = 44100.0;
        assert_eq!(opts.window_shift(), 441);
        assert_eq!(opts.window_size(), 1102); // 44100*0.025 = 1102.5 -> 1102
        assert_eq!(opts.padded_window_size(), 2048);
    }

    #[test]
    fn window_geometry_custom_and_no_pow2() {
        let mut opts = FrameExtractionOptions::default();
        opts.frame_shift_ms = 5.0;
        opts.frame_length_ms = 20.0;
        assert_eq!(opts.window_shift(), 80);
        assert_eq!(opts.window_size(), 320);
        assert_eq!(opts.padded_window_size(), 512);
        opts.round_to_power_of_two = false;
        assert_eq!(opts.padded_window_size(), 320);
    }

    #[test]
    fn window_geometry_float_rounding() {
        // Truncation semantics (static_cast<int32_t>), e.g. 44.1k*0.02=882.0 exact.
        let mut opts = FrameExtractionOptions::default();
        opts.samp_freq = 44100.0;
        opts.frame_length_ms = 20.0;
        assert_eq!(opts.window_size(), 882);
        opts.frame_length_ms = 19.0; // 837.9 -> 837
        assert_eq!(opts.window_size(), 837);
    }

    #[test]
    fn default_options_match_reference() {
        // Mirrors the C++ FbankOptions defaults.
        let opts = FbankOptions::default();
        assert_eq!(opts.mel_opts.num_bins, 23);
        assert_eq!(opts.frame_opts.dither, 0.00003);
        assert_eq!(opts.frame_opts.preemph_coeff, 0.97);
        assert!(opts.frame_opts.remove_dc_offset);
        assert_eq!(opts.frame_opts.window_type, WindowType::Povey);
        assert!(opts.frame_opts.round_to_power_of_two);
        assert!(opts.frame_opts.snip_edges);
        assert!(!opts.use_energy);
        assert!(opts.use_log_fbank);
        assert!(opts.use_power);
        assert!(opts.raw_energy);
        assert!(!opts.htk_compat);
        assert_eq!(opts.mel_opts.low_freq, 20.0);
        assert_eq!(opts.mel_opts.high_freq, 0.0);
    }

    #[test]
    fn display_does_not_panic() {
        let opts = FbankOptions::default();
        let s = opts.to_string();
        assert!(s.contains("num_bins: 23"));
        assert!(s.contains("samp_freq: 16000"));
        let mel = MelBanksOptions::default();
        assert!(mel.to_string().contains("num_bins: 25"));
    }
}
