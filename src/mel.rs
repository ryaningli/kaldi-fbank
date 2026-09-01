//! Mel filterbank, a faithful port of `kaldi-native-fbank/csrc/mel-computations.cc`.
//!
//! Both the Kaldi-style and the librosa-style filterbanks are supported, with
//! all arithmetic performed in f32 exactly like the C++ reference.

use crate::options::{FrameExtractionOptions, MelBanksOptions};

/// HTK-style hz -> mel. All arithmetic in f32, exactly like the reference
/// `1127.0f * logf(1.0f + freq / 700.0f)`.
#[inline]
pub fn mel_scale(freq: f32) -> f32 {
    1127.0_f32 * (1.0_f32 + freq / 700.0_f32).ln()
}

/// HTK-style mel -> hz. All arithmetic in f32, exactly like the reference
/// `700.0f * (expf(mel_freq / 1127.0f) - 1.0f)`.
#[inline]
pub fn inverse_mel_scale(mel_freq: f32) -> f32 {
    700.0_f32 * ((mel_freq / 1127.0_f32).exp() - 1.0_f32)
}

/// Slaney-style hz -> mel. f32 arithmetic like the reference; the constant is
/// `27 / ln(6.4) = 14.545078505785561` rounded to f32.
#[inline]
#[allow(clippy::excessive_precision)] // keep the exact reference literal
pub fn mel_scale_slaney(freq: f32) -> f32 {
    if freq <= 1000.0 {
        freq * 3.0_f32 / 200.0_f32
    } else {
        15.0_f32 + 14.545078505785561_f32 * (freq / 1000.0_f32).ln()
    }
}

/// Slaney-style mel -> hz. f32 arithmetic like the reference; the constant is
/// `ln(6.4) / 27 = 0.06875177742094911` rounded to f32.
#[inline]
#[allow(clippy::excessive_precision)] // keep the exact reference literal
pub fn inverse_mel_scale_slaney(mel_freq: f32) -> f32 {
    if mel_freq <= 15.0 {
        200.0_f32 / 3.0_f32 * mel_freq
    } else {
        1000.0_f32 * ((mel_freq - 15.0_f32) * 0.06875177742094911_f32).exp()
    }
}

fn vtln_warp_freq(
    vtln_low_cutoff: f32,
    vtln_high_cutoff: f32,
    low_freq: f32,
    high_freq: f32,
    vtln_warp_factor: f32,
    freq: f32,
) -> f32 {
    if freq < low_freq || freq > high_freq {
        return freq;
    }

    assert!(vtln_low_cutoff > low_freq && vtln_high_cutoff < high_freq);

    let one = 1.0_f32;
    let l = vtln_low_cutoff * one.max(vtln_warp_factor);
    let h = vtln_high_cutoff * one.min(vtln_warp_factor);
    let scale = 1.0_f32 / vtln_warp_factor;
    let fl = scale * l; // F(l)
    let fh = scale * h; // F(h)
    assert!(l > low_freq && h < high_freq);
    // slope of the left part of the 3-piece linear function
    let scale_left = (fl - low_freq) / (l - low_freq);
    // slope of the right part
    let scale_right = (high_freq - fh) / (high_freq - h);

    if freq < l {
        low_freq + scale_left * (freq - low_freq)
    } else if freq < h {
        scale * freq
    } else {
        high_freq + scale_right * (freq - high_freq)
    }
}

fn vtln_warp_mel_freq(
    vtln_low_cutoff: f32,
    vtln_high_cutoff: f32,
    low_freq: f32,
    high_freq: f32,
    vtln_warp_factor: f32,
    mel_freq: f32,
) -> f32 {
    mel_scale(vtln_warp_freq(
        vtln_low_cutoff,
        vtln_high_cutoff,
        low_freq,
        high_freq,
        vtln_warp_factor,
        inverse_mel_scale(mel_freq),
    ))
}

/// A precomputed mel filterbank: for each bin, the offset of the first nonzero
/// FFT bin plus the vector of weights (exactly like the reference `bins_`).
pub struct MelBanks {
    bins: Vec<(i32, Vec<f32>)>,
    htk_mode: bool,
    num_fft_bins: i32,
}

impl MelBanks {
    pub fn new(
        opts: &MelBanksOptions,
        frame_opts: &FrameExtractionOptions,
        vtln_warp_factor: f32,
    ) -> Self {
        let mut this = Self {
            bins: Vec::new(),
            htk_mode: false,
            num_fft_bins: frame_opts.padded_window_size(),
        };
        if opts.is_librosa {
            this.init_librosa_mel_banks(opts, frame_opts, vtln_warp_factor);
        } else {
            this.init_kaldi_mel_banks(opts, frame_opts, vtln_warp_factor);
        }
        this
    }

    fn init_kaldi_mel_banks(
        &mut self,
        opts: &MelBanksOptions,
        frame_opts: &FrameExtractionOptions,
        vtln_warp_factor: f32,
    ) {
        self.htk_mode = opts.htk_mode;
        let num_bins = opts.num_bins;
        assert!(num_bins >= 3, "Must have at least 3 mel bins");

        let sample_freq = frame_opts.samp_freq;
        let window_length_padded = frame_opts.padded_window_size();
        assert_eq!(window_length_padded % 2, 0);

        let num_fft_bins = window_length_padded / 2;
        let nyquist = 0.5 * sample_freq;

        let low_freq = opts.low_freq;
        let high_freq = if opts.high_freq > 0.0 {
            opts.high_freq
        } else {
            nyquist + opts.high_freq
        };

        assert!(
            low_freq >= 0.0
                && low_freq < nyquist
                && high_freq > 0.0
                && high_freq <= nyquist
                && high_freq > low_freq,
            "Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}"
        );

        let fft_bin_width = sample_freq / window_length_padded as f32;

        let mel_low_freq = mel_scale(low_freq);
        let mel_high_freq = mel_scale(high_freq);

        // Divide by num_bins + 1 because of end-effects where the bins spread
        // out to the sides.
        let mel_freq_delta = (mel_high_freq - mel_low_freq) / ((num_bins + 1) as f32);

        let vtln_low = opts.vtln_low;
        let mut vtln_high = opts.vtln_high;
        if vtln_high < 0.0 {
            vtln_high += nyquist;
        }

        if vtln_warp_factor != 1.0
            && (vtln_low < 0.0
                || vtln_low <= low_freq
                || vtln_low >= high_freq
                || vtln_high <= 0.0
                || vtln_high >= high_freq
                || vtln_high <= vtln_low)
        {
            panic!(
                "Bad values in options: vtln-low {vtln_low} and vtln-high {vtln_high}, \
                 versus low-freq {low_freq} and high-freq {high_freq}"
            );
        }

        self.bins.reserve(num_bins as usize);

        for bin in 0..num_bins {
            let mut left_mel = mel_low_freq + bin as f32 * mel_freq_delta;
            let mut center_mel = mel_low_freq + (bin + 1) as f32 * mel_freq_delta;
            let mut right_mel = mel_low_freq + (bin + 2) as f32 * mel_freq_delta;

            if vtln_warp_factor != 1.0 {
                left_mel = vtln_warp_mel_freq(
                    vtln_low,
                    vtln_high,
                    low_freq,
                    high_freq,
                    vtln_warp_factor,
                    left_mel,
                );
                center_mel = vtln_warp_mel_freq(
                    vtln_low,
                    vtln_high,
                    low_freq,
                    high_freq,
                    vtln_warp_factor,
                    center_mel,
                );
                right_mel = vtln_warp_mel_freq(
                    vtln_low,
                    vtln_high,
                    low_freq,
                    high_freq,
                    vtln_warp_factor,
                    right_mel,
                );
            }

            let mut this_bin = vec![0.0f32; num_fft_bins as usize];
            let mut first_index = -1i32;
            let mut last_index = -1i32;

            for i in 0..num_fft_bins {
                let freq = fft_bin_width * i as f32; // center frequency of this fft bin
                let mel = mel_scale(freq);
                if mel > left_mel && mel < right_mel {
                    let weight = if mel <= center_mel {
                        (mel - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - mel) / (right_mel - center_mel)
                    };
                    this_bin[i as usize] = weight;
                    if first_index == -1 {
                        first_index = i;
                    }
                    last_index = i;
                }
            }

            assert!(
                first_index != -1 && last_index >= first_index,
                "You may have set num_mel_bins too large."
            );

            let size = (last_index + 1 - first_index) as usize;
            let mut v: Vec<f32> =
                this_bin[first_index as usize..first_index as usize + size].to_vec();

            // Replicate a bug in HTK, for testing purposes.
            if opts.htk_mode && bin == 0 && mel_low_freq != 0.0 {
                v[0] = 0.0;
            }

            self.bins.push((first_index, v));
        }
    }

    fn init_librosa_mel_banks(
        &mut self,
        opts: &MelBanksOptions,
        frame_opts: &FrameExtractionOptions,
        _vtln_warp_factor: f32,
    ) {
        self.htk_mode = opts.htk_mode;
        let num_bins = opts.num_bins;
        assert!(num_bins >= 3, "Must have at least 3 mel bins");

        let sample_freq = frame_opts.samp_freq;
        let window_length_padded = frame_opts.padded_window_size();
        assert_eq!(window_length_padded % 2, 0);

        let num_fft_bins = window_length_padded / 2;
        let nyquist = 0.5 * sample_freq;

        let low_freq = opts.low_freq;
        let high_freq = if opts.high_freq > 0.0 {
            opts.high_freq
        } else {
            nyquist + opts.high_freq
        };

        assert!(
            low_freq >= 0.0
                && low_freq < nyquist
                && high_freq > 0.0
                && high_freq <= nyquist
                && high_freq > low_freq,
            "Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}"
        );

        let fft_bin_width = sample_freq / window_length_padded as f32;

        let (mel_low_freq, mel_high_freq) = if opts.use_slaney_mel_scale {
            (mel_scale_slaney(low_freq), mel_scale_slaney(high_freq))
        } else {
            (mel_scale(low_freq), mel_scale(high_freq))
        };

        let mel_freq_delta = (mel_high_freq - mel_low_freq) / ((num_bins + 1) as f32);

        let slaney_norm = if opts.norm.is_empty() {
            false
        } else {
            assert_eq!(opts.norm, "slaney", "Unsupported norm: {}", opts.norm);
            true
        };

        self.bins.reserve(num_bins as usize);

        for bin in 0..num_bins {
            let left_mel = mel_low_freq + bin as f32 * mel_freq_delta;
            let center_mel = mel_low_freq + (bin + 1) as f32 * mel_freq_delta;
            let right_mel = mel_low_freq + (bin + 2) as f32 * mel_freq_delta;

            let mut left_hz = if opts.use_slaney_mel_scale {
                inverse_mel_scale_slaney(left_mel)
            } else {
                inverse_mel_scale(left_mel)
            };
            let mut center_hz = if opts.use_slaney_mel_scale {
                inverse_mel_scale_slaney(center_mel)
            } else {
                inverse_mel_scale(center_mel)
            };
            let mut right_hz = if opts.use_slaney_mel_scale {
                inverse_mel_scale_slaney(right_mel)
            } else {
                inverse_mel_scale(right_mel)
            };

            if opts.floor_to_int_bin {
                let scale = (window_length_padded + 1) as f32 / sample_freq;
                left_hz = (left_hz * scale) as i32 as f32;
                center_hz = (center_hz * scale) as i32 as f32;
                right_hz = (right_hz * scale) as i32 as f32;
            }

            // It is not an error to use num_fft_bins + 1 here; different from Kaldi.
            let mut this_bin = vec![0.0f32; num_fft_bins as usize + 1];
            let mut first_index = -1i32;
            let mut last_index = -1i32;

            for i in 0..=num_fft_bins {
                let hz = if opts.floor_to_int_bin {
                    i as f32
                } else {
                    fft_bin_width * i as f32
                };

                if hz > left_hz && hz < right_hz {
                    let mut weight = if hz <= center_hz {
                        (hz - left_hz) / (center_hz - left_hz)
                    } else {
                        (right_hz - hz) / (right_hz - center_hz)
                    };
                    if slaney_norm {
                        weight *= 2.0 / (right_hz - left_hz);
                    }
                    this_bin[i as usize] = weight;
                    if first_index == -1 {
                        first_index = i;
                    }
                    last_index = i;
                }
            }

            assert!(
                first_index != -1 && last_index >= first_index,
                "You may have set num_mel_bins too large."
            );

            let size = (last_index + 1 - first_index) as usize;
            let v: Vec<f32> = this_bin[first_index as usize..first_index as usize + size].to_vec();
            self.bins.push((first_index, v));
        }
    }

    /// Computes mel energies (not log) from a power spectrum of size
    /// `padded_window_size / 2 + 1`.
    pub fn compute(&self, power_spectrum: &[f32], mel_energies_out: &mut [f32]) {
        let num_bins = self.bins.len();
        for (i, (offset, v)) in self.bins.iter().enumerate().take(num_bins) {
            let mut energy = 0.0f32;
            for (k, w) in v.iter().enumerate() {
                energy += w * power_spectrum[k + *offset as usize];
            }

            // HTK-like flooring, for testing purposes (we prefer dither).
            if self.htk_mode && energy < 1.0 {
                energy = 1.0;
            }

            mel_energies_out[i] = energy;
            debug_assert!(!energy.is_nan());
        }
    }

    pub fn num_bins(&self) -> i32 {
        self.bins.len() as i32
    }

    /// Returns the dense weight matrix of shape `[num_bins, num_fft_bins/2 + 1]`.
    pub fn get_matrix(&self) -> Vec<f32> {
        let num_rows = self.num_bins();
        let num_cols = self.num_fft_bins / 2 + 1;
        let mut ans = vec![0.0f32; (num_rows * num_cols) as usize];
        for (i, (offset, v)) in self.bins.iter().enumerate() {
            let start = i * num_cols as usize + *offset as usize;
            ans[start..start + v.len()].copy_from_slice(v);
        }
        ans
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::options::FrameExtractionOptions;

    // --- mel scale ---------------------------------------------------------

    #[test]
    fn mel_scale_known_values() {
        assert_eq!(mel_scale(0.0), 0.0);
        // 1127 * ln(1 + 1000/700)
        let expected: f32 = 1127.0_f32 * (1.0_f32 + 1000.0_f32 / 700.0_f32).ln();
        assert!((mel_scale(1000.0) - expected).abs() < 1e-4);
        // 1000 Hz is approximately 1000 mel
        assert!((mel_scale(1000.0) - 1000.0).abs() < 1.0);
        // monotonic increasing
        let mut prev = f32::MIN;
        for f in 0..20000 {
            let m = mel_scale(f as f32);
            assert!(m >= prev, "not monotonic at {f}");
            prev = m;
        }
    }

    #[test]
    fn mel_scale_roundtrip() {
        for freq in [0.0f32, 20.0, 200.0, 1000.0, 4000.0, 8000.0, 16000.0] {
            let mel = mel_scale(freq);
            let back = inverse_mel_scale(mel);
            assert!(
                (back - freq).abs() <= freq * 1e-4 + 1e-3,
                "freq={freq} back={back}"
            );
        }
    }

    #[test]
    fn slaney_mel_scale_roundtrip() {
        for freq in [0.0f32, 20.0, 200.0, 999.0, 1001.0, 4000.0, 8000.0, 16000.0] {
            let mel = mel_scale_slaney(freq);
            let back = inverse_mel_scale_slaney(mel);
            assert!(
                (back - freq).abs() <= freq * 1e-3 + 1e-2,
                "freq={freq} back={back}"
            );
        }
    }

    #[test]
    fn slaney_breakpoints() {
        assert_eq!(mel_scale_slaney(1000.0), 15.0);
        assert!((inverse_mel_scale_slaney(15.0) - 1000.0).abs() < 1e-3);
        // linear region below 1000 Hz
        assert!((mel_scale_slaney(500.0) - 7.5).abs() < 1e-5);
    }

    #[test]
    fn htk_and_slaney_scales_differ() {
        // At 4 kHz the two mel scales differ significantly.
        assert!((mel_scale(4000.0) - mel_scale_slaney(4000.0)).abs() > 100.0);
    }

    // --- MelBanks (Kaldi mode) ---------------------------------------------

    fn frame_opts() -> FrameExtractionOptions {
        let mut o = FrameExtractionOptions::default();
        o.dither = 0.0;
        o
    }

    fn kaldi_opts(num_bins: i32) -> MelBanksOptions {
        MelBanksOptions {
            num_bins,
            ..MelBanksOptions::default()
        }
    }

    #[test]
    fn mel_banks_kaldi_basic_properties() {
        let banks = MelBanks::new(&kaldi_opts(80), &frame_opts(), 1.0);
        assert_eq!(banks.num_bins(), 80);
        let matrix = banks.get_matrix();
        let num_cols = 512 / 2 + 1;
        assert_eq!(matrix.len(), 80 * num_cols);
        // weights are non-negative
        assert!(matrix.iter().all(|&w| w >= 0.0));
        // every bin has some nonzero weight
        for (i, row) in matrix.chunks(num_cols).enumerate() {
            assert!(row.iter().any(|&w| w > 0.0), "bin {i} is empty");
        }
    }

    #[test]
    fn mel_banks_triangular_property() {
        // Adjacent triangular bins overlap so that weights sum to 1 in the
        // shared region (Kaldi mel filterbank design).
        let banks = MelBanks::new(&kaldi_opts(80), &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        let num_cols = 257;
        let rows: Vec<&[f32]> = matrix.chunks(num_cols).collect();
        for i in 0..79 {
            for (j, (a, b)) in rows[i].iter().zip(rows[i + 1].iter()).enumerate() {
                let sum = a + b;
                assert!(sum <= 1.0 + 1e-6, "bin {i} col {j} sum={sum}");
                // where both are active the sum is exactly 1
                if *a > 0.0 && *b > 0.0 {
                    assert!((sum - 1.0).abs() < 1e-5, "bin {i} col {j} sum={sum}");
                }
            }
        }
    }

    #[test]
    fn mel_banks_first_last_frequency_extent() {
        let banks = MelBanks::new(&kaldi_opts(80), &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        let num_cols = 257;
        let rows: Vec<&[f32]> = matrix.chunks(num_cols).collect();
        // 16 kHz, 512 FFT: bin width = 31.25 Hz; low_freq=20 -> first bin ~0/1
        let first = rows[0].iter().position(|&w| w > 0.0).unwrap();
        assert!(first <= 2, "first active bin {first}");
        // high_freq=nyquist -> last bin active near 256
        let last = rows[79].iter().rposition(|&w| w > 0.0).unwrap();
        assert!(last >= 250, "last active bin {last}");
    }

    #[test]
    fn mel_banks_htk_mode_bug_replica() {
        let mut opts = kaldi_opts(40);
        opts.htk_mode = true;
        let banks = MelBanks::new(&opts, &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        // HTK bug: first bin's first weight forced to zero (mel_low_freq != 0)
        assert_eq!(matrix[0], 0.0);
    }

    #[test]
    fn mel_banks_htk_mode_flooring_in_compute() {
        let mut opts = kaldi_opts(40);
        opts.htk_mode = true;
        let banks = MelBanks::new(&opts, &frame_opts(), 1.0);
        let mut out = [0.0f32; 40];
        // zero power spectrum -> all energies zero -> floored to 1.0
        let spec = vec![0.0f32; 257];
        banks.compute(&spec, &mut out);
        assert!(
            out.iter().all(|&e| e == 1.0),
            "htk flooring failed: {out:?}"
        );
    }

    #[test]
    fn mel_banks_compute_manual() {
        // A single-bin impulse in the power spectrum selects triangular bins.
        let opts = kaldi_opts(23);
        let banks = MelBanks::new(&opts, &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        let num_cols = 257;
        let rows: Vec<&[f32]> = matrix.chunks(num_cols).collect();
        // choose a column covered by bin 5
        let col = rows[5].iter().position(|&w| w > 0.0).unwrap();
        let mut spec = vec![0.0f32; 257];
        spec[col] = 3.0;
        let mut out = [0.0f32; 23];
        banks.compute(&spec, &mut out);
        assert_eq!(out[5], 3.0 * rows[5][col]);
        assert_eq!(out[6], 3.0 * rows[6][col]);
        // bins far away get 0
        assert_eq!(out[0], 0.0);
        assert_eq!(out[22], 0.0);
    }

    #[test]
    fn mel_banks_different_bins_change_weights() {
        let b23 = MelBanks::new(&kaldi_opts(23), &frame_opts(), 1.0);
        let b80 = MelBanks::new(&kaldi_opts(80), &frame_opts(), 1.0);
        assert_ne!(b23.get_matrix(), b80.get_matrix());
        assert_eq!(b23.num_bins(), 23);
        assert_eq!(b80.num_bins(), 80);
    }

    #[test]
    fn mel_banks_vtln_warp_changes_banks() {
        let opts = kaldi_opts(40);
        let f = frame_opts();
        let plain = MelBanks::new(&opts, &f, 1.0);
        let warped = MelBanks::new(&opts, &f, 1.1);
        assert_ne!(plain.get_matrix(), warped.get_matrix());
    }

    // --- MelBanks (librosa mode) -------------------------------------------

    fn librosa_opts() -> MelBanksOptions {
        MelBanksOptions {
            num_bins: 80,
            low_freq: 0.0,
            is_librosa: true,
            norm: "slaney".into(),
            ..MelBanksOptions::default()
        }
    }

    #[test]
    fn librosa_mel_banks_slaney_norm_continuous_integral_is_one() {
        // norm="slaney" normalizes each triangle so that its *continuous*
        // integral over frequency is 1: sum over discrete bins of
        // w(bin) ~= 1 / fft_bin_width. For 16 kHz / 512 FFT, bin width is
        // 31.25 Hz, so each row sums to ~0.032.
        let banks = MelBanks::new(&librosa_opts(), &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        let num_cols = 257;
        let rows: Vec<&[f32]> = matrix.chunks(num_cols).collect();
        let bin_width = 16000.0 / 512.0; // 31.25
        for (i, row) in rows.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            let expected = 1.0 / bin_width;
            assert!(
                (sum - expected).abs() / expected < 0.15,
                "row {i} sums to {sum}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn librosa_no_norm_rows_sum_to_triangle_area() {
        // Without norm, each row sums to the discrete area of the triangle,
        // which is (right-left)/(2*bin_width) and hence > 1 here.
        let mut opts = librosa_opts();
        opts.norm = String::new();
        let banks = MelBanks::new(&opts, &frame_opts(), 1.0);
        let matrix = banks.get_matrix();
        let num_cols = 257;
        let rows: Vec<&[f32]> = matrix.chunks(num_cols).collect();
        let sums: Vec<f32> = rows.iter().map(|r| r.iter().sum()).collect();
        let min_sum = sums.iter().cloned().fold(f32::MAX, f32::min);
        let max_sum = sums.iter().cloned().fold(f32::MIN, f32::max);
        assert!(min_sum > 1.0, "min row sum {min_sum} should exceed 1");
        assert!(max_sum > min_sum, "row sums vary across bins");
    }

    #[test]
    fn librosa_htk_mel_scale_differs_from_slaney() {
        let mut opts = librosa_opts();
        opts.use_slaney_mel_scale = false;
        let a = MelBanks::new(&opts, &frame_opts(), 1.0);
        let b = MelBanks::new(&librosa_opts(), &frame_opts(), 1.0);
        assert_ne!(a.get_matrix(), b.get_matrix());
    }

    #[test]
    fn librosa_floor_to_int_bin_differs() {
        let mut opts = librosa_opts();
        opts.floor_to_int_bin = true;
        let a = MelBanks::new(&opts, &frame_opts(), 1.0);
        let b = MelBanks::new(&librosa_opts(), &frame_opts(), 1.0);
        assert_ne!(a.get_matrix(), b.get_matrix());
    }

    #[test]
    fn librosa_uses_extra_bin() {
        // librosa scans num_fft_bins+1 columns; verify compute works with a
        // full-length power spectrum.
        let banks = MelBanks::new(&librosa_opts(), &frame_opts(), 1.0);
        let spec = vec![1.0f32; 257];
        let mut out = vec![0.0f32; 80];
        banks.compute(&spec, &mut out);
        assert!(out.iter().all(|e| e.is_finite()));
    }

    #[test]
    #[should_panic(expected = "at least 3")]
    fn mel_banks_requires_three_bins() {
        let opts = kaldi_opts(2);
        MelBanks::new(&opts, &frame_opts(), 1.0);
    }

    #[test]
    #[should_panic(expected = "Bad values")]
    fn mel_banks_rejects_bad_freq_bounds() {
        let mut opts = kaldi_opts(40);
        opts.low_freq = 10000.0; // above nyquist
        MelBanks::new(&opts, &frame_opts(), 1.0);
    }
}
