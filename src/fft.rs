//! Real FFT wrapper around the mainstream [`realfft`] / [`rustfft`] crates.
//!
//! Produces the same packed layout as the reference `kiss_fftr`:
//!
//! ```text
//! in_out[0]     = R[0]     (DC)
//! in_out[1]     = R[n/2]   (Nyquist)
//! in_out[2k]    = R[k]
//! in_out[2k+1]  = I[k]     for 1 <= k < n/2
//! ```

use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

/// n-point real FFT (n even, n >= 2). Plan is precomputed once and reused.
pub struct Rfft {
    fft: std::sync::Arc<dyn RealToComplex<f32>>,
    complex: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl Rfft {
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n % 2 == 0, // `is_multiple_of` needs 1.87; keep MSRV 1.85
            "n should be even and >= 2, got {n}"
        );
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let complex = fft.make_output_vec();
        let scratch = vec![Complex::default(); fft.get_scratch_len()];
        Self {
            fft,
            complex,
            scratch,
        }
    }

    /// Computes the forward real FFT of `in_out` (length `n`) in place,
    /// rewriting it in the packed layout above.
    pub fn compute(&mut self, in_out: &mut [f32]) {
        let n = in_out.len();
        self.fft
            .process_with_scratch(in_out, &mut self.complex, &mut self.scratch)
            .expect("real FFT failed");

        let half = n / 2;
        in_out[0] = self.complex[0].re;
        in_out[1] = self.complex[half].re;
        for i in 1..half {
            in_out[2 * i] = self.complex[i].re;
            in_out[2 * i + 1] = self.complex[i].im;
        }
    }
}

/// Converts the packed FFT output into a power spectrum (first `n/2 + 1`
/// elements), exactly like the reference `ComputePowerSpectrum`.
///
/// The contents of the remaining `n/2 - 1` elements are unspecified, matching
/// the reference behaviour.
pub fn compute_power_spectrum(p: &mut [f32]) {
    let dim = p.len();
    let half_dim = dim / 2;
    let first_energy = p[0] * p[0];
    let last_energy = p[1] * p[1];

    for i in 1..half_dim {
        let real = p[2 * i];
        let im = p[2 * i + 1];
        p[i] = real * real + im * im;
    }
    p[0] = first_energy;
    p[half_dim] = last_energy;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(n^2) real DFT producing the same packed layout as `Rfft`.
    fn naive_dft(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        // R[0] = sum x[j]
        out[0] = x.iter().sum();
        // R[n/2] = sum x[j] * (-1)^j
        let mut nyq = 0.0f32;
        for (j, &v) in x.iter().enumerate() {
            nyq += if j % 2 == 0 { v } else { -v };
        }
        out[1] = nyq;
        for k in 1..n / 2 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, &v) in x.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                re += v as f64 * angle.cos();
                im -= v as f64 * angle.sin();
            }
            out[2 * k] = re as f32;
            out[2 * k + 1] = im as f32;
        }
        out
    }

    #[test]
    fn rfft_matches_naive_dft_small() {
        for n in [8usize, 16, 32] {
            let x: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.7).sin() + i as f32 * 0.01)
                .collect();
            let mut buf = x.clone();
            let mut rfft = Rfft::new(n);
            rfft.compute(&mut buf);
            let expected = naive_dft(&x);
            for i in 0..n {
                assert!(
                    (buf[i] - expected[i]).abs() < 1e-4,
                    "n={n} i={i} got={} expected={}",
                    buf[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn rfft_dc_signal() {
        let n = 16;
        let x = vec![3.0f32; n];
        let mut buf = x.clone();
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut buf);
        // DC bin = sum = 48; all others ~0
        assert!((buf[0] - 48.0).abs() < 1e-4);
        for (i, v) in buf[1..].iter().enumerate() {
            assert!(v.abs() < 1e-4, "i={} val={}", i + 1, v);
        }
    }

    #[test]
    fn rfft_sine_peak() {
        // A sine of k cycles must peak in bin k.
        let n = 64;
        let k = 5;
        let x: Vec<f32> = (0..n)
            .map(|j| (2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32).sin())
            .collect();
        let mut buf = x.clone();
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut buf);
        // find the bin with max power among 1..n/2
        let mut best = 0usize;
        let mut best_p = 0.0f32;
        #[allow(clippy::needless_range_loop)] // packed layout: bins at 2*i / 2*i+1
        for i in 1..n / 2 {
            let p = buf[2 * i] * buf[2 * i] + buf[2 * i + 1] * buf[2 * i + 1];
            if p > best_p {
                best_p = p;
                best = i;
            }
        }
        assert_eq!(best, k, "peak should be at bin {k}");
    }

    #[test]
    fn parseval_theorem() {
        // sum |x|^2 == (1/n) * sum |X|^2
        let n = 512;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();
        let mut buf = x.clone();
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut buf);
        let time_energy: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        // For a real signal, X[n-k] = conj(X[k]): the full spectrum energy is
        // |X0|^2 + |Xn/2|^2 + 2*sum_{k=1}^{n/2-1}|Xk|^2
        let mut freq_energy = 0.0f64;
        freq_energy += (buf[0] as f64).powi(2);
        freq_energy += (buf[1] as f64).powi(2);
        for i in 1..n / 2 {
            let re = buf[2 * i] as f64;
            let im = buf[2 * i + 1] as f64;
            freq_energy += 2.0 * (re * re + im * im);
        }
        // fft of length n (realfft does not normalize)
        let expected = time_energy * n as f64;
        let rel = (freq_energy - expected).abs() / expected;
        assert!(rel < 1e-4, "rel error {rel}");
    }

    #[test]
    fn rfft_requires_even_positive() {
        assert!(std::panic::catch_unwind(|| Rfft::new(0)).is_err());
        assert!(std::panic::catch_unwind(|| Rfft::new(3)).is_err());
        assert!(std::panic::catch_unwind(|| Rfft::new(7)).is_err());
    }

    #[test]
    fn power_spectrum_known_values() {
        // packed layout [R0, Rnyq, R1, I1, R2, I2, ...]
        let mut p = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        compute_power_spectrum(&mut p);
        // p[0] = 2^2 = 4
        assert_eq!(p[0], 4.0);
        // p[4] (= half_dim) = 3^2 = 9 (Nyquist power stored at index n/2)
        assert_eq!(p[4], 9.0);
        // p[1] = 4^2+5^2 = 41
        assert_eq!(p[1], 41.0);
        // p[2] = 6^2+7^2 = 85
        assert_eq!(p[2], 85.0);
        // p[3] = 8^2+9^2 = 145
        assert_eq!(p[3], 145.0);
        // tail beyond n/2 is unspecified; here p[5..] holds 8,9 remnants
    }

    #[test]
    fn power_spectrum_of_dc_signal() {
        let n = 16;
        let mut buf = vec![5.0f32; n];
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut buf);
        compute_power_spectrum(&mut buf);
        assert!((buf[0] - 80.0f32 * 80.0).abs() < 1e-2); // (5*16)^2
        // all other power ~0
        for (i, v) in buf[1..=8].iter().enumerate() {
            assert!(*v < 1e-3, "i={} p={}", i + 1, v);
        }
    }

    #[test]
    fn rfft_plan_reuse_is_consistent() {
        // calling compute twice with the same plan on the same input gives the
        // same result (no hidden state)
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut a = x.clone();
        let mut b = x.clone();
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut a);
        rfft.compute(&mut b);
        assert_eq!(a, b);
    }
}
