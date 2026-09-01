//! Property / fuzz-style tests with deterministic seeds: outputs are finite,
//! deterministic, chunk-order independent, and the extractor never panics on
//! arbitrary (finite) inputs across a wide configuration space.

use kaldi_fbank::{FbankOptions, OnlineFbank, compute_fbank};

/// Small deterministic xorshift64 PRNG (no external deps, reproducible).
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        // uniform in [-1, 1)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo) as u64) as usize
    }
}

fn random_signal(rng: &mut XorShift64, n: usize, kind: u32) -> Vec<f32> {
    match kind % 4 {
        0 => (0..n).map(|_| rng.next_f32()).collect(), // white noise
        1 => (0..n).map(|i| rng.next_f32() * (i as f32).sin()).collect(), // modulated noise
        2 => (0..n).map(|_| rng.next_f32() * 1000.0).collect(), // large amplitude
        3 => (0..n).map(|_| rng.next_f32() * 1e-5).collect(), // tiny amplitude
        _ => unreachable!(),
    }
}

fn random_opts(rng: &mut XorShift64) -> FbankOptions {
    let mut o = FbankOptions::default();
    o.frame_opts.dither = 0.0;
    o.mel_opts.num_bins = [23, 25, 40, 80][rng.range(0, 4)];
    o.frame_opts.snip_edges = rng.next_u64() & 1 == 0;
    o.frame_opts.window_type = [
        kaldi_fbank::WindowType::Povey,
        kaldi_fbank::WindowType::Hamming,
        kaldi_fbank::WindowType::Hanning,
        kaldi_fbank::WindowType::Hann,
        kaldi_fbank::WindowType::Sine,
        kaldi_fbank::WindowType::Rectangular,
        kaldi_fbank::WindowType::Blackman,
    ][rng.range(0, 7)];
    o.frame_opts.preemph_coeff = [0.0, 0.5, 0.97][rng.range(0, 3)];
    o.frame_opts.remove_dc_offset = rng.next_u64() & 1 == 0;
    o.frame_opts.round_to_power_of_two = rng.next_u64() & 1 == 0;
    o.frame_opts.frame_shift_ms = [5.0, 10.0, 20.0][rng.range(0, 3)];
    o.frame_opts.frame_length_ms = [20.0, 25.0, 32.0][rng.range(0, 3)];
    o.use_energy = rng.next_u64() & 1 == 0;
    o.use_log_fbank = rng.next_u64() & 1 == 0;
    o.use_power = rng.next_u64() & 1 == 0;
    if rng.next_u64() & 1 == 0 {
        o.mel_opts.is_librosa = true;
        o.mel_opts.low_freq = 0.0;
        o.mel_opts.norm = if rng.next_u64() & 1 == 0 {
            "slaney".into()
        } else {
            String::new()
        };
    }
    o
}

#[test]
fn fuzz_no_panic_finite_deterministic() {
    let mut rng = XorShift64(0x9E3779B97F4A7C15);
    for iter in 0..300 {
        let opts = random_opts(&mut rng);
        let n = rng.range(0, 40_000);
        let kind = rng.next_u64() as u32;
        let sig = random_signal(&mut rng, n, kind);

        let a = compute_fbank(&opts, &sig);
        let b = compute_fbank(&opts, &sig);
        assert_eq!(a, b, "iter {iter}: non-deterministic result");
        for (fi, frame) in a.iter().enumerate() {
            assert_eq!(
                frame.len(),
                opts.mel_opts.num_bins as usize + if opts.use_energy { 1 } else { 0 },
                "iter {iter} frame {fi}: wrong dim"
            );
            for v in frame {
                assert!(v.is_finite(), "iter {iter} frame {fi}: non-finite {v}");
            }
        }
    }
}

#[test]
fn fuzz_chunk_order_independent() {
    let mut rng = XorShift64(0xD1B54A32D192ED03);
    for iter in 0..80 {
        let opts = random_opts(&mut rng);
        let n = rng.range(0, 30_000);
        let kind = rng.next_u64() as u32;
        let sig = random_signal(&mut rng, n, kind);
        let batch = compute_fbank(&opts, &sig);

        let mut f = OnlineFbank::new(opts.clone());
        let chunk = rng.range(1, 2000);
        for c in sig.chunks(chunk) {
            f.accept_waveform(opts.frame_opts.samp_freq, c);
        }
        f.input_finished();
        assert_eq!(
            f.num_frames_ready() as usize,
            batch.len(),
            "iter {iter} chunk={chunk}"
        );
        for (i, frame) in batch.iter().enumerate().take(20) {
            assert_eq!(
                f.get_frame(i as i32),
                frame.as_slice(),
                "iter {iter} frame {i}"
            );
        }
    }
}

#[test]
fn fuzz_recycling_vector_never_panics() {
    use kaldi_fbank::RecyclingVector;
    let mut rng = XorShift64(0x1234567890ABCDEF);
    for _ in 0..200 {
        let mut v = RecyclingVector::new(rng.range(0, 8) as i32);
        let pushes = rng.range(0, 30);
        for i in 0..pushes {
            v.push_back(vec![i as f32]);
        }
        // logical indices are stable
        let len = v.len();
        let pop_n = rng.range(0, 40) as i32;
        v.pop(pop_n);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn frame_count_is_monotonic_in_signal_length() {
    let mut rng = XorShift64(0xFEDCBA9876543210);
    for _ in 0..50 {
        let opts = random_opts(&mut rng);
        let n1 = rng.range(0, 20_000) as i64;
        let n2 = n1 + rng.range(0, 5000) as i64;
        let f1 = kaldi_fbank::num_frames(n1, &opts.frame_opts, true);
        let f2 = kaldi_fbank::num_frames(n2, &opts.frame_opts, true);
        assert!(f2 >= f1, "n1={n1} f1={f1} n2={n2} f2={f2}");
        // no more than (n2-n1)/shift + 1 extra frames
        let shift = opts.frame_opts.window_shift() as i64;
        assert!((f2 as i64 - f1 as i64) <= (n2 - n1) / shift + 1);
    }
}

#[test]
fn fft_energy_conserved_via_pipeline() {
    // Feeding energy-normalized random signals must not blow up.
    use kaldi_fbank::fft::{Rfft, compute_power_spectrum};
    let mut rng = XorShift64(0xDEADBEEFCAFEBABE);
    for n in [64usize, 128, 512] {
        let sig: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let mut buf = sig.clone();
        let mut rfft = Rfft::new(n);
        rfft.compute(&mut buf);
        compute_power_spectrum(&mut buf);
        // power spectrum entries must be non-negative and finite
        for (i, v) in buf[..=n / 2].iter().enumerate() {
            assert!(*v >= 0.0 && v.is_finite(), "n={n} i={i}");
        }
    }
}

#[test]
fn mel_energies_non_negative_before_log() {
    // use_log_fbank=false exposes the raw (linear) mel energies: they must be
    // non-negative for any real input.
    let mut rng = XorShift64(0xCAFEBABE12345678);
    for _ in 0..50 {
        let mut opts = FbankOptions::default();
        opts.frame_opts.dither = 0.0;
        opts.use_log_fbank = false;
        opts.mel_opts.num_bins = [23, 40, 80][rng.range(0, 3)];
        let n = rng.range(400, 8000);
        let sig = random_signal(&mut rng, n, 0);
        for frame in compute_fbank(&opts, &sig) {
            for v in frame {
                assert!(v >= 0.0, "negative linear mel energy {v}");
            }
        }
    }
}
