//! Per-frame FBank computation, a faithful port of
//! `kaldi-native-fbank/csrc/feature-fbank.cc`.

use std::collections::HashMap;

use crate::feature_window::inner_product;
use crate::fft::{Rfft, compute_power_spectrum};
use crate::mel::MelBanks;
use crate::options::FbankOptions;

/// Computes one frame of FBank features from one windowed frame of signal.
pub struct FbankComputer {
    opts: FbankOptions,
    log_energy_floor: f32,
    /// Cache of mel banks keyed by the VTLN warp factor (as raw bits).
    mel_banks: HashMap<u32, MelBanks>,
    rfft: Rfft,
}

impl FbankComputer {
    pub fn new(opts: FbankOptions) -> Self {
        let mut log_energy_floor = 0.0f32;
        if opts.energy_floor > 0.0 {
            log_energy_floor = opts.energy_floor.ln();
        }
        let rfft = Rfft::new(opts.frame_opts.padded_window_size() as usize);

        let mut this = Self {
            opts,
            log_energy_floor,
            mel_banks: HashMap::new(),
            rfft,
        };
        // We will definitely need the filterbanks for VTLN warping factor 1.0.
        this.get_mel_banks(1.0);
        this
    }

    pub fn opts(&self) -> &FbankOptions {
        &self.opts
    }

    pub fn dim(&self) -> i32 {
        self.opts.mel_opts.num_bins + if self.opts.use_energy { 1 } else { 0 }
    }

    /// If true, compute log_energy before windowing but after dithering and DC
    /// removal.
    pub fn need_raw_log_energy(&self) -> bool {
        self.opts.use_energy && self.opts.raw_energy
    }

    fn get_mel_banks(&mut self, vtln_warp: f32) -> &MelBanks {
        self.mel_banks
            .entry(vtln_warp.to_bits())
            .or_insert_with(|| MelBanks::new(&self.opts.mel_opts, &self.opts.frame_opts, vtln_warp))
    }

    /// Computes one frame of features.
    ///
    /// `signal_raw_log_energy`: the log-energy of the frame prior to windowing
    /// and pre-emphasis (ignored unless `need_raw_log_energy()`); `vtln_warp`:
    /// VTLN warping factor, normally 1.0; `signal_frame`: the windowed frame of
    /// size `padded_window_size()` (used as workspace, modified in place);
    /// `feature`: pre-allocated output of size `dim()`.
    pub fn compute(
        &mut self,
        signal_raw_log_energy: f32,
        vtln_warp: f32,
        signal_frame: &mut [f32],
        feature: &mut [f32],
    ) {
        let padded = self.opts.frame_opts.padded_window_size() as usize;
        let use_energy = self.opts.use_energy;
        let raw_energy = self.opts.raw_energy;
        let use_power = self.opts.use_power;
        let use_log_fbank = self.opts.use_log_fbank;
        let htk_compat = self.opts.htk_compat;
        let num_bins = self.opts.mel_opts.num_bins as usize;
        let mel_offset = if use_energy && !htk_compat { 1 } else { 0 };

        assert_eq!(signal_frame.len(), padded);

        // Compute energy after window function (not the raw one).
        let mut signal_raw_log_energy = signal_raw_log_energy;
        if use_energy && !raw_energy {
            signal_raw_log_energy = inner_product(signal_frame, signal_frame)
                .max(f32::EPSILON)
                .ln();
        }

        self.rfft.compute(signal_frame); // signal_frame is modified in place
        compute_power_spectrum(signal_frame);

        // Use magnitude instead of power if requested.
        if !use_power {
            let half = signal_frame.len() / 2 + 1;
            for v in signal_frame[..half].iter_mut() {
                *v = v.sqrt();
            }
        }

        // mel_energies = feature[mel_offset .. mel_offset + num_bins]
        let mel_energies = &mut feature[mel_offset..mel_offset + num_bins];

        {
            let mel_banks = self.get_mel_banks(vtln_warp);
            mel_banks.compute(signal_frame, mel_energies);
        }

        if use_log_fbank {
            // Avoid log of zero (which should be prevented anyway by dithering).
            for e in mel_energies.iter_mut() {
                let t = (*e).max(f32::EPSILON);
                *e = t.ln();
            }
        }

        // Copy energy as the first value (or last, if htk_compat == true).
        if use_energy {
            let mut sle = signal_raw_log_energy;
            if self.opts.energy_floor > 0.0 && sle < self.log_energy_floor {
                sle = self.log_energy_floor;
            }
            let energy_index = if htk_compat { num_bins } else { 0 };
            feature[energy_index] = sle;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_window::{FeatureWindowFunction, extract_window};

    fn base_opts() -> FbankOptions {
        let mut o = FbankOptions::default();
        o.frame_opts.dither = 0.0;
        o
    }

    /// Runs a full frame through the computer with a simple windowed frame.
    fn compute_frame(opts: &FbankOptions, wave: &[f32], frame: i32) -> (Vec<f32>, Vec<f32>) {
        let mut computer = FbankComputer::new(opts.clone());
        let wf = FeatureWindowFunction::from_frame_opts(&opts.frame_opts);
        let mut window = vec![0.0f32; opts.frame_opts.padded_window_size() as usize];
        let mut raw_log_energy = 0.0f32;
        extract_window(
            0,
            wave,
            frame,
            &opts.frame_opts,
            &wf,
            &mut window,
            computer
                .need_raw_log_energy()
                .then_some(&mut raw_log_energy),
        );
        let window_pre_fft = window.clone(); // compute() overwrites `window` in place
        let mut feature = vec![0.0f32; computer.dim() as usize];
        computer.compute(raw_log_energy, 1.0, &mut window, &mut feature);
        (feature, window_pre_fft)
    }

    #[test]
    fn dim_counts() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 80;
        let c = FbankComputer::new(opts.clone());
        assert_eq!(c.dim(), 80);
        opts.use_energy = true;
        let c = FbankComputer::new(opts);
        assert_eq!(c.dim(), 81);
    }

    #[test]
    fn need_raw_log_energy_flag() {
        let mut opts = base_opts();
        assert!(!FbankComputer::new(opts.clone()).need_raw_log_energy());
        opts.use_energy = true;
        opts.raw_energy = true;
        assert!(FbankComputer::new(opts.clone()).need_raw_log_energy());
        opts.raw_energy = false;
        assert!(!FbankComputer::new(opts).need_raw_log_energy());
    }

    #[test]
    fn compute_output_finite_and_reasonable() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 80;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (feat, _) = compute_frame(&opts, &wave, 5);
        assert_eq!(feat.len(), 80);
        assert!(feat.iter().all(|x| x.is_finite()), "non-finite: {feat:?}");
        // log-fbank of a sine wave: mid-band bins are high, edges low
        let max = feat.iter().cloned().fold(f32::MIN, f32::max);
        let min = feat.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max > -5.0, "max={max}");
        assert!(min < -5.0, "min={min}");
    }

    #[test]
    fn silence_gives_floor_log_value() {
        let opts = base_opts();
        let wave = vec![0.0f32; 32000];
        let (feat, _) = compute_frame(&opts, &wave, 5);
        // all-zero signal -> zero power -> log(max(0, eps)) = log(eps)
        let expected = f32::EPSILON.ln();
        for v in &feat {
            assert!((v - expected).abs() < 1e-4, "v={v} expected={expected}");
        }
    }

    #[test]
    fn energy_placed_first_by_default() {
        // raw_energy=true computes energy *before* preemphasis and windowing;
        // with preemph=0 / remove_dc=false / rectangular window those steps are
        // identity, so the reported energy equals the post-window inner product.
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 80;
        opts.use_energy = true;
        opts.frame_opts.preemph_coeff = 0.0;
        opts.frame_opts.remove_dc_offset = false;
        opts.frame_opts.window_type = crate::options::WindowType::Rectangular;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (feat, window) = compute_frame(&opts, &wave, 5);
        assert_eq!(feat.len(), 81);
        let energy = crate::feature_window::inner_product(&window, &window)
            .max(f32::EPSILON)
            .ln();
        assert!(
            (feat[0] - energy).abs() < 1e-3,
            "feat[0]={} energy={energy}",
            feat[0]
        );
        // mel bins live at 1..81
        assert!(feat[1..].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn energy_placed_last_with_htk_compat() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 80;
        opts.use_energy = true;
        opts.htk_compat = true;
        opts.frame_opts.preemph_coeff = 0.0;
        opts.frame_opts.remove_dc_offset = false;
        opts.frame_opts.window_type = crate::options::WindowType::Rectangular;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (feat, window) = compute_frame(&opts, &wave, 5);
        assert_eq!(feat.len(), 81);
        let energy = crate::feature_window::inner_product(&window, &window)
            .max(f32::EPSILON)
            .ln();
        assert!(
            (feat[80] - energy).abs() < 1e-3,
            "feat[80]={} energy={energy}",
            feat[80]
        );
        // mel bins occupy positions 0..80 in htk mode
        assert!(feat[..80].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn raw_energy_false_uses_post_process_energy() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 40;
        opts.use_energy = true;
        opts.raw_energy = false;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (feat, window) = compute_frame(&opts, &wave, 5);
        // raw_energy=false: energy computed from the fully-processed frame
        let energy = crate::feature_window::inner_product(&window, &window)
            .max(f32::EPSILON)
            .ln();
        assert!(
            (feat[0] - energy).abs() < 1e-3,
            "feat[0]={} energy={energy}",
            feat[0]
        );
    }

    #[test]
    fn energy_floor_clamps_energy() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 80;
        opts.use_energy = true;
        opts.energy_floor = 1.0; // log floor = 0
        let wave = vec![0.0f32; 32000]; // zero signal -> energy ~ -inf -> clamped
        let (feat, _) = compute_frame(&opts, &wave, 5);
        assert!(
            (feat[0] - 0.0).abs() < 1e-6,
            "clamped energy should be 0, got {}",
            feat[0]
        );
    }

    #[test]
    fn magnitude_mode_differs_from_power_mode() {
        let mut power = base_opts();
        power.use_power = true;
        let mut mag = base_opts();
        mag.use_power = false;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.013).sin() + 0.1).collect();
        let (fp, _) = compute_frame(&power, &wave, 3);
        let (fm, _) = compute_frame(&mag, &wave, 3);
        assert!(fp.iter().all(|x| x.is_finite()));
        assert!(fm.iter().all(|x| x.is_finite()));
        // magnitude-weighted energies differ from power-weighted ones
        assert_ne!(fp, fm);
    }

    #[test]
    fn linear_fbank_is_exp_of_log_fbank() {
        let mut logm = base_opts();
        logm.use_log_fbank = true;
        let mut lin = base_opts();
        lin.use_log_fbank = false;
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.007).cos()).collect();
        let (fl, _) = compute_frame(&logm, &wave, 3);
        let (fnl, _) = compute_frame(&lin, &wave, 3);
        for (l, n) in fl.iter().zip(fnl.iter()) {
            assert!(
                (l - n.ln().max(f32::EPSILON.ln())).abs() < 1e-3,
                "log={l} linear={n}"
            );
        }
    }

    #[test]
    fn compute_is_deterministic() {
        let opts = base_opts();
        let wave: Vec<f32> = (0..32000).map(|i| (i * 31 % 997) as f32 / 997.0).collect();
        let (a, _) = compute_frame(&opts, &wave, 7);
        let (b, _) = compute_frame(&opts, &wave, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn different_frames_differ() {
        let opts = base_opts();
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (f0, _) = compute_frame(&opts, &wave, 0);
        let (f1, _) = compute_frame(&opts, &wave, 1);
        assert_ne!(f0, f1);
    }

    #[test]
    fn vtln_warp_supported_via_cache() {
        let mut opts = base_opts();
        opts.mel_opts.num_bins = 40;
        let mut computer = FbankComputer::new(opts.clone());
        let wf = FeatureWindowFunction::from_frame_opts(&opts.frame_opts);
        let mut window = vec![0.0f32; 512];
        let wave: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.01).sin()).collect();
        extract_window(0, &wave, 0, &opts.frame_opts, &wf, &mut window, None);
        let mut a = vec![0.0f32; 40];
        let mut b = vec![0.0f32; 40];
        computer.compute(0.0, 1.0, &mut window.clone(), &mut a);
        computer.compute(0.0, 1.1, &mut window.clone(), &mut b);
        assert_ne!(a, b);
    }
}
