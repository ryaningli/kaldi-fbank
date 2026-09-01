//! Online (streaming) FBank extraction, a faithful port of
//! `kaldi-native-fbank/csrc/online-feature.cc`.

use std::collections::VecDeque;

use crate::feature_fbank::FbankComputer;
use crate::feature_window::{
    FeatureWindowFunction, extract_window, first_sample_of_frame, num_frames,
};
use crate::options::FbankOptions;

/// Storage for feature vectors with an optional cap on memory usage; old
/// elements can be dropped while keeping frame indices stable.
pub struct RecyclingVector {
    items: VecDeque<Vec<f32>>,
    items_to_hold: i32,
    first_available_index: i32,
}

impl RecyclingVector {
    /// `items_to_hold <= 0` means keep everything.
    pub fn new(items_to_hold: i32) -> Self {
        Self {
            items: VecDeque::new(),
            items_to_hold: if items_to_hold == 0 {
                -1
            } else {
                items_to_hold
            },
            first_available_index: 0,
        }
    }

    pub fn at(&self, index: i32) -> &[f32] {
        if index < self.first_available_index {
            panic!(
                "Attempted to retrieve a feature vector that was already removed \
                 (index = {index}; first_available_index = {}; size = {})",
                self.first_available_index,
                self.len()
            );
        }
        &self.items[index as usize - self.first_available_index as usize]
    }

    pub fn push_back(&mut self, item: Vec<f32>) {
        if self.items_to_hold > 0 && self.items.len() == self.items_to_hold as usize {
            self.items.pop_front();
            self.first_available_index += 1;
        }
        self.items.push_back(item);
    }

    /// Size as if no recycling had happened.
    pub fn len(&self) -> i32 {
        self.first_available_index + self.items.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Discards the first `n` frames.
    pub fn pop(&mut self, n: i32) {
        for _ in 0..n {
            if self.items.pop_front().is_none() {
                break;
            }
            self.first_available_index += 1;
        }
    }
}

/// Online FBank feature extractor: accepts waveform in chunks, computes frames
/// as soon as they become available, and (optionally) recycles old frames.
pub struct OnlineFbank {
    computer: FbankComputer,
    window_function: FeatureWindowFunction,
    features: RecyclingVector,
    input_finished: bool,
    /// Number of samples already discarded, i.e. prior to `waveform_remainder_`.
    waveform_offset: i64,
    /// Short piece of waveform kept after extracting all whole frames.
    waveform_remainder: Vec<f32>,
    // Reusable scratch buffers (no per-frame allocation).
    window: Vec<f32>,
    feature: Vec<f32>,
}

impl OnlineFbank {
    pub fn new(opts: FbankOptions) -> Self {
        let computer = FbankComputer::new(opts);
        let frame_opts = computer.opts().frame_opts.clone();
        let dim = computer.dim() as usize;
        let window_function = FeatureWindowFunction::from_frame_opts(&frame_opts);
        let padded = frame_opts.padded_window_size() as usize;
        Self {
            computer,
            window_function,
            features: RecyclingVector::new(-1),
            input_finished: false,
            waveform_offset: 0,
            waveform_remainder: Vec::new(),
            window: vec![0.0; padded],
            feature: vec![0.0; dim],
        }
    }

    pub fn dim(&self) -> i32 {
        self.computer.dim()
    }

    pub fn frame_shift_in_seconds(&self) -> f32 {
        self.computer.opts().frame_opts.frame_shift_ms / 1000.0
    }

    pub fn num_frames_ready(&self) -> i32 {
        self.features.len()
    }

    /// `IsLastFrame()` is only true after `input_finished()` for the final frame.
    pub fn is_last_frame(&self, frame: i32) -> bool {
        self.input_finished && frame == self.num_frames_ready() - 1
    }

    pub fn get_frame(&self, frame: i32) -> &[f32] {
        self.features.at(frame)
    }

    /// Feeds more waveform. The sampling rate must match the options.
    pub fn accept_waveform(&mut self, sampling_rate: f32, waveform: &[f32]) {
        if waveform.is_empty() {
            return; // Nothing to do.
        }
        assert!(
            !self.input_finished,
            "accept_waveform called after input_finished() was called."
        );
        assert_eq!(
            sampling_rate,
            self.computer.opts().frame_opts.samp_freq,
            "sampling rate mismatch: got {sampling_rate}, expected {}",
            self.computer.opts().frame_opts.samp_freq
        );

        self.waveform_remainder.extend_from_slice(waveform);
        self.compute_features();
    }

    /// Tells the extractor that no more waveform is coming; flushes the last
    /// frame(s) when `snip_edges == false`.
    pub fn input_finished(&mut self) {
        self.input_finished = true;
        self.compute_features();
    }

    /// Discards the first `n` frames.
    pub fn pop(&mut self, n: i32) {
        self.features.pop(n);
    }

    fn compute_features(&mut self) {
        let frame_opts = self.computer.opts().frame_opts.clone();
        let num_samples_total = self.waveform_offset + self.waveform_remainder.len() as i64;
        let num_frames_old = self.features.len();
        let num_frames_new = num_frames(num_samples_total, &frame_opts, self.input_finished);
        assert!(num_frames_new >= num_frames_old);

        // Note: this online feature-extraction code does not support VTLN.
        let vtln_warp = 1.0f32;
        let need_raw_log_energy = self.computer.need_raw_log_energy();

        for frame in num_frames_old..num_frames_new {
            let mut raw_log_energy = 0.0f32;
            extract_window(
                self.waveform_offset,
                &self.waveform_remainder,
                frame,
                &frame_opts,
                &self.window_function,
                &mut self.window,
                need_raw_log_energy.then_some(&mut raw_log_energy),
            );

            self.computer.compute(
                raw_log_energy,
                vtln_warp,
                &mut self.window,
                &mut self.feature,
            );
            self.features.push_back(self.feature.clone());
        }

        // Discard any portion of the signal that will not be necessary to
        // compute frames in the future.
        let first_sample_of_next_frame = first_sample_of_frame(num_frames_new, &frame_opts);
        let samples_to_discard = first_sample_of_next_frame - self.waveform_offset;

        if samples_to_discard > 0 {
            let samples_to_discard = samples_to_discard as usize;
            if samples_to_discard >= self.waveform_remainder.len() {
                self.waveform_offset += self.waveform_remainder.len() as i64;
                self.waveform_remainder.clear();
            } else {
                self.waveform_remainder.drain(..samples_to_discard);
                self.waveform_offset += samples_to_discard as i64;
            }
        }
    }
}

/// Batch convenience API: computes all FBank frames for a whole waveform in
/// one call. Equivalent to feeding the whole signal then calling
/// `input_finished()`.
pub fn compute_fbank(opts: &FbankOptions, samples: &[f32]) -> Vec<Vec<f32>> {
    let mut fbank = OnlineFbank::new(opts.clone());
    fbank.accept_waveform(opts.frame_opts.samp_freq, samples);
    fbank.input_finished();
    let n = fbank.num_frames_ready();
    (0..n).map(|i| fbank.get_frame(i).to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::FbankOptions;

    fn opts16k() -> FbankOptions {
        let mut o = FbankOptions::default();
        o.frame_opts.dither = 0.0;
        o.mel_opts.num_bins = 80;
        o
    }

    fn signal(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.013).sin() * 0.5).collect()
    }

    // --- RecyclingVector ----------------------------------------------------

    #[test]
    fn recycling_vector_basic() {
        let mut v = RecyclingVector::new(-1);
        assert!(v.is_empty());
        for i in 0..5 {
            v.push_back(vec![i as f32]);
        }
        assert_eq!(v.len(), 5);
        for i in 0..5 {
            assert_eq!(v.at(i), &[i as f32]);
        }
    }

    #[test]
    fn recycling_vector_capacity_eviction() {
        let mut v = RecyclingVector::new(3);
        for i in 0..10 {
            v.push_back(vec![i as f32]);
        }
        assert_eq!(v.len(), 10); // logical size keeps growing
        // only the last 3 items are physically stored
        assert_eq!(v.at(7), &[7.0]);
        assert_eq!(v.at(9), &[9.0]);
    }

    #[test]
    fn recycling_vector_pop() {
        let mut v = RecyclingVector::new(-1);
        for i in 0..6 {
            v.push_back(vec![i as f32]);
        }
        v.pop(2);
        assert_eq!(v.len(), 6); // logical size unchanged
        // remaining physical items: at(2) is the first stored one
        assert_eq!(v.at(2), &[2.0]);
        assert_eq!(v.at(5), &[5.0]);
        v.pop(4); // pops the remaining 4 stored items
        assert_eq!(v.len(), 6);
        // all items physically removed: accessing them now panics
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.at(5))).is_err());
        v.pop(100); // popping more than stored is safe
        assert_eq!(v.len(), 6);
        assert_eq!(v.first_available_index, 6);
    }

    #[test]
    #[should_panic(expected = "already removed")]
    fn recycling_vector_at_removed_index_panics() {
        let mut v = RecyclingVector::new(2);
        for i in 0..4 {
            v.push_back(vec![i as f32]);
        }
        let _ = v.at(0); // evicted
    }

    #[test]
    fn recycling_vector_empty_pop_is_safe() {
        let mut v = RecyclingVector::new(-1);
        v.pop(5);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    // --- OnlineFbank --------------------------------------------------------

    #[test]
    fn empty_input_produces_no_frames() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &[]);
        f.input_finished();
        assert_eq!(f.num_frames_ready(), 0);
    }

    #[test]
    fn sub_frame_input_produces_no_frames_snip_true() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &[0.0; 399]); // < 400 samples
        f.input_finished();
        assert_eq!(f.num_frames_ready(), 0);
    }

    #[test]
    fn one_sample_chunks_match_batch() {
        let opts = opts16k();
        let sig = signal(32000);
        let batch = compute_fbank(&opts, &sig);
        let mut f = OnlineFbank::new(opts.clone());
        for s in &sig {
            f.accept_waveform(16000.0, &[*s]);
        }
        f.input_finished();
        assert_eq!(f.num_frames_ready() as usize, batch.len());
        for (i, frame) in batch.iter().enumerate() {
            assert_eq!(f.get_frame(i as i32), frame.as_slice(), "frame {i}");
        }
    }

    #[test]
    fn arbitrary_chunk_sizes_match_batch() {
        let opts = opts16k();
        let sig = signal(100_000);
        let batch = compute_fbank(&opts, &sig);
        for chunk in [1usize, 7, 99, 1000, 1600, 16000, 100_000] {
            let mut f = OnlineFbank::new(opts.clone());
            for c in sig.chunks(chunk) {
                f.accept_waveform(16000.0, c);
            }
            f.input_finished();
            assert_eq!(f.num_frames_ready() as usize, batch.len(), "chunk={chunk}");
            for (i, frame) in batch.iter().enumerate().take(50) {
                assert_eq!(
                    f.get_frame(i as i32),
                    frame.as_slice(),
                    "chunk={chunk} frame={i}"
                );
            }
        }
    }

    #[test]
    fn snip_edges_false_streaming_matches_batch() {
        let mut opts = opts16k();
        opts.frame_opts.snip_edges = false;
        let sig = signal(50_000);
        let batch = compute_fbank(&opts, &sig);
        let mut f = OnlineFbank::new(opts.clone());
        for c in sig.chunks(1600) {
            f.accept_waveform(16000.0, c);
        }
        f.input_finished();
        assert_eq!(f.num_frames_ready() as usize, batch.len());
        for (i, frame) in batch.iter().enumerate() {
            assert_eq!(f.get_frame(i as i32), frame.as_slice(), "frame {i}");
        }
    }

    #[test]
    fn input_finished_is_idempotent() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &signal(16000));
        f.input_finished();
        let n = f.num_frames_ready();
        f.input_finished(); // calling again must not change anything
        assert_eq!(f.num_frames_ready(), n);
    }

    #[test]
    fn frames_appear_before_input_finished() {
        // With snip_edges=true and enough samples, frames are ready immediately.
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &signal(16000));
        assert!(f.num_frames_ready() > 0);
    }

    #[test]
    fn is_last_frame_semantics() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &signal(16000));
        let n = f.num_frames_ready();
        assert!(!f.is_last_frame(n - 1), "not finished yet");
        f.input_finished();
        assert!(f.is_last_frame(n - 1));
        assert!(!f.is_last_frame(n - 2));
    }

    #[test]
    fn dim_and_frame_shift_api() {
        let f = OnlineFbank::new(opts16k());
        assert_eq!(f.dim(), 80);
        assert!((f.frame_shift_in_seconds() - 0.01).abs() < 1e-9);
        let mut o = opts16k();
        o.frame_opts.frame_shift_ms = 20.0;
        let f = OnlineFbank::new(o);
        assert!((f.frame_shift_in_seconds() - 0.02).abs() < 1e-9);
    }

    #[test]
    fn pop_then_get_remaining_frames() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &signal(32000));
        f.input_finished();
        let n = f.num_frames_ready();
        f.pop(10);
        assert_eq!(f.num_frames_ready(), n);
        // old frames still addressable
        let _ = f.get_frame(10);
        let frame = f.get_frame(n - 1);
        assert_eq!(frame.len(), 80);
    }

    #[test]
    #[should_panic(expected = "sampling rate mismatch")]
    fn wrong_sampling_rate_panics() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(8000.0, &[0.0; 100]);
    }

    #[test]
    #[should_panic(expected = "input_finished")]
    fn accept_after_input_finished_panics() {
        let mut f = OnlineFbank::new(opts16k());
        f.accept_waveform(16000.0, &[0.0; 1000]);
        f.input_finished();
        f.accept_waveform(16000.0, &[0.0; 100]);
    }

    #[test]
    fn compute_fbank_matches_manual_online() {
        let opts = opts16k();
        let sig = signal(100_000);
        let frames = compute_fbank(&opts, &sig);
        // reference: full signal in one call + input_finished
        let mut f = OnlineFbank::new(opts.clone());
        f.accept_waveform(16000.0, &sig);
        f.input_finished();
        assert_eq!(frames.len() as i32, f.num_frames_ready());
        for (i, fr) in frames.iter().enumerate() {
            assert_eq!(fr, f.get_frame(i as i32));
        }
    }

    #[test]
    fn long_signal_no_panic_and_correct_frame_count() {
        // 600s at 16 kHz = 9.6M samples -> 59998 frames
        let opts = opts16k();
        let mut f = OnlineFbank::new(opts);
        for c in signal(9_600_000).chunks(16000) {
            f.accept_waveform(16000.0, c);
        }
        f.input_finished();
        assert_eq!(f.num_frames_ready(), 59_998);
    }
}
