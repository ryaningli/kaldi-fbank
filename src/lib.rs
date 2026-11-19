//! # kaldi-fbank — Pure-Rust Kaldi-compatible online FBank feature extractor
//!
//! A clean-room, pure-Rust reimplementation of
//! [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)'s
//! FBank pipeline, designed to be output-consistent (within FFT rounding) and
//! at least as fast as the C++ reference.
//!
//! The heavy numeric work (real FFT) is delegated to the mainstream
//! [`realfft`](https://crates.io/crates/realfft) /
//! [`rustfft`](https://crates.io/crates/rustfft) crates; everything else is a
//! faithful port of the Kaldi feature-extraction algorithm.
//!
//! # Example
//!
//! ```
//! use kaldi_fbank::{FbankOptions, OnlineFbank};
//!
//! let mut opts = FbankOptions::default();
//! opts.frame_opts.dither = 0.0;
//! opts.mel_opts.num_bins = 80;
//!
//! let mut fbank = OnlineFbank::new(opts);
//! // feed a 1-second chunk of 16 kHz audio
//! let chunk: Vec<f32> = vec![0.0; 16000];
//! fbank.accept_waveform(16000.0, &chunk);
//! fbank.input_finished();
//!
//! let n = fbank.num_frames_ready();
//! let dim = fbank.dim();
//! let frame0: &[f32] = fbank.get_frame(0);
//! assert_eq!(frame0.len(), dim as usize);
//! assert!(n > 0);
//! ```

pub mod feature_fbank;
pub mod feature_window;
pub mod fft;
pub mod mel;
pub mod online;
pub mod options;

pub use feature_fbank::FbankComputer;
pub use feature_window::{
    FeatureWindowFunction, KaldiRand, dither, extract_window, first_sample_of_frame, get_window,
    inner_product, num_frames, process_window,
};
pub use mel::{MelBanks, inverse_mel_scale, inverse_mel_scale_slaney, mel_scale, mel_scale_slaney};
pub use online::{OnlineFbank, RecyclingVector, compute_fbank};
pub use options::{
    CenterPadMode, FbankOptions, FrameExtractionOptions, MelBanksOptions, WindowType,
    round_up_to_nearest_power_of_two,
};
