# kaldi-fbank

A **pure-Rust, Kaldi-compatible online FBank feature extractor** — a drop-in
replacement for [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank).

kaldi-fbank is a faithful, from-scratch Rust port of the FBank pipeline of
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) (and
hence of [Kaldi](https://github.com/kaldi-asr/kaldi)'s feature extraction): the
same options, the same frame/window/mel-bank math, the same online streaming API.

- **Pure Rust**: no C/C++ dependencies, no `build.rs` linking of external
  libraries; compiles on any platform.
- **No reinvented wheels**: the real FFT uses the mainstream
  [`realfft`](https://crates.io/crates/realfft) /
  [`rustfft`](https://crates.io/crates/rustfft) crates (precomputed plans, SIMD
  acceleration); the CLI uses [`clap`](https://crates.io/crates/clap). The
  rest is a faithful port of the Kaldi feature-extraction algorithm.
- **Output-consistent**: on real speech, frame-by-frame max absolute error vs
  kaldi-native-fbank (C++) is about `1e-5 ~ 3e-4` (only FFT-implementation
  rounding differences; 604/604 cases pass within the `1e-3` tolerance).
- **Faster**: with identical inputs and options, computation is **~3~4.5x
  faster** than the C++ reference (up to 4.45x across 33 benchmark cases; long
  audio and production configs cluster at 3.0x~4.4x — see
  [docs/benchmark.md](docs/benchmark.md)).

## Usage

```rust
use kaldi_fbank::{FbankOptions, OnlineFbank};

let mut opts = FbankOptions::default();
opts.frame_opts.dither = 0.0;     // deterministic output (default 0.00003, matching kaldi-native-fbank; Kaldi itself defaults to 1.0)
opts.mel_opts.num_bins = 80;      // common sherpa / whisper configuration

let mut fbank = OnlineFbank::new(opts);
fbank.accept_waveform(16000.0, &samples_chunk); // streaming, chunked input
// ... more chunks ...
fbank.input_finished();

let n = fbank.num_frames_ready();
for i in 0..n {
    let frame: &[f32] = fbank.get_frame(i); // dim-dimensional feature
    // ...
}
```

Batch convenience API:

```rust
let frames: Vec<Vec<f32>> = kaldi_fbank::compute_fbank(&opts, &samples);
```

## Project layout

```
src/
  options.rs          # FrameExtractionOptions / MelBanksOptions / FbankOptions
  feature_window.rs   # framing, DC removal, pre-emphasis, windowing, dither (Kaldi-compatible RNG)
  mel.rs              # mel filterbanks (Kaldi style + librosa style)
  fft.rs              # realfft/rustfft wrapper, kiss_fftr-compatible packed layout
  feature_fbank.rs    # FbankComputer: per-frame FFT -> power spectrum -> mel -> log
  online.rs           # OnlineFbank online streaming + RecyclingVector
examples/fbank-cli.rs # CLI with the same flags as ref/fbank_cli.cc (benchmark/comparison tool)
ref/fbank_cli.cc      # C++ reference CLI built on kaldi-native-fbank (for comparison/benchmarks only)
scripts/              # consistency matrix, benchmark comparison, golden generation
tests/golden_test.rs  # golden-value + unit tests (cargo test)
benches/fbank_bench.rs# criterion benchmarks (cargo bench)
```

## Consistency with kaldi-native-fbank

The reference implementation is built as follows:

```bash
cd /path/to/kaldi-native-fbank   # https://github.com/csukuangfj/kaldi-native-fbank
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
      -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF -DKALDI_NATIVE_FBANK_ENABLE_CHECK=OFF ..
make -j
```

Run the consistency matrix (short/medium/long audio x 31 configs x 5 chunk
sizes, 600+ cases):

```bash
cargo build --release --example fbank-cli   # build the CLI (a benchmark tool)
scripts/run_matrix.sh          # full matrix (includes 300s audio)
scripts/run_matrix.sh --quick  # quick matrix (without 300s audio)
```

Coverage: bins (23/40/80), snip-edges, all 7 window types, preemph/DC/energy
dimension, log / power spectrum / non-power-of-two window lengths, VTLN
frequency bounds, librosa style (incl. slaney normalization), varying frame
shift/length, 8 kHz & 44.1 kHz sample rates, and extreme chunking (one sample
per frame).

## Performance

```bash
scripts/bench_compare.sh   # prints C++ vs Rust RTF and speedups
cargo bench --bench fbank_bench
```

Summary (local machine, 300s 16 kHz audio, 80 bins, 10 ms shift; full tables
in [docs/benchmark.md](docs/benchmark.md)):

| Implementation | 300s compute time | RTF | FPS |
|----------------|-------------------|-----|-----|
| kaldi-native-fbank (C++) | ~0.15 s | ~0.00048 | ~200k |
| **kaldi-fbank (Rust)** | **~0.045 s** | **~0.00014** | **~650k** |

> Note: kaldi-native-fbank's `Rfft` re-allocates a kiss_fft plan on every
> frame, while rustfft plans once and reuses it — the main source of the gap;
> the Rust side also avoids per-frame heap allocations.

## Tests

**130 tests**, all passing with `cargo test --release` (clippy: 0 warnings):

```bash
cargo test --release   # all 130 tests
cargo clippy --all-targets -- -D warnings
```

Coverage:

| Category | Count | What it covers |
|----------|-------|----------------|
| Unit tests (in each module) | 96 | window function values, framing/frame-count formulas, reflection padding, preprocessing order, mel-scale round-trips, triangular filterbank properties, HTK-mode, librosa normalization, FFT vs naive DFT, Parseval, RecyclingVector, API contracts (sample-rate mismatch / post-finish panics), etc. |
| Edge cases (`tests/edge_cases.rs`) | 20 | zero-length / single-sample / exactly-one-frame inputs, all-zero / constant / pure-tone / extreme-value signals, frame-count cross-checks, 8k/44.1k, non-power-of-two windows, snip_edges reflection, energy-dimension sizes, determinism |
| Golden consistency (`tests/golden_test.rs`) | 7 | **33 configs x 3 sample-rate signals** compared frame-by-frame against kaldi-native-fbank (C++) (tolerance 2e-3; both batch and chunked-streaming paths) |
| Property/fuzz (`tests/property_test.rs`) | 6 | 300 random signals x random configs: no panic, finite output, determinism; 80 random chunkings equal batch; frame-count monotonicity; non-negative power spectrum |
| Doc example | 1 | lib.rs doc test |

Additionally `scripts/run_matrix.sh` provides an end-to-end consistency matrix
against the C++ reference (31 configs x 5 chunk sizes x 4 durations +
8k/44.1 kHz, 604 cases, `PASS=604 FAIL=0`).

Golden reference files are generated by `scripts/gen_golden.sh` (requires the
locally built C++ reference CLI).

Note: with `dither != 0` the C++ reference seeds from the system `rand()`
(non-reproducible), so bit-exact alignment only makes sense with `dither = 0`
(the production standard configuration); kaldi-fbank exposes a fixed seed for
reproducibility.

## Edge device (RK3576 / aarch64): cross-compilation & benchmarks

Deployed and benchmarked on an **RK3576 (rk3576-0203, 4xA53 + 4xA72, Debian 12)**.

Cross-compile (pure-Rust deps, statically linked with rust-lld, zero dynamic
dependencies):

```bash
rustup target add aarch64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld" cargo build --release --target aarch64-unknown-linux-musl     --example fbank-cli
# produces target/aarch64-unknown-linux-musl/release/examples/fbank-cli (static, 1.6 MB)
# on the device, keep the artifact name expected by scripts/bench_arm.sh:
#   cp examples/fbank-cli /tmp/knf-arm/fbank_cli_rs
```

On-device results (C++ reference compiled natively on the device with g++ -O3,
the optimal arm baseline; 33 benchmark cases):

| Metric | C++ (arm native) | **Rust (aarch64 cross)** |
|--------|------------------|--------------------------|
| 600s compute time | ~1.65 s | **~0.46 s** |
| RTF | ~0.00275 | **~0.00077** |
| Frames/s | ~36,000 | **~130,000** |
| Speedup | — | **2.4x ~ 4.5x** (3.2x~4.4x for long audio) |

- On-device consistency matrix: 30 configs x 3 audio files = **90/90 PASS**,
  `max_abs_diff <= 3.4e-4`.
- Local x86-64 33-case benchmark: speedup **1.82x ~ 4.45x** (3.0x~4.4x for long
  audio and production configs).
- Full reports: [docs/benchmark.md](docs/benchmark.md) and
  [docs/benchmark-arm.md](docs/benchmark-arm.md); on-device scripts `scripts/bench_arm.sh`, `scripts/matrix_arm.sh` (with
  `scripts/compare_fbank_plain.py`, no numpy needed).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

The FBank algorithm and the API are ported from
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
(Apache-2.0), which itself ports Kaldi's
[feature-extraction code](https://github.com/kaldi-asr/kaldi) (Apache-2.0).
This project is an independent Rust implementation of that algorithm.
