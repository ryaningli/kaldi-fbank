# kaldi-fbank

> Pure-Rust, Kaldi-compatible online FBank feature extraction. Streaming-first API, faithful math, zero C/C++ dependencies.

[![CI](https://github.com/ryaningli/kaldi-fbank/actions/workflows/ci.yml/badge.svg)](https://github.com/ryaningli/kaldi-fbank/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/kaldi-fbank.svg)](https://crates.io/crates/kaldi-fbank)
[![docs.rs](https://img.shields.io/docsrs/kaldi-fbank.svg)](https://docs.rs/kaldi-fbank)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust: 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/tools/install)
[![Pure Rust](https://img.shields.io/badge/pure%20Rust-zero%20C%2FC%2B%2B%20deps-9cf.svg)](#-features)

**English** | [简体中文](README.zh-CN.md)

## ✨ Features

- **Pure Rust** — the only runtime dependencies are [`realfft`](https://crates.io/crates/realfft) / [`rustfft`](https://crates.io/crates/rustfft), both pure Rust. No C/C++ code, no `build.rs` linking, no cmake — it compiles anywhere the Rust toolchain does, and cross-compiles to a fully static musl binary in one command.
- **Kaldi-faithful math** — a from-scratch Rust port of the FBank pipeline of [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) (itself a port of [Kaldi](https://github.com/kaldi-asr/kaldi)'s feature extraction): same options, same framing / dither / pre-emphasis / windowing / mel-bank math, same streaming API shape. On real speech, the per-frame max absolute error vs the C++ reference is `1e-5 ~ 3e-4` — FFT-implementation rounding only.
- **Streaming-first** — `accept_waveform` takes chunks of any size (down to a single sample), frames are ready as soon as they are complete, and `get_frame(i)` returns a borrowed `&[f32]` with no per-frame allocation. Streaming unboundedly? `pop(n)` releases the oldest frames and keeps memory flat.
- **Fast** — ~3–4.5× the C++ reference at identical inputs and options (33-case benchmark suite; long audio and production configs cluster at 3.0×–4.4×). Main source of the gap: rustfft plans once and reuses the plan, while kaldi-native-fbank's `Rfft` re-allocates a kiss_fft plan every frame.
- **Every knob carried over** — 7 window types (a real enum with `FromStr` / `Display`), snip-edges vs reflection padding, energy dimensions (`use_energy` / `raw_energy` / `htk_compat`), log vs linear filterbanks, power vs magnitude spectra, Kaldi/HTK vs librosa mel style (incl. slaney norm), VTLN frequency bounds, non-power-of-two window lengths, and a fixed-seed dither for reproducible output.
- **Centered framing (new in 0.2.0)** — `frame_opts.center = true` follows the `torch.stft` / torchaudio / NeMo `center=True` convention: windows centered on the frame-shift grid and torchaudio-compatible frame counts. An extension beyond the Kaldi C++ reference; pairs with the librosa/slaney mel style for drop-in NeMo `FilterbankFeatures` frontends.
- **Center pad modes (new in 0.3.0)** — `frame_opts.center_pad_mode` picks the boundary padding of centered framing: `Constant` (zero-pad, torchaudio / NeMo) or `Reflect` (edge-exclusive mirroring, `numpy.pad(..., mode="reflect")` / librosa / Whisper).
- **Model presets (new in 0.3.0)** — `FbankOptions::whisper(n_mels)` builds the complete Whisper / Qwen3-ASR feature recipe in one call: periodic Hann, 400-point window / 160-point shift, 400-point FFT, reflect-padded centered framing, librosa/slaney mel, power spectrum, linear output (the Whisper log normalization stays with the caller).
- **Tested hard** — **140** `cargo test` tests (unit, edge-case, golden, property/fuzz, centered-framing & reflect-padding compatibility) plus a **604-case** end-to-end consistency matrix against the locally built C++ reference: `PASS=604 FAIL=0`.

## 📦 Installation

```bash
cargo add kaldi-fbank
```

or in `Cargo.toml`:

```toml
[dependencies]
kaldi-fbank = "0.1"
```

No system prerequisites — no C compiler, no cmake, no vendored C sources. `cargo build` is all there is.

## 🚀 Quick start

### Streaming (chunked input)

```rust
use kaldi_fbank::{FbankOptions, OnlineFbank};

let mut opts = FbankOptions::default();
opts.frame_opts.dither = 0.0;    // deterministic output (default 0.00003, matching
                                 // kaldi-native-fbank; Kaldi itself defaults to 1.0)
opts.mel_opts.num_bins = 80;     // common sherpa / whisper configuration

let mut fbank = OnlineFbank::new(opts);
fbank.accept_waveform(16000.0, &chunk_1);  // feed audio as it arrives —
fbank.accept_waveform(16000.0, &chunk_2);  // chunks of any size, even 1 sample
fbank.input_finished();                    // flush the tail

for i in 0..fbank.num_frames_ready() {
    let frame: &[f32] = fbank.get_frame(i);  // dim-dimensional feature, borrowed
    // ...
}
```

### Batch (whole signal at once)

```rust
let frames: Vec<Vec<f32>> = kaldi_fbank::compute_fbank(&opts, &samples);
```

### Picking a window

```rust
use kaldi_fbank::WindowType;

opts.frame_opts.window_type = WindowType::Hann;            // Povey is the default
opts.frame_opts.window_type = "hanning".parse().unwrap();  // FromStr, same names as Kaldi
```

## 📐 API tour

| Item | Purpose |
| --- | --- |
| `FbankOptions` | Nested options: `frame_opts` (`FrameExtractionOptions`) + `mel_opts` (`MelBanksOptions`) + energy / log / power switches. `Default` matches kaldi-native-fbank. |
| `OnlineFbank` | Streaming extractor — `accept_waveform` / `input_finished` / `num_frames_ready` / `get_frame` / `pop`. |
| `compute_fbank` | Batch helper — whole signal in, `Vec<Vec<f32>>` out. |
| `FbankComputer` | The per-frame kernel (FFT → power spectrum → mel → log), usable without the streaming wrapper. |

## 🔬 Fidelity & testing

Correctness is pinned against the C++ reference at two levels.

**In-repo tests** — 130 tests, `cargo test` all green (clippy: 0 warnings):

| Category | Count | What it covers |
| --- | --- | --- |
| Unit tests (in each module) | 96 | window function values, framing/frame-count formulas, reflection padding, preprocessing order, mel-scale round-trips, triangular filterbank properties, HTK mode, librosa normalization, FFT vs naive DFT, Parseval, RecyclingVector, API contracts (sample-rate mismatch / post-finish panics) |
| Edge cases (`tests/edge_cases.rs`) | 20 | zero-length / single-sample / exactly-one-frame inputs, all-zero / constant / pure-tone / extreme-value signals, frame-count cross-checks, 8k / 44.1k, non-power-of-two windows, snip-edges reflection, energy-dimension sizes, determinism |
| Golden consistency (`tests/golden_test.rs`) | 7 | 33 configs × 3 sample-rate signals, frame-by-frame vs kaldi-native-fbank (C++), tolerance 2e-3, batch and chunked-streaming paths both covered |
| Property/fuzz (`tests/property_test.rs`) | 6 | 300 random signals × random configs: no panic, finite output, determinism; 80 random chunkings equal batch; frame-count monotonicity; non-negative power spectrum |
| Doc example | 1 | `lib.rs` doc test |

**End-to-end consistency matrix** — 31 configs × 5 chunk sizes × 4 durations + 8 kHz/44.1 kHz = **604 cases**, frame-by-frame against the C++ reference within tolerance `1e-3`: `PASS=604 FAIL=0`.

Build the reference and run the matrix:

```bash
cd /path/to/kaldi-native-fbank    # https://github.com/csukuangfj/kaldi-native-fbank
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
      -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF -DKALDI_NATIVE_FBANK_ENABLE_CHECK=OFF ..
make -j

cd /path/to/kaldi-fbank
cargo build --release --example fbank-cli    # the Rust side of the comparison
scripts/run_matrix.sh --quick                # quick matrix (no 300s audio)
scripts/run_matrix.sh                        # full matrix
```

Matrix coverage: bins (23/40/80), snip-edges, all 7 window types, preemph / DC / energy dimensions, log / power spectrum / non-power-of-two window lengths, VTLN frequency bounds, librosa style (incl. slaney norm), varying frame shift/length, 8 kHz & 44.1 kHz sample rates, and extreme chunking (one sample per call).

> With `dither != 0` the C++ reference seeds from the system `rand()` (non-reproducible), so alignment is only meaningful with `dither = 0` — the production standard configuration. kaldi-fbank exposes a fixed seed, so its dithered output is reproducible.

## ⚡ Performance

Identical inputs and options; the C++ side built `Release` / `-O3`. Full tables in [docs/benchmark.md](docs/benchmark.md).

### x86-64

| Implementation | 300s audio compute | RTF | frames/s |
| --- | --- | --- | --- |
| kaldi-native-fbank (C++) | ~0.15 s | ~0.00048 | ~200k |
| **kaldi-fbank (Rust)** | **~0.045 s** | **~0.00014** | **~650k** |

Speedup across the 33-case suite: **1.82× ~ 4.45×** (3.0× ~ 4.4× for long audio and production configs).

> Why the gap: kaldi-native-fbank's `Rfft` re-allocates a kiss_fft plan **every frame**, while rustfft plans once and reuses it; the Rust side also avoids per-frame heap allocation.

### Edge device — RK3576 (4×A53 + 4×A72, Debian 12)

C++ reference compiled natively on the device with `g++ -O3` — the optimal arm baseline.

| Metric | C++ (arm native) | **Rust (aarch64 cross)** |
| --- | --- | --- |
| 600s audio compute | ~1.65 s | **~0.46 s** |
| RTF | ~0.00275 | **~0.00077** |
| frames/s | ~36,000 | **~130,000** |
| Speedup | — | **2.4× ~ 4.5×** |

On-device consistency: 30 configs × 3 audio files = **90/90 PASS**, `max_abs_diff ≤ 3.4e-4`. Full report in [docs/benchmark-arm.md](docs/benchmark-arm.md); on-device scripts `scripts/bench_arm.sh` / `scripts/matrix_arm.sh` (with `scripts/compare_fbank_plain.py` — no numpy needed).

Cross-compile a fully static binary (pure-Rust deps + rust-lld, zero dynamic libraries):

```bash
rustup target add aarch64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld" \
  cargo build --release --target aarch64-unknown-linux-musl --example fbank-cli
# → target/aarch64-unknown-linux-musl/release/examples/fbank-cli (static, 1.6 MB)
```

## 🛠️ Development

```bash
cargo fmt --all                             # format
cargo clippy --all-targets -- -D warnings   # lint (warnings as errors)
cargo test                                  # all 130 tests
cargo bench --bench fbank_bench             # criterion benchmarks
```

Quality gates before every commit: fmt green, clippy green, tests green. CI (Linux) runs the same three; pushing a `v*` tag publishes to crates.io after verifying it matches the `Cargo.toml` version.

The consistency matrix additionally needs the C++ reference built locally — see [🔬 Fidelity & testing](#-fidelity--testing).

## 🧱 Architecture

The pipeline is Kaldi's, module by module:

```
PCM samples
   │  accept_waveform() — chunked, streaming
   ▼
feature_window.rs     framing · dither · DC removal · pre-emphasis · windowing
   ▼
fft.rs                real FFT via realfft/rustfft, kiss_fftr-compatible packed layout
   ▼
feature_fbank.rs      power spectrum → mel filterbanks → log    (FbankComputer)
   ▼
online.rs             OnlineFbank: frame cache (RecyclingVector), get_frame(i) → &[f32]
```

```
src/
  options.rs           # FrameExtractionOptions / MelBanksOptions / FbankOptions / WindowType
  feature_window.rs    # framing, dither (Kaldi-compatible RNG), DC removal, pre-emphasis, windows
  mel.rs               # mel filterbanks (Kaldi/HTK style + librosa style)
  fft.rs               # realfft/rustfft wrapper
  feature_fbank.rs     # FbankComputer: per-frame FFT → power spectrum → mel → log
  online.rs            # OnlineFbank streaming + RecyclingVector
examples/fbank-cli.rs  # CLI with the same flags as ref/fbank_cli.cc (comparison/benchmark tool)
ref/fbank_cli.cc       # C++ reference CLI built on kaldi-native-fbank (comparison only)
scripts/               # consistency matrix, benchmark comparison, golden generation
tests/                 # golden / edge-case / property tests
benches/fbank_bench.rs # criterion benchmarks
```

## 📚 Documentation

- [docs.rs/kaldi-fbank](https://docs.rs/kaldi-fbank) — API reference
- [docs/benchmark.md](docs/benchmark.md) — x86-64 benchmark tables
- [docs/benchmark-arm.md](docs/benchmark-arm.md) — RK3576 on-device benchmarks

## 🤝 Contributing

Contributions are welcome. For anything touching the numeric path, keep `cargo test` green and, where possible, extend the consistency matrix — output parity with the C++ reference is this project's core contract.

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).

The FBank algorithm and API are ported from [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) (Apache-2.0), which itself ports Kaldi's [feature-extraction code](https://github.com/kaldi-asr/kaldi) (Apache-2.0). This project is an independent Rust implementation of that algorithm.
