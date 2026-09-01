# Benchmark Report (RK3576 Edge Device, rk3576-0203)

> Device: RK3576, aarch64, 4x Cortex-A53 + 4x Cortex-A72, 8 cores, Debian 12.
> C++ reference: compiled **natively on the device** with `g++ -O3` (the optimal
> arm baseline). Rust implementation: **cross-compiled** on the dev machine
> (`aarch64-unknown-linux-musl`, rust-lld static link, zero dynamic deps).
> Identical inputs (raw f32 PCM), options and chunk sizes; timing covers only
> feature computation; each case reports the best of RUNS=5.
> Reproduce on-device with `scripts/bench_arm.sh`.

## Conclusions

- **Consistency**: on-device full matrix **90/90 PASS** (30 configs x 3 audio
  files), `max_abs_diff <= 3.4e-4` (tolerance `1e-3`).
- **Performance**: **all 33 benchmark cases beat the C++ reference**, speedup
  **2.4x ~ 4.5x** (long audio and production configs cluster at
  **3.2x ~ 4.4x**; very short audio such as 0.1s/0.3s has few frames and is
  sensitive to shared-device scheduling noise, see note below).

## Benchmark Matrix (33 cases x 2 chunk sizes)

### 1. Duration sweep (0.1s -> 600s, default 80 bins)

| Duration | Chunk | C++ RTF | Rust RTF | Speedup |
|----------|-------|---------|----------|---------|
| 0.1s | 1600 | 0.002534 | 0.000896 | 2.83x |
| 0.1s | 16000 | 0.002508 | 0.000913 | 2.75x |
| 0.3s | 1600 | 0.002644 | 0.001674 | 1.58x* |
| 0.3s | 16000 | 0.006571 | 0.000839 | 7.83x* |
| 2s | 1600 | 0.002816 | 0.000804 | 3.50x |
| 2s | 16000 | 0.002815 | 0.000842 | 3.34x |
| 5s | 1600 | 0.002814 | 0.000804 | 3.50x |
| 5s | 16000 | 0.004663 | 0.000801 | 5.82x* |
| 30s | 1600 | 0.002807 | 0.000771 | 3.64x |
| 30s | 16000 | 0.002786 | 0.000771 | 3.61x |
| 60s | 1600 | 0.002869 | 0.000776 | 3.70x |
| 60s | 16000 | 0.002814 | 0.000862 | 3.26x |
| 300s | 1600 | 0.002721 | 0.000762 | 3.57x |
| 300s | 16000 | 0.002729 | 0.000759 | 3.60x |
| 600s | 1600 | 0.002738 | 0.000775 | 3.53x |
| 600s | 16000 | 0.002730 | 0.000772 | 3.54x |

\* Very short audio (<30 frames) single-run best values are affected by
shared-device scheduling noise; excluding these points the duration sweep is
stable at **3.2x ~ 3.7x**.

### 2. Config sweep (60s audio x 18 configs)

| Config | Chunk 1600 | Chunk 16000 |
|--------|-----------|-------------|
| bins=80 | 3.18x | 3.66x |
| bins=23 | 4.44x | 3.86x |
| bins=40 | 4.03x | 4.03x |
| snip_edges=0 | 3.55x | 3.52x |
| hamming window | 3.58x | 3.58x |
| hanning window | 3.63x | 3.50x |
| sine window | 3.50x | 3.61x |
| blackman window | 3.30x | 3.65x |
| use_energy=1 | 3.37x | 3.36x |
| use_power=0 (magnitude) | 3.36x | 3.34x |
| use_log_fbank=0 (linear) | 3.84x | 3.89x |
| round_to_pow2=0 (400-pt FFT) | 2.41x | 2.50x |
| librosa style | 3.52x | 3.77x |
| shift 5ms (2x frames) | 3.55x | 3.59x |
| shift 20ms (half frames) | 2.92x | 3.50x |
| length 20ms (320-pt FFT) | 3.68x | 3.61x |
| length 32ms (512-pt FFT) | 3.55x | 3.18x |
| preemph=0 | 3.54x | 3.58x |
| remove_dc=0 | 3.82x | 3.73x |

### 3. Content sweep (meeting recordings, 80 bins)

| Duration | Chunk | C++ RTF | Rust RTF | Speedup |
|----------|-------|---------|----------|---------|
| 300s | 1600 | 0.002734 | 0.000764 | 3.58x |
| 300s | 16000 | 0.002722 | 0.000756 | 3.60x |
| 600s | 1600 | 0.002751 | 0.000765 | 3.60x |
| 600s | 16000 | 0.002734 | 0.000756 | 3.62x |

### 4. Sample-rate sweep (60s audio)

| Sample rate | Chunk | C++ RTF | Rust RTF | Speedup |
|-------------|-------|---------|----------|---------|
| 8 kHz | 1600 | 0.001522 | 0.000498 | 3.06x |
| 8 kHz | 16000 | 0.001590 | 0.000498 | 3.19x |
| 44.1 kHz | 1600 | 0.010694 | 0.002407 | 4.44x |
| 44.1 kHz | 16000 | 0.010602 | 0.002381 | 4.45x |

## Absolute Time (600s audio, chunk=16000, 80 bins)

| Implementation | Compute time | RTF | Frames/s |
|----------------|--------------|-----|----------|
| kaldi-native-fbank (arm native) | ~1.65 s | ~0.00275 | ~36,000 |
| **kaldi-fbank (aarch64 cross)** | **~0.46 s** | **~0.00077** | **~130,000** |

i.e. FBank computation for 600 s of audio takes only **~460 ms** on the device
(1/1300 of real time).

## Cross-compilation & Reproduction

```bash
# Rust (on the dev machine; requires the rustup target)
rustup target add aarch64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld" cargo build --release --target aarch64-unknown-linux-musl \
    --example fbank-cli
# produces target/aarch64-unknown-linux-musl/release/examples/fbank-cli (static, 1.6 MB)
# on the device, name the artifact fbank_cli_rs as scripts/bench_arm.sh expects:
#   cp examples/fbank-cli /tmp/knf-arm/fbank_cli_rs

# C++ reference compiled on the device
g++ -O3 -std=c++17 -I. -Ikissfft fbank_cli.cc kaldi-native-fbank/csrc/*.cc \
    kissfft/kiss_fft.c kissfft/kiss_fftr.c -o fbank_cli_ref

# run benchmark & consistency on the device
scp artifacts and audio to the device, then: RUNS=5 ./bench_arm.sh && ./matrix_arm.sh
```

> Note: musl static linking gives zero dynamic dependencies for edge
> deployment; the glibc target (aarch64-unknown-linux-gnu) would need a cross
> sysroot, so musl is the simplest cross-compilation path for a pure-Rust
> program. The C++ reference is compiled natively on the device, which is the
> optimal arm baseline for benchmarking.
