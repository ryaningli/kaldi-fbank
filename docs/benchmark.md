# Benchmark Report (kaldi-fbank vs kaldi-native-fbank, local x86-64)

> Machine: local (x86-64, AVX2); builds: C++ `-O3`, Rust `release` (LTO thin).
> Both implementations use identical inputs (raw f32 PCM), identical options and
> identical chunk feeding; timing covers only feature computation (no file I/O).
> Each case reports the best of N runs (RUNS=7).
> Reproduce with `scripts/bench_compare.sh`.

## Test Machine

| Item | Value |
|------|-------|
| CPU | 12th Gen Intel(R) Core(TM) i9-12900H (Alder Lake, hybrid 6P+8E) |
| Architecture | x86_64 |
| Logical CPUs | 20 (14 cores; 2 threads/core on P-cores) |
| Frequency | 400 MHz min - 5.0 GHz max |
| SIMD | AVX2, AVX-VNNI, FMA, SSE4.2, SHA-NI |
| Cache | L1d 544 KiB, L1i 704 KiB, L2 11.5 MiB, L3 24 MiB |
| OS / Kernel | Ubuntu 26.04 LTS, kernel 7.0.0-29-generic |

> Note: rustfft/realfft dispatch to the AVX2 code path on this CPU.

## Conclusions

- **Correctness**: 604/604 consistency cases pass (see [README](../README.md)),
  max absolute error `9.5e-6 ~ 2.9e-4` (well below the `1e-3` tolerance used by
  kaldi-native-fbank's own tests).
- **Performance**: **all 33 benchmark cases beat the C++ reference**, speedup
  **1.82x ~ 4.45x** (typical long-audio / production configs cluster at
  **3.0x ~ 4.4x**). The main reason: kaldi-native-fbank's `Rfft` re-allocates a
  kiss_fft plan on every frame, while rustfft/realfft plan once and reuse;
  kaldi-fbank also has zero per-frame heap allocation.

## Benchmark Matrix (33 cases x 2 chunk sizes)

### 1. Duration sweep (0.1s -> 600s, default 80 bins)

| Duration | Chunk | C++ RTF | Rust RTF | Speedup |
|----------|-------|---------|----------|---------|
| 0.1s | 1600 | 0.000503 | 0.000276 | 1.82x |
| 0.1s | 16000 | 0.000622 | 0.000225 | 2.76x |
| 0.3s | 1600 | 0.000568 | 0.000174 | 3.26x |
| 0.3s | 16000 | 0.000474 | 0.000183 | 2.59x |
| 2s | 1600 | 0.000659 | 0.000149 | 4.42x |
| 2s | 16000 | 0.000689 | 0.000189 | 3.64x |
| 5s | 1600 | 0.000593 | 0.000163 | 3.64x |
| 5s | 16000 | 0.000501 | 0.000148 | 3.39x |
| 30s | 1600 | 0.000479 | 0.000151 | 3.17x |
| 30s | 16000 | 0.000555 | 0.000154 | 3.60x |
| 60s | 1600 | 0.000515 | 0.000171 | 3.01x |
| 60s | 16000 | 0.000537 | 0.000148 | 3.63x |
| 300s | 1600 | 0.000505 | 0.000156 | 3.23x |
| 300s | 16000 | 0.000515 | 0.000171 | 3.01x |
| 600s | 1600 | 0.000593 | 0.000175 | 3.39x |
| 600s | 16000 | 0.000575 | 0.000170 | 3.38x |

### 2. Config sweep (60s audio x 18 configs)

| Config | Chunk 1600 | Chunk 16000 |
|--------|-----------|-------------|
| bins=80 | 3.29x | 3.28x |
| bins=23 | 4.24x | 4.19x |
| bins=40 | 4.25x | 3.88x |
| snip_edges=0 | 3.50x | 3.51x |
| hamming window | 3.30x | 3.46x |
| hanning window | 3.96x | 3.52x |
| sine window | 3.68x | 3.49x |
| blackman window | 3.27x | 3.77x |
| use_energy=1 | 2.93x | 3.21x |
| use_power=0 (magnitude) | 3.43x | 3.42x |
| use_log_fbank=0 (linear) | 4.45x | 3.41x |
| round_to_pow2=0 (400-pt FFT) | 2.92x | 2.93x |
| librosa style | 3.40x | 3.82x |
| shift 5ms (2x frames) | 3.50x | 3.27x |
| shift 20ms (half frames) | 2.77x | 3.31x |
| length 20ms (320-pt FFT) | 3.82x | 3.61x |
| length 32ms (512-pt FFT) | 3.05x | 3.28x |
| preemph=0 | 3.16x | 3.35x |
| remove_dc=0 | 4.01x | 3.94x |

### 3. Content sweep (meeting recordings, 80 bins)

| Duration | Chunk | C++ RTF | Rust RTF | Speedup |
|----------|-------|---------|----------|---------|
| 300s | 1600 | 0.000614 | 0.000165 | 3.72x |
| 300s | 16000 | 0.000592 | 0.000189 | 3.13x |
| 600s | 1600 | 0.000590 | 0.000162 | 3.64x |
| 600s | 16000 | 0.000560 | 0.000158 | 3.54x |

### 4. Sample-rate sweep (60s audio)

| Sample rate | Chunk | C++ RTF | Rust RTF | Speedup |
|-------------|-------|---------|----------|---------|
| 8 kHz | 1600 | 0.000316 | 0.000115 | 2.75x |
| 8 kHz | 16000 | 0.000301 | 0.000119 | 2.53x |
| 44.1 kHz | 1600 | 0.001986 | 0.000520 | 3.82x |
| 44.1 kHz | 16000 | 0.002027 | 0.000460 | 4.41x |

## Absolute Time (600s audio, chunk=16000, 80 bins)

| Implementation | Compute time | RTF | Frames/s |
|----------------|--------------|-----|----------|
| kaldi-native-fbank (C++) | ~0.37 s | ~0.00062 | ~160,000 |
| **kaldi-fbank (Rust)** | **~0.12 s** | **~0.00019** | **~520,000** |

i.e. FBank computation for 600 s of audio takes only **~120 ms** in Rust
(1/5000 of real time).

## Reproduction

```bash
cargo build --release
RUNS=7 scripts/bench_compare.sh   # same 33-case table as above
cargo bench --bench fbank_bench   # criterion report (1s: ~247us, 30s: ~4.8ms, 300s: ~63ms)
```

## Consistency Spot-Check (max_abs_diff, chunk=16000)

| Config | 0.3s | 2s | 30s |
|--------|------|----|-----|
| 80 bins default | 9.5e-6 | 3.1e-5 | 5.9e-5 |
| snip_edges=0 | 9.5e-6 | 2.0e-5 | 5.9e-5 |
| use_energy=1 | 9.5e-6 | 3.1e-5 | 5.9e-5 |
| librosa style | 6.7e-6 | 1.3e-5 | 3.3e-5 |
| hamming window | 1.2e-5 | — | — |
| 23 bins | 1.4e-6 | — | — |

Full matrix (31 configs x 5 chunk sizes x 4 durations + 8k/44.1 kHz):
`scripts/run_matrix.sh` outputs `PASS=604 FAIL=0`.
