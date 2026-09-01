# AGENTS.md

Guidance for AI coding agents working in this repo — Claude Code, Cursor, Codex,
GitHub Copilot, or any other tool. This file is the single canonical agent guide.

## Project Overview

kaldi-fbank is a pure-Rust reimplementation of the FBank (log-mel filterbank)
feature-extraction pipeline of
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) — itself
extracted from Kaldi. Two pillars hold the project up:

- **Output fidelity**: frame-by-frame max absolute error vs the C++ reference is
  about `1e-5 ~ 3e-4` on real speech (FFT-rounding only); 604/604 consistency-
  matrix cases pass at `1e-3` (`scripts/run_matrix.sh`), and the in-repo golden
  suite (`cargo test`) passes at `2e-3`.
- **Speed**: ~3~4.5× faster than the C++ reference at identical options
  (up to 4.45× on the 33-case local benchmark; long-audio and production
  configs cluster at 3.0~4.4×).

Everything under `src/` is a deliberate port. When fidelity and "more idiomatic
Rust" conflict, **fidelity wins**.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/options.rs` | `FrameExtractionOptions` / `MelBanksOptions` / `FbankOptions` and their defaults |
| `src/feature_window.rs` | framing, DC removal, pre-emphasis, window functions, Kaldi-compatible dither RNG |
| `src/fft.rs` | realfft/rustfft wrapper producing the kiss_fftr packed layout |
| `src/mel.rs` | mel filterbanks (Kaldi style + librosa style) |
| `src/feature_fbank.rs` | `FbankComputer`: one frame → FFT → power spectrum → mel → log |
| `src/online.rs` | `OnlineFbank` streaming API, `RecyclingVector`, batch `compute_fbank` |
| `examples/fbank-cli.rs` | CLI mirroring the C++ reference CLI (runs on dev-dependencies only) |
| `ref/fbank_cli.cc` | C++ reference CLI — the oracle for golden generation & benchmarks (local-only, see Golden Data Discipline) |
| `tests/golden_test.rs` + `tests/golden/` | golden-value tests against C++ reference outputs |
| `tests/property_test.rs` | property/fuzz tests: chunk-independence, determinism, energy conservation |
| `tests/edge_cases.rs` | boundary inputs |
| `benches/fbank_bench.rs` | criterion benchmarks |
| `scripts/` | golden generation, consistency matrix, C++-vs-Rust benchmark scripts |
| `docs/` | benchmark reports |

`data/` (large audio corpora), `ref/` (C++ oracle), and `results/` (local runs)
are **local-only and gitignored** — never commit them.

## Build Commands

```bash
cargo build                              # library
cargo test                               # all suites (unit + golden + property + edge + doc)
cargo test --lib                         # unit tests only
cargo test --test golden_test            # golden suite
cargo bench                              # criterion benchmarks
cargo run --release --example fbank-cli -- in.f32 out.fbank --num-bins 80 --bench
cargo fmt
cargo clippy --all-targets -- -D warnings
```

> Rust builds are slow. Avoid `cargo clean` — it invalidates incremental caches
> and forces a full rebuild. Only run it when artifacts are actually broken.

## Non-negotiable Constraints

- **English only** — source code, comments, errors, and commit messages. This is
  an international open-source project. User-facing docs may carry non-English
  translations (e.g. `README.zh-CN.md`), but the English version stays
  authoritative.
- **Zero `unsafe`** — the crate contains none; keep it that way.
- **Runtime dependencies stay at `realfft` + `rustfft`** — the dependency
  surface is a selling point. `clap` (CLI) and `criterion` (benches) live in
  `[dev-dependencies]` only; anything a library user would link must be
  justified before being added.
- **`cargo fmt` and `cargo clippy --all-targets -- -D warnings` green before
  every commit.**
- **Panics are part of the API contract.** CHECK-style panics (sampling-rate
  mismatch, feeding after `input_finished()`, invalid window type) follow the
  Kaldi/knf tradition. Any new panic on a public function gets a `# Panics`
  section in its doc comment.

## Porting Fidelity Rules

Numeric behavior must match kaldi-native-fbank except for FFT rounding. Do not
"improve" these paths casually:

- f32 accumulation in `inner_product` and energy computation (the reference does
  **not** accumulate in f64);
- window coefficients computed in f64, then cast to f32 (`get_window`);
- truncation semantics everywhere (`static_cast<int32_t>` → `as i32`);
- the kiss_fftr packed spectrum layout (`fft.rs`);
- the glibc `rand_r` + Kaldi `RandGauss` replica used for dithering;
- **per-frame fresh RNG seeding** — load-bearing: it makes streaming output
  identical to batch output regardless of chunk boundaries, and is pinned by
  `fuzz_chunk_order_independent`. Changing the seeding policy requires updating
  the property tests and docs in the same change.

Deliberate, documented divergences:

- `dither_seed`: the C++ reference seeds from `rand()` (irreproducible); we take
  a fixed seed (default `27437`) so runs are reproducible.
- Defaults are `dither = 0.00003`, povey window, 23 bins — matching
  kaldi-native-fbank's C++ `FrameExtractionOptions` default (`0.00003f`,
  which is Kaldi's `1.0` rescaled). Set `dither` explicitly when
  cross-implementation parity matters.

## Golden Data Discipline

- `tests/golden/*.fbank` are generated by the C++ reference CLI via
  `scripts/gen_golden.sh` — **never hand-edit golden files**.
- Extending coverage = add the case to the matrix, regenerate with the C++
  oracle, commit config and outputs together.
- Any change that shifts numerics re-runs the **full** golden matrix, not just
  the touched case.

## Benchmark Methodology

- Compare against the C++ reference with `scripts/bench_compare.sh` (same input,
  same options, best-of-N, matched builds).
- Measure inside one warmed-up process; never time cold starts or short-lived
  processes.
- Reports live in `docs/`. Rerun before quoting numbers after any numeric
  change; do not cite stale reports.

## Git Commit Convention

Conventional Commits, English descriptions:

```
<type>(<scope>): <description>
```

- **type**: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`
- **scope**: module name (`fft`, `mel`, `online`, `options`, `feature-window`, `bench`, `docs`, …)
- **breaking**: append `!` to the subject.

Branches: `feat/<desc>` · `fix/<desc>` · `refactor/<desc>`.

## Release

- Bump `version` in `Cargo.toml`, commit, then push a `v<version>` tag — CI
  runs the full matrix and publishes to crates.io automatically
  (`.github/workflows/ci.yml`).
- Never `cargo publish` locally: the local registry mirror is a CN replacement
  source, and a manual publish bypasses the CI checks.
- The tag must match the `Cargo.toml` version; CI refuses the tag otherwise.
