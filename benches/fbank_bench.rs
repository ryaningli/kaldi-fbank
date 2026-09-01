//! Criterion benchmark for the kaldi-fbank library: measures FBank extraction
//! throughput (frames/sec and real-time factor) for short / medium / long
//! signals. Run with `cargo bench --bench fbank_bench`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kaldi_fbank::{FbankOptions, compute_fbank};

/// 16 kHz mono deterministic pseudo-speech (sum of sines), 1 second.
fn make_signal(seconds: usize) -> Vec<f32> {
    let n = 16000 * seconds;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16000.0;
            (0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin())
                * 0.5
        })
        .collect()
}

fn fbank_opts() -> FbankOptions {
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.mel_opts.num_bins = 80;
    opts
}

fn bench_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("fbank_extract");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &seconds in &[1usize, 30, 300] {
        let signal = make_signal(seconds);
        group.bench_with_input(
            BenchmarkId::new("whole_signal", format!("{seconds}s")),
            &signal,
            |b, sig| {
                b.iter(|| {
                    let f = compute_fbank(&fbank_opts(), black_box(sig));
                    black_box(f.len())
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_extract);
criterion_main!(benches);
