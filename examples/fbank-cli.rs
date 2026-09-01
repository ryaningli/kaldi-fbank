//! fbank-cli — computes FBank features with kaldi-fbank, mirroring the reference
//! `ref/fbank_cli.cc` CLI (same flags, same I/O layout) so both can be driven
//! by identical command lines for consistency and benchmark comparison.
//!
//! Reads raw f32 PCM (little endian, mono), writes fbank features as raw f32
//! `[num_frames x dim]`, and with `--bench` prints timing / RTF to stderr.

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use clap::Parser;

use kaldi_fbank::{FbankOptions, OnlineFbank, WindowType};

/// Accepts `0|1|true|false` so command lines stay compatible with the C++
/// reference CLI.
fn parse_bool(s: &str) -> Result<bool, String> {
    match s {
        "0" | "false" | "no" => Ok(false),
        "1" | "true" | "yes" => Ok(true),
        other => Err(format!(
            "invalid boolean value '{other}' (expected 0/1/true/false)"
        )),
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "fbank-cli",
    about = "Pure-Rust FBank feature extractor (kaldi-fbank)"
)]
struct Args {
    /// Raw f32 PCM input file (little endian, mono).
    input: String,
    /// Output f32 feature file.
    output: String,

    /// Sampling rate.
    #[arg(long, default_value_t = 16000.0)]
    samp_freq: f32,
    /// Frame shift in ms.
    #[arg(long, default_value_t = 10.0)]
    frame_shift_ms: f32,
    /// Frame length in ms.
    #[arg(long, default_value_t = 25.0)]
    frame_length_ms: f32,
    /// Number of mel bins.
    #[arg(long, default_value_t = 80)]
    num_bins: i32,
    /// Lower frequency cutoff.
    #[arg(long, default_value_t = 20.0)]
    low_freq: f32,
    /// Upper frequency cutoff (0 => Nyquist; negative => Nyquist + value).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    high_freq: f32,
    /// Preemphasis coefficient.
    #[arg(long, default_value_t = 0.97)]
    preemph: f32,
    /// Dither amount.
    #[arg(long, default_value_t = 0.0)]
    dither: f32,
    /// Snip edges.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    snip_edges: bool,
    /// Remove DC offset.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    remove_dc: bool,
    /// Round window size to power of two.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    round_to_pow2: bool,
    /// Window type.
    #[arg(long, value_parser = clap::value_parser!(WindowType), default_value_t = WindowType::Povey)]
    window_type: WindowType,
    /// Append energy dimension.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_parser = parse_bool)]
    use_energy: bool,
    /// Use log fbank.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    use_log_fbank: bool,
    /// Use power (else magnitude).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    use_power: bool,
    /// HTK compat (energy last).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_parser = parse_bool)]
    htk_compat: bool,
    /// Raw energy (energy before preemphasis).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    raw_energy: bool,
    /// Energy floor.
    #[arg(long, default_value_t = 0.0)]
    energy_floor: f32,
    /// Librosa-style mel banks.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_parser = parse_bool)]
    is_librosa: bool,
    /// Librosa norm: "" or "slaney".
    #[arg(long, default_value = "slaney")]
    norm: String,
    /// Use Slaney mel scale (librosa mode).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_parser = parse_bool)]
    use_slaney_mel_scale: bool,
    /// Floor to integer bin (librosa mode).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_parser = parse_bool)]
    floor_to_int_bin: bool,
    /// Samples per AcceptWaveform call.
    #[arg(long, default_value_t = 16000)]
    chunk_size: usize,
    /// Print timing + RTF to stderr.
    #[arg(long)]
    bench: bool,
}

fn load_raw_f32(path: &str) -> Vec<f32> {
    let mut data = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {path}: {e}"))
        .read_to_end(&mut data)
        .expect("read failed");
    assert!(
        data.len() % 4 == 0,
        "input file size {} is not a multiple of 4 bytes",
        data.len()
    );
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let args = Args::parse();

    let mut opts = FbankOptions::default();
    opts.frame_opts.samp_freq = args.samp_freq;
    opts.frame_opts.frame_shift_ms = args.frame_shift_ms;
    opts.frame_opts.frame_length_ms = args.frame_length_ms;
    opts.frame_opts.dither = args.dither;
    opts.frame_opts.preemph_coeff = args.preemph;
    opts.frame_opts.snip_edges = args.snip_edges;
    opts.frame_opts.remove_dc_offset = args.remove_dc;
    opts.frame_opts.round_to_power_of_two = args.round_to_pow2;
    opts.frame_opts.window_type = args.window_type;
    opts.mel_opts.num_bins = args.num_bins;
    opts.mel_opts.low_freq = args.low_freq;
    opts.mel_opts.high_freq = args.high_freq;
    opts.use_energy = args.use_energy;
    opts.use_log_fbank = args.use_log_fbank;
    opts.use_power = args.use_power;
    opts.htk_compat = args.htk_compat;
    opts.raw_energy = args.raw_energy;
    opts.energy_floor = args.energy_floor;
    opts.mel_opts.is_librosa = args.is_librosa;
    opts.mel_opts.norm = args.norm;
    opts.mel_opts.use_slaney_mel_scale = args.use_slaney_mel_scale;
    opts.mel_opts.floor_to_int_bin = args.floor_to_int_bin;

    let samples = load_raw_f32(&args.input);

    let mut fbank = OnlineFbank::new(opts);
    let t0 = Instant::now();
    for chunk in samples.chunks(args.chunk_size) {
        fbank.accept_waveform(args.samp_freq, chunk);
    }
    fbank.input_finished();
    let t1 = Instant::now();

    let num_frames = fbank.num_frames_ready();
    let dim = fbank.dim() as usize;

    let mut out =
        File::create(&args.output).unwrap_or_else(|e| panic!("Cannot create {}: {e}", args.output));
    let mut row = Vec::with_capacity(dim * num_frames as usize);
    for i in 0..num_frames {
        row.extend_from_slice(fbank.get_frame(i));
    }
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.write_all(&bytes).expect("write failed");

    if args.bench {
        let secs = t1.duration_since(t0).as_secs_f64();
        let audio_secs = samples.len() as f64 / args.samp_freq as f64;
        eprintln!(
            "[bench] frames={num_frames} dim={dim} compute_s={secs:.6} audio_s={audio_secs:.3} \
             rtf={:.6} fps={:.1}",
            secs / audio_secs,
            num_frames as f64 / secs
        );
    }
}
