// fbank_cli.cc
//
// Reference CLI for FBank feature extraction based on kaldi-native-fbank.
//
// Reads raw f32 PCM (little-endian, mono) from a file, computes FBank
// features with the stock kaldi-native-fbank library, writes the features
// (f32, [num_frames x dim]) to an output file, and (optionally) reports
// timing to stderr.
//
// Build:
//   g++ -O3 -std=c++17 -I<kaldi-native-fbank-root> fbank_cli.cc \
//       -L<build/lib> -lkaldi-native-fbank-core -o fbank_cli
//
// Usage:
//   fbank_cli [options] input.raw output.raw

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace {

void Usage(const char *prog) {
  std::cerr
      << "Usage: " << prog << " [options] input.raw output.raw\n"
      << "  input.raw : raw f32 PCM (little endian, mono)\n"
      << "  output.raw: fbank features, f32, [num_frames x dim]\n"
      << "Options:\n"
      << "  --samp-freq <f>       sampling rate (default 16000)\n"
      << "  --frame-shift-ms <f>  frame shift in ms (default 10)\n"
      << "  --frame-length-ms <f> frame length in ms (default 25)\n"
      << "  --num-bins <n>        mel bins (default 80)\n"
      << "  --low-freq <f>        (default 20)\n"
      << "  --high-freq <f>       (default 0 => nyquist)\n"
      << "  --preemph <f>         preemphasis coeff (default 0.97)\n"
      << "  --dither <f>          dither amount (default 0)\n"
      << "  --snip-edges <0|1>    (default 1)\n"
      << "  --remove-dc <0|1>     remove dc offset (default 1)\n"
      << "  --round-to-pow2 <0|1> (default 1)\n"
      << "  --window-type <s>     povey|hamming|hanning|hann|sine|rectangular|blackman (default povey)\n"
      << "  --use-energy <0|1>    append energy dim (default 0)\n"
      << "  --use-log-fbank <0|1> (default 1)\n"
      << "  --use-power <0|1>     (default 1)\n"
      << "  --htk-compat <0|1>    (default 0)\n"
      << "  --raw-energy <0|1>    (default 1)\n"
      << "  --energy-floor <f>    (default 0)\n"
      << "  --is-librosa <0|1>    librosa-style mel banks (default 0)\n"
      << "  --norm <s>            librosa norm: \"\" or slaney (default slaney)\n"
      << "  --use-slaney-mel-scale <0|1> (default 1)\n"
      << "  --floor-to-int-bin <0|1>     (default 0)\n"
      << "  --chunk-size <n>      samples per AcceptWaveform (default 16000)\n"
      << "  --bench               print timing + RTF to stderr\n";
}

struct Args {
  knf::FbankOptions opts;
  int32_t chunk_size = 16000;
  bool bench = false;
  std::string input;
  std::string output;
};

bool ParseBool(const std::string &s, bool *out) {
  if (s == "1" || s == "true" || s == "yes") {
    *out = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "no") {
    *out = false;
    return true;
  }
  return false;
}

bool ParseArgs(int argc, char **argv, Args *args) {
  int i = 1;
  auto next = [&](const char *name) -> const char * {
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << name << "\n";
      exit(-1);
    }
    return argv[++i];
  };
  for (; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--samp-freq") {
      args->opts.frame_opts.samp_freq = std::stof(next(a.c_str()));
    } else if (a == "--frame-shift-ms") {
      args->opts.frame_opts.frame_shift_ms = std::stof(next(a.c_str()));
    } else if (a == "--frame-length-ms") {
      args->opts.frame_opts.frame_length_ms = std::stof(next(a.c_str()));
    } else if (a == "--num-bins") {
      args->opts.mel_opts.num_bins = std::stoi(next(a.c_str()));
    } else if (a == "--low-freq") {
      args->opts.mel_opts.low_freq = std::stof(next(a.c_str()));
    } else if (a == "--high-freq") {
      args->opts.mel_opts.high_freq = std::stof(next(a.c_str()));
    } else if (a == "--preemph") {
      args->opts.frame_opts.preemph_coeff = std::stof(next(a.c_str()));
    } else if (a == "--dither") {
      args->opts.frame_opts.dither = std::stof(next(a.c_str()));
    } else if (a == "--snip-edges") {
      if (!ParseBool(next(a.c_str()), &args->opts.frame_opts.snip_edges)) {
        std::cerr << "Bad bool for --snip-edges\n";
        return false;
      }
    } else if (a == "--remove-dc") {
      if (!ParseBool(next(a.c_str()), &args->opts.frame_opts.remove_dc_offset)) {
        std::cerr << "Bad bool for --remove-dc\n";
        return false;
      }
    } else if (a == "--round-to-pow2") {
      if (!ParseBool(next(a.c_str()),
                     &args->opts.frame_opts.round_to_power_of_two)) {
        std::cerr << "Bad bool for --round-to-pow2\n";
        return false;
      }
    } else if (a == "--window-type") {
      args->opts.frame_opts.window_type = next(a.c_str());
    } else if (a == "--use-energy") {
      if (!ParseBool(next(a.c_str()), &args->opts.use_energy)) {
        std::cerr << "Bad bool for --use-energy\n";
        return false;
      }
    } else if (a == "--use-log-fbank") {
      if (!ParseBool(next(a.c_str()), &args->opts.use_log_fbank)) {
        std::cerr << "Bad bool for --use-log-fbank\n";
        return false;
      }
    } else if (a == "--use-power") {
      if (!ParseBool(next(a.c_str()), &args->opts.use_power)) {
        std::cerr << "Bad bool for --use-power\n";
        return false;
      }
    } else if (a == "--htk-compat") {
      if (!ParseBool(next(a.c_str()), &args->opts.htk_compat)) {
        std::cerr << "Bad bool for --htk-compat\n";
        return false;
      }
    } else if (a == "--raw-energy") {
      if (!ParseBool(next(a.c_str()), &args->opts.raw_energy)) {
        std::cerr << "Bad bool for --raw-energy\n";
        return false;
      }
    } else if (a == "--energy-floor") {
      args->opts.energy_floor = std::stof(next(a.c_str()));
    } else if (a == "--is-librosa") {
      if (!ParseBool(next(a.c_str()), &args->opts.mel_opts.is_librosa)) {
        std::cerr << "Bad bool for --is-librosa\n";
        return false;
      }
    } else if (a == "--norm") {
      args->opts.mel_opts.norm = next(a.c_str());
    } else if (a == "--use-slaney-mel-scale") {
      if (!ParseBool(next(a.c_str()), &args->opts.mel_opts.use_slaney_mel_scale)) {
        std::cerr << "Bad bool for --use-slaney-mel-scale\n";
        return false;
      }
    } else if (a == "--floor-to-int-bin") {
      if (!ParseBool(next(a.c_str()), &args->opts.mel_opts.floor_to_int_bin)) {
        std::cerr << "Bad bool for --floor-to-int-bin\n";
        return false;
      }
    } else if (a == "--chunk-size") {
      args->chunk_size = std::stoi(next(a.c_str()));
    } else if (a == "--bench") {
      args->bench = true;
    } else if (a == "--help" || a == "-h") {
      Usage(argv[0]);
      exit(0);
    } else if (a.size() > 0 && a[0] == '-') {
      std::cerr << "Unknown option: " << a << "\n";
      return false;
    } else {
      if (args->input.empty()) {
        args->input = a;
      } else if (args->output.empty()) {
        args->output = a;
      } else {
        std::cerr << "Too many positional arguments\n";
        return false;
      }
    }
  }
  if (args->input.empty() || args->output.empty()) {
    std::cerr << "Need input and output files\n";
    return false;
  }
  return true;
}

std::vector<float> LoadRawF32(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    std::cerr << "Failed to open " << path << "\n";
    exit(-1);
  }
  is.seekg(0, std::ios::end);
  std::streamsize size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<float> data(size / sizeof(float));
  is.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

}  // namespace

int main(int argc, char **argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return -1;
  }

  std::vector<float> samples = LoadRawF32(args.input);

  knf::OnlineFbank fbank(args.opts);

  auto t0 = std::chrono::steady_clock::now();
  int32_t n = static_cast<int32_t>(samples.size());
  for (int32_t i = 0; i < n; i += args.chunk_size) {
    int32_t len = std::min(args.chunk_size, n - i);
    fbank.AcceptWaveform(args.opts.frame_opts.samp_freq, samples.data() + i,
                         len);
  }
  fbank.InputFinished();
  auto t1 = std::chrono::steady_clock::now();

  int32_t num_frames = fbank.NumFramesReady();
  int32_t dim = fbank.Dim();

  // Write [num_frames x dim] f32 rows.
  std::ofstream os(args.output, std::ios::binary);
  for (int32_t i = 0; i < num_frames; ++i) {
    const float *frame = fbank.GetFrame(i);
    os.write(reinterpret_cast<const char *>(frame), dim * sizeof(float));
  }

  if (args.bench) {
    double secs =
        std::chrono::duration<double>(t1 - t0).count();
    double audio_secs = n / static_cast<double>(args.opts.frame_opts.samp_freq);
    std::cerr << "[bench] frames=" << num_frames << " dim=" << dim
              << " compute_s=" << secs << " audio_s=" << audio_secs
              << " rtf=" << (secs / audio_secs)
              << " fps=" << (num_frames / secs) << "\n";
  }
  return 0;
}
