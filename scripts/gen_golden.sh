#!/usr/bin/env bash
# gen_golden.sh — regenerates the golden reference files used by
# tests/golden_test.rs. Requires the C++ reference CLI to be built
# (ref/fbank_cli).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/ref/fbank_cli"
GOLD="$ROOT/tests/golden"
mkdir -p "$GOLD"

# --- signals ---------------------------------------------------------------
# Deterministic pseudo-speech: sum of sines with amplitude envelope, 2s @ 16k.
python3 - "$GOLD/pspeech_2s.f32" << 'EOF'
import math, struct, sys
out = sys.argv[1]
sr = 16000
n = sr * 2
freqs = [(440.0, 0.30), (1200.0, 0.20), (3000.0, 0.15), (7000.0, 0.10)]
with open(out, "wb") as f:
    for i in range(n):
        t = i / sr
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        s = sum(a * math.sin(2 * math.pi * fr * t) for fr, a in freqs) * env * 0.5
        f.write(struct.pack("<f", s))
print(f"wrote {out}")
EOF
# 8k and 44.1k versions of the same 2s signal
python3 - "$GOLD/pspeech_2s_8k.f32" "$GOLD/pspeech_2s_44k.f32" << 'EOF'
import math, struct, sys
for out, sr in zip(sys.argv[1:], (8000, 44100)):
    n = sr * 2
    freqs = [(440.0, 0.30), (1200.0, 0.20), (3000.0, 0.15), (7000.0, 0.10)]
    with open(out, "wb") as f:
        for i in range(n):
            t = i / sr
            env = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
            s = sum(a * math.sin(2 * math.pi * fr * t) for fr, a in freqs) * env * 0.5
            f.write(struct.pack("<f", s))
    print(f"wrote {out} @ {sr}")
EOF

# --- cases: name|signal|args -------------------------------------------------
declare -a CASES=(
  # bins
  "bins80|pspeech_2s.f32|--num-bins 80 --dither 0"
  "bins23|pspeech_2s.f32|--num-bins 23 --dither 0"
  "bins40|pspeech_2s.f32|--num-bins 40 --dither 0"
  "bins80_snip0|pspeech_2s.f32|--num-bins 80 --dither 0 --snip-edges 0"
  "bins40_snip0|pspeech_2s.f32|--num-bins 40 --dither 0 --snip-edges 0"
  # windows
  "hamming|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type hamming"
  "hanning|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type hanning"
  "hann|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type hann"
  "sine|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type sine"
  "rectangular|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type rectangular"
  "blackman|pspeech_2s.f32|--num-bins 80 --dither 0 --window-type blackman"
  # energy variants
  "energy|pspeech_2s.f32|--num-bins 80 --dither 0 --use-energy 1"
  "energy_raw0|pspeech_2s.f32|--num-bins 80 --dither 0 --use-energy 1 --raw-energy 0"
  "energy_htk|pspeech_2s.f32|--num-bins 80 --dither 0 --use-energy 1 --htk-compat 1"
  "energy_floor|pspeech_2s.f32|--num-bins 80 --dither 0 --use-energy 1 --energy-floor 1.0"
  # spectrum variants
  "magnitude|pspeech_2s.f32|--num-bins 80 --dither 0 --use-power 0"
  "linear|pspeech_2s.f32|--num-bins 80 --dither 0 --use-log-fbank 0"
  # window/frame geometry
  "nopow2|pspeech_2s.f32|--num-bins 80 --dither 0 --round-to-pow2 0"
  "shift5ms|pspeech_2s.f32|--num-bins 80 --dither 0 --frame-shift-ms 5"
  "shift20ms|pspeech_2s.f32|--num-bins 80 --dither 0 --frame-shift-ms 20"
  "len20ms|pspeech_2s.f32|--num-bins 80 --dither 0 --frame-length-ms 20"
  "len32ms|pspeech_2s.f32|--num-bins 80 --dither 0 --frame-length-ms 32"
  # preprocessing
  "preemph0|pspeech_2s.f32|--num-bins 80 --dither 0 --preemph 0"
  "removedc0|pspeech_2s.f32|--num-bins 80 --dither 0 --remove-dc 0"
  # freq bounds
  "lowfreq0|pspeech_2s.f32|--num-bins 80 --dither 0 --low-freq 0"
  "highfreq_neg|pspeech_2s.f32|--num-bins 80 --dither 0 --high-freq -2000"
  # librosa variants
  "librosa|pspeech_2s.f32|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm slaney"
  "librosa_nonorm|pspeech_2s.f32|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm EMPTY"
  "librosa_noslaney|pspeech_2s.f32|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm EMPTY --use-slaney-mel-scale 0"
  "librosa_floor|pspeech_2s.f32|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --floor-to-int-bin 1"
  # sample rates
  "bins80_8k|pspeech_2s_8k.f32|--num-bins 80 --dither 0 --samp-freq 8000"
  "bins40_snip0_8k|pspeech_2s_8k.f32|--num-bins 40 --dither 0 --samp-freq 8000 --snip-edges 0"
  "bins80_44k|pspeech_2s_44k.f32|--num-bins 80 --dither 0 --samp-freq 44100"
  "bins80_snip0_44k|pspeech_2s_44k.f32|--num-bins 80 --dither 0 --samp-freq 44100 --snip-edges 0"
)

for c in "${CASES[@]}"; do
  name="${c%%|*}"; rest="${c#*|}"; sig="${rest%%|*}"; args="${rest#*|}"
  out="$GOLD/pspeech_${name}.fbank"
  arglist=()
  for tok in $args; do
    if [[ "$tok" == "EMPTY" ]]; then arglist+=(""); else arglist+=("$tok"); fi
  done
  "$REF" "${arglist[@]}" "$GOLD/$sig" "$out"
  echo "generated $out"
done

echo "golden files: $(ls "$GOLD"/*.fbank | wc -l)"
