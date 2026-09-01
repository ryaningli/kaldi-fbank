#!/usr/bin/env bash
# run_matrix.sh — runs the C++ reference and the Rust implementation across a
# matrix of audio lengths / configs / chunk sizes and compares outputs.
#
# Usage: scripts/run_matrix.sh [--quick]

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/ref/fbank_cli"
RS="$ROOT/target/release/examples/fbank-cli"
AUDIO="$ROOT/data/audio"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

QUICK="${1:-}"

declare -a FILES=(speech_tiny_0.3s speech_short_2s speech_medium_30s)
if [[ "$QUICK" != "--quick" ]]; then
  FILES+=(speech_long_300s)
fi

# name|extra args (audio file name appended separately)
declare -a CONFIGS=(
  "default80|--num-bins 80 --dither 0"
  "default23|--num-bins 23 --dither 0"
  "bins40|--num-bins 40 --dither 0"
  "snip0|--num-bins 80 --dither 0 --snip-edges 0"
  "snip0_bins40|--num-bins 40 --dither 0 --snip-edges 0"
  "hamming|--num-bins 80 --dither 0 --window-type hamming"
  "hanning|--num-bins 80 --dither 0 --window-type hanning"
  "hann|--num-bins 80 --dither 0 --window-type hann"
  "sine|--num-bins 80 --dither 0 --window-type sine"
  "rectangular|--num-bins 80 --dither 0 --window-type rectangular"
  "blackman|--num-bins 80 --dither 0 --window-type blackman"
  "preemph0|--num-bins 80 --dither 0 --preemph 0"
  "remove_dc0|--num-bins 80 --dither 0 --remove-dc 0"
  "energy|--num-bins 80 --dither 0 --use-energy 1"
  "energy_raw0|--num-bins 80 --dither 0 --use-energy 1 --raw-energy 0"
  "energy_htk|--num-bins 80 --dither 0 --use-energy 1 --htk-compat 1"
  "energy_floor|--num-bins 80 --dither 0 --use-energy 1 --energy-floor 1.0"
  "magnitude|--num-bins 80 --dither 0 --use-power 0"
  "linear|--num-bins 80 --dither 0 --use-log-fbank 0"
  "nopow2|--num-bins 80 --dither 0 --round-to-pow2 0"
  "lowfreq0|--num-bins 80 --dither 0 --low-freq 0"
  "highfreq_neg|--num-bins 80 --dither 0 --high-freq -2000"
  "librosa|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm slaney"
  "librosa_noslaney|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm EMPTY --use-slaney-mel-scale 0"
  "librosa_floor|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --floor-to-int-bin 1"
  "librosa_nonorm|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm EMPTY"
  "win_shift5|--num-bins 80 --dither 0 --frame-shift-ms 5"
  "win_shift20|--num-bins 80 --dither 0 --frame-shift-ms 20"
  "win_len20|--num-bins 80 --dither 0 --frame-length-ms 20"
  "win_len32|--num-bins 80 --dither 0 --frame-length-ms 32"
)

declare -a CHUNKS=(1 100 1600 16000 1000000)

PASS=0
FAIL=0
FAILED_CASES=()

run_case() {
  local name="$1" wavfile="$2" args="$3" chunk="$4" sr="$5"
  local ref_out="$WORK/ref.bin" rs_out="$WORK/rs.bin"
  # Translate the EMPTY sentinel into a real empty-string argument so that
  # word-splitting cannot mangle it.
  local -a arglist=()
  for tok in $args; do
    if [[ "$tok" == "EMPTY" ]]; then arglist+=(""); else arglist+=("$tok"); fi
  done
  if ! "$REF" "${arglist[@]}" --chunk-size "$chunk" "$wavfile" "$ref_out" 2>/dev/null; then
    echo "FAIL(ref) $name [$wavfile] chunk=$chunk args='$args'"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$name [$wavfile] chunk=$chunk args='$args'")
    return
  fi
  if ! "$RS" "${arglist[@]}" --chunk-size "$chunk" "$wavfile" "$rs_out" 2>/dev/null; then
    echo "FAIL(rs) $name [$wavfile] chunk=$chunk args='$args'"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$name [$wavfile] chunk=$chunk args='$args'")
    return
  fi
  local result maxdiff
  result="$(cd "$ROOT" && python3 scripts/compare_fbank.py "$ref_out" "$rs_out")"
  if [[ $? -ne 0 ]]; then
    echo "FAIL(diff) $name [$wavfile] chunk=$chunk args='$args'"
    echo "    $result"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$name [$wavfile] chunk=$chunk args='$args'")
  else
    maxdiff="$(echo "$result" | grep max_abs_diff | sed 's/.*max_abs_diff=\([^ ]*\).*/\1/')"
    echo "PASS $name [$wavfile] chunk=$chunk max_abs_diff=$maxdiff"
    PASS=$((PASS + 1))
  fi
}

echo "== consistency matrix =="
for file in "${FILES[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%|*}"
    args="${cfg#*|}"
    for chunk in "${CHUNKS[@]}"; do
      run_case "$name" "$AUDIO/$file.f32" "$args" "$chunk" 16000
    done
  done
done

echo
echo "== 8 kHz sample-rate test =="
ffmpeg -y -v error -i "$AUDIO/speech_medium_30s.wav" -ac 1 -ar 8000 -f f32le "$WORK/speech8k.f32" 2>/dev/null
run_case "default80-8k" "$WORK/speech8k.f32" "--num-bins 80 --dither 0 --samp-freq 8000" 16000 8000
run_case "snip0-8k" "$WORK/speech8k.f32" "--num-bins 80 --dither 0 --samp-freq 8000 --snip-edges 0" 16000 8000

echo
echo "== 44.1 kHz sample-rate test =="
ffmpeg -y -v error -i "$AUDIO/speech_short_2s.wav" -ac 1 -ar 44100 -f f32le "$WORK/speech44k.f32" 2>/dev/null
run_case "default80-44k" "$WORK/speech44k.f32" "--num-bins 80 --dither 0 --samp-freq 44100" 16000 44100
run_case "snip0-44k" "$WORK/speech44k.f32" "--num-bins 80 --dither 0 --samp-freq 44100 --snip-edges 0" 16000 44100

echo
echo "PASS=$PASS FAIL=$FAIL"
if [[ $FAIL -gt 0 ]]; then
  printf 'failed:\n'
  printf '  %s\n' "${FAILED_CASES[@]}"
  exit 1
fi
exit 0
