#!/usr/bin/env bash
# bench_compare.sh — benchmarks the C++ reference vs the Rust implementation on
# the same inputs and prints a comparison table. Each case is run `RUNS` times;
# the reported number is the best (minimum) RTF.
#
# Usage: scripts/bench_compare.sh
#   RUNS=<n>   number of repeats per case (default 7)

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/ref/fbank_cli"
RS="$ROOT/target/release/examples/fbank-cli"
AUDIO="$ROOT/data/audio"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

RUNS="${RUNS:-7}"

# Case = "group|label|audiofile|args" (audiofile is the basename in data/audio,
# appended with `.f32`; a leading `/` means a full path).
declare -a BENCH_CASES=(
  # ---- 1. duration sweep: 0.1s -> 600s, default 80 bins ----
  "duration|tiny-0.1s|speech_tiny_0.1s|--num-bins 80 --dither 0"
  "duration|tiny-0.3s|speech_tiny_0.3s|--num-bins 80 --dither 0"
  "duration|short-2s|speech_short_2s|--num-bins 80 --dither 0"
  "duration|short-5s|speech_short_5s|--num-bins 80 --dither 0"
  "duration|medium-30s|speech_medium_30s|--num-bins 80 --dither 0"
  "duration|medium-60s|speech_medium_60s|--num-bins 80 --dither 0"
  "duration|long-300s|speech_long_300s|--num-bins 80 --dither 0"
  "duration|long-600s|speech_long_600s|--num-bins 80 --dither 0"
  # ---- 2. config sweep: 60s audio x many configs ----
  "config|bins80|speech_medium_60s|--num-bins 80 --dither 0"
  "config|bins23|speech_medium_60s|--num-bins 23 --dither 0"
  "config|bins40|speech_medium_60s|--num-bins 40 --dither 0"
  "config|snip0|speech_medium_60s|--num-bins 80 --dither 0 --snip-edges 0"
  "config|hamming|speech_medium_60s|--num-bins 80 --dither 0 --window-type hamming"
  "config|hanning|speech_medium_60s|--num-bins 80 --dither 0 --window-type hanning"
  "config|sine|speech_medium_60s|--num-bins 80 --dither 0 --window-type sine"
  "config|blackman|speech_medium_60s|--num-bins 80 --dither 0 --window-type blackman"
  "config|energy|speech_medium_60s|--num-bins 80 --dither 0 --use-energy 1"
  "config|magnitude|speech_medium_60s|--num-bins 80 --dither 0 --use-power 0"
  "config|linear|speech_medium_60s|--num-bins 80 --dither 0 --use-log-fbank 0"
  "config|nopow2|speech_medium_60s|--num-bins 80 --dither 0 --round-to-pow2 0"
  "config|librosa|speech_medium_60s|--num-bins 80 --dither 0 --is-librosa 1 --low-freq 0 --norm slaney"
  "config|shift5ms|speech_medium_60s|--num-bins 80 --dither 0 --frame-shift-ms 5"
  "config|shift20ms|speech_medium_60s|--num-bins 80 --dither 0 --frame-shift-ms 20"
  "config|len20ms|speech_medium_60s|--num-bins 80 --dither 0 --frame-length-ms 20"
  "config|len32ms|speech_medium_60s|--num-bins 80 --dither 0 --frame-length-ms 32"
  "config|preemph0|speech_medium_60s|--num-bins 80 --dither 0 --preemph 0"
  "config|remove_dc0|speech_medium_60s|--num-bins 80 --dither 0 --remove-dc 0"
  # ---- 3. content sweep: meeting recordings ----
  "content|meeting-300s|meeting_300s|--num-bins 80 --dither 0"
  "content|meeting-600s|meeting_600s|--num-bins 80 --dither 0"
  # ---- 4. sample-rate sweep ----
  "samplerate|8k-60s|speech8k_60s|--num-bins 80 --dither 0 --samp-freq 8000"
  "samplerate|44.1k-60s|speech44k_60s|--num-bins 80 --dither 0 --samp-freq 44100"
)

best_rtf() {
  local bin="$1" audio="$2" args="$3" chunk="$4"
  local best=""
  # shellcheck disable=SC2086
  for _ in $(seq 1 "$RUNS"); do
    local line rtf
    line="$("$bin" $args --chunk-size "$chunk" --bench "$audio" "$WORK/out.bin" 2>&1 >/dev/null)"
    rtf="$(echo "$line" | sed -n 's/.*rtf=\([0-9.e-]*\).*/\1/p')"
    if [[ -z "$rtf" ]]; then
      echo "ERROR"
      return
    fi
    if [[ -z "$best" ]] || awk -v a="$rtf" -v b="$best" 'BEGIN{exit !(a<b)}'; then
      best="$rtf"
    fi
  done
  echo "$best"
}

# (8k / 44.1kHz sample-rate files are pre-generated in data/audio)

echo "== kaldi-fbank vs kaldi-native-fbank (best of $RUNS), local x86-64 =="
echo "group | case | audio_len | chunk | C++ rtf | Rust rtf | speedup"
echo "------|------|-----------|-------|----------|----------|--------"

prev_group=""
for c in "${BENCH_CASES[@]}"; do
  group="${c%%|*}"; rest="${c#*|}"
  name="${rest%%|*}"; rest="${rest#*|}"
  audio="${rest%%|*}"; args="${rest#*|}"
  if [[ "$group" != "$prev_group" ]]; then
    echo "---- $group ----"
    prev_group="$group"
  fi
  if [[ "$audio" == /* ]]; then
    full_audio="$audio"
  else
    full_audio="$AUDIO/$audio.f32"
  fi
  bytes=$(stat -c %s "$full_audio")
  samples=$((bytes / 4))
  sr="$(echo "$args" | sed -n 's/.*--samp-freq \([0-9]*\).*/\1/p')"
  [[ -z "$sr" ]] && sr=16000
  len_sec=$(awk "BEGIN{printf \"%.1f\", $samples/$sr}")
  for chunk in 1600 16000; do
    ref_rtf="$(best_rtf "$REF" "$full_audio" "$args" "$chunk")"
    rs_rtf="$(best_rtf "$RS" "$full_audio" "$args" "$chunk")"
    if [[ "$ref_rtf" == "ERROR" || "$rs_rtf" == "ERROR" ]]; then
      echo "$group | $name | ${len_sec}s | $chunk | ERROR"
      continue
    fi
    speedup="$(awk -v a="$ref_rtf" -v b="$rs_rtf" 'BEGIN{printf "%.2f", a/b}')"
    echo "$group | $name | ${len_sec}s | $chunk | $ref_rtf | $rs_rtf | ${speedup}x"
  done
done

echo
echo "== absolute time & throughput (600s audio, chunk=16000, 80 bins) =="
for i in 1 2 3; do
  "$REF" --num-bins 80 --dither 0 --chunk-size 16000 --bench "$AUDIO/speech_long_600s.f32" "$WORK/o1.bin" 2>&1 >/dev/null
  "$RS" --num-bins 80 --dither 0 --chunk-size 16000 --bench "$AUDIO/speech_long_600s.f32" "$WORK/o2.bin" 2>&1 >/dev/null
done
