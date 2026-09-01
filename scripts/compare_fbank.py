#!/usr/bin/env python3
"""Compare two raw f32 FBank output files (both [num_frames x dim]).

Pure standard library (no numpy), so it runs anywhere with python3.
"""
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) % 4 == 0, f"{path}: size {len(data)} not a multiple of 4"
    return struct.unpack(f"<{len(data) // 4}f", data)


def main():
    if len(sys.argv) != 3:
        print("usage: compare_fbank.py ref.raw rs.raw")
        sys.exit(2)
    ref = load(sys.argv[1])
    rs = load(sys.argv[2])
    n = len(ref)
    if n != len(rs):
        print(f"FAIL: size mismatch ref={n} rs={len(rs)}")
        sys.exit(1)

    max_abs = 0.0
    sum_abs = 0.0
    max_rel = 0.0
    worst = 0
    for i in range(n):
        d = abs(ref[i] - rs[i])
        if d > max_abs:
            max_abs = d
            worst = i
        sum_abs += d
        rel = d / (abs(ref[i]) if abs(ref[i]) > 1e-12 else 1e-12)
        if rel > max_rel:
            max_rel = rel

    # infer plausible (frames, dim); report all dims that divide evenly
    plausible = [d for d in (80, 23, 25, 40, 41, 81, 128, 161, 32, 64) if n % d == 0]
    if len(plausible) == 1:
        nd = plausible[0]
        print(f"total_floats={n} frames={n // nd} dim={nd}")
    else:
        print(f"total_floats={n} possible_dims={plausible}")
    print(f"max_abs_diff={max_abs:.6e}  mean_abs_diff={sum_abs / n:.6e}  max_rel_diff={max_rel:.6e}")
    print(f"worst at flat_idx={worst}: ref={ref[worst]:.6f} rs={rs[worst]:.6f}")
    ok = max_abs < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
