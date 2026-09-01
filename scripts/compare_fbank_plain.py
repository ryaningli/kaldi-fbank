#!/usr/bin/env python3
"""No-numpy f32 output comparator (for edge devices without numpy)."""
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) % 4 == 0, f"{path}: size {len(data)} not multiple of 4"
    return struct.unpack(f"<{len(data)//4}f", data)


def main():
    if len(sys.argv) != 3:
        print("usage: compare_fbank_plain.py ref.raw rs.raw")
        sys.exit(2)
    ref = load(sys.argv[1])
    rs = load(sys.argv[2])
    if len(ref) != len(rs):
        print(f"FAIL: size mismatch ref={len(ref)} rs={len(rs)}")
        sys.exit(1)
    max_abs = 0.0
    sum_abs = 0.0
    max_rel = 0.0
    worst = 0
    n = len(ref)
    for i in range(n):
        d = abs(ref[i] - rs[i])
        if d > max_abs:
            max_abs = d
            worst = i
        sum_abs += d
        r = d / (abs(ref[i]) if abs(ref[i]) > 1e-12 else 1e-12)
        if r > max_rel:
            max_rel = r
    print(f"total_floats={n}")
    print(
        f"max_abs_diff={max_abs:.6e}  mean_abs_diff={sum_abs/n:.6e}  max_rel_diff={max_rel:.6e}"
    )
    print(f"worst at flat_idx={worst}: ref={ref[worst]:.6f} rs={rs[worst]:.6f}")
    ok = max_abs < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
