#!/usr/bin/env python3
"""Compare two period_search output files column by column.

Columns: period rms chisq dark lambda beta
Reports exact-print agreement and numeric deltas for the pole columns
(dark/lambda/beta), which the project validator ignores.
"""
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                rows.append(parts[:6])
    return rows


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"line count differs: {len(a)} vs {len(b)}")
    names = ["period", "rms", "chisq", "dark", "lambda", "beta"]
    exact = [0] * 6
    lam_off5 = beta_off5 = 0
    flipped = []
    for i in range(n):
        for c in range(6):
            if a[i][c] == b[i][c]:
                exact[c] += 1
        dl = abs(float(a[i][4]) - float(b[i][4]))
        dl = min(dl, 360 - dl)
        db = abs(float(a[i][5]) - float(b[i][5]))
        if dl > 5:
            lam_off5 += 1
        if db > 5:
            beta_off5 += 1
        if dl > 5 or db > 5:
            flipped.append(
                f"  line {i+1:3d}: {sys.argv[1].split('/')[-1]}=({a[i][4]:>4},{a[i][5]:>4})"
                f"  {sys.argv[2].split('/')[-1]}=({b[i][4]:>4},{b[i][5]:>4})  per {a[i][0]} vs {b[i][0]}"
            )
    print(f"{n} lines compared: {sys.argv[1]} vs {sys.argv[2]}")
    print("exact-print agreement per column:")
    for c in range(6):
        print(f"  {names[c]:7s} {exact[c]}/{n}")
    print(f"lambda off by >5 deg: {lam_off5}/{n}")
    print(f"beta   off by >5 deg: {beta_off5}/{n}")
    if "-v" in sys.argv:
        print("flipped lines (>5 deg):")
        print("\n".join(flipped))


if __name__ == "__main__":
    main()
