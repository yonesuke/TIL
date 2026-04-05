# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Benchmark: C++ tridiagonal variants vs scipy.

Variants
--------
scipy          scipy.linalg.solve_banded  (LAPACK DGTSV)
cpp_thomas     pure Thomas algorithm      (sequential loop)
cpp_bwd_scan   loop fwd + affine scan bwd (std::inclusive_scan)
cpp_const_t    const-coeff Thomas         (scalar l/d/u)
cpp_const_bwd  const-coeff bwd scan       (scalar l/d/u)

Build
-----
    python setup.py build_ext --inplace

Run
---
    python benchmark.py
"""

import glob, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ── build C++ extension if needed ────────────────────────────────────────────
def _build():
    if glob.glob(os.path.join(HERE, "tridiagonal_cpp*.so")):
        return
    print("Building C++ extension ...", flush=True)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=HERE, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

_build()
import tridiagonal_cpp as cpp


# ── solvers ───────────────────────────────────────────────────────────────────
def scipy_solve(lo, di, up, b):
    n = len(di)
    ab = np.zeros((3, n))
    ab[0, 1:]  = up
    ab[1, :]   = di
    ab[2, :-1] = lo
    return solve_banded((1, 1), ab, b)


# ── correctness check ─────────────────────────────────────────────────────────
def verify():
    rng = np.random.default_rng(0)
    n = 30
    lo = rng.standard_normal(n - 1) * 0.5
    di = np.abs(rng.standard_normal(n)) + 3.0
    up = rng.standard_normal(n - 1) * 0.5
    b  = rng.standard_normal(n)
    lo_s, di_s, up_s = 0.4, 3.0, 0.3  # constant-coeff reference
    b_c = rng.standard_normal(n)

    ref_vec   = scipy_solve(lo, di, up, b)
    ref_const = scipy_solve(np.full(n-1, lo_s), np.full(n, di_s),
                            np.full(n-1, up_s), b_c)

    cases = [
        ("cpp_thomas",    cpp.solve_thomas(lo, di, up, b),           ref_vec),
        ("cpp_bwd_scan",  cpp.solve_bwd_scan(lo, di, up, b),         ref_vec),
        ("cpp_const_t",   cpp.solve_const_thomas(lo_s, di_s, up_s, b_c),  ref_const),
        ("cpp_const_bwd", cpp.solve_const_bwd_scan(lo_s, di_s, up_s, b_c), ref_const),
    ]
    print("=== correctness (max |err|) ===")
    for name, got, ref in cases:
        print(f"  {name:18s}  {np.abs(got - ref).max():.2e}")
    print()


# ── timing helper ─────────────────────────────────────────────────────────────
def timeit(fn, warmup=10, iters=500):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e6  # µs/call


# ── benchmark ─────────────────────────────────────────────────────────────────
def benchmark():
    rng  = np.random.default_rng(42)
    ns   = [100, 300, 1_000, 3_000, 10_000, 30_000]
    lo_s, di_s, up_s = 0.4, 3.0, 0.3

    keys  = ["scipy", "cpp_thomas", "cpp_bwd_scan", "cpp_const_t", "cpp_const_bwd"]
    times = {k: [] for k in keys}

    print(f"{'n':>7}  {'scipy':>8}  {'cpp_thomas':>11}  {'cpp_bwd':>9}  "
          f"{'const_t':>9}  {'const_bwd':>11}  (µs/call)")
    print("─" * 72)

    for n in ns:
        lo = rng.standard_normal(n - 1) * 0.5
        di = np.abs(rng.standard_normal(n)) + 3.0
        up = rng.standard_normal(n - 1) * 0.5
        b  = rng.standard_normal(n)
        bc = rng.standard_normal(n)

        r = {
            "scipy":        timeit(lambda: scipy_solve(lo, di, up, b)),
            "cpp_thomas":   timeit(lambda: cpp.solve_thomas(lo, di, up, b)),
            "cpp_bwd_scan": timeit(lambda: cpp.solve_bwd_scan(lo, di, up, b)),
            "cpp_const_t":  timeit(lambda: cpp.solve_const_thomas(lo_s, di_s, up_s, bc)),
            "cpp_const_bwd":timeit(lambda: cpp.solve_const_bwd_scan(lo_s, di_s, up_s, bc)),
        }
        for k in keys: times[k].append(r[k])
        print(f"{n:>7}  {r['scipy']:>7.1f}µ  {r['cpp_thomas']:>10.1f}µ  "
              f"{r['cpp_bwd_scan']:>8.1f}µ  {r['cpp_const_t']:>8.1f}µ  "
              f"{r['cpp_const_bwd']:>10.1f}µ")

    return ns, times


# ── plot ──────────────────────────────────────────────────────────────────────
def plot(ns, times):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    styles = {
        "scipy":         ("C0", "o", "-",  "scipy (LAPACK DGTSV)"),
        "cpp_thomas":    ("C2", "^", "-",  "C++ Thomas"),
        "cpp_bwd_scan":  ("C3", "v", "-",  "C++ bwd scan"),
        "cpp_const_t":   ("C2", "^", "--", "C++ const Thomas"),
        "cpp_const_bwd": ("C3", "v", "--", "C++ const bwd scan"),
    }

    # left: absolute time
    ax = axes[0]
    for key, (col, mk, ls, label) in styles.items():
        ax.plot(ns, times[key], color=col, marker=mk, linestyle=ls,
                label=label, lw=1.8, ms=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("System size  n", fontsize=12)
    ax.set_ylabel("Time per solve  [µs]", fontsize=12)
    ax.set_title("Absolute time", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

    # right: speedup over scipy
    ax = axes[1]
    for key, (col, mk, ls, label) in styles.items():
        if key == "scipy":
            continue
        sp = [s / t for s, t in zip(times["scipy"], times[key])]
        ax.plot(ns, sp, color=col, marker=mk, linestyle=ls,
                label=label, lw=1.8, ms=6)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("System size  n", fontsize=12)
    ax.set_ylabel("Speedup vs scipy", fontsize=12)
    ax.set_title("Speedup over scipy.solve_banded", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Tridiagonal solver: C++ Thomas variants vs scipy  (-O3 -march=native)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "benchmark.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    verify()
    ns, times = benchmark()
    plot(ns, times)
