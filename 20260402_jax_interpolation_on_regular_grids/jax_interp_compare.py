# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax[cpu]",
#   "scipy",
#   "numpy",
# ]
# ///
"""
scipy vs jax_interp: correspondence table, accuracy tests, and speed benchmarks.

Correspondence table
--------------------
method/BC/oob     jax_interp call                    scipy equivalent
─────────────────────────────────────────────────────────────────────
nearest           interp1d(..., method='nearest')    interp1d(kind='nearest')           [LEGACY]
linear            interp1d(..., method='linear')     interp1d(kind='linear')            [LEGACY]
                                                     make_interp_spline(x,y, k=1)
quadratic         interp1d(..., method='quadratic')  interp1d(kind='quadratic')         [DIFFERENT ALGORITHM]
cubic/natural     interp1d(..., bc='natural')        CubicSpline(bc_type='natural')
cubic/not-a-knot  interp1d(..., bc='not-a-knot')     CubicSpline(bc_type='not-a-knot')  [default]
cubic/clamped     interp1d(..., bc='clamped', fp=..) CubicSpline(bc_type=((1,fp_l),(1,fp_r)))
cubic/periodic    interp1d(..., bc='periodic')       CubicSpline(bc_type='periodic')
bspline           interp1d(..., method='bspline')    make_interp_spline(x, y, k=3)      [not-a-knot]
─────────────────────────────────────────────────────────────────────
oob='clamp'       interp1d(..., oob='clamp')         interp1d(bounds_error=False, fill_value=(y[0],y[-1]))
oob='extrapolate' interp1d(..., oob='extrapolate')   interp1d(fill_value='extrapolate')
                                                     make_interp_spline()(xq, extrapolate=True)  [default]
oob='periodic'    interp1d(..., oob='periodic')      BSpline()(xq, extrapolate='periodic')       [bspline only]
oob='reflect'     interp1d(..., oob='reflect')       NO SCIPY EQUIVALENT
─────────────────────────────────────────────────────────────────────
2D linear         interp2d(..., method='linear')     RegularGridInterpolator(method='linear')
2D bspline        interp2d(..., method='bspline')    RegularGridInterpolator(method='cubic')  [>= 1.13]
2D cubic          interp2d(..., method='cubic')      NO DIRECT EQUIVALENT (custom 16-coeff bicubic)
─────────────────────────────────────────────────────────────────────

Notes
-----
- "quadratic" uses 3-point Lagrange (ours) vs quadratic B-spline (scipy) -> different algorithms
- "bspline" and scipy "cubic" (>= 1.13) should match to machine epsilon on uniform grids
- "2D cubic" uses the 16-coefficient bicubic with precomputed node derivatives; scipy has no direct
  equivalent (it uses the full NdBSpline tensor-product approach for method='cubic')
- oob='reflect' has no scipy equivalent at all
"""

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import (
    interp1d as scipy_interp1d,
    CubicSpline,
    make_interp_spline,
    RegularGridInterpolator,
)

from jax_interp import interp1d, interp2d

jax.config.update("jax_enable_x64", True)

# ── helpers ──────────────────────────────────────────────────────────────────

EPS64 = float(np.finfo(np.float64).eps)   # ~2.22e-16
EPS32 = float(np.finfo(np.float32).eps)   # ~1.19e-07


def _ok(err, tol=1e-10):
    return "OK " if err <= tol else "FAIL"


def _bench_jax(fn, *args, n_warmup=5, n_runs=50, **kwargs):
    """Time a JIT-compiled JAX function (exclude compile time)."""
    f = jax.jit(fn, static_argnames=["even", "method", "bc", "oob"]) if callable(fn) else fn
    # warmup
    for _ in range(n_warmup):
        jax.block_until_ready(f(*args, **kwargs))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e6  # microseconds


def _bench_scipy(fn, *args, n_warmup=5, n_runs=50, **kwargs):
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e6  # microseconds


# ── test data ─────────────────────────────────────────────────────────────────

def make_data_1d(n=50, func=None):
    xs = np.linspace(0.0, 2 * np.pi, n)
    if func is None:
        ys = np.sin(xs) + 0.3 * np.cos(3 * xs)
    else:
        ys = func(xs)
    xq = np.linspace(0.05, 2 * np.pi - 0.05, 500)   # interior only
    return xs, ys, xq


def make_data_2d(nx=25, ny=30):
    xs = np.linspace(0.0, 2.0, nx)
    ys = np.linspace(0.0, 1.5, ny)
    vals = np.sin(np.pi * xs[:, None]) * np.cos(np.pi * ys[None, :])
    xq = np.random.default_rng(0).uniform(0.1, 1.9, 200)
    yq = np.random.default_rng(1).uniform(0.1, 1.4, 200)
    return xs, ys, vals, xq, yq


# ─────────────────────────────────────────────────────────────────────────────
# 1-D ACCURACY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_1d_accuracy():
    print("\n" + "=" * 70)
    print("1-D ACCURACY vs scipy  (n=50 knots, 500 interior query points)")
    print("=" * 70)
    print(f"{'case':<38} {'max|err|':>12}  {'status':>6}")
    print("-" * 70)

    # Multiple test functions
    funcs = {
        "sin+cos": lambda x: np.sin(x) + 0.3 * np.cos(3 * x),
        "exp":     lambda x: np.exp(0.5 * x),
        "x^3-2x":  lambda x: x ** 3 - 2 * x,
    }

    rows = []   # collect for return

    for fname, func in funcs.items():
        xs, ys, xq = make_data_1d(n=50, func=func)
        a, b = float(xs[0]), float(xs[-1])
        xs_ep = jnp.array([a, b])   # endpoints for even=True (default)
        ys_j = jnp.array(ys)
        xq_j = jnp.array(xq)

        # 1. nearest
        ref = scipy_interp1d(xs, ys, kind="nearest")(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="nearest"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("nearest", fname, err))
        print(f"  nearest          [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        # 2. linear
        ref = scipy_interp1d(xs, ys, kind="linear")(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="linear"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("linear", fname, err))
        print(f"  linear           [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        # 3. quadratic  [different algorithm - note separately]
        ref_sci = scipy_interp1d(xs, ys, kind="quadratic")(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="quadratic"))
        err_vs_sci = float(np.max(np.abs(got - ref_sci)))
        err_vs_true = float(np.max(np.abs(got - func(xq))))
        rows.append(("quadratic[vs_scipy]", fname, err_vs_sci))
        print(f"  quadratic vs sci [{fname:<10}]  {err_vs_sci:12.3e}  [diff algo]")

        # 4. cubic / natural
        ref = CubicSpline(xs, ys, bc_type="natural")(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="cubic", bc="natural"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("cubic/natural", fname, err))
        print(f"  cubic/natural    [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        # 5. cubic / not-a-knot
        ref = CubicSpline(xs, ys, bc_type="not-a-knot")(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="cubic", bc="not-a-knot"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("cubic/not-a-knot", fname, err))
        print(f"  cubic/not-a-knot [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        # 6. cubic / clamped  (fp = exact derivatives of func)
        if fname == "exp":
            fp_l = 0.5 * np.exp(0.5 * a)   # exp(0.5x) -> 0.5*exp(0.5x)
            fp_r = 0.5 * np.exp(0.5 * b)
        elif fname == "x^3-2x":
            fp_l = 3 * a ** 2 - 2
            fp_r = 3 * b ** 2 - 2
        else:  # sin+cos: deriv = cos(x) - 0.9*sin(3x)
            fp_l = float(np.cos(a) - 0.9 * np.sin(3 * a))
            fp_r = float(np.cos(b) - 0.9 * np.sin(3 * b))
        ref = CubicSpline(xs, ys, bc_type=((1, fp_l), (1, fp_r)))(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="cubic", bc="clamped", fp=(fp_l, fp_r)))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("cubic/clamped", fname, err))
        print(f"  cubic/clamped    [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        # 7. cubic / periodic  (use periodic data)
        xs_per = np.linspace(0.0, 2 * np.pi, 51)
        ys_per = np.sin(xs_per) + 0.3 * np.cos(3 * xs_per)   # periodic: y[0]==y[-1]
        xq_per = np.linspace(0.1, 2 * np.pi - 0.1, 500)
        xs_per_ep = jnp.array([float(xs_per[0]), float(xs_per[-1])])
        ref = CubicSpline(xs_per, ys_per, bc_type="periodic")(xq_per)
        got = np.array(interp1d(xs_per_ep, jnp.array(ys_per), jnp.array(xq_per), method="cubic", bc="periodic"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("cubic/periodic", fname, err))
        print(f"  cubic/periodic   [sin+cos]    {err:12.3e}  {_ok(err, 1e-10)}")

        # 8. bspline
        spl = make_interp_spline(xs, ys, k=3)
        ref = spl(xq)
        got = np.array(interp1d(xs_ep, ys_j, xq_j, method="bspline"))
        err = float(np.max(np.abs(got - ref)))
        rows.append(("bspline", fname, err))
        print(f"  bspline          [{fname:<10}]  {err:12.3e}  {_ok(err, 1e-10)}")

        print()

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-BOUNDS ACCURACY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_oob_accuracy():
    print("=" * 70)
    print("OUT-OF-BOUNDS ACCURACY vs scipy")
    print("=" * 70)
    print(f"{'case':<38} {'max|err|':>12}  {'status':>6}")
    print("-" * 70)

    xs = np.linspace(0.0, 1.0, 30)
    ys = np.sin(3 * xs) + 0.4 * xs ** 2
    a, b = 0.0, 1.0
    xs_ep = jnp.array([a, b])
    ys_j = jnp.array(ys)
    # Query points: mix of inside and outside
    xq = np.linspace(-0.3, 1.3, 200)
    xq_j = jnp.array(xq)

    rows = []

    # oob='clamp' + linear
    ref = scipy_interp1d(xs, ys, kind="linear", bounds_error=False,
                          fill_value=(ys[0], ys[-1]))(xq)
    got = np.array(interp1d(xs_ep, ys_j, xq_j, method="linear", oob="clamp"))
    err = float(np.max(np.abs(got - ref)))
    rows.append(("clamp/linear", err))
    print(f"  clamp + linear   [sin+x^2]    {err:12.3e}  {_ok(err, 1e-10)}")

    # oob='extrapolate' + linear
    ref = scipy_interp1d(xs, ys, kind="linear", bounds_error=False,
                          fill_value="extrapolate")(xq)
    got = np.array(interp1d(xs_ep, ys_j, xq_j, method="linear", oob="extrapolate"))
    err = float(np.max(np.abs(got - ref)))
    rows.append(("extrapolate/linear", err))
    print(f"  extrapolate+linear[sin+x^2]   {err:12.3e}  {_ok(err, 1e-10)}")

    # oob='extrapolate' + cubic (default CubicSpline behavior)
    cs = CubicSpline(xs, ys)  # not-a-knot by default, extrapolates by default
    ref = cs(xq)
    got = np.array(interp1d(xs_ep, ys_j, xq_j, method="cubic", bc="not-a-knot", oob="extrapolate"))
    err = float(np.max(np.abs(got - ref)))
    rows.append(("extrapolate/cubic/nak", err))
    print(f"  extrapolate+cubic/nak [sin+x^2]{err:12.3e}  {_ok(err, 1e-10)}")

    # oob='periodic' + bspline  vs scipy BSpline extrapolate='periodic'
    xs_per = np.linspace(0.0, 2 * np.pi, 40)
    ys_per = np.sin(xs_per)
    xs_per_ep = jnp.array([float(xs_per[0]), float(xs_per[-1])])
    ys_per_j = jnp.array(ys_per)
    xq_per = np.linspace(-np.pi, 3 * np.pi, 300)
    xq_per_j = jnp.array(xq_per)
    spl = make_interp_spline(xs_per, ys_per, k=3)
    ref = spl(xq_per, extrapolate="periodic")
    got = np.array(interp1d(xs_per_ep, ys_per_j, xq_per_j, method="bspline", oob="periodic"))
    err = float(np.max(np.abs(got - ref)))
    rows.append(("periodic/bspline", err))
    print(f"  periodic+bspline [sin]        {err:12.3e}  {_ok(err, 1e-10)}")

    # oob='reflect': no scipy equivalent -> just show values
    got_ref = np.array(interp1d(xs_ep, ys_j, xq_j, method="linear", oob="reflect"))
    print(f"  reflect+linear   [sin+x^2]   (no scipy equivalent)")

    print()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2-D ACCURACY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_2d_accuracy():
    print("=" * 70)
    print("2-D ACCURACY vs scipy  (25x30 grid, 200 query points)")
    print("=" * 70)
    print(f"{'case':<38} {'max|err|':>12}  {'status':>6}")
    print("-" * 70)

    xs, ys_ax, vals, xq, yq = make_data_2d(nx=25, ny=30)
    a_x, b_x = float(xs[0]), float(xs[-1])
    a_y, b_y = float(ys_ax[0]), float(ys_ax[-1])
    xs_ep = jnp.array([a_x, b_x])
    ys_ep = jnp.array([a_y, b_y])
    vals_j = jnp.array(vals)
    xq_j, yq_j = jnp.array(xq), jnp.array(yq)

    rows = []

    # 2D linear (even)
    pts = np.column_stack([xq, yq])
    rgi = RegularGridInterpolator((xs, ys_ax), vals, method="linear")
    ref = rgi(pts)
    got = np.array(interp2d(xs_ep, ys_ep, vals_j, xq_j, yq_j, method="linear"))
    err = float(np.max(np.abs(got - ref)))
    rows.append(("2d/linear", err, 1e-10))
    print(f"  2d linear        [sin*cos]    {err:12.3e}  {_ok(err, 1e-10)}")

    # 2D bspline vs RegularGridInterpolator(method='cubic') >= scipy 1.13
    # NOTE: scipy's make_ndbspl uses gcrotmk (iterative sparse solver, atol=1e-6) internally,
    # so it only has O(1e-5) accuracy.  Our separable direct-solve is more accurate;
    # the expected difference vs scipy is ~1e-5, not machine epsilon.
    rgi_c = RegularGridInterpolator((xs, ys_ax), vals, method="cubic")
    ref_c = rgi_c(pts)
    got_bs = np.array(interp2d(xs_ep, ys_ep, vals_j, xq_j, yq_j, method="bspline"))
    err_bs = float(np.max(np.abs(got_bs - ref_c)))
    rows.append(("2d/bspline vs rgi/cubic", err_bs, 1e-4))
    print(f"  2d bspline vs rgi/cubic       {err_bs:12.3e}  {_ok(err_bs, 1e-4)}  [scipy atol=1e-6]")

    # 2D cubic (even, our 16-coeff bicubic): no direct scipy equivalent, show vs true function
    # O(h^4) error expected; not included in EXACT-match count (no scipy equivalent)
    true_vals = np.sin(np.pi * xq) * np.cos(np.pi * yq)
    got_cub = np.array(interp2d(xs_ep, ys_ep, vals_j, xq_j, yq_j, method="cubic"))
    err_true = float(np.max(np.abs(got_cub - true_vals)))
    # not appended to rows (no scipy comparison to make)
    print(f"  2d cubic vs true function     {err_true:12.3e}  (no scipy equiv, O(h^4) expected)")

    # ── Uneven grid tests: compare even=False on a UNIFORM grid vs even=True ──
    # If both implementations are correct they must agree to machine epsilon.
    print()
    print("  [uneven grid: even=False on same uniform knots must match even=True]")

    xs_full = jnp.array(xs)      # full grid arrays (nx,)
    ys_full = jnp.array(ys_ax)   # full grid arrays (ny,)

    # 2D linear uneven
    got_lin_e = np.array(interp2d(xs_ep,   ys_ep,   vals_j, xq_j, yq_j, method="linear"))
    got_lin_u = np.array(interp2d(xs_full, ys_full, vals_j, xq_j, yq_j, method="linear", even=False))
    err_lin_u = float(np.max(np.abs(got_lin_e - got_lin_u)))
    rows.append(("2d/linear/uneven vs even", err_lin_u, 1e-10))
    print(f"  2d linear uneven vs even      {err_lin_u:12.3e}  {_ok(err_lin_u, 1e-10)}")

    # 2D bspline uneven
    got_bs_e = np.array(interp2d(xs_ep,   ys_ep,   vals_j, xq_j, yq_j, method="bspline"))
    got_bs_u = np.array(interp2d(xs_full, ys_full, vals_j, xq_j, yq_j, method="bspline", even=False))
    err_bs_u = float(np.max(np.abs(got_bs_e - got_bs_u)))
    rows.append(("2d/bspline/uneven vs even", err_bs_u, 1e-9))
    print(f"  2d bspline uneven vs even     {err_bs_u:12.3e}  {_ok(err_bs_u, 1e-9)}")

    # 2D cubic uneven
    got_cub_e = np.array(interp2d(xs_ep,   ys_ep,   vals_j, xq_j, yq_j, method="cubic"))
    got_cub_u = np.array(interp2d(xs_full, ys_full, vals_j, xq_j, yq_j, method="cubic", even=False))
    err_cub_u = float(np.max(np.abs(got_cub_e - got_cub_u)))
    rows.append(("2d/cubic/uneven vs even", err_cub_u, 1e-9))
    print(f"  2d cubic uneven vs even       {err_cub_u:12.3e}  {_ok(err_cub_u, 1e-9)}")

    print()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SPEED BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

def _run_speed():
    print("=" * 70)
    print("SPEED BENCHMARKS  (median over 50 runs, in microseconds)")
    print("All JAX times exclude JIT compilation (post-warmup)")
    print("=" * 70)

    sizes = [
        ("small ", 50,    500),
        ("medium", 500,   5000),
        ("large ", 5000,  50000),
    ]

    print(f"\n{'case':<35} {'size':>6}  {'JAX(us)':>10}  {'scipy(us)':>10}  {'speedup':>8}")
    print("-" * 75)

    for label, n_knots, n_query in sizes:
        xs = np.linspace(0.0, 2 * np.pi, n_knots)
        ys = np.sin(xs) + 0.3 * np.cos(3 * xs)
        xq = np.linspace(0.1, 2 * np.pi - 0.1, n_query)
        a, b = float(xs[0]), float(xs[-1])
        xs_ep = jnp.array([a, b])
        ys_j = jnp.array(ys)
        xq_j = jnp.array(xq)

        # nearest
        jax_fn = jax.jit(interp1d, static_argnames=["even", "method", "bc", "oob"])
        t_jax = _bench_jax(jax_fn, xs_ep, ys_j, xq_j, method="nearest")
        f_sci = scipy_interp1d(xs, ys, kind="nearest")
        t_sci = _bench_scipy(f_sci, xq)
        print(f"  nearest                [{label}]  n={n_knots:>5}  {t_jax:10.1f}  {t_sci:10.1f}  {t_sci/t_jax:8.1f}x")

        # linear
        t_jax = _bench_jax(jax_fn, xs_ep, ys_j, xq_j, method="linear")
        f_sci = scipy_interp1d(xs, ys, kind="linear")
        t_sci = _bench_scipy(f_sci, xq)
        print(f"  linear                 [{label}]  n={n_knots:>5}  {t_jax:10.1f}  {t_sci:10.1f}  {t_sci/t_jax:8.1f}x")

        # cubic/not-a-knot
        t_jax = _bench_jax(jax_fn, xs_ep, ys_j, xq_j, method="cubic", bc="not-a-knot")
        f_sci = CubicSpline(xs, ys, bc_type="not-a-knot")
        t_sci_eval = _bench_scipy(f_sci, xq)
        # scipy CubicSpline: construction + eval
        t_sci_build = _bench_scipy(lambda: CubicSpline(xs, ys, bc_type="not-a-knot"))
        t_sci_total = t_sci_build + t_sci_eval
        print(f"  cubic/not-a-knot       [{label}]  n={n_knots:>5}  {t_jax:10.1f}  {t_sci_total:10.1f}  {t_sci_total/t_jax:8.1f}x  (build+eval)")

        # bspline
        t_jax = _bench_jax(jax_fn, xs_ep, ys_j, xq_j, method="bspline")
        spl = make_interp_spline(xs, ys, k=3)
        t_sci_eval = _bench_scipy(spl, xq)
        t_sci_build = _bench_scipy(lambda: make_interp_spline(xs, ys, k=3))
        t_sci_total = t_sci_build + t_sci_eval
        print(f"  bspline                [{label}]  n={n_knots:>5}  {t_jax:10.1f}  {t_sci_total:10.1f}  {t_sci_total/t_jax:8.1f}x  (build+eval)")

    # 2D benchmarks
    print()
    print(f"{'case':<35} {'size':>12}  {'JAX(us)':>10}  {'scipy(us)':>10}  {'speedup':>8}")
    print("-" * 80)

    sizes_2d = [
        ("small ", 15,  15,  100),
        ("medium", 50,  50,  1000),
        ("large ", 200, 200, 5000),
    ]

    for label, nx, ny, nq in sizes_2d:
        xs = np.linspace(0.0, 2.0, nx)
        ys_ax = np.linspace(0.0, 1.5, ny)
        vals = np.sin(np.pi * xs[:, None]) * np.cos(np.pi * ys_ax[None, :])
        xq = np.random.default_rng(0).uniform(0.1, 1.9, nq)
        yq = np.random.default_rng(1).uniform(0.1, 1.4, nq)
        a_x, b_x = float(xs[0]), float(xs[-1])
        a_y, b_y = float(ys_ax[0]), float(ys_ax[-1])
        xs_ep = jnp.array([a_x, b_x])
        ys_ep = jnp.array([a_y, b_y])
        vals_j = jnp.array(vals)
        xq_j, yq_j = jnp.array(xq), jnp.array(yq)
        pts = np.column_stack([xq, yq])

        jax_fn_2d = jax.jit(interp2d, static_argnames=["even", "method", "bc", "oob"])

        # 2D linear
        t_jax = _bench_jax(jax_fn_2d, xs_ep, ys_ep, vals_j, xq_j, yq_j, method="linear")
        rgi = RegularGridInterpolator((xs, ys_ax), vals, method="linear")
        t_sci = _bench_scipy(rgi, pts)
        size_str = f"{nx}x{ny}+{nq}q"
        print(f"  2d linear              [{label}]  {size_str:>12}  {t_jax:10.1f}  {t_sci:10.1f}  {t_sci/t_jax:8.1f}x")

        # 2D bspline
        t_jax = _bench_jax(jax_fn_2d, xs_ep, ys_ep, vals_j, xq_j, yq_j, method="bspline")
        t_sci_build = _bench_scipy(lambda: RegularGridInterpolator((xs, ys_ax), vals, method="cubic"))
        rgi_c = RegularGridInterpolator((xs, ys_ax), vals, method="cubic")
        t_sci_eval = _bench_scipy(rgi_c, pts)
        t_sci_total = t_sci_build + t_sci_eval
        print(f"  2d bspline vs rgi/cub  [{label}]  {size_str:>12}  {t_jax:10.1f}  {t_sci_total:10.1f}  {t_sci_total/t_jax:8.1f}x  (build+eval)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CORRESPONDENCE TABLE
# ─────────────────────────────────────────────────────────────────────────────

CORRESPONDENCE = [
    # (jax method, jax bc/oob, scipy call, match_type)
    # match_type: 'exact'=machine eps, 'diff_algo'=different algorithm, 'none'=no equiv
    ("nearest",          "",          "interp1d(kind='nearest')  [legacy]",                               "exact"),
    ("linear",           "",          "interp1d(kind='linear')  [legacy] / make_interp_spline(k=1)",      "exact"),
    ("quadratic",        "",          "interp1d(kind='quadratic')  [DIFFERENT ALGORITHM]",                "diff_algo"),
    ("cubic",            "natural",   "CubicSpline(bc_type='natural')",                                   "exact"),
    ("cubic",            "not-a-knot","CubicSpline(bc_type='not-a-knot')  [default]",                    "exact"),
    ("cubic",            "clamped",   "CubicSpline(bc_type=((1,fp_l),(1,fp_r)))",                         "exact"),
    ("cubic",            "periodic",  "CubicSpline(bc_type='periodic')",                                  "exact"),
    ("bspline",          "",          "make_interp_spline(x,y,k=3)  [not-a-knot]",                       "exact"),
    ("oob=clamp",        "linear",    "interp1d(bounds_error=False, fill_value=(y[0],y[-1]))",            "exact"),
    ("oob=extrapolate",  "linear",    "interp1d(fill_value='extrapolate')",                               "exact"),
    ("oob=extrapolate",  "cubic",     "CubicSpline()(xq)  [default extrapolation]",                      "exact"),
    ("oob=periodic",     "bspline",   "make_interp_spline()(xq, extrapolate='periodic')",                 "exact"),
    ("oob=reflect",      "",          "NO SCIPY EQUIVALENT",                                              "none"),
    ("2d/linear",        "",          "RegularGridInterpolator(method='linear')",                         "exact"),
    ("2d/bspline",       "",          "RegularGridInterpolator(method='cubic')  [>= scipy 1.13]",         "approx"),
    ("2d/cubic",         "",          "NO DIRECT SCIPY EQUIVALENT  (16-coeff bicubic)",                   "none"),
]


def _print_table():
    print("\n" + "=" * 90)
    print("CORRESPONDENCE TABLE: jax_interp  vs  scipy.interpolate")
    print("=" * 90)
    print(f"{'jax method':<18} {'bc/oob':<14} {'scipy equivalent':<46} {'match'}")
    print("-" * 90)
    for jax_method, bc_oob, scipy_call, match in CORRESPONDENCE:
        marker = {"exact": "EXACT", "approx": "APPROX", "diff_algo": "DIFF ALGO", "none": "NO EQUIV"}[match]
        print(f"  {jax_method:<16} {bc_oob:<14} {scipy_call:<46} {marker}")
    print("-" * 90)
    print("  EXACT     = matches scipy to machine epsilon on uniform grids")
    print("  APPROX    = same algorithm, differs ~1e-5 because scipy uses iterative solver (atol=1e-6)")
    print("  DIFF ALGO = same interpolation class, different algorithm / end conditions")
    print("  NO EQUIV  = no scipy equivalent")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _print_table()
    rows_1d = _run_1d_accuracy()
    rows_oob = _run_oob_accuracy()
    rows_2d = _run_2d_accuracy()
    _run_speed()

    # summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tol_default = 1e-10
    # Collect (name, err, tol) triples for all tests with scipy equivalents
    all_tests = [(r[0], r[2], tol_default) for r in rows_1d if "vs_sci" not in r[0]]
    all_tests += [(name, err, tol_default) for name, err in rows_oob]
    all_tests += rows_2d  # already (name, err, tol) triples
    n_pass = sum(1 for _, e, t in all_tests if e <= t)
    n_total = len(all_tests)
    print(f"  Tests with EXACT match (err <= per-test tol): {n_pass}/{n_total}")
    for name, err, t in all_tests:
        if err > t:
            print(f"  FAIL: {name}  err={err:.3e}  (tol={t:.0e})")
    if n_pass == n_total:
        print("  All EXACT-match tests passed.")
