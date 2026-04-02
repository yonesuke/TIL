# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax[cpu]",
# ]
# ///
"""
JAX interpolation on evenly or unevenly spaced regular grids.

Public API
----------
interp1d(xs, values, xq, *, even, method, bc, oob, fp)
interp2d(xs, ys_g, values, xq, yq, *, even, method, bc, oob, fp)

xs, ys_g : For even=True, pass just the two endpoints as a 2-element array.
           For even=False, pass the full knot array.
even     : True (default) → uniform grid, uses fast direct index formula.
           False          → non-uniform grid, uses searchsorted.
method   : 'nearest' | 'linear' | 'quadratic' | 'cubic' | 'bspline'
bc       : 'natural' | 'not-a-knot' | 'clamped' | 'periodic'   (spline end conditions)
oob      : 'clamp' | 'extrapolate' | 'periodic' | 'reflect'     (out-of-bounds)
fp       : endpoint first derivatives for bc='clamped'
           1D: (fp_left, fp_right)
           2D: (fpx_left, fpx_right, fpy_left, fpy_right)

JIT usage
---------
    xs = jnp.array([0., 1.])
    f = jax.jit(interp1d, static_argnames=['even', 'method', 'bc', 'oob'])
    y = f(xs, values, xq, method='cubic', bc='not-a-knot')

Notes
-----
- method='cubic'   : solves tridiagonal system for second derivatives M (CubicSpline style)
- method='bspline' : not-a-knot cubic (matches scipy RegularGridInterpolator method='cubic' >= 1.13)
- method='nearest' : not differentiable (grad returns zero)
- bc is ignored for non-cubic/bspline methods
- not-a-knot requires n >= 4 knots
- periodic BC assumes values[0] ≈ values[-1]
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

# ── [A] Grid utilities ─────────────────────────────────────────────────────────


def _cell_index(a: float, h: float, xq: jax.Array, n: int) -> jax.Array:
    """Return cell index i s.t. x[i] <= xq < x[i+1] on a uniform grid.

    n = number of knots.  Valid range: [0, n-2].
    """
    raw = jnp.floor((xq - a) / h).astype(jnp.int32)
    return jnp.clip(raw, 0, n - 2)


def _cell_index_uneven(xs: jax.Array, xq: jax.Array) -> jax.Array:
    """Return cell index i s.t. xs[i] <= xq < xs[i+1] on a non-uniform grid."""
    i = jnp.searchsorted(xs, xq, side="right") - 1
    return jnp.clip(i, 0, xs.shape[0] - 2)


def _apply_oob(a: float, b: float, xq: jax.Array, oob: str) -> jax.Array:
    """Transform xq according to the out-of-bounds rule."""
    L = b - a
    if oob == "clamp":
        return jnp.clip(xq, a, b)
    elif oob == "periodic":
        return a + jnp.mod(xq - a, L)
    elif oob == "reflect":
        t = jnp.mod(xq - a, 2.0 * L)
        return a + jnp.where(t <= L, t, 2.0 * L - t)
    elif oob == "extrapolate":
        return xq
    else:
        raise ValueError(f"Unknown oob={oob!r}. Choose: clamp, extrapolate, periodic, reflect")


# ── [B] Thomas + tridiagonal solvers ──────────────────────────────────────────


def _thomas_141(d: jax.Array) -> jax.Array:
    """Solve A x = d for the uniform 1,4,1 tridiagonal matrix via Thomas scan.

    Forward sweep uses inf sentinel; back-substitution uses zero sentinel.
    """
    def fwd(carry, i):
        b_prev, dp_prev, d_arr = carry
        w = 1.0 / b_prev
        b_i = 4.0 - w
        dp_i = d_arr[i] - w * dp_prev
        return (b_i, dp_i, d_arr.at[i].set(dp_i)), b_i

    inf = jnp.array(jnp.inf, dtype=d.dtype)
    zero = jnp.zeros((), dtype=d.dtype)
    (_, _, d), b = lax.scan(fwd, (inf, zero, d), jnp.arange(d.shape[0]))

    def bwd(x_next, i):
        x_i = (d[i] - x_next) / b[i]
        return x_i, x_i

    _, x = lax.scan(bwd, zero, jnp.arange(d.shape[0]), reverse=True)
    return x


def _thomas_nak(d: jax.Array) -> jax.Array:
    """Solve A x = d for the not-a-knot 1D uniform spline tridiagonal system.

    Matrix structure (size m = n-1):
      Row 0:   [6, 0, 0, ...]      diag[0]=6, du[0]=0
      Row 1..m-2: [1, 4, 1]
      Row m-1: [..., 0, 6]         diag[m-1]=6, dl[m-1]=0

    Uses lax.scan for JIT-compatibility.
    """
    m = d.shape[0]

    def fwd(carry, i):
        b_prev, dp_prev, d_arr = carry
        # dl[m-1]=0 → w=0 at last row; inf sentinel → w=0 at first row
        w = jnp.where(i == m - 1, 0.0, 1.0 / b_prev)
        # du[0]=0 → at row 1, the upper-diagonal coefficient from row 0 is 0
        du_prev = jnp.where(i == 1, 0.0, 1.0)
        diag_i = jnp.where((i == 0) | (i == m - 1), 6.0, 4.0)
        b_i = diag_i - w * du_prev
        dp_i = d_arr[i] - w * dp_prev
        return (b_i, dp_i, d_arr.at[i].set(dp_i)), b_i

    inf = jnp.array(jnp.inf, dtype=d.dtype)
    zero = jnp.zeros((), dtype=d.dtype)
    (_, _, d), b = lax.scan(fwd, (inf, zero, d), jnp.arange(m))

    def bwd(x_next, i):
        # du[0]=0 → at row 0, upper-diagonal is 0 → x[0] = d[0]/b[0]
        du_i = jnp.where(i == 0, 0.0, 1.0)
        x_i = (d[i] - du_i * x_next) / b[i]
        return x_i, x_i

    _, x = lax.scan(bwd, zero, jnp.arange(m), reverse=True)
    return x


def _tridiag_solve(dl: jax.Array, diag: jax.Array, du: jax.Array, rhs: jax.Array) -> jax.Array:
    """Thin wrapper around lax.linalg.tridiagonal_solve for 1D systems.

    Convention: dl[0] and du[-1] are ignored by the XLA kernel.
    """
    return lax.linalg.tridiagonal_solve(dl, diag, du, rhs[:, None])[:, 0]


# ── [C] Uniform cubic spline coefficient builders ─────────────────────────────


def _spline_natural(ys: jax.Array, h: float) -> jax.Array:
    """Natural cubic spline: M[0] = M[n] = 0.  Returns M of shape (n+1,)."""
    rhs = 6.0 * (ys[2:] - 2.0 * ys[1:-1] + ys[:-2]) / h ** 2
    M_inner = _thomas_141(rhs)
    return jnp.concatenate([jnp.zeros(1, dtype=ys.dtype), M_inner, jnp.zeros(1, dtype=ys.dtype)])


def _spline_notaknot(ys: jax.Array, h: float) -> jax.Array:
    """Not-a-knot cubic spline.  Returns M of shape (n+1,)."""
    n = ys.shape[0] - 1
    m = n - 1
    rhs = 6.0 * (ys[2:] - 2.0 * ys[1:-1] + ys[:-2]) / h ** 2  # shape (m,)
    M_inner = _thomas_nak(rhs)
    M0 = 2.0 * M_inner[0] - M_inner[1]
    Mn = 2.0 * M_inner[-1] - M_inner[-2]
    return jnp.concatenate([M0[None], M_inner, Mn[None]])


def _spline_clamped(ys: jax.Array, h: float, fp_left: float, fp_right: float) -> jax.Array:
    """Clamped cubic spline: S'(x_0) = fp_left, S'(x_n) = fp_right.

    Full (n+1)×(n+1) system with unknowns M[0]..M[n].
    """
    n = ys.shape[0] - 1
    m = n + 1

    rhs_int = 6.0 * (ys[2:] - 2.0 * ys[1:-1] + ys[:-2]) / h ** 2
    rhs_0 = (6.0 / h) * ((ys[1] - ys[0]) / h - fp_left)
    rhs_n = (6.0 / h) * (fp_right - (ys[-1] - ys[-2]) / h)
    rhs = jnp.concatenate([rhs_0[None], rhs_int, rhs_n[None]])

    diag = jnp.full(m, 4.0, dtype=ys.dtype).at[0].set(2.0).at[-1].set(2.0)
    dl = jnp.ones(m, dtype=ys.dtype).at[0].set(0.0)
    du = jnp.ones(m, dtype=ys.dtype).at[-1].set(0.0)

    return _tridiag_solve(dl, diag, du, rhs)


def _spline_periodic(ys: jax.Array, h: float) -> jax.Array:
    """Periodic cubic spline: M[0] = M[n].  Uses Sherman-Morrison."""
    n = ys.shape[0] - 1

    y_period = ys[0:n]
    y_prev = jnp.roll(y_period, 1)
    y_next = jnp.roll(y_period, -1)
    rhs = 6.0 * (y_next - 2.0 * y_period + y_prev) / h ** 2

    gamma = jnp.array(-4.0, dtype=ys.dtype)
    diag_mod = jnp.full(n, 4.0, dtype=ys.dtype).at[0].set(4.0 - gamma).at[-1].set(4.0 - 1.0 / gamma)
    dl = jnp.ones(n, dtype=ys.dtype).at[0].set(0.0)
    du = jnp.ones(n, dtype=ys.dtype).at[-1].set(0.0)

    u = jnp.zeros(n, dtype=ys.dtype).at[0].set(gamma).at[-1].set(1.0)

    x1 = _tridiag_solve(dl, diag_mod, du, rhs)
    x2 = _tridiag_solve(dl, diag_mod, du, u)

    inv_gamma = 1.0 / gamma
    v_x1 = x1[0] + inv_gamma * x1[-1]
    v_x2 = x2[0] + inv_gamma * x2[-1]

    M_inner = x1 - (v_x1 / (1.0 + v_x2)) * x2
    return jnp.concatenate([M_inner, M_inner[0:1]])


def _spline_M(ys: jax.Array, h: float, bc: str, fp_left: float = 0.0, fp_right: float = 0.0) -> jax.Array:
    """Dispatch to uniform spline builder."""
    if bc == "natural":
        return _spline_natural(ys, h)
    elif bc == "not-a-knot":
        return _spline_notaknot(ys, h)
    elif bc == "clamped":
        return _spline_clamped(ys, h, fp_left, fp_right)
    elif bc == "periodic":
        return _spline_periodic(ys, h)
    else:
        raise ValueError(f"Unknown bc={bc!r}. Choose: natural, not-a-knot, clamped, periodic")


# ── [D] B-spline (uniform, alias for not-a-knot) ──────────────────────────────


def _bspline_coeffs(ys: jax.Array, h: float) -> jax.Array:
    """Not-a-knot M values; matches scipy's cubic B-spline collocation (uniform grid)."""
    return _spline_notaknot(ys, h)


# ── [E] Uneven cubic spline builders ──────────────────────────────────────────


def _spline_natural_uneven(ys: jax.Array, xs: jax.Array) -> jax.Array:
    """Natural cubic spline on non-uniform knots.  Returns M of shape (n+1,)."""
    h = xs[1:] - xs[:-1]                                                       # (n,)
    rhs = 6.0 * ((ys[2:] - ys[1:-1]) / h[1:] - (ys[1:-1] - ys[:-2]) / h[:-1])  # (n-1,)
    dl = h[:-1]                                                                 # XLA ignores dl[0]
    diag = 2.0 * (h[:-1] + h[1:])
    du = h[1:]                                                                  # XLA ignores du[-1]
    M_inner = _tridiag_solve(dl, diag, du, rhs)
    return jnp.concatenate([jnp.zeros(1, dtype=ys.dtype), M_inner, jnp.zeros(1, dtype=ys.dtype)])


def _spline_notaknot_uneven(ys: jax.Array, xs: jax.Array) -> jax.Array:
    """Not-a-knot cubic spline on non-uniform knots.  Requires n >= 3 intervals.

    Not-a-knot condition: S'''(x) continuous at x[1] and x[n-1].
    Substituting M[0] = ((h0+h1)*M1 - h0*M2) / h1 into row 1 modifies the
    first tridiagonal row; similarly for the last row.  Recovers M[0] and M[n]
    after solving the reduced (n-1)×(n-1) interior system.
    """
    h = xs[1:] - xs[:-1]                                                          # (n,)
    rhs = 6.0 * ((ys[2:] - ys[1:-1]) / h[1:] - (ys[1:-1] - ys[:-2]) / h[:-1])   # (n-1,)

    dl = h[:-1]
    diag = 2.0 * (h[:-1] + h[1:])
    du = h[1:]

    # Not-a-knot modification of first row
    diag = diag.at[0].set((h[0] + h[1]) * (h[0] + 2.0 * h[1]) / h[1])
    du = du.at[0].set((h[1] ** 2 - h[0] ** 2) / h[1])

    # Not-a-knot modification of last row
    diag = diag.at[-1].set((h[-2] + h[-1]) * (2.0 * h[-2] + h[-1]) / h[-2])
    dl = dl.at[-1].set((h[-2] ** 2 - h[-1] ** 2) / h[-2])

    M_inner = _tridiag_solve(dl, diag, du, rhs)

    M0 = ((h[0] + h[1]) * M_inner[0] - h[0] * M_inner[1]) / h[1]
    Mn = ((h[-2] + h[-1]) * M_inner[-1] - h[-1] * M_inner[-2]) / h[-2]
    return jnp.concatenate([M0[None], M_inner, Mn[None]])


def _spline_clamped_uneven(ys: jax.Array, xs: jax.Array, fp_left: float, fp_right: float) -> jax.Array:
    """Clamped cubic spline on non-uniform knots.  Returns M of shape (n+1,)."""
    h = xs[1:] - xs[:-1]   # (n,)
    m = h.shape[0] + 1      # n+1

    rhs_int = 6.0 * ((ys[2:] - ys[1:-1]) / h[1:] - (ys[1:-1] - ys[:-2]) / h[:-1])
    rhs_0 = (6.0 / h[0]) * ((ys[1] - ys[0]) / h[0] - fp_left)
    rhs_n = (6.0 / h[-1]) * (fp_right - (ys[-1] - ys[-2]) / h[-1])
    rhs = jnp.concatenate([rhs_0[None], rhs_int, rhs_n[None]])

    diag = jnp.concatenate([
        jnp.array([2.0], dtype=ys.dtype),
        2.0 * (h[:-1] + h[1:]),
        jnp.array([2.0], dtype=ys.dtype),
    ])
    # dl[k] = h[k-1] for k=1..n,  dl[0]=0 (ignored by XLA)
    dl = jnp.concatenate([jnp.zeros(1, dtype=ys.dtype), h])
    # du[0]=1, du[k]=h[k] for k=1..n-1, du[n]=0 (ignored by XLA)
    du = jnp.concatenate([jnp.array([1.0], dtype=ys.dtype), h[1:], jnp.zeros(1, dtype=ys.dtype)])

    return _tridiag_solve(dl, diag, du, rhs)


def _spline_periodic_uneven(ys: jax.Array, xs: jax.Array) -> jax.Array:
    """Periodic cubic spline on non-uniform knots.  M[n] = M[0].

    Uses Sherman-Morrison to solve the cyclic tridiagonal system.
    """
    h = xs[1:] - xs[:-1]   # (n,)
    n = h.shape[0]

    y_period = ys[:n]
    y_next = jnp.concatenate([y_period[1:], y_period[0:1]])
    y_prev = jnp.concatenate([y_period[-1:], y_period[:-1]])
    h_prev = jnp.concatenate([h[-1:], h[:-1]])   # h[k-1 mod n]
    rhs = 6.0 * ((y_next - y_period) / h - (y_period - y_prev) / h_prev)

    diag = 2.0 * (h_prev + h)
    dl = jnp.concatenate([jnp.zeros(1, dtype=ys.dtype), h[:-1]])   # dl[0] ignored
    du = jnp.concatenate([h[:-1], jnp.zeros(1, dtype=ys.dtype)])   # du[-1] ignored

    # Sherman-Morrison: cyclic corners A[0,n-1] = A[n-1,0] = h[n-1]
    c = h[-1]
    gamma = -c
    diag_mod = diag.at[0].add(-gamma).at[-1].add(c)

    u_vec = jnp.zeros(n, dtype=ys.dtype).at[0].set(gamma).at[-1].set(c)

    x1 = _tridiag_solve(dl, diag_mod, du, rhs)
    x2 = _tridiag_solve(dl, diag_mod, du, u_vec)

    # v = [1, 0,...,0, -1]  (since c/gamma = -1)
    v_x1 = x1[0] - x1[-1]
    v_x2 = x2[0] - x2[-1]
    M_inner = x1 - (v_x1 / (1.0 + v_x2)) * x2
    return jnp.concatenate([M_inner, M_inner[0:1]])


def _spline_M_uneven(
    ys: jax.Array, xs: jax.Array, bc: str, fp_left: float = 0.0, fp_right: float = 0.0
) -> jax.Array:
    """Dispatch to uneven spline builder."""
    if bc == "natural":
        return _spline_natural_uneven(ys, xs)
    elif bc == "not-a-knot":
        return _spline_notaknot_uneven(ys, xs)
    elif bc == "clamped":
        return _spline_clamped_uneven(ys, xs, fp_left, fp_right)
    elif bc == "periodic":
        return _spline_periodic_uneven(ys, xs)
    else:
        raise ValueError(f"Unknown bc={bc!r}. Choose: natural, not-a-knot, clamped, periodic")


# ── [F] 1D interpolation kernels ──────────────────────────────────────────────
# nearest / linear / cubic work identically for even and uneven — the caller
# pre-computes i (cell index) and t (local coordinate ∈ [0,1]).
# quadratic and cubic-uneven have separate variants.


def _interp1d_nearest(ys: jax.Array, i: jax.Array, t: jax.Array) -> jax.Array:
    n = ys.shape[0]
    j = jnp.clip(jnp.where(t >= 0.5, i + 1, i), 0, n - 1)
    return ys[j]


def _interp1d_linear(ys: jax.Array, i: jax.Array, t: jax.Array) -> jax.Array:
    return ys[i] * (1.0 - t) + ys[i + 1] * t


def _interp1d_quadratic(ys: jax.Array, i: jax.Array, t: jax.Array) -> jax.Array:
    """3-point Lagrange on a uniform grid.  Stencil: i0 = clip(i-1, 0, n-3)."""
    n = ys.shape[0]
    i0 = jnp.clip(i - 1, 0, n - 3)
    u = t + (i - i0).astype(ys.dtype)  # u ∈ [0, 2]
    y0, y1, y2 = ys[i0], ys[i0 + 1], ys[i0 + 2]
    L0 = (u - 1.0) * (u - 2.0) / 2.0
    L1 = -u * (u - 2.0)
    L2 = u * (u - 1.0) / 2.0
    return L0 * y0 + L1 * y1 + L2 * y2


def _interp1d_quadratic_uneven(
    ys: jax.Array, xs: jax.Array, xq2: jax.Array, i: jax.Array
) -> jax.Array:
    """3-point Lagrange on a non-uniform grid using actual knot coordinates."""
    n = ys.shape[0]
    i0 = jnp.clip(i - 1, 0, n - 3)
    x0, x1, x2 = xs[i0], xs[i0 + 1], xs[i0 + 2]
    xq = xq2
    L0 = (xq - x1) * (xq - x2) / ((x0 - x1) * (x0 - x2))
    L1 = (xq - x0) * (xq - x2) / ((x1 - x0) * (x1 - x2))
    L2 = (xq - x0) * (xq - x1) / ((x2 - x0) * (x2 - x1))
    return L0 * ys[i0] + L1 * ys[i0 + 1] + L2 * ys[i0 + 2]


def _interp1d_cubic(ys: jax.Array, i: jax.Array, t: jax.Array, M: jax.Array, h: float) -> jax.Array:
    """Cubic spline evaluation from M (second derivatives) on a uniform grid.

    S(t) = y_i*(1-t) + y_{i+1}*t - h²/6 * t*(1-t) * [M_i*(2-t) + M_{i+1}*(1+t)]
    """
    return (
        ys[i] * (1.0 - t)
        + ys[i + 1] * t
        - (h ** 2 / 6.0) * t * (1.0 - t) * (M[i] * (2.0 - t) + M[i + 1] * (1.0 + t))
    )


def _interp1d_cubic_uneven(
    ys: jax.Array, xs: jax.Array, i: jax.Array, t: jax.Array, M: jax.Array
) -> jax.Array:
    """Cubic spline evaluation from M on a non-uniform grid.  h_i = xs[i+1]-xs[i]."""
    h_i = xs[i + 1] - xs[i]
    return (
        ys[i] * (1.0 - t)
        + ys[i + 1] * t
        - (h_i ** 2 / 6.0) * t * (1.0 - t) * (M[i] * (2.0 - t) + M[i + 1] * (1.0 + t))
    )


# ── [G] interp1d (public) ──────────────────────────────────────────────────────


def interp1d(
    xs: jax.Array,
    values: jax.Array,
    xq: jax.Array,
    *,
    even: bool = True,
    method: str = "linear",
    bc: str = "not-a-knot",
    oob: str = "clamp",
    fp: tuple = (0.0, 0.0),
) -> jax.Array:
    """Interpolate on a 1D grid.

    Parameters
    ----------
    xs     : array — for even=True, pass [a, b] (two endpoints); for even=False,
             pass the full sorted knot array of shape (n,).
    values : array of shape (n,) — function values at knots.
    xq     : array — query points.
    even   : True (default) → uniform grid; False → non-uniform grid.
    method : 'nearest' | 'linear' | 'quadratic' | 'cubic' | 'bspline'
    bc     : 'natural' | 'not-a-knot' | 'clamped' | 'periodic'
    oob    : 'clamp' | 'extrapolate' | 'periodic' | 'reflect'
    fp     : (fp_left, fp_right) — endpoint derivatives for bc='clamped'

    Notes
    -----
    JIT: jax.jit(interp1d, static_argnames=['even', 'method', 'bc', 'oob'])
    """
    n = values.shape[0]
    a, b = xs[0], xs[-1]

    xq2 = _apply_oob(a, b, xq, oob)

    if even:
        h = (b - a) / (n - 1)
        i = _cell_index(a, h, xq2, n)
        t = (xq2 - (a + i.astype(values.dtype) * h)) / h
    else:
        i = _cell_index_uneven(xs, xq2)
        h_i = xs[i + 1] - xs[i]
        t = (xq2 - xs[i]) / h_i

    if method == "nearest":
        return _interp1d_nearest(values, i, t)

    elif method == "linear":
        return _interp1d_linear(values, i, t)

    elif method == "quadratic":
        if even:
            return _interp1d_quadratic(values, i, t)
        else:
            return _interp1d_quadratic_uneven(values, xs, xq2, i)

    elif method == "cubic":
        if even:
            h = (b - a) / (n - 1)
            M = _spline_M(values, h, bc, fp[0], fp[1])
            return _interp1d_cubic(values, i, t, M, h)
        else:
            M = _spline_M_uneven(values, xs, bc, fp[0], fp[1])
            return _interp1d_cubic_uneven(values, xs, i, t, M)

    elif method == "bspline":
        if even:
            h = (b - a) / (n - 1)
            M = _bspline_coeffs(values, h)
            return _interp1d_cubic(values, i, t, M, h)
        else:
            # Not-a-knot cubic on uneven grid (matches scipy's B-spline for uniform grids)
            M = _spline_notaknot_uneven(values, xs)
            return _interp1d_cubic_uneven(values, xs, i, t, M)

    else:
        raise ValueError(f"Unknown method={method!r}. Choose: nearest, linear, quadratic, cubic, bspline")


# ── [H] 2D helpers ─────────────────────────────────────────────────────────────


def _spline_deriv_at_nodes(ys: jax.Array, M: jax.Array, h: float) -> jax.Array:
    """Compute S'(x_i) at each knot (uniform grid).

    S'(x_i) = (y[i+1]-y[i])/h - h/6*(2*M[i]+M[i+1])   for i = 0..n-1
    S'(x_n) = (y[n]-y[n-1])/h + h/6*(M[n-1]+2*M[n])
    """
    d_int = (ys[1:] - ys[:-1]) / h - (h / 6.0) * (2.0 * M[:-1] + M[1:])
    d_last = (ys[-1] - ys[-2]) / h + (h / 6.0) * (M[-2] + 2.0 * M[-1])
    return jnp.concatenate([d_int, d_last[None]])


def _spline_deriv_at_nodes_uneven(ys: jax.Array, xs: jax.Array, M: jax.Array) -> jax.Array:
    """Compute S'(x_i) at each knot (non-uniform grid).

    S'(x_i) = (y[i+1]-y[i])/h[i] - h[i]/6*(2*M[i]+M[i+1])  for i=0..n-1
    S'(x_n) = (y[n]-y[n-1])/h[n-1] + h[n-1]/6*(M[n-1]+2*M[n])
    """
    h = xs[1:] - xs[:-1]   # (n,)
    d_int = (ys[1:] - ys[:-1]) / h - (h / 6.0) * (2.0 * M[:-1] + M[1:])
    d_last = (ys[-1] - ys[-2]) / h[-1] + (h[-1] / 6.0) * (M[-2] + 2.0 * M[-1])
    return jnp.concatenate([d_int, d_last[None]])


# ── [I] 2D interpolation kernels ──────────────────────────────────────────────

# Bicubic 16×16 coefficient matrix from Numerical Recipes §3.6
_A_BICUBIC = jnp.array([
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
    [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
    [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
    [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
    [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
    [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
    [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1],
], dtype=jnp.float64)


def _interp2d_nearest(values, ix, iy, t, s):
    nx, ny = values.shape
    jx = jnp.clip(jnp.where(t >= 0.5, ix + 1, ix), 0, nx - 1)
    jy = jnp.clip(jnp.where(s >= 0.5, iy + 1, iy), 0, ny - 1)
    return values[jx, jy]


def _interp2d_linear(values, ix, iy, t, s):
    f00 = values[ix,     iy    ]
    f10 = values[ix + 1, iy    ]
    f01 = values[ix,     iy + 1]
    f11 = values[ix + 1, iy + 1]
    return f00 * (1.0 - t) * (1.0 - s) + f10 * t * (1.0 - s) + f01 * (1.0 - t) * s + f11 * t * s


def _interp2d_quadratic(values, ix, iy, t, s):
    """Separable 3-point Lagrange quadratic (uniform grid)."""
    nx, ny = values.shape
    ix0 = jnp.clip(ix - 1, 0, nx - 3)
    iy0 = jnp.clip(iy - 1, 0, ny - 3)
    ux = t + (ix - ix0).astype(values.dtype)
    uy = s + (iy - iy0).astype(values.dtype)

    Lx0 = (ux - 1.0) * (ux - 2.0) / 2.0
    Lx1 = -ux * (ux - 2.0)
    Lx2 = ux * (ux - 1.0) / 2.0
    Ly0 = (uy - 1.0) * (uy - 2.0) / 2.0
    Ly1 = -uy * (uy - 2.0)
    Ly2 = uy * (uy - 1.0) / 2.0

    gx0 = Lx0 * values[ix0,     iy0] + Lx1 * values[ix0,     iy0 + 1] + Lx2 * values[ix0,     iy0 + 2]
    gx1 = Lx0 * values[ix0 + 1, iy0] + Lx1 * values[ix0 + 1, iy0 + 1] + Lx2 * values[ix0 + 1, iy0 + 2]
    gx2 = Lx0 * values[ix0 + 2, iy0] + Lx1 * values[ix0 + 2, iy0 + 1] + Lx2 * values[ix0 + 2, iy0 + 2]
    return Ly0 * gx0 + Ly1 * gx1 + Ly2 * gx2


def _interp2d_quadratic_uneven(values, xs_g, ys_g, ix, iy, xq2, yq2):
    """Separable 3-point Lagrange quadratic (non-uniform grid)."""
    nx, ny = values.shape
    ix0 = jnp.clip(ix - 1, 0, nx - 3)
    iy0 = jnp.clip(iy - 1, 0, ny - 3)

    x0, x1, x2 = xs_g[ix0], xs_g[ix0 + 1], xs_g[ix0 + 2]
    y0, y1, y2 = ys_g[iy0], ys_g[iy0 + 1], ys_g[iy0 + 2]
    xq, yq = xq2, yq2

    Lx0 = (xq - x1) * (xq - x2) / ((x0 - x1) * (x0 - x2))
    Lx1 = (xq - x0) * (xq - x2) / ((x1 - x0) * (x1 - x2))
    Lx2 = (xq - x0) * (xq - x1) / ((x2 - x0) * (x2 - x1))
    Ly0 = (yq - y1) * (yq - y2) / ((y0 - y1) * (y0 - y2))
    Ly1 = (yq - y0) * (yq - y2) / ((y1 - y0) * (y1 - y2))
    Ly2 = (yq - y0) * (yq - y1) / ((y2 - y0) * (y2 - y1))

    gx0 = Lx0 * values[ix0,     iy0] + Lx1 * values[ix0,     iy0 + 1] + Lx2 * values[ix0,     iy0 + 2]
    gx1 = Lx0 * values[ix0 + 1, iy0] + Lx1 * values[ix0 + 1, iy0 + 1] + Lx2 * values[ix0 + 1, iy0 + 2]
    gx2 = Lx0 * values[ix0 + 2, iy0] + Lx1 * values[ix0 + 2, iy0 + 1] + Lx2 * values[ix0 + 2, iy0 + 2]
    return Ly0 * gx0 + Ly1 * gx1 + Ly2 * gx2


def _interp2d_cubic_scalar(values, fx, fy, fxy, ix, iy, t, s, hx, hy):
    """Bicubic interpolation at a SINGLE query point (uniform hx, hy).

    Requires precomputed node derivatives fx, fy, fxy (all shape (nx, ny)).
    """
    F = jnp.stack([
        values[ix,     iy    ], values[ix + 1, iy    ], values[ix,     iy + 1], values[ix + 1, iy + 1],
        fx[ix,     iy    ] * hx, fx[ix + 1, iy    ] * hx, fx[ix,     iy + 1] * hx, fx[ix + 1, iy + 1] * hx,
        fy[ix,     iy    ] * hy, fy[ix + 1, iy    ] * hy, fy[ix,     iy + 1] * hy, fy[ix + 1, iy + 1] * hy,
        fxy[ix,     iy    ] * hx * hy, fxy[ix + 1, iy    ] * hx * hy,
        fxy[ix,     iy + 1] * hx * hy, fxy[ix + 1, iy + 1] * hx * hy,
    ]).astype(jnp.float64)
    coef = (_A_BICUBIC @ F).reshape(4, 4)
    tv = jnp.stack([jnp.ones_like(t), t, t ** 2, t ** 3]).astype(jnp.float64)
    sv = jnp.stack([jnp.ones_like(s), s, s ** 2, s ** 3]).astype(jnp.float64)
    return jnp.einsum("ij,i,j->", coef, tv, sv).astype(values.dtype)


def _interp2d_cubic_scalar_uneven(values, fx, fy, fxy, ix, iy, t, s, hx, hy):
    """Bicubic interpolation at a SINGLE query point (non-uniform, per-cell h).

    hx and hy are the cell widths for this specific query point, passed
    explicitly to avoid dynamic array indexing inside vmap.
    """
    F = jnp.stack([
        values[ix,     iy    ], values[ix + 1, iy    ], values[ix,     iy + 1], values[ix + 1, iy + 1],
        fx[ix,     iy    ] * hx, fx[ix + 1, iy    ] * hx, fx[ix,     iy + 1] * hx, fx[ix + 1, iy + 1] * hx,
        fy[ix,     iy    ] * hy, fy[ix + 1, iy    ] * hy, fy[ix,     iy + 1] * hy, fy[ix + 1, iy + 1] * hy,
        fxy[ix,     iy    ] * hx * hy, fxy[ix + 1, iy    ] * hx * hy,
        fxy[ix,     iy + 1] * hx * hy, fxy[ix + 1, iy + 1] * hx * hy,
    ]).astype(jnp.float64)
    coef = (_A_BICUBIC @ F).reshape(4, 4)
    tv = jnp.stack([jnp.ones_like(t), t, t ** 2, t ** 3]).astype(jnp.float64)
    sv = jnp.stack([jnp.ones_like(s), s, s ** 2, s ** 3]).astype(jnp.float64)
    return jnp.einsum("ij,i,j->", coef, tv, sv).astype(values.dtype)


def _interp2d_bspline_scalar(values, My, hy, ix, iy, t, s, hx):
    """Separable not-a-knot tensor-product bspline (uniform hx, hy).

    My[k,:] = not-a-knot M values for row k (y-direction), shape (nx, ny).
    """
    # Step 1: evaluate y-spline for each x-row → g[k]
    g = jax.vmap(lambda row, M_row: _interp1d_cubic(row, iy, s, M_row, hy))(values, My)
    # Step 2: fit not-a-knot spline through g, evaluate at t
    Mg = _spline_notaknot(g, hx)
    return _interp1d_cubic(g, ix, t, Mg, hx)


def _interp2d_bspline_scalar_uneven(values, My, xs_g, ys_g, ix, iy, t, s):
    """Separable not-a-knot tensor-product bspline (non-uniform grids).

    My[k,:] = not-a-knot M values for row k (y-direction), shape (nx, ny).
    """
    # Step 1: evaluate y-spline (uneven) for each x-row → g[k]
    g = jax.vmap(lambda row, M_row: _interp1d_cubic_uneven(row, ys_g, iy, s, M_row))(values, My)
    # Step 2: fit not-a-knot uneven spline through g, evaluate at t
    Mg = _spline_notaknot_uneven(g, xs_g)
    return _interp1d_cubic_uneven(g, xs_g, ix, t, Mg)


# ── [J] interp2d (public) ──────────────────────────────────────────────────────


def interp2d(
    xs: jax.Array,
    ys_g: jax.Array,
    values: jax.Array,
    xq: jax.Array,
    yq: jax.Array,
    *,
    even: bool = True,
    method: str = "linear",
    bc: str = "not-a-knot",
    oob: str = "clamp",
    fp: tuple = (0.0, 0.0, 0.0, 0.0),
) -> jax.Array:
    """Interpolate on a 2D grid.

    Parameters
    ----------
    xs, ys_g : For even=True, pass the two endpoints [a, b].
               For even=False, pass the full sorted knot arrays.
    values   : array of shape (nx, ny) — values[i, j] at (xs[i], ys_g[j]).
    xq, yq   : arrays of shape (nq,) — query points.
    even     : True (default) → uniform grid; False → non-uniform grid.
    method   : 'nearest' | 'linear' | 'quadratic' | 'cubic' | 'bspline'
    bc       : spline end conditions (for cubic/bspline only)
    oob      : out-of-bounds handling
    fp       : (fpx_left, fpx_right, fpy_left, fpy_right) for bc='clamped'

    Notes
    -----
    JIT: jax.jit(interp2d, static_argnames=['even', 'method', 'bc', 'oob'])
    """
    nx, ny = values.shape
    ax, bx = xs[0], xs[-1]
    ay, by = ys_g[0], ys_g[-1]

    xq2 = _apply_oob(ax, bx, xq, oob)
    yq2 = _apply_oob(ay, by, yq, oob)

    if even:
        hx = (bx - ax) / (nx - 1)
        hy = (by - ay) / (ny - 1)
        ix = _cell_index(ax, hx, xq2, nx)
        iy = _cell_index(ay, hy, yq2, ny)
        t = (xq2 - (ax + ix.astype(values.dtype) * hx)) / hx
        s = (yq2 - (ay + iy.astype(values.dtype) * hy)) / hy
    else:
        ix = _cell_index_uneven(xs, xq2)
        iy = _cell_index_uneven(ys_g, yq2)
        hx_arr = xs[ix + 1] - xs[ix]
        hy_arr = ys_g[iy + 1] - ys_g[iy]
        t = (xq2 - xs[ix]) / hx_arr
        s = (yq2 - ys_g[iy]) / hy_arr

    if method == "nearest":
        return _interp2d_nearest(values, ix, iy, t, s)

    elif method == "linear":
        return _interp2d_linear(values, ix, iy, t, s)

    elif method == "quadratic":
        if even:
            return _interp2d_quadratic(values, ix, iy, t, s)
        else:
            return jax.vmap(
                lambda ix_i, iy_i, xq_i, yq_i: _interp2d_quadratic_uneven(
                    values, xs, ys_g, ix_i, iy_i, xq_i, yq_i
                )
            )(ix, iy, xq2, yq2)

    elif method == "cubic":
        if even:
            hx = (bx - ax) / (nx - 1)
            hy = (by - ay) / (ny - 1)

            # x-direction: fit spline along columns (axis 0) with hx step
            def fit_x_col(col): return _spline_M(col, hx, bc, fp[0], fp[1])
            # y-direction: fit spline along rows (axis 1) with hy step
            def fit_y_row(row): return _spline_M(row, hy, bc, fp[2], fp[3])

            Mx = jax.vmap(fit_x_col)(values.T).T      # (nx, ny): x-direction M values
            My = jax.vmap(fit_y_row)(values)           # (nx, ny): y-direction M values

            # Node first derivatives
            fx = jax.vmap(lambda col, M: _spline_deriv_at_nodes(col, M, hx))(values.T, Mx.T).T
            fy = jax.vmap(lambda row, M: _spline_deriv_at_nodes(row, M, hy))(values, My)

            # Cross derivatives: x-spline through fy (fy varies along x at fixed y)
            Mxy = jax.vmap(fit_x_col)(fy.T).T
            fxy = jax.vmap(lambda col, M: _spline_deriv_at_nodes(col, M, hx))(fy.T, Mxy.T).T

            return jax.vmap(
                lambda ix_i, iy_i, t_i, s_i: _interp2d_cubic_scalar(
                    values, fx, fy, fxy, ix_i, iy_i, t_i, s_i, hx, hy
                )
            )(ix, iy, t, s)

        else:
            # x-direction: fit along columns (xs grid)
            def fit_x_col(col): return _spline_M_uneven(col, xs, bc, fp[0], fp[1])
            # y-direction: fit along rows (ys_g grid)
            def fit_y_row(row): return _spline_M_uneven(row, ys_g, bc, fp[2], fp[3])

            Mx = jax.vmap(fit_x_col)(values.T).T      # (nx, ny)
            My = jax.vmap(fit_y_row)(values)           # (nx, ny)

            fx = jax.vmap(lambda col, M: _spline_deriv_at_nodes_uneven(col, xs, M))(values.T, Mx.T).T
            fy = jax.vmap(lambda row, M: _spline_deriv_at_nodes_uneven(row, ys_g, M))(values, My)

            Mxy = jax.vmap(fit_x_col)(fy.T).T
            fxy = jax.vmap(lambda col, M: _spline_deriv_at_nodes_uneven(col, xs, M))(fy.T, Mxy.T).T

            return jax.vmap(
                lambda ix_i, iy_i, t_i, s_i, hx_i, hy_i: _interp2d_cubic_scalar_uneven(
                    values, fx, fy, fxy, ix_i, iy_i, t_i, s_i, hx_i, hy_i
                )
            )(ix, iy, t, s, hx_arr, hy_arr)

    elif method == "bspline":
        if even:
            hx = (bx - ax) / (nx - 1)
            hy = (by - ay) / (ny - 1)
            My = jax.vmap(lambda row: _spline_notaknot(row, hy))(values)
            return jax.vmap(
                lambda ix_i, iy_i, t_i, s_i: _interp2d_bspline_scalar(
                    values, My, hy, ix_i, iy_i, t_i, s_i, hx
                )
            )(ix, iy, t, s)
        else:
            My = jax.vmap(lambda row: _spline_notaknot_uneven(row, ys_g))(values)
            return jax.vmap(
                lambda ix_i, iy_i, t_i, s_i: _interp2d_bspline_scalar_uneven(
                    values, My, xs, ys_g, ix_i, iy_i, t_i, s_i
                )
            )(ix, iy, t, s)

    else:
        raise ValueError(f"Unknown method={method!r}. Choose: nearest, linear, quadratic, cubic, bspline")


# ── [K] Cached wrappers ────────────────────────────────────────────────────────


def make_interp1d(
    xs: jax.Array,
    values: jax.Array,
    *,
    even: bool = True,
    method: str = "linear",
    bc: str = "not-a-knot",
    oob: str = "clamp",
    fp: tuple = (0.0, 0.0),
):
    """Return a JIT-compiled callable with spline coefficients pre-computed.

    Useful when the same knot values are queried many times.
    xs    : for even=True, [a, b]; for even=False, full knot array.
    """
    from functools import partial

    n = values.shape[0]
    a, b = xs[0], xs[-1]

    if method in ("cubic", "bspline"):
        if even:
            h = (b - a) / (n - 1)
            M = _bspline_coeffs(values, h) if method == "bspline" else _spline_M(values, h, bc, fp[0], fp[1])

            @jax.jit
            def _call_even(xq):
                xq2 = _apply_oob(a, b, xq, oob)
                i = _cell_index(a, h, xq2, n)
                t = (xq2 - (a + i.astype(values.dtype) * h)) / h
                return _interp1d_cubic(values, i, t, M, h)

            return _call_even

        else:
            M = (
                _spline_notaknot_uneven(values, xs)
                if method == "bspline"
                else _spline_M_uneven(values, xs, bc, fp[0], fp[1])
            )

            @jax.jit
            def _call_uneven(xq):
                xq2 = _apply_oob(a, b, xq, oob)
                i = _cell_index_uneven(xs, xq2)
                t = (xq2 - xs[i]) / (xs[i + 1] - xs[i])
                return _interp1d_cubic_uneven(values, xs, i, t, M)

            return _call_uneven

    else:
        return jax.jit(
            partial(interp1d, xs, values, even=even, method=method, bc=bc, oob=oob, fp=fp)
        )


# ── [L] Verification ───────────────────────────────────────────────────────────


def _verify() -> None:
    import numpy as np

    print("=" * 60)
    print("jax_interp verification")
    print("=" * 60)

    # ── 1D even ──────────────────────────────────────────────────────────────
    xs_e = jnp.linspace(0.0, 1.0, 11)
    ys_e = jnp.sin(xs_e * 3.0)
    xq = jnp.linspace(0.05, 0.95, 200)
    ref = jnp.sin(xq * 3.0)
    xs_ep = jnp.array([0.0, 1.0])  # endpoints only for even=True

    print("\n  --- 1D even grid ---")
    for method in ["linear", "quadratic", "cubic", "bspline"]:
        err_k = float(jnp.max(jnp.abs(interp1d(xs_ep, ys_e, xs_e, method=method) - ys_e)))
        print(f"  knot interp  {method:10s}  max|err|={err_k:.2e}")

    # h^4 convergence
    print("\n  O(h^4) convergence - cubic not-a-knot - sin(3x):")
    ns = [11, 21, 41]
    errs = []
    for n_pts in ns:
        xsi = jnp.linspace(0.0, 1.0, n_pts)
        ysi = jnp.sin(xsi * 3.0)
        e = float(jnp.max(jnp.abs(
            interp1d(jnp.array([0.0, 1.0]), ysi, jnp.linspace(0.2, 0.8, 100), method="cubic")
            - jnp.sin(jnp.linspace(0.2, 0.8, 100) * 3.0)
        )))
        errs.append(e)
        print(f"    n={n_pts:3d}  max|err|={e:.2e}")
    if len(errs) >= 2 and errs[1] > 0:
        print(f"    ratio 11→21: {errs[0]/errs[1]:.1f}x  (expect ~16x)")

    # OOB
    print("\n  Out-of-bounds:")
    xq_oob = jnp.array([-0.2, 0.5, 1.2])
    for oob_mode in ["clamp", "extrapolate", "periodic", "reflect"]:
        r = interp1d(xs_ep, ys_e, xq_oob, method="linear", oob=oob_mode)
        print(f"    oob={oob_mode:11s}  {np.array(r)}")

    # JAX compat
    print("\n  JAX compatibility (even):")
    f_jit = jax.jit(interp1d, static_argnames=["even", "method", "bc", "oob"])
    r = f_jit(xs_ep, ys_e, xq, method="cubic", bc="not-a-knot")
    print(f"    jit(cubic)  OK  shape={r.shape}")

    g = jax.grad(lambda v: jnp.sum(interp1d(xs_ep, v, xq, method="linear")))(ys_e)
    print(f"    grad(linear) OK  shape={g.shape}  finite={bool(jnp.all(jnp.isfinite(g)))}")

    # ── 1D uneven ────────────────────────────────────────────────────────────
    print("\n  --- 1D uneven grid ---")
    rng = np.random.default_rng(42)
    xs_u = jnp.array(np.sort(rng.uniform(0.0, 1.0, 11)))
    xs_u = xs_u.at[0].set(0.0).at[-1].set(1.0)
    ys_u = jnp.sin(xs_u * 3.0)

    for method in ["linear", "quadratic", "cubic", "bspline"]:
        err_k = float(jnp.max(jnp.abs(interp1d(xs_u, ys_u, xs_u, even=False, method=method) - ys_u)))
        print(f"  knot interp  {method:10s}  max|err|={err_k:.2e}")

    # Compare even vs uneven on same data (uniform knots → should agree)
    xs_same = jnp.linspace(0.0, 1.0, 11)
    ys_same = jnp.sin(xs_same * 3.0)
    xq_mid = jnp.linspace(0.1, 0.9, 50)
    for method in ["linear", "cubic"]:
        r_e = interp1d(jnp.array([0.0, 1.0]), ys_same, xq_mid, method=method)
        r_u = interp1d(xs_same, ys_same, xq_mid, even=False, method=method)
        diff = float(jnp.max(jnp.abs(r_e - r_u)))
        print(f"  even vs uneven [{method:6s}]  diff={diff:.2e}  (expect ~machine-eps)")

    # ── 2D even ──────────────────────────────────────────────────────────────
    print("\n  --- 2D even grid ---")
    xs2 = jnp.linspace(0.0, 1.0, 7)
    ys2 = jnp.linspace(0.0, 1.0, 7)
    vals2 = xs2[:, None] + ys2[None, :]
    xq2 = jax.random.uniform(jax.random.PRNGKey(1), (50,))
    yq2 = jax.random.uniform(jax.random.PRNGKey(2), (50,))

    for method in ["linear", "cubic"]:
        r2 = interp2d(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]), vals2, xq2, yq2, method=method)
        err = float(jnp.max(jnp.abs(r2 - (xq2 + yq2))))
        print(f"  2d {method:8s}  f=x+y  max|err|={err:.2e}")

    # ── 2D uneven ────────────────────────────────────────────────────────────
    print("\n  --- 2D uneven grid ---")
    xs2u = jnp.array(np.sort(rng.uniform(0.0, 1.0, 7)))
    xs2u = xs2u.at[0].set(0.0).at[-1].set(1.0)
    ys2u = jnp.array(np.sort(rng.uniform(0.0, 1.0, 7)))
    ys2u = ys2u.at[0].set(0.0).at[-1].set(1.0)
    vals2u = xs2u[:, None] + ys2u[None, :]

    for method in ["linear", "cubic"]:
        r2u = interp2d(xs2u, ys2u, vals2u, xq2, yq2, even=False, method=method)
        err = float(jnp.max(jnp.abs(r2u - (xq2 + yq2))))
        print(f"  2d uneven {method:8s}  f=x+y  max|err|={err:.2e}")

    print("\nAll checks done.")


if __name__ == "__main__":
    _verify()
