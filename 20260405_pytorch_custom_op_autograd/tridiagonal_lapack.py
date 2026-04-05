# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "scipy",
#   "numpy",
# ]
# ///
"""Tridiagonal solver: LAPACK gtsv, two autograd integrations.

Supports both single RHS (n,) and batched RHS (n, k).

Backward derivation
-------------------
Single RHS: Ax = b, scalar loss L
  dL/db  = A^{-T} (dL/dx)        [solve A^T v = dL/dx, v shape (n,)]
  dL/dA  = -(dL/db) x^T          [outer product, restrict to tridiagonal entries]

  grad_rhs   [i]   = v[i]
  grad_diag  [i]   = -v[i]   * x[i]
  grad_upper [i]   = -v[i]   * x[i+1]   (A[i, i+1])
  grad_lower [i]   = -v[i+1] * x[i]     (A[i+1, i])

Batched RHS: AX = B with X, B of shape (n, k), scalar loss L
  dL/dB  = A^{-T} (dL/dX)        [solve A^T V = dL/dX, V shape (n, k)]

  grad_rhs   [i, :]  = v[i, :]
  grad_diag  [i]     = -sum_j v[i,j]   * x[i,j]
  grad_upper [i]     = -sum_j v[i,j]   * x[i+1,j]
  grad_lower [i]     = -sum_j v[i+1,j] * x[i,j]

A^T is tridiagonal with lower/upper swapped, so the same gtsv call handles it.

Two implementations
-------------------
thomas_solve_lapack   : torch.autograd.Function wrapper (legacy, not compile-friendly)
thomas_solve_custom   : torch.library.custom_op + register_autograd (torch.compile OK)
"""

import time

import numpy as np
import torch
from scipy.linalg import solve_banded
from scipy.linalg.lapack import get_lapack_funcs


# ---------------------------------------------------------------------------
# LAPACK kernel (shared by both implementations)
# ---------------------------------------------------------------------------


def _gtsv(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Call LAPACK ?gtsv (copies inputs to protect originals)."""
    dl = lower.copy()
    d = diag.copy()
    du = upper.copy()
    b = rhs.copy()
    (gtsv,) = get_lapack_funcs(("gtsv",), (dl, d, du, b))
    _, _, _, x, info = gtsv(dl, d, du, b)
    if info != 0:
        raise RuntimeError(f"LAPACK gtsv failed: info={info}")
    return x


def _compute_grads(
    v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tridiagonal gradient from adjoint v and solution x."""
    if x.dim() == 1:
        grad_diag = -v * x
        grad_upper = -v[:-1] * x[1:]
        grad_lower = -v[1:] * x[:-1]
    else:  # (n, k) batched
        grad_diag = -(v * x).sum(dim=1)
        grad_upper = -(v[:-1] * x[1:]).sum(dim=1)
        grad_lower = -(v[1:] * x[:-1]).sum(dim=1)
    return grad_lower, grad_diag, grad_upper


# ---------------------------------------------------------------------------
# Implementation 1: torch.autograd.Function (not torch.compile friendly)
# ---------------------------------------------------------------------------


class _TridiagonalSolveFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, diag, upper, rhs):
        x_np = _gtsv(
            lower.detach().numpy(),
            diag.detach().numpy(),
            upper.detach().numpy(),
            rhs.detach().numpy(),
        )
        x = torch.tensor(x_np, dtype=diag.dtype)
        ctx.save_for_backward(lower, diag, upper, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        lower, diag, upper, x = ctx.saved_tensors
        # A^T solve: swap lower/upper
        v_np = _gtsv(
            upper.detach().numpy(),
            diag.detach().numpy(),
            lower.detach().numpy(),
            grad_output.detach().numpy(),
        )
        v = torch.tensor(v_np, dtype=diag.dtype)
        grad_lower, grad_diag, grad_upper = _compute_grads(v, x)
        return grad_lower, grad_diag, grad_upper, v


def thomas_solve_lapack(lower, diag, upper, rhs):
    """Tridiagonal solve via LAPACK (autograd.Function). Not torch.compile friendly."""
    return _TridiagonalSolveFn.apply(lower, diag, upper, rhs)


# ---------------------------------------------------------------------------
# Implementation 2: torch.library.custom_op (torch.compile friendly)
# ---------------------------------------------------------------------------


@torch.library.custom_op("mylib::tridiagonal_solve", mutates_args=())
def _tridiagonal_solve_kernel(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    x = _gtsv(lower.numpy(), diag.numpy(), upper.numpy(), rhs.numpy())
    return torch.tensor(x, dtype=diag.dtype)


@_tridiagonal_solve_kernel.register_fake
def _(lower, diag, upper, rhs):
    return rhs.new_empty(rhs.shape)


def _setup_context(ctx, inputs, output):
    lower, diag, upper, _ = inputs
    ctx.save_for_backward(lower, diag, upper, output)


def _backward(ctx, grad_output):
    lower, diag, upper, x = ctx.saved_tensors
    # A^T solve: swap lower/upper — reuse the same custom op so .numpy()
    # stays inside an opaque kernel that torch.compile can handle.
    v = _tridiagonal_solve_kernel(upper, diag, lower, grad_output)
    grad_lower, grad_diag, grad_upper = _compute_grads(v, x)
    return grad_lower, grad_diag, grad_upper, v


torch.library.register_autograd(
    "mylib::tridiagonal_solve",
    _backward,
    setup_context=_setup_context,
)


def thomas_solve_custom(lower, diag, upper, rhs):
    """Tridiagonal solve via LAPACK (custom_op). torch.compile friendly."""
    return _tridiagonal_solve_kernel(lower, diag, upper, rhs)


# ---------------------------------------------------------------------------
# Pure-torch Thomas algorithm (reference)
# ---------------------------------------------------------------------------


def thomas_solve_torch(lower, diag, upper, rhs):
    """Solve tridiagonal system via pure-torch Thomas algorithm (autograd via tape)."""
    n = diag.shape[0]
    c: list[torch.Tensor] = []
    d: list[torch.Tensor] = []

    c.append(upper[0] / diag[0])
    d.append(rhs[0] / diag[0])

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c.append(upper[i] / denom)
        d.append((rhs[i] - lower[i - 1] * d[i - 1]) / denom)

    x: list[torch.Tensor] = [torch.zeros((), dtype=diag.dtype)] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return torch.stack(x)


# ---------------------------------------------------------------------------
# scipy reference
# ---------------------------------------------------------------------------


def scipy_solve(lower, diag, upper, rhs):
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(42)
    n = 10

    lower_np = rng.standard_normal(n - 1) * 0.5
    upper_np = rng.standard_normal(n - 1) * 0.5
    diag_np = np.abs(rng.standard_normal(n)) + 3.0
    rhs_np = rng.standard_normal(n)
    rhs_batch_np = rng.standard_normal((n, 5))

    def to_tensor(a, dtype=torch.float64, grad=False):
        return torch.tensor(a, dtype=dtype, requires_grad=grad)

    args64 = [to_tensor(a) for a in [lower_np, diag_np, upper_np, rhs_np]]
    args64_grad = [
        to_tensor(a, grad=True) for a in [lower_np, diag_np, upper_np, rhs_np]
    ]
    args64_batch_grad = [
        *[to_tensor(a, grad=True) for a in [lower_np, diag_np, upper_np]],
        to_tensor(rhs_batch_np, grad=True),
    ]

    x_scipy = scipy_solve(lower_np, diag_np, upper_np, rhs_np)

    # --- forward correctness -----------------------------------------------
    print("=== forward correctness ===")
    for name, fn in [
        ("autograd.Function", thomas_solve_lapack),
        ("custom_op        ", thomas_solve_custom),
    ]:
        err = np.abs(x_scipy - fn(*args64).numpy()).max()
        print(
            f"  {name}  single  max err = {err:.2e}  [{'OK' if err < 1e-12 else 'FAIL'}]"
        )

        x_batch = fn(*args64[:-1], to_tensor(rhs_batch_np))
        x_batch_ref = scipy_solve(lower_np, diag_np, upper_np, rhs_batch_np)
        err_b = np.abs(x_batch_ref - x_batch.numpy()).max()
        print(
            f"  {name}  batch   max err = {err_b:.2e}  [{'OK' if err_b < 1e-12 else 'FAIL'}]"
        )

    # --- gradcheck ---------------------------------------------------------
    print("\n=== gradcheck ===")
    for name, fn in [
        ("autograd.Function", thomas_solve_lapack),
        ("custom_op        ", thomas_solve_custom),
    ]:
        p1 = torch.autograd.gradcheck(fn, args64_grad, eps=1e-6, atol=1e-5)
        p2 = torch.autograd.gradcheck(fn, args64_batch_grad, eps=1e-6, atol=1e-5)
        print(f"  {name}  single={p1}  batch={p2}")

    # --- torch.compile -----------------------------------------------------
    print("\n=== torch.compile ===")

    def model(lo, di, up, b):
        return thomas_solve_custom(lo, di, up, b).pow(2).sum()

    compiled = torch.compile(model)
    lo, di, up, b = [
        to_tensor(a, grad=True) for a in [lower_np, diag_np, upper_np, rhs_np]
    ]
    loss = compiled(lo, di, up, b)
    loss.backward()
    print(f"  compiled forward OK  loss={loss.item():.6f}")
    print(f"  compiled backward OK grad_rhs norm={b.grad.norm().item():.6f}")

    # check for graph breaks
    expl = torch._dynamo.explain(model)(lo, di, up, b)
    n_breaks = len(expl.break_reasons)
    print(f"  graph breaks: {n_breaks}  [{'OK' if n_breaks == 0 else 'WARN'}]")

    # --- speed comparison --------------------------------------------------
    print("\n=== speed (n=1000, 1000 iters) ===")
    n_large = 1000
    lo = to_tensor(rng.standard_normal(n_large - 1) * 0.5)
    di = to_tensor(np.abs(rng.standard_normal(n_large)) + 3.0)
    up = to_tensor(rng.standard_normal(n_large - 1) * 0.5)
    b1 = to_tensor(rng.standard_normal(n_large))
    b100 = to_tensor(rng.standard_normal((n_large, 100)))
    iters = 1000

    timings = {}
    for label, fn, rhs_arg in [
        ("autograd.Function single", thomas_solve_lapack, b1),
        ("custom_op         single", thomas_solve_custom, b1),
        ("custom_op         batch ", thomas_solve_custom, b100),
        ("torch Thomas      single", thomas_solve_torch, b1),
    ]:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(lo, di, up, rhs_arg)
        timings[label] = time.perf_counter() - t0
        print(
            f"  {label}: {timings[label] * 1e3:.1f} ms  ({timings[label] / iters * 1e6:.1f} us/iter)"
        )


if __name__ == "__main__":
    main()
