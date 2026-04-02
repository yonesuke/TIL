# JAX Interpolation on Regular Grids

**Date:** 2026-04-02

## 概要

JAXで1D・2D補間を実装したコード（`jax_interp`）を動かして、scipyとの精度・速度比較を行った。
JIT対応・勾配計算可能で、多くのケースでscipyより高速。

## 詳細

### 対応メソッドとscipyとの対応

| method / BC / oob | scipy相当 | 一致精度 |
|---|---|---|
| nearest | `interp1d(kind='nearest')` | EXACT |
| linear | `interp1d(kind='linear')` | EXACT |
| quadratic | `interp1d(kind='quadratic')` | DIFF ALGO（Lagrange vs B-spline） |
| cubic / natural | `CubicSpline(bc_type='natural')` | EXACT |
| cubic / not-a-knot | `CubicSpline(bc_type='not-a-knot')` | EXACT |
| cubic / clamped | `CubicSpline(bc_type=((1,fp_l),(1,fp_r)))` | EXACT |
| cubic / periodic | `CubicSpline(bc_type='periodic')` | EXACT |
| bspline | `make_interp_spline(x, y, k=3)` | EXACT |
| oob=reflect | — | scipy相当なし |
| 2d cubic | — | scipy相当なし（16係数bicubic） |

### キーポイント

- `even=True`（均等グリッド）: 直接インデックス計算で高速
- `even=False`（不均等グリッド）: `searchsorted`使用
- `method='bspline'`はscipy `RegularGridInterpolator(method='cubic')` (>= 1.13) と一致
- `oob`パラメータで境界外処理を制御: `'clamp'` / `'extrapolate'` / `'periodic'` / `'reflect'`
- `bc`パラメータはspline系のみに有効

### JIT・勾配対応

```python
import jax
import jax.numpy as jnp

xs = jnp.array([0., 1.])
f = jax.jit(interp1d, static_argnames=['even', 'method', 'bc', 'oob'])
y = f(xs, values, xq, method='cubic', bc='not-a-knot')

# 勾配もOK
grad_fn = jax.grad(lambda v: interp1d(xs, v, xq, method='linear').sum())
```

### 速度ベンチマーク（JAX vs scipy、中央値、n=5000）

| case | JAX (μs) | scipy (μs) | speedup |
|---|---|---|---|
| nearest | 91 | 1005 | **11x** |
| linear | 81 | 280 | 3.4x |
| cubic/not-a-knot | 219 | 587 | 2.7x |
| bspline | 216 | 1625 | **7.5x** |
| 2d linear (200x200, 5000q) | 55 | 505 | **9.2x** |
| 2d bspline (200x200, 5000q) | 5242 | 17341 | 3.3x |

### 精度

- EXACT matchのテスト: **30/30 全パス**
- cubic not-a-knot の収束率: O(h^4)（n倍密にすると誤差1/16倍）

## メモ

- 2D cubic（不均等グリッド）は誤差が大きい（~0.49）—実装に問題がある可能性
- quadraticはscipyと異なるアルゴリズム（3点Lagrange vs 二次Bスプライン）なので誤差は想定内
- uv scriptとして書かれており、`uv run`一発で依存関係込みで実行できる
