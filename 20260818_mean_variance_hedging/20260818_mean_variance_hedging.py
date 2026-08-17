# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.20",
#     "matplotlib>=3.8.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
European Call Option Daily Hedging via High-Dimensional Generic Sigmoid Basis & Simplex-Constrained QP

本スクリプトでは、以下の枠組みでヨーロピアン・コールオプションの最適ヘッジポジションを求めます:
1. 状態変数 (S, t) をもとにモデルフリーな汎用ロジスティック・シグモイド基底群 (D = 160) を構成
   - 空間中心: ATM 周辺を高密度にサンプリング (16点)
   - スケール幅: 局所ステップから広域まで (5点)
   - 時間変調: 拡散スケール sqrt(tau) による正規化
2. モンテカルロシミュレーション (10万パス, 250ステップ/日次ヘッジ) の累積ゲインを jax.lax.scan で省メモリ集計
3. ヘッジ PnL の平均分散目的関数 max_w { E[Pi] - (lambda / 2) Var(Pi) } を二次計画問題 (QP) として定式化
4. 単体制約 (w >= 0, sum(w) = 1) を課して FISTA (加速射影勾配法) で求解
   - 0 <= Delta(S, t) <= 1 の有界性と単調増加性 (dDelta/dS >= 0) を数学的に 100% 保証
   - L1 正則化 (Lasso) 効果により有効な基底のみを自動スパース選択
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. Black-Scholes ベンチマーク (理論解)
# ==========================================
def bs_call_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, tau, r, sigma):
    tau = jnp.maximum(tau, 1e-6)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * jnp.sqrt(tau))
    return norm.cdf(d1)


# ==========================================
# 2. 高次元・汎用ロジスティック・シグモイド基底 (D = 160)
# ==========================================
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(z, -30.0, 30.0)))


def compute_high_dim_sigmoid_basis(
    S: float, t: float, K: float, T: float
) -> jax.Array:
    tau_norm = jnp.maximum((T - t) / T, 1e-4)
    sqrt_tau = jnp.sqrt(tau_norm)

    # ATM 近傍を高密度にサンプリングした 16 点
    centers = K * jnp.array([
        0.70, 0.80, 0.88, 0.92, 0.95, 0.98, 1.00,
        1.02, 1.05, 1.08, 1.12, 1.20, 1.30,
        0.96, 1.00, 1.04,
    ])

    # 5 つの幅スケール
    scales = jnp.array([0.03, 0.07, 0.15, 0.30, 0.60])

    basis_list = []
    # タイプ 1: 純粋な空間スケール
    for c in centers:
        for sc in scales:
            z = (S - c) / (sc * K * sqrt_tau)
            basis_list.append(sigmoid(z))

    # タイプ 2: 時間の線形推移補正
    for c in centers:
        for sc in scales:
            z = (S * (1.0 + 0.05 * tau_norm) - c) / (sc * K * sqrt_tau)
            basis_list.append(sigmoid(z))

    return jnp.array(basis_list)


vmap_step_basis = vmap(compute_high_dim_sigmoid_basis, in_axes=(0, None, None, None))
vmap_step_bs_delta = vmap(bs_call_delta, in_axes=(0, None, None, None, None))


# ==========================================
# 3. モンテカルロシミュレーション & scan 特徴量抽出
# ==========================================
@jit(static_argnames=("M", "N"))
def simulate_and_extract_features(
    key, S0, K, T, r, mu_drift, sigma, M=100000, N=250
):
    dt = T / N
    t_grid = jnp.linspace(0.0, T, N + 1)
    D = jnp.exp(-r * t_grid)

    Z = jax.random.normal(key, shape=(M, N))
    drift = (mu_drift - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt) * Z
    log_returns = jnp.hstack([jnp.zeros((M, 1)), drift + diffusion])
    S_paths = S0 * jnp.exp(jnp.cumsum(log_returns, axis=1))

    S_tilde = S_paths * D[None, :]
    delta_S_tilde = S_tilde[:, 1:] - S_tilde[:, :-1]

    def step_fn(carry, k):
        X_acc, X_bs_acc = carry
        S_k = S_paths[:, k]
        t_k = t_grid[k]
        tau_k = T - t_k
        dS = delta_S_tilde[:, k]

        phi_k = vmap_step_basis(S_k, t_k, K, T)
        X_acc = X_acc + phi_k * dS[:, None]

        bs_delta_k = vmap_step_bs_delta(S_k, K, tau_k, r, sigma)
        X_bs_acc = X_bs_acc + bs_delta_k * dS

        return (X_acc, X_bs_acc), None

    dim_basis = 16 * 5 * 2
    init_X = jnp.zeros((M, dim_basis))
    init_X_bs = jnp.zeros(M)

    (X, X_bs), _ = jax.lax.scan(step_fn, (init_X, init_X_bs), jnp.arange(N))

    payoff = jnp.maximum(S_paths[:, -1] - K, 0.0)
    y = D[-1] * payoff

    return S_paths, t_grid, X, X_bs, y


# ==========================================
# 4. 単体制約付き二次計画法 (QP on Simplex)
# ==========================================
def project_simplex_np(v, z=1.0):
    v_sorted = np.sort(v)[::-1]
    cssv = np.cumsum(v_sorted) - z
    ind = np.arange(1, len(v) + 1)
    cond = v_sorted - cssv / ind > 0
    rho = np.count_nonzero(cond)
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def solve_qp_simplex(Q, q, max_iter=2000, tol=1e-7):
    Q_np = np.array(Q)
    q_np = np.array(q)
    D = len(q_np)

    L = float(np.linalg.norm(Q_np, 2)) + 1e-4
    step_size = 1.0 / L

    w = np.ones(D) / D
    y_k = w.copy()
    t_k = 1.0

    for _ in range(max_iter):
        grad = Q_np @ y_k + q_np
        w_next = project_simplex_np(y_k - step_size * grad, z=1.0)

        diff = np.linalg.norm(w_next - w)
        if diff < tol:
            w = w_next
            break

        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        y_k = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        w = w_next
        t_k = t_next

    return jnp.array(w)


def build_qp_matrices(X, y, lam=1000.0, reg=1e-5):
    mu_X = jnp.mean(X, axis=0)
    mu_y = jnp.mean(y)
    Sigma_XX = jnp.cov(X, rowvar=False)
    Sigma_Xy = jnp.mean((X - mu_X) * (y - mu_y)[:, None], axis=0)

    Q = lam * Sigma_XX + reg * jnp.eye(X.shape[1])
    q = -(mu_X + lam * Sigma_Xy)

    return Q, q, mu_X, mu_y, Sigma_XX, Sigma_Xy


# ==========================================
# 5. メイン実行 & プロット生成
# ==========================================
def main():
    S0 = 100.0
    K = 100.0
    T = 1.0
    N = 250  # 日次ヘッジ (250ステップ)
    r = 0.05
    mu_drift = 0.10
    sigma = 0.20
    M = 100000

    print("=== European Call Option Daily Hedging (Generic Sigmoid + Simplex QP) ===")
    print(f"S0={S0}, K={K}, T={T}, r={r}, mu={mu_drift}, sigma={sigma}")
    print(f"Paths M={M}, Rebalance steps N={N} (Daily)\n")

    key = jax.random.PRNGKey(42)
    S_paths, t_grid, X, X_bs, y = simulate_and_extract_features(
        key, S0, K, T, r, mu_drift, sigma, M=M, N=N
    )

    bs_price = bs_call_price(S0, K, T, r, sigma)
    c0 = bs_price

    # 1. ベンチマーク (BS Delta 理論ヘッジ)
    bs_pnl = np.array(c0 + X_bs - y)
    bs_mean = float(np.mean(bs_pnl))
    bs_std = float(np.std(bs_pnl))
    bs_q01 = float(np.percentile(bs_pnl, 1.0))

    # 2. 単体制約付き QP 解 (純粋ヘッジ lambda = 1000)
    Q, q, _, _, _, _ = build_qp_matrices(X, y, lam=1000.0)
    w_qp = solve_qp_simplex(Q, q)
    high_dim_pnl = np.array(c0 + X @ w_qp - y)

    m_high = float(np.mean(high_dim_pnl))
    s_high = float(np.std(high_dim_pnl))
    q01_high = float(np.percentile(high_dim_pnl, 1.0))
    active_basis_count = int(np.count_nonzero(np.array(w_qp) > 1e-4))

    print(f"【BS Delta 理論ヘッジ】     : E[PnL] = {bs_mean:8.4f} | Std = {bs_std:7.4f} | 1% Quantile = {bs_q01:8.4f}")
    print(f"【高次元シグモイド (D=160)】: E[PnL] = {m_high:8.4f} | Std = {s_high:7.4f} | 1% Quantile = {q01_high:8.4f}")
    print(f"有効基底数 (w > 1e-4)       : {active_basis_count} / {len(w_qp)} (単体制約による自動スパース選択)")
    print(f"BS 理論解との 1% Quantile 差: {abs(q01_high - bs_q01):.4f}\n")

    # 3. デルタ比較プロット
    S_test = jnp.linspace(60.0, 140.0, 300)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tau_cases = [1.0, 0.2, 0.02]
    for i, tau_val in enumerate(tau_cases):
        t_val = T - tau_val
        phi_test = vmap(compute_high_dim_sigmoid_basis, in_axes=(0, None, None, None))(
            S_test, t_val, K, T
        )
        learned_delta = phi_test @ w_qp
        bs_delta = bs_call_delta(S_test, K, tau_val, r, sigma)

        axes[i].plot(
            S_test, bs_delta, "k--", label="Black-Scholes Delta (Benchmark)", linewidth=2.2
        )
        axes[i].plot(
            S_test,
            learned_delta,
            "r-",
            label="Generic Sigmoid + Simplex QP (D=160)",
            linewidth=1.8,
        )
        axes[i].set_title(f"Remaining Time $\\tau = {tau_val:.2f}$ yr")
        axes[i].set_xlabel("Stock Price $S$")
        axes[i].set_ylabel("Hedge Ratio $\\Delta$")
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("delta_comparison.png", dpi=150)
    print("Saved: delta_comparison.png")

    # 4. PnL 分布比較プロット
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-6, 6, 120)

    plt.hist(
        bs_pnl,
        bins=bins,
        density=True,
        alpha=0.6,
        color="royalblue",
        label=f"BS Delta Hedge (std={bs_std:.2f}, 1%Q={bs_q01:.2f})",
        edgecolor="royalblue",
        linewidth=1.5,
        histtype="stepfilled",
    )
    plt.hist(
        high_dim_pnl,
        bins=bins,
        density=True,
        alpha=0.5,
        color="crimson",
        label=f"High-Dim Sigmoid QP D=160 (std={s_high:.2f}, 1%Q={q01_high:.2f})",
        edgecolor="crimson",
        linewidth=1.5,
        histtype="stepfilled",
    )

    plt.axvline(bs_q01, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axvline(q01_high, color="crimson", linestyle="--", linewidth=1.5)

    plt.title("Hedging PnL Distribution: BS Benchmark vs High-Dim Sigmoid QP (D=160)", fontsize=13)
    plt.xlabel("Discounted PnL $\\Pi_T$", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.xlim(-6, 6)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10.5)
    plt.tight_layout()
    plt.savefig("pnl_distribution_comparison.png", dpi=150)
    print("Saved: pnl_distribution_comparison.png")


if __name__ == "__main__":
    main()
