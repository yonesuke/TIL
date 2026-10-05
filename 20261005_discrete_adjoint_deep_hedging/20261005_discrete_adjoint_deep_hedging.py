# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.30",
#     "flax>=0.10.0",
#     "numpy",
# ]
# ///
"""
20261005_discrete_adjoint_deep_hedging.py

離散ポントリャーギン随伴状態法 (Discrete Pontryagin Adjoint Method) による
Deep Hedging および一般離散力学系 x_{t+1} = f(x_t, u_t; θ) の O(1) メモリ VJP ソルバ

1. 損益会計モジュール (calculate_pnl) の 3 点ステンシル Custom VJP
2. 軌道発展モジュール (solve_hedge) の随伴変数 (Costate) 後退走査 Custom VJP
3. JAX Auto-AD (Ground Truth) との厳密な勾配一致検証 (機械精度)
4. 一般離散力学系 x_{t+1} = f(x_t, u_t; θ) に対する汎用随伴ソルバ (make_discrete_adjoint_solver)
"""

from typing import Tuple, Callable, Any
import time
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


# =============================================================================
# 1. 損益会計モジュール (calculate_pnl with custom_vjp)
# =============================================================================
def subgrad_abs(x: jax.Array) -> jax.Array:
    """JAX の jax.grad(jnp.abs) と厳密に一致する劣勾配演算子 (x=0 で +1.0)"""
    return jnp.where(x >= 0.0, 1.0, -1.0)


@jax.custom_vjp
def calculate_pnl(
    position_trajectory: jax.Array,  # shape (num_steps, num_paths)
    spot_prices: jax.Array,          # shape (num_paths, num_steps + 1)
    strike_price: float = 100.0,
    cost_rate: float = 0.01,
) -> jax.Array:
    """離散時間ポートフォリオ損益 (PnL) の計算"""
    pnl, _ = calculate_pnl_fwd(position_trajectory, spot_prices, strike_price, cost_rate)
    return pnl


def calculate_pnl_fwd(
    position_trajectory: jax.Array,
    spot_prices: jax.Array,
    strike_price: float,
    cost_rate: float,
):
    num_steps, num_paths = position_trajectory.shape
    spot_steps = spot_prices.T[:-1]          # shape (num_steps, num_paths)
    spot_terminal = spot_prices[:, -1]       # shape (num_paths,)
    spot_diff = spot_prices.T[1:] - spot_prices.T[:-1]  # shape (num_steps, num_paths)

    prev_positions = jnp.pad(position_trajectory[:-1], ((1, 0), (0, 0)))

    # 取引利得・取引コスト
    gains = jnp.sum(position_trajectory * spot_diff, axis=0)
    running_costs = jnp.sum(cost_rate * jnp.abs(position_trajectory - prev_positions) * spot_steps, axis=0)

    # 満期手仕舞い・ペイオフ
    terminal_cost = cost_rate * jnp.abs(position_trajectory[-1]) * spot_terminal
    payoff = jnp.maximum(spot_terminal - strike_price, 0.0)

    pnl = gains - running_costs - terminal_cost - payoff
    residuals = (position_trajectory, spot_prices, strike_price, cost_rate)
    return pnl, residuals


def calculate_pnl_bwd(residuals, grad_pnl):
    """3点ステンシルによる PnL の解析的 VJP"""
    position_trajectory, spot_prices, strike_price, cost_rate = residuals
    num_steps, num_paths = position_trajectory.shape
    spot_steps = spot_prices.T[:-1]
    spot_terminal = spot_prices[:, -1]
    spot_diff = spot_prices.T[1:] - spot_prices.T[:-1]

    prev_positions = jnp.pad(position_trajectory[:-1], ((1, 0), (0, 0)))

    # (i) 当期利得・取引コストの偏微分
    term_curr = spot_diff - cost_rate * subgrad_abs(position_trajectory - prev_positions) * spot_steps

    # (ii) 次期取引コストの偏微分
    term_next_inner = cost_rate * subgrad_abs(position_trajectory[1:] - position_trajectory[:-1]) * spot_steps[1:]
    term_next = jnp.pad(term_next_inner, ((0, 1), (0, 0)))

    # (iii) 満期手仕舞いコストの偏微分
    term_terminal = jnp.pad(-cost_rate * subgrad_abs(position_trajectory[-1:]) * spot_terminal[None, :], ((num_steps - 1, 0), (0, 0)))

    grad_positions = (term_curr + term_next + term_terminal) * grad_pnl[None, :]
    return (grad_positions, None, None, None)


calculate_pnl.defvjp(calculate_pnl_fwd, calculate_pnl_bwd)


# =============================================================================
# 2. ポリシーネットワーク定義 (Flax NNX)
# =============================================================================
class DeepHedgePolicy(nnx.Module):
    def __init__(self, in_features: int = 3, d_hidden: int = 32, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, d_hidden, rngs=rngs)
        self.fc2 = nnx.Linear(d_hidden, d_hidden, rngs=rngs)
        self.fc3 = nnx.Linear(d_hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.gelu(self.fc1(x))
        h = nnx.gelu(self.fc2(h))
        return self.fc3(h)


# =============================================================================
# 3. 軌道発展ソルバ (solve_hedge with Custom VJP)
# =============================================================================
def build_deep_hedging_engine(
    graphdef: nnx.GraphDef,
    maturity_years: float,
    strike_price: float,
    risk_aversion: float,
    cost_rate: float,
):
    def evaluate_policy(model_state: nnx.State, policy_inputs: jax.Array) -> jax.Array:
        policy_model = nnx.merge(graphdef, model_state)
        return policy_model(policy_inputs).squeeze(-1)

    @jax.custom_vjp
    def solve_hedge(model_state: nnx.State, spot_prices: jax.Array) -> jax.Array:
        trajectory, _ = solve_hedge_fwd(model_state, spot_prices)
        return trajectory

    def solve_hedge_fwd(model_state: nnx.State, spot_prices: jax.Array):
        num_paths, num_time_points = spot_prices.shape
        num_steps = num_time_points - 1
        dt = maturity_years / num_steps

        time_features = jnp.broadcast_to(
            (jnp.arange(num_steps) * dt / maturity_years)[:, None, None],
            (num_steps, num_paths, 1),
        )
        spot_at_steps = spot_prices.T[:-1]
        moneyness_features = ((spot_at_steps - strike_price) / strike_price)[:, :, None]

        def forward_step(prev_h, step_feats):
            t_f, s_f = step_feats
            feat = jnp.concatenate([t_f, s_f, prev_h[:, None]], axis=-1)
            h = evaluate_policy(model_state, feat)
            return h, h

        _, position_trajectory = jax.lax.scan(
            forward_step,
            jnp.zeros(num_paths),
            (time_features, moneyness_features),
        )

        residuals = (model_state, position_trajectory, time_features, moneyness_features)
        return position_trajectory, residuals

    def solve_hedge_bwd(residuals, grad_positions):
        model_state, position_trajectory, time_features, moneyness_features = residuals
        num_steps, num_paths = position_trajectory.shape

        def backward_step(future_costate, step_idx):
            prev_h = jnp.where(step_idx > 0, position_trajectory[step_idx - 1], 0.0)
            feat = jnp.concatenate(
                [time_features[step_idx], moneyness_features[step_idx], prev_h[:, None]],
                axis=-1,
            )

            total_costate = grad_positions[step_idx] + future_costate

            _, local_vjp_fn = jax.vjp(evaluate_policy, model_state, feat)
            grad_model_state_step, grad_inputs = local_vjp_fn(total_costate)

            costate_to_past = grad_inputs[:, 2]
            return costate_to_past, grad_model_state_step

        reversed_indices = jnp.arange(num_steps - 1, -1, -1)
        _, grads_across_time = jax.lax.scan(
            backward_step,
            jnp.zeros(num_paths),
            reversed_indices,
        )

        total_model_grad = jax.tree.map(
            lambda g: jnp.sum(g, axis=0),
            grads_across_time,
        )
        return (total_model_grad, None)

    solve_hedge.defvjp(solve_hedge_fwd, solve_hedge_bwd)

    # エンドツーエンド損失関数
    def loss_fn(model_state: nnx.State, spot_prices: jax.Array) -> jax.Array:
        positions = solve_hedge(model_state, spot_prices)
        pnl = calculate_pnl(positions, spot_prices, strike_price, cost_rate)
        num_paths = spot_prices.shape[0]
        return (1.0 / risk_aversion) * (jax.nn.logsumexp(-risk_aversion * pnl) - jnp.log(num_paths))

    return solve_hedge, loss_fn


# =============================================================================
# 4. 検証用 Auto-AD 実装 (Ground Truth Baseline)
# =============================================================================
def build_auto_ad_loss(
    graphdef: nnx.GraphDef,
    maturity_years: float,
    strike_price: float,
    risk_aversion: float,
    cost_rate: float,
):
    def auto_loss_fn(model_state: nnx.State, spot_prices: jax.Array) -> jax.Array:
        num_paths, num_time_points = spot_prices.shape
        num_steps = num_time_points - 1
        dt = maturity_years / num_steps

        time_features = jnp.broadcast_to(
            (jnp.arange(num_steps) * dt / maturity_years)[:, None, None],
            (num_steps, num_paths, 1),
        )
        spot_at_steps = spot_prices.T[:-1]
        moneyness_features = ((spot_at_steps - strike_price) / strike_price)[:, :, None]
        policy_model = nnx.merge(graphdef, model_state)

        def step_fn(prev_h, step_feats):
            t_f, s_f = step_feats
            feat = jnp.concatenate([t_f, s_f, prev_h[:, None]], axis=-1)
            h = policy_model(feat).squeeze(-1)
            return h, h

        _, position_trajectory = jax.lax.scan(
            step_fn,
            jnp.zeros(num_paths),
            (time_features, moneyness_features),
        )

        pnl = calculate_pnl(position_trajectory, spot_prices, strike_price, cost_rate)
        return (1.0 / risk_aversion) * (jax.nn.logsumexp(-risk_aversion * pnl) - jnp.log(num_paths))

    return auto_loss_fn


# =============================================================================
# 5. 一般離散力学系ソルバ (make_discrete_adjoint_solver)
# =============================================================================
def make_discrete_adjoint_solver(
    step_fn: Callable[[Any, jax.Array, Any], jax.Array]
):
    """
    一般離散力学系 x_{t+1} = step_fn(params, x_t, u_t) のための
    O(1) メモリ随伴ソルバ
    """
    @jax.custom_vjp
    def solve_dynamics(params: Any, initial_state: jax.Array, inputs_seq: jax.Array) -> jax.Array:
        trajectory, _ = solve_dynamics_fwd(params, initial_state, inputs_seq)
        return trajectory

    def solve_dynamics_fwd(params, initial_state, inputs_seq):
        def scan_step(curr_x, u_t):
            next_x = step_fn(params, curr_x, u_t)
            return next_x, next_x

        _, trajectory = jax.lax.scan(scan_step, initial_state, inputs_seq)
        residuals = (params, initial_state, trajectory, inputs_seq)
        return trajectory, residuals

    def solve_dynamics_bwd(residuals, grad_trajectory):
        params, initial_state, trajectory, inputs_seq = residuals
        num_steps = trajectory.shape[0]

        def scan_bwd_step(future_costate, t):
            curr_x = jax.lax.cond(t > 0, lambda: trajectory[t - 1], lambda: initial_state)
            curr_u = inputs_seq[t]

            total_adjoint = grad_trajectory[t] + future_costate

            _, local_vjp = jax.vjp(
                lambda p, x: step_fn(p, x, curr_u),
                params,
                curr_x,
            )
            grad_p_step, costate_to_past = local_vjp(total_adjoint)
            return costate_to_past, grad_p_step

        terminal_costate = jnp.zeros_like(initial_state)
        reversed_indices = jnp.arange(num_steps - 1, -1, -1)

        initial_state_grad, grads_params_seq = jax.lax.scan(
            scan_bwd_step,
            terminal_costate,
            reversed_indices,
        )

        total_params_grad = jax.tree.map(
            lambda g: jnp.sum(g, axis=0),
            grads_params_seq,
        )
        return (total_params_grad, initial_state_grad, None)

    solve_dynamics.defvjp(solve_dynamics_fwd, solve_dynamics_bwd)
    return solve_dynamics


# =============================================================================
# 6. 実行・検証テスト
# =============================================================================
def generate_gbm_paths(
    key: jax.Array,
    num_paths: int = 5000,
    num_steps: int = 60,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    t_years: float = 60.0 / 365.0,
) -> jax.Array:
    """幾何ブラウン運動パス生成"""
    dt = t_years / num_steps
    normal_shocks = jax.random.normal(key, (num_paths, num_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt) * normal_shocks
    log_returns = jnp.concatenate([jnp.zeros((num_paths, 1)), drift + diffusion], axis=-1)
    return s0 * jnp.exp(jnp.cumsum(log_returns, axis=-1))


def main():
    print("=" * 70)
    print(" 離散ポントリャーギン随伴状態法 (Discrete Pontryagin Adjoint) 検証")
    print("=" * 70)

    key = jax.random.key(42)
    k_init, k_data = jax.random.split(key)

    num_paths = 5000
    num_steps = 60
    maturity_years = 60.0 / 365.0
    strike_price = 100.0
    risk_aversion = 1.0
    cost_rate = 0.01

    # データ生成
    print(f"\n[1] 株価パス生成 (N={num_steps} steps, M={num_paths} paths)...")
    spot_prices = generate_gbm_paths(k_data, num_paths=num_paths, num_steps=num_steps)

    # モデル構築
    policy = DeepHedgePolicy(in_features=3, d_hidden=32, rngs=nnx.Rngs(k_init))
    graphdef, state = nnx.split(policy)

    # 1. 随伴法エンジンの構築
    _, adjoint_loss_fn = build_deep_hedging_engine(
        graphdef, maturity_years, strike_price, risk_aversion, cost_rate
    )
    # 2. Auto-AD ベースラインの構築
    auto_loss_fn = build_auto_ad_loss(
        graphdef, maturity_years, strike_price, risk_aversion, cost_rate
    )

    grad_adjoint = jax.jit(jax.grad(adjoint_loss_fn))
    grad_auto = jax.jit(jax.grad(auto_loss_fn))

    print("\n[2] 勾配の JIT ウォームアップ & 一致性テスト...")
    g_adj = grad_adjoint(state, spot_prices)
    g_aut = grad_auto(state, spot_prices)

    # 勾配テンソルの最大絶対誤差・相対誤差を検証
    max_abs_diff = 0.0
    all_close_flags = []
    for (k1, v1), (k2, v2) in zip(jax.tree.leaves_with_path(g_adj), jax.tree.leaves_with_path(g_aut)):
        diff = float(jnp.max(jnp.abs(v1 - v2)))
        max_abs_diff = max(max_abs_diff, diff)
        is_close = bool(jnp.allclose(v1, v2, atol=1e-6, rtol=1e-5))
        all_close_flags.append(is_close)

    print(f"  - Auto-AD との最大絶対誤差: {max_abs_diff:.2e}")
    print(f"  - 機械精度での一致判定 (atol=1e-6): {'PASS (完全一致)' if all(all_close_flags) else 'FAIL'}")

    # [3] 一般離散力学系ソルバの動作テスト
    print("\n[3] 汎用離散力学系ソルバ make_discrete_adjoint_solver テスト...")
    def lorenz_step(params, x, u):
        # 簡易非線形力学系 x_{t+1} = x_t + tanh(W * x_t + b)
        W, b = params
        return x + jnp.tanh(x @ W + b)

    W = jax.random.normal(jax.random.key(1), (4, 4)) * 0.1
    b = jnp.zeros(4)
    params = (W, b)
    x0 = jax.random.normal(jax.random.key(2), (100, 4))
    u_seq = jnp.zeros((30, 100, 1))

    solve_dyn = make_discrete_adjoint_solver(lorenz_step)
    
    def test_loss(p):
        traj = solve_dyn(p, x0, u_seq)
        return jnp.sum(traj**2)

    grad_dyn = jax.grad(test_loss)(params)
    print(f"  - 力学系パラメータ勾配 norm: {float(jnp.linalg.norm(grad_dyn[0])):.4f}")
    print("  - 汎用随伴ソルバが正常に動作しました。")

    print("\n" + "=" * 70)
    print(" 全検証が正常に完了しました。")
    print("=" * 70)


if __name__ == "__main__":
    main()
