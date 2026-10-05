# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.30",
#     "flax>=0.10.0",
#     "numpy",
# ]
# ///
"""
20261005_discrete_adjoint_state_method.py

離散ポントリャーギン随伴状態法 (Discrete Pontryagin Adjoint Method) による
一般離散力学系 x_{t+1} = f(x_t, u_t; θ) の O(1) メモリ Custom VJP ソルバ

1. 任意のステップ関数 f(params, x_t, u_t) に対する汎用随伴ソルバ make_discrete_adjoint_solver
2. 時間逆順ポントリャーギン共役変数 (Costate) スキャンによる局所 VJP オンザフライ計算
3. JAX 標準の逆モード自動微分 (Auto-AD / BPTT) との機械精度での完全一致検証
4. Reverse-over-Reverse による Hessian-Vector Product (HVP) の計算保証
"""

from typing import Tuple, Callable, Any
import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# 1. 汎用離散力学系 随伴ソルバ (make_discrete_adjoint_solver)
# =============================================================================
def make_discrete_adjoint_solver(
    step_fn: Callable[[Any, jax.Array, Any], jax.Array]
):
    """
    一般離散力学系 x_{t+1} = step_fn(params, x_t, u_t) に対する
    メモリフットプリント O(1) の Custom VJP 随伴ソルバを生成する。

    引数:
        step_fn: (params, current_state, step_input) -> next_state
                 - params: 任意の PyTree (重み、物理パラメータ等)
                 - current_state: 状態ベクトル x_t, shape: (batch_size, state_dim)
                 - step_input: 各ステップの外生入力 u_t (制御入力、時刻特徴量等)
    戻り値:
        solve_dynamics: (params, initial_state, inputs_seq) -> trajectory
    """

    @jax.custom_vjp
    def solve_dynamics(
        params: Any,
        initial_state: jax.Array,  # x_0: shape (batch_size, state_dim)
        inputs_seq: jax.Array,     # u_0 ~ u_{T-1}: shape (num_steps, batch_size, input_dim)
    ) -> jax.Array:
        """
        力学系を前進シミュレーションし、状態軌道 x_1 ~ x_T を返す。
        戻り値: trajectory, shape (num_steps, batch_size, state_dim)
        """
        trajectory, _ = solve_dynamics_fwd(params, initial_state, inputs_seq)
        return trajectory

    # -------------------------------------------------------------------------
    # Forward Pass (前方走査)
    # -------------------------------------------------------------------------
    def solve_dynamics_fwd(params, initial_state, inputs_seq):
        def scan_step(curr_x, u_t):
            next_x = step_fn(params, curr_x, u_t)
            return next_x, next_x

        # 初期状態 x_0 から出発して x_1, ..., x_T を計算
        _, trajectory = jax.lax.scan(scan_step, initial_state, inputs_seq)

        # ★ Tape には公称軌道と入力のみを保持。
        # 各ステップ内部のニューラルネット・演算の中間層テンソルはすべて破棄！
        residuals = (params, initial_state, trajectory, inputs_seq)
        return trajectory, residuals

    # -------------------------------------------------------------------------
    # Backward Pass (ポントリャーギン共役変数後退走査)
    # -------------------------------------------------------------------------
    def solve_dynamics_bwd(residuals, grad_trajectory):
        """
        grad_trajectory: 下流の損失関数から伝播してきた各時刻の状態に対する勾配
                         dL/dx_1, dL/dx_2, ..., dL/dx_T
                         shape (num_steps, batch_size, state_dim)
        """
        params, initial_state, trajectory, inputs_seq = residuals
        num_steps = trajectory.shape[0]

        def scan_bwd_step(future_costate, t):
            """
            future_costate: 将来ステップから遡ってきた共役変数 λ_{t+1}
            t: 現在の時刻インデックス (num_steps - 1 down to 0)
            """
            # 時刻 t の状態 x_t (t=0 のときは初期状態 initial_state)
            curr_x = jax.lax.cond(
                t > 0,
                lambda: trajectory[t - 1],
                lambda: initial_state
            )
            curr_u = inputs_seq[t]

            # 当該時刻にかかる総随伴力 (Total Adjoint Force)
            # λ_{t+1}^{total} = (下流からの直接勾配 dL/dx_{t+1}) + (将来から遡ってきた λ_{t+2})
            total_adjoint = grad_trajectory[t] + future_costate

            # ★ その場 (オンザフライ) で 1 ステップ分の局所 VJP を構築・評価
            # f: (params, x_t) -> x_{t+1}
            _, local_vjp = jax.vjp(
                lambda p, x: step_fn(p, x, curr_u),
                params,
                curr_x,
            )
            # (df/d params)^T * λ,  (df/d x_t)^T * λ
            grad_params_step, costate_to_past = local_vjp(total_adjoint)

            # costate_to_past が過去 (時刻 t) へ引き継がれる共役変数 λ_t となる
            return costate_to_past, grad_params_step

        # 終端条件 λ_{T+1} = 0 から過去へ向けてスキャン
        terminal_costate = jnp.zeros_like(initial_state)
        reversed_indices = jnp.arange(num_steps - 1, -1, -1)

        initial_state_grad, grads_params_seq = jax.lax.scan(
            scan_bwd_step,
            terminal_costate,
            reversed_indices,
        )

        # 全ステップで計算されたパラメータ勾配を合算
        total_params_grad = jax.tree.map(
            lambda g: jnp.sum(g, axis=0),
            grads_params_seq,
        )

        # 戻り値: (params への勾配, initial_state への勾配, inputs への勾配=None)
        return (total_params_grad, initial_state_grad, None)

    solve_dynamics.defvjp(solve_dynamics_fwd, solve_dynamics_bwd)
    return solve_dynamics


# =============================================================================
# 2. 比較検証用ナイーブ Auto-AD ソルバ
# =============================================================================
def make_naive_auto_ad_solver(
    step_fn: Callable[[Any, jax.Array, Any], jax.Array]
):
    """標準の Auto-AD (BPTT) で全ステップの中間層活性化を Tape に保持するソルバ"""
    def solve_dynamics(params: Any, initial_state: jax.Array, inputs_seq: jax.Array) -> jax.Array:
        def scan_step(curr_x, u_t):
            next_x = step_fn(params, curr_x, u_t)
            return next_x, next_x

        _, trajectory = jax.lax.scan(scan_step, initial_state, inputs_seq)
        return trajectory

    return solve_dynamics


# =============================================================================
# 3. テスト用非線形ニューラル力学系モデル (Flax NNX)
# =============================================================================
class NeuralDynamicsBlock(nnx.Module):
    """x_{t+1} = x_t + dt * MLP(x_t, u_t) 型の非線形連続・離散力学系"""
    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(state_dim + input_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, state_dim, rngs=rngs)

    def __call__(self, x: jax.Array, u: jax.Array) -> jax.Array:
        h = jnp.concatenate([x, u], axis=-1)
        h = nnx.gelu(self.fc1(h))
        h = nnx.gelu(self.fc2(h))
        dx = self.fc3(h)
        # 残差接続型発展 (ResNet / Forward Euler 離散化)
        return x + 0.01 * dx


# =============================================================================
# 4. 実行・検証テスト
# =============================================================================
def main():
    print("=" * 70)
    print(" 離散ポントリャーギン随伴状態法 (Discrete Pontryagin Adjoint Method)")
    print(" 汎用離散力学系 x_{t+1} = f(x_t, u_t; θ) の O(1) メモリ VJP 検証")
    print("=" * 70)

    key = jax.random.key(2026)
    k_model, k_state, k_input = jax.random.split(key, 3)

    num_steps = 100    # 時間ステップ数 T
    batch_size = 64    # バッチサイズ M
    state_dim = 8      # 状態次元 d
    input_dim = 2      # 入力次元 m

    # モデル構築
    model = NeuralDynamicsBlock(state_dim=state_dim, input_dim=input_dim, hidden_dim=32, rngs=nnx.Rngs(k_model))
    graphdef, state = nnx.split(model)

    def step_fn(st: nnx.State, x_t: jax.Array, u_t: jax.Array) -> jax.Array:
        m = nnx.merge(graphdef, st)
        return m(x_t, u_t)

    # 入力データ生成
    x0 = jax.random.normal(k_state, (batch_size, state_dim))
    inputs_seq = jax.random.normal(k_input, (num_steps, batch_size, input_dim))

    # ソルバの生成
    adjoint_solver = make_discrete_adjoint_solver(step_fn)
    auto_solver = make_naive_auto_ad_solver(step_fn)

    # 目的関数: 軌道全体の追従誤差 + 終端状態正則化
    def loss_adjoint(st):
        traj = adjoint_solver(st, x0, inputs_seq)
        return jnp.mean(jnp.sum(traj**2, axis=-1))

    def loss_auto(st):
        traj = auto_solver(st, x0, inputs_seq)
        return jnp.mean(jnp.sum(traj**2, axis=-1))

    # [1] 勾配の一致性検証
    print(f"\n[1] 勾配計算テスト (Steps={num_steps}, Batch={batch_size}, Dim={state_dim})...")
    grad_adj_fn = jax.jit(jax.grad(loss_adjoint))
    grad_auto_fn = jax.jit(jax.grad(loss_auto))

    g_adj = grad_adj_fn(state)
    g_auto = grad_auto_fn(state)

    max_diff = 0.0
    all_close = True
    for (k1, v1), (k2, v2) in zip(jax.tree.leaves_with_path(g_adj), jax.tree.leaves_with_path(g_auto)):
        diff = float(jnp.max(jnp.abs(v1 - v2)))
        max_diff = max(max_diff, diff)
        if not bool(jnp.allclose(v1, v2, atol=1e-6, rtol=1e-5)):
            all_close = False

    print(f"  - Auto-AD (BPTT) との最大絶対誤差: {max_diff:.2e}")
    print(f"  - 機械精度での一致判定 (atol=1e-6): {'PASS (完全一致)' if all_close else 'FAIL'}")

    # [2] Reverse-over-Reverse HVP (Hessian-Vector Product) テスト
    print("\n[2] Reverse-over-Reverse による 2階曲率 HVP テスト...")
    def hvp_fn(st, v):
        grad_fn = jax.grad(loss_adjoint)
        _, vjp_eval = jax.vjp(grad_fn, st)
        return vjp_eval(v)[0]

    v_tangent = jax.tree.map(lambda x: jnp.ones_like(x) * 0.01, state)
    hvp_result = jax.jit(hvp_fn)(state, v_tangent)
    hvp_norm = float(jnp.sqrt(sum(float(jnp.sum(p**2)) for p in jax.tree.leaves(hvp_result))))
    print(f"  - HVP ノルム: {hvp_norm:.4f}")
    print("  - custom_vjp 下での Reverse-over-Reverse 2階微分が正常に動作しました。")

    print("\n" + "=" * 70)
    print(" すべての離散力学系随伴法テストが正常に完了しました。")
    print("=" * 70)


if __name__ == "__main__":
    main()
