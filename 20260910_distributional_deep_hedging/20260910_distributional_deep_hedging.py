# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax",
#     "flax>=0.11.0",
#     "optax",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Distributional Deep Hedging with Actor-Critic (JAX / Flax NNX)
==============================================================
A minimal, model-free implementation of:
1. Actor: Hedging ratio delta(t, S)
2. Critic: Distributional Value (Implicit Quantile Network for future PnL)
3. Distributional Temporal Difference (TD) Learning with Quantile Huber Loss
4. Pure database/replay-buffer transition based (no analytical pricing in loss)
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# =============================================================================
# 1. Market & Option Specs (Normalized units: K = 1.0)
# =============================================================================
S0 = 100.0          # Base asset price
K = 100.0           # Strike price
R = 0.0             # Risk-free rate (for simplicity)
SIGMA = 0.20        # Volatility (20%)
T_EXP = 30.0 / 365.0  # Maturity: 30 days (~0.0822 yr)
N_STEPS = 30        # Daily rebalancing (30 steps)
DT = T_EXP / N_STEPS

# =============================================================================
# 2. Black-Scholes Analytical Benchmark (For Ground Truth Comparison)
# =============================================================================
def bs_delta(s, tau, k=K, r=R, sigma=SIGMA):
    """Analytical BS Delta for European Call."""
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return 0.5 * (1.0 + jax.scipy.special.erf(d1 / np.sqrt(2.0)))

# =============================================================================
# 3. Networks: Actor and Distributional Critic (IQN) in Flax NNX
# =============================================================================
class Actor(nnx.Module):
    """Policy Network: Maps (moneyness, tau) -> delta in [0, 1]."""
    def __init__(self, hidden_dim: int = 64, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(2, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x):
        # x: (batch, 2) where x = [log(S/K), tau_mat]
        h = nnx.gelu(self.fc1(x))
        h = nnx.gelu(self.fc2(h))
        # Sigmoid restricts hedge ratio delta to [0, 1] for European Call
        return nnx.sigmoid(self.fc_out(h)).squeeze(-1)


class DistributionalCritic(nnx.Module):
    """
    IQN Critic: Maps (state, action, tau_quantile) -> Quantile of future PnL.
    Uses Cosine Embedding for tau in [0, 1].
    """
    def __init__(self, hidden_dim: int = 64, num_cosines: int = 32, rngs: nnx.Rngs = None):
        self.num_cosines = num_cosines
        # State + Action projection: [log(S/K), tau_mat, delta] -> hidden_dim
        self.fc_sa = nnx.Linear(3, hidden_dim, rngs=rngs)
        # Quantile embedding projection
        self.fc_tau = nnx.Linear(num_cosines, hidden_dim, rngs=rngs)
        # Combined layers
        self.fc1 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, sa, tau):
        # sa: (batch, 3) [log(S/K), tau_mat, delta]
        # tau: (batch, num_quantiles) in [0, 1]
        batch_size = sa.shape[0]
        num_q = tau.shape[1]

        # 1. State-Action feature: (batch, 1, hidden_dim)
        phi_sa = nnx.gelu(self.fc_sa(sa))[:, None, :]

        # 2. Cosine embedding for tau: (batch, num_q, num_cosines)
        i_vec = jnp.arange(self.num_cosines)[None, None, :]  # (1, 1, num_cosines)
        cos_tau = jnp.cos(jnp.pi * i_vec * tau[:, :, None])  # (batch, num_q, num_cosines)
        phi_tau = nnx.gelu(self.fc_tau(cos_tau))              # (batch, num_q, hidden_dim)

        # 3. Element-wise fusion (Hadamard product)
        h = phi_sa * phi_tau                                 # (batch, num_q, hidden_dim)
        h = nnx.gelu(self.fc1(h))
        out = self.fc_out(h)                                 # (batch, num_q, 1)
        return out.squeeze(-1)                               # (batch, num_q)


# =============================================================================
# 4. Quantile Huber Loss (Distributional TD)
# =============================================================================
def quantile_huber_loss(y_target, q_pred, tau, kappa=0.01):
    """
    y_target: (batch, num_target_q)
    q_pred:   (batch, num_pred_q)
    tau:      (batch, num_pred_q)
    """
    diff = y_target[:, None, :] - q_pred[:, :, None]  # (batch, num_pred_q, num_target_q)

    # Huber loss
    abs_diff = jnp.abs(diff)
    huber = jnp.where(abs_diff <= kappa, 0.5 * (diff**2), kappa * (abs_diff - 0.5 * kappa))

    # Asymmetric quantile penalty
    tau_expanded = tau[:, :, None]  # (batch, num_pred_q, 1)
    weight = jnp.abs(tau_expanded - (diff < 0).astype(jnp.float32))

    loss = weight * (huber / kappa)
    return jnp.mean(loss)


# =============================================================================
# 5. Data Pipeline: Pure Transition Database
# =============================================================================
def generate_market_database(num_paths=10000, seed=42):
    """
    Simulates historical stock price transitions (or load real market tick data).
    Each record: (step_idx, S_t, S_{t+1}, is_terminal)
    """
    rng = np.random.default_rng(seed)
    dt = DT
    z = rng.standard_normal((num_paths, N_STEPS))
    
    # Simulate spot paths: S0 = 100
    s_paths = np.zeros((num_paths, N_STEPS + 1))
    s_paths[:, 0] = S0 * np.exp(rng.uniform(-0.15, 0.15, size=num_paths))
    for t in range(N_STEPS):
        drift = (R - 0.5 * SIGMA**2) * dt
        diff = SIGMA * np.sqrt(dt) * z[:, t]
        s_paths[:, t + 1] = s_paths[:, t] * np.exp(drift + diff)

    # Flatten into transition tuples
    steps = np.tile(np.arange(N_STEPS), num_paths)
    s_t = s_paths[:, :-1].flatten()
    s_next = s_paths[:, 1:].flatten()
    is_terminal = (steps == (N_STEPS - 1)).astype(np.float32)

    return {
        "step": steps,
        "s_t": s_t,
        "s_next": s_next,
        "is_terminal": is_terminal,
        "total_transitions": len(steps)
    }


# =============================================================================
# 6. Training Pipeline
# =============================================================================
def train():
    print("=" * 70)
    print("Training Distributional Deep Hedging (Actor-Critic) in JAX / Flax NNX")
    print("=" * 70)

    # 1. Prepare Market Data
    db = generate_market_database(num_paths=10000, seed=123)
    num_data = db["total_transitions"]
    print(f"Total Market Transitions in DB: {num_data:,}")

    # 2. Initialize Models
    rngs = nnx.Rngs(42)
    actor = Actor(hidden_dim=64, rngs=rngs)
    critic = DistributionalCritic(hidden_dim=64, num_cosines=32, rngs=rngs)
    target_critic = DistributionalCritic(hidden_dim=64, num_cosines=32, rngs=rngs)

    opt_actor = nnx.Optimizer(actor, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
    opt_critic = nnx.Optimizer(critic, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

    # Sync target network initial weights
    nnx.update(target_critic, nnx.state(critic, nnx.Param))

    # Batch parameters
    batch_size = 1024
    num_q = 32
    num_epochs = 15
    steps_per_epoch = num_data // batch_size

    # Loss Functions
    def critic_loss_fn(critic, target_critic, actor, batch, tau_pred, tau_target):
        s_t = batch["s_t"]
        s_next = batch["s_next"]
        step = batch["step"]
        is_term = batch["is_terminal"]

        tau_mat = (N_STEPS - step) * DT
        log_m = jnp.log(s_t / K)
        x_curr = jnp.stack([log_m, tau_mat], axis=-1)

        delta = actor(x_curr)
        sa_curr = jnp.stack([log_m, tau_mat, delta], axis=-1)

        # 1-step PnL: delta * (S_{t+1} - S_t) / K
        r_pnl = delta * (s_next - s_t) / K

        # Next state
        tau_mat_next = jnp.maximum((N_STEPS - (step + 1)) * DT, 0.0)
        log_m_next = jnp.log(s_next / K)
        x_next = jnp.stack([log_m_next, tau_mat_next], axis=-1)
        next_delta = actor(x_next)
        sa_next = jnp.stack([log_m_next, tau_mat_next, next_delta], axis=-1)

        # Target distribution
        q_next = target_critic(sa_next, tau_target)
        intermediate_target = r_pnl[:, None] + q_next
        payoff = jnp.maximum(0.0, s_next / K - 1.0)
        terminal_target = (r_pnl - payoff)[:, None]

        y_target = jnp.where(is_term[:, None] > 0.5, terminal_target, intermediate_target)
        y_target = jax.lax.stop_gradient(y_target)

        q_pred = critic(sa_curr, tau_pred)
        loss = quantile_huber_loss(y_target, q_pred, tau_pred)
        return loss

    def actor_loss_fn(actor, critic, batch, tau_sample):
        s_t = batch["s_t"]
        step = batch["step"]
        tau_mat = (N_STEPS - step) * DT
        log_m = jnp.log(s_t / K)
        x_curr = jnp.stack([log_m, tau_mat], axis=-1)

        delta = actor(x_curr)
        sa_curr = jnp.stack([log_m, tau_mat, delta], axis=-1)
        q_pred = critic(sa_curr, tau_sample)

        # Minimize tail loss (maximize lower tail PnL)
        cvar_loss = -jnp.mean(q_pred)
        return cvar_loss

    @nnx.jit
    def train_step(actor, critic, target_critic, opt_actor, opt_critic, batch, key):
        k1, k2, k3 = jax.random.split(key, 3)
        tau_pred = jax.random.uniform(k1, (batch["s_t"].shape[0], num_q), minval=0.01, maxval=0.99)
        tau_target = jax.random.uniform(k2, (batch["s_t"].shape[0], num_q), minval=0.01, maxval=0.99)
        tau_cvar = jax.random.uniform(k3, (batch["s_t"].shape[0], num_q), minval=0.01, maxval=0.20)

        # 1. Update Critic
        grad_critic_fn = nnx.value_and_grad(critic_loss_fn)
        c_loss, grads_c = grad_critic_fn(critic, target_critic, actor, batch, tau_pred, tau_target)
        opt_critic.update(critic, grads_c)

        # 2. Update Actor
        grad_actor_fn = nnx.value_and_grad(actor_loss_fn)
        a_loss, grads_a = grad_actor_fn(actor, critic, batch, tau_cvar)
        opt_actor.update(actor, grads_a)

        # 3. Polyak averaging target update
        c_state = nnx.state(critic, nnx.Param)
        t_state = nnx.state(target_critic, nnx.Param)
        polyak_tau = 0.05
        new_target = jax.tree.map(lambda c, t: (1.0 - polyak_tau) * t + polyak_tau * c, c_state, t_state)
        nnx.update(target_critic, new_target)

        return c_loss, a_loss

    # Training Loop
    t0 = time.time()
    rng_key = jax.random.PRNGKey(0)

    for epoch in range(num_epochs):
        perm = np.random.permutation(num_data)
        epoch_c_loss = []
        epoch_a_loss = []

        for step_i in range(steps_per_epoch):
            idx = perm[step_i * batch_size : (step_i + 1) * batch_size]
            batch = {
                "step": jnp.array(db["step"][idx]),
                "s_t": jnp.array(db["s_t"][idx]),
                "s_next": jnp.array(db["s_next"][idx]),
                "is_terminal": jnp.array(db["is_terminal"][idx]),
            }

            rng_key, subkey = jax.random.split(rng_key)
            c_loss, a_loss = train_step(actor, critic, target_critic, opt_actor, opt_critic, batch, subkey)
            epoch_c_loss.append(float(c_loss))
            epoch_a_loss.append(float(a_loss))

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}]  Critic TD Loss: {np.mean(epoch_c_loss):.6f} | Actor CVaR Loss: {np.mean(epoch_a_loss):.6f}", flush=True)

    print(f"Training finished in {time.time() - t0:.2f} seconds.")

    # =========================================================================
    # 7. Validation & Visualization
    # =========================================================================
    print("Evaluating learned policy against Black-Scholes benchmark...")
    
    s_test = np.linspace(70.0, 130.0, 100)
    tau_test = 15.0 / 365.0  # At mid-life (15 days left)
    
    # BS Delta
    bs_deltas = bs_delta(s_test, tau_test)
    
    # NN Actor Delta
    x_eval = jnp.stack([jnp.log(s_test / K), jnp.full_like(s_test, tau_test)], axis=-1)
    nn_deltas = np.array(actor(x_eval))

    # PnL Quantiles from Critic for At-The-Money (S = 100)
    tau_grid = np.linspace(0.01, 0.99, 50)
    sa_atm = jnp.array([[0.0, tau_test, nn_deltas[50]]])
    tau_in = jnp.array(tau_grid)[None, :]
    pnl_quantiles = np.array(critic(sa_atm, tau_in)).flatten() * K

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Delta comparison
    axes[0].plot(s_test, bs_deltas, "k--", label="Black-Scholes Delta (Theoretical)")
    axes[0].plot(s_test, nn_deltas, "r-", linewidth=2, label="Actor Learned Delta (Model-Free TD)")
    axes[0].set_title(f"Hedge Ratio Delta (tau = 15 days, K = {K})")
    axes[0].set_xlabel("Spot Price S")
    axes[0].set_ylabel("Hedge Ratio delta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Critic PnL Quantile Distribution
    axes[1].plot(tau_grid, pnl_quantiles, "b-o", markersize=3, label="Critic Quantile Function Q(tau)")
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_title("Critic Predicted Hedging PnL Distribution (ATM, tau = 15d)")
    axes[1].set_xlabel("Quantile Level tau in [0, 1]")
    axes[1].set_ylabel("Hedging PnL ($)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_path = Path(__file__).parent / "20260910_distributional_deep_hedging.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Results figure saved to: {out_path}")


if __name__ == "__main__":
    train()
