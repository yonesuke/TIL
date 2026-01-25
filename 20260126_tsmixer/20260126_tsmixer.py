# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.20",
#     "flax>=0.10.0",
#     "optax",
#     "numpy",
# ]
# ///
"""
TSMixer: Time-Series Mixer for Time Series Forecasting

Implementation based on:
"TSMixer: An All-MLP Architecture for Time Series Forecasting"
https://arxiv.org/abs/2303.06053

This implementation uses JAX and Flax NNX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


class ResBlock(nnx.Module):
    """Residual block of TSMixer.

    Consists of:
    1. Temporal Linear: Mixing along the time dimension
    2. Feature Linear: Mixing along the feature/channel dimension
    """

    def __init__(
        self,
        input_length: int,
        n_channels: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_length = input_length
        self.n_channels = n_channels
        self.ff_dim = ff_dim

        # Temporal mixing layers
        # Normalize over channel dimension (last axis)
        self.temporal_norm = nnx.RMSNorm(num_features=n_channels, rngs=rngs)
        self.temporal_dense = nnx.Linear(
            in_features=input_length,
            out_features=input_length,
            rngs=rngs,
        )
        self.temporal_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        # Feature mixing layers
        self.feature_norm = nnx.RMSNorm(num_features=n_channels, rngs=rngs)
        self.feature_dense1 = nnx.Linear(
            in_features=n_channels,
            out_features=ff_dim,
            rngs=rngs,
        )
        self.feature_dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.feature_dense2 = nnx.Linear(
            in_features=ff_dim,
            out_features=n_channels,
            rngs=rngs,
        )
        self.feature_dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: [Batch, Input Length, Channel]

        # Temporal Linear
        residual = x
        x = self.temporal_norm(x)
        x = jnp.transpose(x, (0, 2, 1))  # [Batch, Channel, Input Length]
        x = self.temporal_dense(x)
        x = nnx.gelu(x)
        x = jnp.transpose(x, (0, 2, 1))  # [Batch, Input Length, Channel]
        x = self.temporal_dropout(x)
        x = x + residual

        # Feature Linear
        residual = x
        x = self.feature_norm(x)
        x = self.feature_dense1(x)  # [Batch, Input Length, FF_Dim]
        x = nnx.gelu(x)
        x = self.feature_dropout1(x)
        x = self.feature_dense2(x)  # [Batch, Input Length, Channel]
        x = self.feature_dropout2(x)
        x = x + residual

        return x


class TSMixer(nnx.Module):
    """TSMixer model for time series forecasting.

    Args:
        input_length: Length of input sequence
        pred_length: Length of prediction sequence
        n_channels: Number of input channels/features
        n_blocks: Number of residual blocks
        ff_dim: Hidden dimension in feature mixing MLP
        dropout_rate: Dropout rate
        target_slice: Optional slice for selecting target channels in output
        rngs: Random number generators for initialization
    """

    def __init__(
        self,
        input_length: int,
        pred_length: int,
        n_channels: int,
        n_blocks: int = 2,
        ff_dim: int = 64,
        dropout_rate: float = 0.1,
        target_slice: slice | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_length = input_length
        self.pred_length = pred_length
        self.n_channels = n_channels
        self.target_slice = target_slice

        # Calculate output channels
        self.output_channels = n_channels
        if target_slice is not None:
            # Estimate output channels from slice
            dummy = jnp.zeros(n_channels)
            self.output_channels = dummy[target_slice].shape[0]

        # Stack of residual blocks (use nnx.List for proper pytree handling)
        self.blocks = nnx.List([
            ResBlock(
                input_length=input_length,
                n_channels=n_channels,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            for _ in range(n_blocks)
        ])

        # Output projection
        self.output_dense = nnx.Linear(
            in_features=input_length,
            out_features=pred_length,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor of shape [Batch, Input Length, Channel]

        Returns:
            Output tensor of shape [Batch, Pred Length, Output Channel]
        """
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Select target channels if specified
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]

        # Project to prediction length
        x = jnp.transpose(x, (0, 2, 1))  # [Batch, Channel, Input Length]
        x = self.output_dense(x)  # [Batch, Channel, Pred Length]
        x = jnp.transpose(x, (0, 2, 1))  # [Batch, Pred Length, Channel]

        return x


def create_synthetic_data(
    batch_size: int = 32,
    input_length: int = 96,
    pred_length: int = 24,
    n_channels: int = 7,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Create synthetic time series data for demonstration."""
    if key is None:
        key = jax.random.key(42)

    key1, key2 = jax.random.split(key)

    # Generate input sequences
    t = jnp.linspace(0, 4 * jnp.pi, input_length + pred_length)
    base_signal = jnp.sin(t) + 0.5 * jnp.sin(2 * t)

    # Create multi-channel data with noise
    x_data = []
    y_data = []

    for i in range(batch_size):
        key1, subkey = jax.random.split(key1)
        noise = jax.random.normal(subkey, (input_length + pred_length, n_channels)) * 0.1
        channel_weights = jax.random.uniform(
            jax.random.fold_in(key2, i), (n_channels,), minval=0.5, maxval=1.5
        )
        signal = base_signal[:, None] * channel_weights + noise
        x_data.append(signal[:input_length])
        y_data.append(signal[input_length : input_length + pred_length])

    return jnp.stack(x_data), jnp.stack(y_data)


def mse_loss(model: TSMixer, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean squared error loss."""
    pred = model(x)
    return jnp.mean((pred - y) ** 2)


@nnx.jit
def train_step(
    model: TSMixer,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Single training step."""
    loss, grads = nnx.value_and_grad(mse_loss)(model, x, y)
    optimizer.update(model, grads)
    return loss


def main():
    print("=" * 60)
    print("TSMixer: Time-Series Mixer for Time Series Forecasting")
    print("=" * 60)

    # Hyperparameters
    input_length = 96
    pred_length = 24
    n_channels = 7
    n_blocks = 2
    ff_dim = 64
    dropout_rate = 0.1
    batch_size = 32
    n_epochs = 100

    print(f"\nModel Configuration:")
    print(f"  Input length: {input_length}")
    print(f"  Prediction length: {pred_length}")
    print(f"  Number of channels: {n_channels}")
    print(f"  Number of blocks: {n_blocks}")
    print(f"  Feed-forward dimension: {ff_dim}")
    print(f"  Dropout rate: {dropout_rate}")

    # Initialize model
    rngs = nnx.Rngs(0)
    model = TSMixer(
        input_length=input_length,
        pred_length=pred_length,
        n_channels=n_channels,
        n_blocks=n_blocks,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    # Count parameters
    params = nnx.state(model, nnx.Param)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Total parameters: {n_params:,}")

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    key = jax.random.key(42)
    x_train, y_train = create_synthetic_data(
        batch_size=batch_size,
        input_length=input_length,
        pred_length=pred_length,
        n_channels=n_channels,
        key=key,
    )
    print(f"  Training data shape: X={x_train.shape}, Y={y_train.shape}")

    # Validation data
    x_val, y_val = create_synthetic_data(
        batch_size=batch_size // 4,
        input_length=input_length,
        pred_length=pred_length,
        n_channels=n_channels,
        key=jax.random.key(123),
    )

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 40)

    for epoch in range(n_epochs):
        # Training step
        loss = train_step(model, optimizer, x_train, y_train)

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Validation loss (with dropout disabled)
            model.eval()
            val_loss = mse_loss(model, x_val, y_val)
            model.train()
            print(f"Epoch {epoch + 1:3d} | Train Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")

    print("-" * 40)
    print("Training complete!")

    # Final evaluation
    model.eval()
    final_val_loss = mse_loss(model, x_val, y_val)
    print(f"\nFinal Validation Loss: {final_val_loss:.6f}")

    # Inference example
    print("\nInference Example:")
    sample_input = x_val[:1]
    sample_output = model(sample_input)
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output shape: {sample_output.shape}")
    print(f"  Output range: [{float(sample_output.min()):.4f}, {float(sample_output.max()):.4f}]")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
