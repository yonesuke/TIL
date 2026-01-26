# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.20",
#     "flax>=0.10.0",
#     "optax",
#     "rax",
#     "numpy",
# ]
# ///
"""
Rax: Learning-to-Rank with JAX

Implementation of a ranking model using:
- Rax: Google's Learning-to-Rank library
- Flax NNX: Neural network library for JAX

Reference:
- https://github.com/google/rax
- "Rax: Composable Learning-to-Rank Using JAX" (KDD 2022)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rax
from flax import nnx


class RankingMLP(nnx.Module):
    """Simple MLP for document scoring.

    Takes document features and outputs a relevance score.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Build layers
        self.layers = nnx.List([])
        self.norms = nnx.List([])
        self.dropouts = nnx.List([])

        current_dim = in_features
        for _ in range(n_layers):
            self.layers.append(
                nnx.Linear(in_features=current_dim, out_features=hidden_dim, rngs=rngs)
            )
            self.norms.append(nnx.LayerNorm(num_features=hidden_dim, rngs=rngs))
            self.dropouts.append(nnx.Dropout(rate=dropout_rate, rngs=rngs))
            current_dim = hidden_dim

        # Output layer (single score per document)
        self.output_layer = nnx.Linear(
            in_features=hidden_dim, out_features=1, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Document features of shape [batch, list_size, features]
               or [list_size, features]

        Returns:
            Scores of shape [batch, list_size] or [list_size]
        """
        # Handle both batched and unbatched inputs
        input_shape = x.shape
        if len(input_shape) == 2:
            x = x[None, :, :]  # Add batch dimension

        batch_size, list_size, _ = x.shape

        # Reshape to process all documents
        x = x.reshape(-1, self.in_features)  # [batch * list_size, features]

        # Apply MLP layers
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = nnx.gelu(x)
            x = dropout(x)

        # Output score
        x = self.output_layer(x)  # [batch * list_size, 1]
        x = x.squeeze(-1)  # [batch * list_size]
        x = x.reshape(batch_size, list_size)  # [batch, list_size]

        # Remove batch dimension if input was unbatched
        if len(input_shape) == 2:
            x = x.squeeze(0)

        return x


class ListwiseRanker(nnx.Module):
    """Listwise ranking model using Rax losses.

    Supports multiple ranking losses:
    - softmax: Listwise softmax cross-entropy
    - pairwise_logistic: Pairwise logistic loss (RankNet)
    - pairwise_hinge: Pairwise hinge loss (Ranking SVM)
    - approx_ndcg: Approximate NDCG optimization
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        loss_type: str = "softmax",
        *,
        rngs: nnx.Rngs,
    ):
        self.loss_type = loss_type
        self.scorer = RankingMLP(
            in_features=in_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute scores for documents.

        Args:
            features: Document features [batch, list_size, features]

        Returns:
            Scores [batch, list_size]
        """
        return self.scorer(features)

    def compute_loss(
        self, features: jax.Array, labels: jax.Array, where: jax.Array | None = None
    ) -> jax.Array:
        """Compute ranking loss.

        Args:
            features: Document features [batch, list_size, features]
            labels: Relevance labels [batch, list_size]
            where: Optional mask [batch, list_size]

        Returns:
            Scalar loss value
        """
        scores = self(features)

        # Select loss function
        if self.loss_type == "softmax":
            # Listwise softmax loss
            loss_fn = lambda s, l, w: rax.softmax_loss(s, l, where=w)
        elif self.loss_type == "pairwise_logistic":
            # RankNet-style pairwise loss
            loss_fn = lambda s, l, w: rax.pairwise_logistic_loss(s, l, where=w)
        elif self.loss_type == "pairwise_hinge":
            # Ranking SVM-style loss
            loss_fn = lambda s, l, w: rax.pairwise_hinge_loss(s, l, where=w)
        elif self.loss_type == "listmle":
            # ListMLE loss
            loss_fn = lambda s, l, w: rax.listmle_loss(s, l, where=w)
        elif self.loss_type == "approx_ndcg":
            # Approximate NDCG optimization
            loss_fn = lambda s, l, w: rax.approx_t12n(rax.ndcg_metric, temperature=1.0)(
                s, l, where=w
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute loss for each query in batch
        if len(scores.shape) == 1:
            return loss_fn(scores, labels, where)

        # Vectorized over batch
        batch_losses = jax.vmap(loss_fn)(scores, labels, where)
        return jnp.mean(batch_losses)


def create_synthetic_ranking_data(
    n_queries: int = 100,
    list_size: int = 10,
    n_features: int = 16,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Create synthetic ranking data.

    Generates document features and relevance labels where
    relevance is correlated with a hidden "true" relevance signal.

    Args:
        n_queries: Number of queries (batches)
        list_size: Number of documents per query
        n_features: Feature dimension per document
        key: Random key

    Returns:
        features: [n_queries, list_size, n_features]
        labels: [n_queries, list_size] with values in {0, 1, 2, 3}
    """
    if key is None:
        key = jax.random.key(42)

    key1, key2, key3 = jax.random.split(key, 3)

    # Generate random features
    features = jax.random.normal(key1, (n_queries, list_size, n_features))

    # Create a "true" relevance signal based on some features
    # Relevance depends on first few features
    true_weights = jnp.zeros(n_features).at[:4].set(jnp.array([1.0, 0.5, 0.3, 0.2]))
    true_scores = jnp.einsum("qdf,f->qd", features, true_weights)

    # Add noise
    noise = jax.random.normal(key2, true_scores.shape) * 0.5
    noisy_scores = true_scores + noise

    # Convert to discrete labels (0-3 scale)
    # Use percentiles to create balanced labels
    percentiles = jnp.percentile(noisy_scores.flatten(), jnp.array([25, 50, 75]))

    labels = jnp.zeros_like(noisy_scores)
    labels = jnp.where(noisy_scores > percentiles[0], 1.0, labels)
    labels = jnp.where(noisy_scores > percentiles[1], 2.0, labels)
    labels = jnp.where(noisy_scores > percentiles[2], 3.0, labels)

    return features, labels


def evaluate_metrics(
    model: ListwiseRanker, features: jax.Array, labels: jax.Array
) -> dict[str, float]:
    """Evaluate ranking metrics.

    Args:
        model: Trained ranking model
        features: Document features [batch, list_size, features]
        labels: Relevance labels [batch, list_size]

    Returns:
        Dictionary of metric names to values
    """
    scores = model(features)

    # Compute metrics per query then average
    def compute_query_metrics(s, l):
        return {
            "ndcg": rax.ndcg_metric(s, l),
            "ndcg@3": rax.ndcg_metric(s, l, topn=3),
            "ndcg@5": rax.ndcg_metric(s, l, topn=5),
            "mrr": rax.mrr_metric(s, l),
            "precision@3": rax.precision_metric(s, l, topn=3),
        }

    batch_metrics = jax.vmap(compute_query_metrics)(scores, labels)

    return {k: float(jnp.mean(v)) for k, v in batch_metrics.items()}


@nnx.jit
def train_step(
    model: ListwiseRanker,
    optimizer: nnx.Optimizer,
    features: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Single training step."""

    def loss_fn(m):
        return m.compute_loss(features, labels)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def main():
    print("=" * 60)
    print("Rax: Learning-to-Rank with JAX and Flax NNX")
    print("=" * 60)

    # Hyperparameters
    n_features = 16
    hidden_dim = 64
    n_layers = 2
    dropout_rate = 0.1
    loss_type = "softmax"  # Options: softmax, pairwise_logistic, pairwise_hinge, listmle
    n_epochs = 100
    learning_rate = 1e-3

    # Data parameters
    n_train_queries = 200
    n_val_queries = 50
    list_size = 10

    print(f"\nModel Configuration:")
    print(f"  Features: {n_features}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {n_layers}")
    print(f"  Loss type: {loss_type}")
    print(f"\nData Configuration:")
    print(f"  Train queries: {n_train_queries}")
    print(f"  List size: {list_size}")

    # Initialize model
    rngs = nnx.Rngs(0)
    model = ListwiseRanker(
        in_features=n_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        loss_type=loss_type,
        rngs=rngs,
    )

    # Count parameters
    params = nnx.state(model, nnx.Param)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Total parameters: {n_params:,}")

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=learning_rate), wrt=nnx.Param)

    # Generate data
    print(f"\nGenerating synthetic ranking data...")
    key = jax.random.key(42)
    key_train, key_val = jax.random.split(key)

    train_features, train_labels = create_synthetic_ranking_data(
        n_queries=n_train_queries,
        list_size=list_size,
        n_features=n_features,
        key=key_train,
    )
    val_features, val_labels = create_synthetic_ranking_data(
        n_queries=n_val_queries,
        list_size=list_size,
        n_features=n_features,
        key=key_val,
    )
    print(f"  Train: {train_features.shape}, Val: {val_features.shape}")

    # Initial metrics
    model.eval()
    init_metrics = evaluate_metrics(model, val_features, val_labels)
    print(f"\nInitial Metrics:")
    for k, v in init_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 50)

    model.train()
    for epoch in range(n_epochs):
        loss = train_step(model, optimizer, train_features, train_labels)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            metrics = evaluate_metrics(model, val_features, val_labels)
            model.train()
            print(
                f"Epoch {epoch + 1:3d} | Loss: {loss:.4f} | "
                f"NDCG: {metrics['ndcg']:.4f} | MRR: {metrics['mrr']:.4f}"
            )

    print("-" * 50)

    # Final evaluation
    model.eval()
    final_metrics = evaluate_metrics(model, val_features, val_labels)
    print(f"\nFinal Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Compare different loss functions
    print("\n" + "=" * 60)
    print("Comparing Loss Functions")
    print("=" * 60)

    loss_types = ["softmax", "pairwise_logistic", "pairwise_hinge", "listmle"]
    results = {}

    for lt in loss_types:
        print(f"\nTraining with {lt} loss...")
        rngs = nnx.Rngs(0)
        m = ListwiseRanker(
            in_features=n_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            loss_type=lt,
            rngs=rngs,
        )
        opt = nnx.Optimizer(m, optax.adam(learning_rate=learning_rate), wrt=nnx.Param)

        m.train()
        for _ in range(n_epochs):
            train_step(m, opt, train_features, train_labels)

        m.eval()
        results[lt] = evaluate_metrics(m, val_features, val_labels)

    print("\n" + "-" * 50)
    print("Results Summary:")
    print("-" * 50)
    print(f"{'Loss Type':<20} {'NDCG':<10} {'NDCG@3':<10} {'MRR':<10}")
    print("-" * 50)
    for lt, metrics in results.items():
        print(f"{lt:<20} {metrics['ndcg']:.4f}     {metrics['ndcg@3']:.4f}     {metrics['mrr']:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
