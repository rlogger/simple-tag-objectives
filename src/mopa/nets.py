"""Shared network modules for loading MAPPO actors and greedy rollouts.

``ActorLogits`` is param-compatible with ``scripts.train_mappo.Actor`` but
returns raw logits (no Distrax) so offline eval / BC can ``argmax`` without
the trainer stack.
"""
from __future__ import annotations

import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ActorLogits(nn.Module):
    """Two-layer MLP policy head returning raw action logits."""

    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):  # noqa: ANN001
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
