"""Team helpers used by batched evaluation rollouts."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def select_active(active: Any, new: Any, old: Any) -> Any:
    """Keep ``old`` wherever ``active`` is False (freeze finished episodes)."""
    mask = active
    while mask.ndim < new.ndim:
        mask = mask[..., None]
    return jnp.where(mask, new, old)


def freeze_tree(active: Any, new_tree: Any, old_tree: Any) -> Any:
    """Apply :func:`select_active` across a pytree of arrays."""
    return jax.tree.map(lambda n, o: select_active(active, n, o), new_tree, old_tree)
