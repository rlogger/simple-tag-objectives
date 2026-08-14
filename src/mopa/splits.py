"""Leakage-safe train/validation splits for trajectory data."""
from __future__ import annotations

import numpy as np


def episode_validation_mask(
    episode_ids: np.ndarray,
    rng_seed: int,
    val_frac: float = 0.2,
) -> np.ndarray:
    """Return a validation mask that keeps whole episodes together."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    eps = np.unique(episode_ids)
    if len(eps) < 2:
        raise ValueError("episode-level split requires at least two episodes")

    n_val = int(len(eps) * val_frac)
    n_val = min(max(1, n_val), len(eps) - 1)
    rng = np.random.RandomState(rng_seed)
    val_eps = rng.choice(eps, n_val, replace=False)
    return np.isin(episode_ids, val_eps)
