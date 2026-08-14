"""Leakage-safe train/validation splits for trajectory data."""
from __future__ import annotations

import numpy as np


def episode_validation_mask(
    episode_ids: np.ndarray,
    rng_seed: int,
    val_frac: float = 0.2,
) -> np.ndarray:
    """Return a validation mask that keeps whole episodes together."""
    return group_validation_mask(episode_ids, rng_seed=rng_seed, val_frac=val_frac)


def group_validation_mask(
    group_ids: np.ndarray,
    rng_seed: int,
    val_frac: float = 0.2,
) -> np.ndarray:
    """Hold out a fraction of groups; all rows of a group share train/val."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    groups = np.unique(group_ids)
    if len(groups) < 2:
        raise ValueError("group split requires at least two distinct groups")

    n_val = int(len(groups) * val_frac)
    n_val = min(max(1, n_val), len(groups) - 1)
    rng = np.random.RandomState(rng_seed)
    val_groups = rng.choice(groups, n_val, replace=False)
    return np.isin(group_ids, val_groups)


def checkpoint_validation_mask(
    ckpt_ids: np.ndarray,
    rng_seed: int,
    val_frac: float = 0.2,
) -> np.ndarray:
    """Hold out whole checkpoint seeds (no episode from a val ckpt in train)."""
    return group_validation_mask(ckpt_ids, rng_seed=rng_seed, val_frac=val_frac)


def leave_one_checkpoint_out_folds(ckpt_ids: np.ndarray) -> list[np.ndarray]:
    """Return one validation mask per distinct checkpoint seed."""
    ckpt_ids = np.asarray(ckpt_ids)
    folds = []
    for cid in np.unique(ckpt_ids):
        folds.append(ckpt_ids == cid)
    if len(folds) < 2:
        raise ValueError("leave-one-checkpoint-out requires at least two checkpoints")
    return folds
