"""Behaviour-cloning sample construction for predator policies.

Defaults target the 1v1 objective-typed env (one predator). Multi-predator
teams are supported via ``n_predators``.
"""
from __future__ import annotations

import numpy as np

from mopa.features import EP_LEN


def valid_bc_steps(t0: int, t1: int, ep_len: int = EP_LEN) -> tuple[int, ...]:
    """Steps with a valid one-step velocity feature.

    Auto-reset lands at indices ``0, ep_len+1, …``; velocity at those indices
    spans two episodes and must be excluded.
    """
    if t0 < 0 or t1 <= t0:
        raise ValueError(f"invalid step range [{t0}, {t1})")
    period = ep_len + 1
    return tuple(t for t in range(max(1, t0), t1) if t % period != 0)


def predator_state_features(
    preds: np.ndarray,
    prey: np.ndarray,
    prev_preds: np.ndarray,
    prev_prey: np.ndarray,
) -> np.ndarray:
    """Build per-predator BC features for one timestep.

    ``preds`` / ``prev_preds``: ``(N, P, 2)``. Returns ``(N, P, 4 + 4*P)`` —
    concatenated positions, velocity proxy, and predator-id one-hot.
    """
    if preds.ndim != 3 or preds.shape[-1] != 2:
        raise ValueError("preds must have shape (N, P, 2)")
    if prev_preds.shape != preds.shape:
        raise ValueError("prev_preds must match preds")
    if prey.shape != (len(preds), 2) or prev_prey.shape != prey.shape:
        raise ValueError("prey and prev_prey must have shape (N, 2)")

    n, p, _ = preds.shape
    pos = np.concatenate([preds.reshape(n, p * 2), prey], -1)
    prev_pos = np.concatenate([prev_preds.reshape(n, p * 2), prev_prey], -1)
    vel = pos - prev_pos
    feat_dim = pos.shape[-1] + vel.shape[-1] + p
    feats = np.zeros((n, p, feat_dim), np.float32)
    for i in range(p):
        pid = np.zeros((n, p), np.float32)
        pid[:, i] = 1.0
        feats[:, i] = np.concatenate([pos, vel, pid], -1)
    return feats


def build_predator_samples(
    ds: dict[str, np.ndarray],
    t0: int,
    t1: int,
    ep_len: int = EP_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(state, action, episode_id)`` samples over a step interval."""
    prey = ds["prey_pos"]
    preds = ds["pred_pos"]
    acts = ds["pred_act"]
    if prey.ndim != 3 or prey.shape[-1] != 2:
        raise ValueError("ds['prey_pos'] must have shape (N, T + 1, 2)")
    if preds.ndim != 4 or preds.shape[-1] != 2:
        raise ValueError("ds['pred_pos'] must have shape (N, T + 1, P, 2)")
    n_pred = preds.shape[2]
    if acts.shape[:2] != (len(prey), preds.shape[1] - 1):
        raise ValueError("ds['pred_act'] must have shape (N, T, P)")
    if acts.shape[2] != n_pred:
        raise ValueError("pred_act predator dim must match pred_pos")
    if t1 > acts.shape[1]:
        raise ValueError(f"t1={t1} exceeds action horizon {acts.shape[1]}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    ep_ids = np.arange(len(prey), dtype=np.int32)
    for t in valid_bc_steps(t0, t1, ep_len):
        feats = predator_state_features(
            preds[:, t],
            prey[:, t],
            preds[:, t - 1],
            prey[:, t - 1],
        )
        for p in range(n_pred):
            states.append(feats[:, p])
            actions.append(acts[:, t, p])
            episodes.append(ep_ids)

    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.int32),
        np.concatenate(episodes).astype(np.int32),
    )
