"""Pure trajectory feature extraction utilities (NumPy-only)."""
from __future__ import annotations

import numpy as np

EP_LEN = 100
OCC_BINS = 8
OCC_RANGE = (-2.0, 2.0)


def episode_lengths(
    capture_t: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Valid step counts matching ``survival_time`` in rollouts.

    ``capture_t[i] >= 0`` means episode ``i`` ended at that env step; otherwise
    the episode ran the full ``num_steps`` horizon. Lengths are clipped to
    ``[1, num_steps]`` for safe sequence indexing.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    ct = np.asarray(capture_t, dtype=np.int32)
    captured = ct >= 0
    lengths = np.where(captured, ct, num_steps).astype(np.int32)
    return np.clip(lengths, 1, num_steps)


def length_matched_prefix(
    lengths: np.ndarray,
    k: int,
) -> np.ndarray:
    """Clip requested prefix ``k`` to each episode's true length."""
    if k < 1:
        raise ValueError("k must be >= 1")
    lengths = np.asarray(lengths, dtype=np.int32)
    return np.minimum(lengths, k).astype(np.int32)


def predator_sequence_features(
    prey_pos: np.ndarray,
    pred_pos: np.ndarray,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Build per-step sequence features ``(N, T, F)`` from positions.

    Uses action-horizon steps ``t = 0..T-1`` with positions at ``t`` (after the
    previous transition / initial state). Post-length steps are zeroed when
    ``lengths`` is provided.
    """
    if prey_pos.ndim != 3 or prey_pos.shape[-1] != 2:
        raise ValueError("prey_pos must have shape (N, T + 1, 2)")
    if pred_pos.ndim != 4 or pred_pos.shape[-1] != 2:
        raise ValueError("pred_pos must have shape (N, T + 1, P, 2)")
    if prey_pos.shape[:2] != pred_pos.shape[:2]:
        raise ValueError("prey_pos and pred_pos horizon must match")

    n, t_plus, _ = prey_pos.shape
    t_max = t_plus - 1
    # Absolute positions at each action step + one-step velocity proxy.
    prey_t = prey_pos[:, :t_max]
    pred_t = pred_pos[:, :t_max].reshape(n, t_max, -1)
    prey_v = prey_pos[:, 1 : t_max + 1] - prey_pos[:, :t_max]
    pred_v = (
        pred_pos[:, 1 : t_max + 1] - pred_pos[:, :t_max]
    ).reshape(n, t_max, -1)
    seq = np.concatenate([prey_t, pred_t, prey_v, pred_v], axis=-1).astype(np.float32)
    if lengths is not None:
        lens = np.clip(np.asarray(lengths, np.int32), 1, t_max)
        mask = np.arange(t_max)[None, :] < lens[:, None]
        seq = seq * mask[..., None].astype(np.float32)
    return seq


def window(pos: np.ndarray, k: int, kmax: int | None = None) -> np.ndarray:
    """Flatten a fixed first-episode window with masked future steps."""
    if kmax is None:
        kmax = EP_LEN
    if not 0 <= k <= kmax:
        raise ValueError(f"k must be in [0, {kmax}], got {k}")
    ep = pos[:, :kmax]
    flat = ep.reshape(len(ep), kmax, -1).copy()
    flat[:, k:] = 0.0
    return flat.reshape(len(ep), -1).astype(np.float32)


def standardize(
    X: np.ndarray,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features with train-set statistics."""
    if mu is None:
        mu = X.mean(0)
    if sd is None:
        sd = X.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu, sd


def occupancy(
    pos: np.ndarray,
    t0: int,
    t1: int,
    bins: int = OCC_BINS,
    value_range: tuple[float, float] = OCC_RANGE,
) -> np.ndarray:
    """Per-episode normalized 2-D occupancy histogram over steps ``[t0, t1)``."""
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError("pos must have shape (N, T + 1, 2)")
    if not 0 <= t0 < t1 <= pos.shape[1]:
        raise ValueError(f"invalid occupancy interval [{t0}, {t1})")

    out = np.zeros((len(pos), bins * bins), np.float32)
    hist_range = [value_range, value_range]
    for i in range(len(pos)):
        h, _, _ = np.histogram2d(
            pos[i, t0:t1, 0],
            pos[i, t0:t1, 1],
            bins=bins,
            range=hist_range,
        )
        h = h.ravel().astype(np.float32)
        total = h.sum()
        out[i] = h / total if total > 0 else h
    return out
