"""Latent-conditioned behaviour cloning for the predator team.

Protocol guarantees:
  * Episode-level train/val split — no pooled-timestep leakage.
  * ``z`` is computed from the prey's first ``CTX`` steps only; BC samples
    start at ``t = CTX``, so conditioning never peeks at the future.
  * Every reported number is mean ± std over training seeds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal
from numpy.typing import NDArray

from mopa.features import EP_LEN
from mopa.samples import build_predator_samples
from mopa.splits import episode_validation_mask
from mopa.types import BCRunStats

BC_HID = 128
BC_STEPS = 4000
BC_BATCH = 256

Arrayf = NDArray[np.floating[Any]]
Arrayi = NDArray[np.integer[Any]]

__all__ = [
    "BCNet",
    "BCRunStats",
    "BC_BATCH",
    "BC_HID",
    "BC_STEPS",
    "bc_comparison",
    "build_samples",
    "train_eval_bc",
]


class BCNet(nn.Module):
    """Two-layer MLP mapping state (+ optional latent) to discrete action logits."""

    n_actions: int = 5

    @nn.compact
    def __call__(self, x):  # noqa: ANN001
        x = nn.relu(nn.Dense(BC_HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(BC_HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))(x)


def build_samples(
    ds: Mapping[str, np.ndarray],
    ctx: int,
    ep_len: int = EP_LEN,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi]:
    """Build ``(state, action, episode_id)`` samples for every predator.

    State = absolute positions of all agents + velocity proxy + predator id
    one-hot. Valid steps: ``t in [ctx, t_max)`` excluding the auto-reset
    boundary.
    """
    if t_max is None:
        t_max = min(ep_len, int(ds["pred_act"].shape[1]))
    return build_predator_samples(ds, ctx, t_max, ep_len=ep_len)


def train_eval_bc(
    S: Arrayf,
    A: Arrayi,
    ep: Arrayi,
    rng_seed: int,
    val_frac: float = 0.2,
    steps: int = BC_STEPS,
) -> float:
    """Episode-level split, train a BC net, return held-out action accuracy."""
    vmask = episode_validation_mask(ep, rng_seed=rng_seed, val_frac=val_frac)
    Str, Atr, Sva, Ava = S[~vmask], A[~vmask], S[vmask], A[vmask]

    mu, sd = Str.mean(0), Str.std(0) + 1e-6
    Str, Sva = (Str - mu) / sd, (Sva - mu) / sd

    net = BCNet()
    key = jax.random.PRNGKey(rng_seed)
    key, ki = jax.random.split(key)
    params = net.init(ki, Str[:1])
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Sj, Aj = jnp.asarray(Str), jnp.asarray(Atr)
    n = len(Str)

    def loss_fn(p, s, a):  # noqa: ANN001
        logits = net.apply(p, s)
        return optax.softmax_cross_entropy_with_integer_labels(logits, a).mean()

    @jax.jit
    def upd(params, opt, idx):  # noqa: ANN001
        g = jax.grad(loss_fn)(params, Sj[idx], Aj[idx])
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for _ in range(steps):
        key, bk = jax.random.split(key)
        idx = jax.random.choice(bk, n, (min(BC_BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx)

    pred = np.asarray(net.apply(params, jnp.asarray(Sva))).argmax(-1)
    return float((pred == Ava).mean())


def bc_comparison(
    ds: Mapping[str, np.ndarray],
    z_dict: Mapping[str, np.ndarray | None],
    ctx: int,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, BCRunStats]:
    """Run BC for each conditioning variant.

    ``z_dict`` maps variant name → per-episode conditioning array ``(N, d)``,
    or ``None`` for the unconditioned baseline.
    """
    S0, A, ep = build_samples(ds, ctx)
    results: dict[str, BCRunStats] = {}
    for name, z in z_dict.items():
        if z is None:
            S = S0
        else:
            # Append raw z; train_eval_bc standardizes from the train fold only
            # so val episodes never influence feature scale.
            S = np.concatenate([S0, np.asarray(z, dtype=np.float32)[ep]], -1)
        runs = tuple(train_eval_bc(S, A, ep, s) for s in seeds)
        v = np.asarray(runs)
        results[name] = BCRunStats(mean=float(v.mean()), std=float(v.std()), runs=runs)
        suffix = f",{name}" if z is not None else ""
        print(f"  BC pi(a|s{suffix}) : {v.mean():.4f} +/- {v.std():.4f}")
    return results
