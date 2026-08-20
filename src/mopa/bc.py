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
from mopa.samples import build_predator_samples, build_predator_samples_with_time
from mopa.splits import episode_validation_mask, group_validation_mask
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
    "build_samples_with_time",
    "train_eval_bc",
    "train_eval_bc_metrics",
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
    boundary and post-capture frozen frames when ``capture_t`` is present.
    """
    if t_max is None:
        t_max = min(ep_len, int(ds["pred_act"].shape[1]))
    capture_t = ds["capture_t"] if "capture_t" in ds else None
    return build_predator_samples(
        dict(ds), ctx, t_max, ep_len=ep_len, capture_t=capture_t
    )


def build_samples_with_time(
    ds: Mapping[str, np.ndarray],
    ctx: int,
    ep_len: int = EP_LEN,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi, Arrayi, Arrayi]:
    """Build samples plus ``(timestep, predator_id)`` causal provenance."""
    if t_max is None:
        t_max = min(ep_len, int(ds["pred_act"].shape[1]))
    capture_t = ds["capture_t"] if "capture_t" in ds else None
    return build_predator_samples_with_time(
        dict(ds), ctx, t_max, ep_len=ep_len, capture_t=capture_t
    )


def train_eval_bc(
    S: Arrayf,
    A: Arrayi,
    ep: Arrayi,
    rng_seed: int,
    val_frac: float = 0.2,
    steps: int = BC_STEPS,
    group_ids: Arrayi | None = None,
    split_seed: int | None = None,
    validation_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Train a BC net and return held-out action accuracy.

    By default splits by episode. When ``group_ids`` is provided (e.g.
    checkpoint seeds broadcast to sample rows), holds out whole groups.
    ``split_seed`` defaults to ``rng_seed`` for backward compatibility.
    """
    return float(
        train_eval_bc_metrics(
            S,
            A,
            ep,
            rng_seed,
            val_frac=val_frac,
            steps=steps,
            group_ids=group_ids,
            split_seed=split_seed,
            validation_mask=validation_mask,
        )["accuracy"]
    )


def train_eval_bc_metrics(
    S: Arrayf,
    A: Arrayi,
    ep: Arrayi,
    rng_seed: int,
    val_frac: float = 0.2,
    steps: int = BC_STEPS,
    group_ids: Arrayi | None = None,
    split_seed: int | None = None,
    validation_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float | int]:
    """Train BC once and return held-out accuracy and action NLL.

    ``validation_mask`` is the preferred experiment-driver interface: it is a
    sample-aligned mask derived from the single manifest split and therefore
    stays identical across model initialization seeds.  The legacy split
    arguments remain for small standalone calls.
    """
    if validation_mask is not None:
        vmask = np.asarray(validation_mask, dtype=bool)
        if vmask.shape != (len(ep),):
            raise ValueError("validation_mask must align with samples")
    else:
        split_rng = rng_seed if split_seed is None else split_seed
        if group_ids is None:
            vmask = episode_validation_mask(ep, rng_seed=split_rng, val_frac=val_frac)
        else:
            if len(group_ids) != len(ep):
                raise ValueError("group_ids must align with samples")
            vmask = group_validation_mask(
                group_ids, rng_seed=split_rng, val_frac=val_frac
            )
    Str, Atr, Sva, Ava = S[~vmask], A[~vmask], S[vmask], A[vmask]
    if len(Str) == 0 or len(Sva) == 0:
        raise ValueError("BC split must contain both train and validation samples")

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

    logits = np.asarray(net.apply(params, jnp.asarray(Sva)), dtype=np.float32)
    pred = logits.argmax(-1)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = -float(log_probs[np.arange(len(Ava)), Ava].mean())
    return {
        "accuracy": float((pred == Ava).mean()),
        "nll": nll,
        "n_train": int(len(Str)),
        "n_val": int(len(Sva)),
    }


def bc_comparison(
    ds: Mapping[str, np.ndarray],
    z_dict: Mapping[str, np.ndarray | None],
    ctx: int,
    seeds: Sequence[int] = (0, 1, 2),
    group_ids: Arrayi | None = None,
    split: str = "episode",
    split_seed: int = 0,
) -> dict[str, BCRunStats]:
    """Run BC for each conditioning variant.

    ``z_dict`` maps variant name → per-episode conditioning array ``(N, d)``,
    or ``None`` for the unconditioned baseline.

    ``split="checkpoint"`` uses ``ds['ckpt_seed']`` (or provided ``group_ids``)
    broadcast onto sample rows so held-out checkpoints never appear in train.
    """
    S0, A, ep = build_samples(ds, ctx)
    sample_groups = None
    if split == "checkpoint":
        if group_ids is None:
            if "ckpt_seed" not in ds:
                raise ValueError("checkpoint split requires ckpt_seed on the dataset")
            group_ids = np.asarray(ds["ckpt_seed"])
        sample_groups = np.asarray(group_ids, dtype=np.int32)[ep]
    elif split != "episode":
        raise ValueError(f"unknown split={split!r}")

    results: dict[str, BCRunStats] = {}
    for name, z in z_dict.items():
        if z is None:
            S = S0
        else:
            # Append raw z; train_eval_bc standardizes from the train fold only
            # so val episodes never influence feature scale.
            S = np.concatenate([S0, np.asarray(z, dtype=np.float32)[ep]], -1)
        runs = tuple(
            train_eval_bc(
                S,
                A,
                ep,
                s,
                group_ids=sample_groups,
                split_seed=split_seed,
            )
            for s in seeds
        )
        v = np.asarray(runs)
        results[name] = BCRunStats(mean=float(v.mean()), std=float(v.std()), runs=runs)
        suffix = f",{name}" if z is not None else ""
        print(f"  BC pi(a|s{suffix}) : {v.mean():.4f} +/- {v.std():.4f}")
    return results
