#!/usr/bin/env python3
"""Runnable Part 1 foundation experiment with JSON artifact output.

Smoke-scale defaults. Full matrix seeds/steps should be set explicitly.
Requires MAPPO specialist checkpoints under ``--logdir`` unless ``--dry-run-checks``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running without editable install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logdir", type=Path, default=Path("logs") / "MPE_simple_tag_v3")
    p.add_argument("--n-eps", type=int, default=16)
    p.add_argument("--ckpt-seeds", type=str, default="0,1,2")
    p.add_argument("--encoder-seeds", type=str, default="0,1,2")
    p.add_argument("--bc-seeds", type=str, default="0,1,2")
    p.add_argument("--lat", type=int, default=2)
    p.add_argument("--ctx", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.34)
    p.add_argument("--split", choices=("checkpoint", "episode"), default="checkpoint")
    p.add_argument("--encoder-steps", type=int, default=200)
    p.add_argument("--bc-steps", type=int, default=200)
    p.add_argument("--hid", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("artifacts/part1_run.json"))
    p.add_argument("--dataset-cache", type=Path, default=None)
    p.add_argument("--skip-bc", action="store_true")
    p.add_argument(
        "--dry-run-checks",
        action="store_true",
        help="Validate checkpoint files and write a stub manifest, then exit.",
    )
    return p


def _params_path(logdir: Path, pred_type: str, team: str, seed_idx: int) -> Path:
    """Mirror ``mopa.data._params_path`` without importing JaxMARL."""
    alg = f"mappo_objectives_{pred_type}"
    return (
        logdir
        / f"{alg}_MPE_simple_tag_v3_{team}_actor_seed0_vmap{seed_idx}.safetensors"
    )


def _checkpoint_refs(logdir: Path, ckpt_seeds: tuple[int, ...]) -> list[dict]:
    from mopa.manifest import file_sha256

    objective_types = ("capture", "risk", "curious")
    refs = []
    for pred_type in objective_types:
        for team in ("pred", "prey"):
            for seed in ckpt_seeds:
                path = _params_path(logdir, pred_type, team, seed)
                entry = {
                    "alg": f"mappo_objectives_{pred_type}",
                    "team": team,
                    "seed": seed,
                    "path": str(path),
                    "exists": path.is_file(),
                }
                if path.is_file():
                    entry["sha256"] = file_sha256(path)
                refs.append(entry)
    return refs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_seeds = _parse_ints(args.ckpt_seeds)
    encoder_seeds = _parse_ints(args.encoder_seeds)
    bc_seeds = _parse_ints(args.bc_seeds)

    from mopa.manifest import build_manifest, write_manifest

    config = {
        "n_eps": args.n_eps,
        "ckpt_seeds": list(ckpt_seeds),
        "encoder_seeds": list(encoder_seeds),
        "bc_seeds": list(bc_seeds),
        "lat": args.lat,
        "ctx": args.ctx,
        "val_frac": args.val_frac,
        "split": args.split,
        "logdir": str(args.logdir),
        "encoder_steps": args.encoder_steps,
        "bc_steps": args.bc_steps,
        "hid": args.hid,
        "skip_bc": args.skip_bc,
    }
    checkpoints = _checkpoint_refs(args.logdir, ckpt_seeds)
    missing = [c for c in checkpoints if not c["exists"]]

    if args.dry_run_checks:
        metrics = {
            "dry_run": True,
            "checkpoints_ok": len(missing) == 0,
            "missing_count": len(missing),
        }
        manifest = build_manifest(
            config=config,
            checkpoints=checkpoints,
            metrics=metrics,
            split={"mode": args.split},
            cwd=_ROOT,
        )
        write_manifest(args.out, manifest)
        print(f"Wrote dry-run manifest to {args.out}")
        return 0 if not missing else 2

    if missing:
        print(f"Missing {len(missing)} checkpoint files under {args.logdir}", file=sys.stderr)
        for c in missing[:6]:
            print(f"  - {c['path']}", file=sys.stderr)
        return 2

    import jax

    from mopa.belief import fit_latent_belief
    from mopa.data import objective_dataset
    from mopa.encoders import (
        collapse_diagnostics,
        encode_jepa_gru,
        gather_prefix_latents,
        train_jepa_gru_with_params,
    )
    from mopa.features import episode_lengths, predator_sequence_features, window
    from mopa.metrics import (
        expected_calibration_error,
        survival_time_probe_acc,
        train_only_metrics,
    )
    from mopa.splits import checkpoint_validation_mask, episode_validation_mask

    if args.dataset_cache and args.dataset_cache.is_file():
        raw = dict(np.load(args.dataset_cache, allow_pickle=False))
        from mopa.types import ObjectiveDataset

        ds = ObjectiveDataset(**{k: raw[k] for k in ObjectiveDataset.__dataclass_fields__})
    else:
        ds = objective_dataset(
            n_eps=args.n_eps,
            ckpt_seeds=ckpt_seeds,
            logdir=args.logdir,
        )
        if args.dataset_cache is not None:
            args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.dataset_cache, **ds.as_dict())

    num_steps = int(ds.pred_act.shape[1])
    lengths = episode_lengths(ds.capture_t, num_steps)
    X_seq = predator_sequence_features(ds.prey_pos, ds.pred_pos, lengths)

    # Checkpoint-grouped holdout (default).
    if args.split == "checkpoint":
        vmask = checkpoint_validation_mask(
            ds.ckpt_seed, rng_seed=0, val_frac=args.val_frac
        )
    else:
        ep_ids = np.arange(len(ds.label))
        vmask = episode_validation_mask(ep_ids, rng_seed=0, val_frac=args.val_frac)

    train_idx = np.where(~vmask)[0]
    val_idx = np.where(vmask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        print("Empty train or val split", file=sys.stderr)
        return 1

    # Train-only feature scale for window baselines / survival control context.
    Xc_all = window(ds.prey_pos, min(args.ctx, num_steps), kmax=num_steps)
    mu = Xc_all[train_idx].mean(0)
    sd = Xc_all[train_idx].std(0) + 1e-6
    Xc = ((Xc_all - mu) / sd).astype(np.float32)

    probe_runs = []
    ari_runs = []
    collapse_runs = []
    ece_runs = []
    z_full_by_seed: dict[int, np.ndarray] = {}

    for seed in encoder_seeds:
        z, params, _ = train_jepa_gru_with_params(
            X_seq[train_idx],
            lengths[train_idx],
            jax.random.PRNGKey(seed),
            lat=args.lat,
            hid=args.hid,
            steps=args.encoder_steps,
        )
        # Frozen encode of the full pool with train-only params.
        z_all = encode_jepa_gru(
            params, X_seq, lengths, lat=args.lat, hid=args.hid
        )
        z_ep = gather_prefix_latents(z_all, lengths)
        z_full_by_seed[seed] = z_ep
        m = train_only_metrics(
            z_ep[train_idx],
            ds.label[train_idx],
            z_ep[val_idx],
            ds.label[val_idx],
            n_classes=3,
            seed=seed,
        )
        probe_runs.append(m["probe"])
        ari_runs.append(m["ari"])
        collapse_runs.append(collapse_diagnostics(z_ep[train_idx]))
        belief = fit_latent_belief(
            z_ep[train_idx], ds.label[train_idx], z_ep[val_idx]
        )
        ece_runs.append(
            expected_calibration_error(belief.probs, ds.label[val_idx])
        )

    survival_probe = survival_time_probe_acc(
        ds.survival_time[val_idx], ds.label[val_idx], cv=min(3, len(val_idx))
    )

    metrics: dict = {
        "heldout_probe_mean": float(np.mean(probe_runs)),
        "heldout_probe_std": float(np.std(probe_runs)),
        "heldout_probe_all": [float(x) for x in probe_runs],
        "heldout_ari_mean": float(np.mean(ari_runs)),
        "heldout_ari_std": float(np.std(ari_runs)),
        "heldout_ece_mean": float(np.mean(ece_runs)),
        "heldout_ece_std": float(np.std(ece_runs)),
        "collapse": collapse_runs,
        "survival_time_probe_control": float(survival_probe),
        "window_context_dim": int(Xc.shape[1]),
    }

    if not args.skip_bc:
        from mopa.bc import build_samples, train_eval_bc

        # Use first encoder seed's full-episode z as point-latent baseline.
        z0 = z_full_by_seed[encoder_seeds[0]]
        S0, A, ep = build_samples(ds.as_dict(), args.ctx)
        sample_groups = None
        if args.split == "checkpoint":
            sample_groups = np.asarray(ds.ckpt_seed, dtype=np.int32)[ep]

        def _bc(name: str, S):
            runs = []
            for s in bc_seeds:
                runs.append(
                    train_eval_bc(
                        S,
                        A,
                        ep,
                        s,
                        steps=args.bc_steps,
                        group_ids=sample_groups,
                    )
                )
            return {
                "mean": float(np.mean(runs)),
                "std": float(np.std(runs)),
                "runs": [float(x) for x in runs],
                "baseline": name == "z",
                "label": (
                    "point-latent BC baseline (not belief mixture)"
                    if name == "z"
                    else "unconditioned BC"
                ),
            }

        metrics["bc"] = {
            "none": _bc("none", S0),
            "z": _bc("z", np.concatenate([S0, z0[ep]], -1)),
        }

    split_info = {
        "mode": args.split,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_ckpt_seeds": sorted(
            {int(x) for x in np.unique(ds.ckpt_seed[val_idx])}
        )
        if args.split == "checkpoint"
        else [],
        "train_ckpt_seeds": sorted(
            {int(x) for x in np.unique(ds.ckpt_seed[train_idx])}
        )
        if args.split == "checkpoint"
        else [],
    }

    manifest = build_manifest(
        config=config,
        checkpoints=checkpoints,
        metrics=metrics,
        split=split_info,
        cwd=_ROOT,
    )
    write_manifest(args.out, manifest)
    print(
        f"held-out probe {metrics['heldout_probe_mean']:.3f} "
        f"+/- {metrics['heldout_probe_std']:.3f} "
        f"(survival control {metrics['survival_time_probe_control']:.3f})"
    )
    print(f"Wrote manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
