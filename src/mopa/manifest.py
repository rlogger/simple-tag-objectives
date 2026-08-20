"""Artifact manifests for reproducible Part 1 runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
STAGE_STATES = {"not_run", "blocked", "smoke_passed", "full_run_finished"}


def _hex_string(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def git_sha(cwd: Path | None = None) -> str | None:
    """Best-effort git commit SHA; ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(cwd: Path | None = None) -> bool | None:
    """Best-effort working-tree state; ``None`` outside a Git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(
    names: tuple[str, ...] = (
        "numpy",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "scikit-learn",
        "jaxmarl",
        "distrax",
        "opponent-modeling",
    ),
) -> dict[str, str]:
    pins: dict[str, str] = {}
    for name in names:
        try:
            pins[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pins[name] = "unknown"
    return pins


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_manifest(
    *,
    config: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    metrics: dict[str, Any],
    split: dict[str, Any] | None = None,
    stages: dict[str, dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    run_kind: str = "smoke",
    notes: list[str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Assemble a versioned Part 1 JSON artifact."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(cwd),
        "git_dirty": git_dirty(cwd),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_pins": package_versions(),
        "config": config,
        "checkpoints": checkpoints,
        "split": split or {},
        "run_kind": run_kind,
        "stages": stages or {},
        "artifacts": artifacts or [],
        "metrics": metrics,
        "notes": notes
        or [
            "Strategy labels used by evaluation probes are not encoder inputs.",
            "A dry or smoke run is infrastructure evidence, not a full experiment result.",
        ],
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on malformed or overstated run manifests."""
    required = {
        "schema_version",
        "git_sha",
        "git_dirty",
        "created_at",
        "dependency_pins",
        "config",
        "checkpoints",
        "split",
        "run_kind",
        "stages",
        "artifacts",
        "metrics",
        "notes",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version does not match this package")
    if manifest["run_kind"] not in {"dry_run", "smoke", "full"}:
        raise ValueError("run_kind must be dry_run, smoke, or full")
    if not isinstance(manifest["stages"], dict):
        raise ValueError("stages must be a mapping")
    if manifest["run_kind"] == "full":
        if not _hex_string(manifest["git_sha"], 40) or manifest["git_dirty"] is not False:
            raise ValueError("a full manifest requires a clean, committed Git state")
        config = manifest["config"]
        required_config = {
            "n_eps",
            "ckpt_seeds",
            "encoder_seeds",
            "bc_seeds",
            "encoder_steps",
            "bc_steps",
            "split",
            "prey_objective",
            "synthetic",
            "skip_bc",
            "effective_run_kind",
        }
        if not isinstance(config, dict) or required_config - set(config):
            raise ValueError("a full manifest is missing required Part 1 config")
        for name in ("ckpt_seeds", "encoder_seeds", "bc_seeds"):
            seeds = config[name]
            if (
                not isinstance(seeds, list)
                or len(seeds) < 3
                or not all(
                    isinstance(seed, int) and not isinstance(seed, bool)
                    for seed in seeds
                )
                or len(seeds) != len(set(seeds))
            ):
                raise ValueError(f"full config needs three distinct {name}")
        if (
            not _positive_int(config["n_eps"])
            or not _positive_int(config["encoder_steps"])
            or not _positive_int(config["bc_steps"])
            or config["split"] != "checkpoint"
            or config["prey_objective"] == "matched"
            or config["synthetic"] is not False
            or config["skip_bc"] is not False
            or config["effective_run_kind"] != "full"
        ):
            raise ValueError("full config violates the reference protocol")
        checkpoints = manifest["checkpoints"]
        if (
            not isinstance(checkpoints, list)
            or len(checkpoints) != 4 * len(config["ckpt_seeds"])
            or any(
                checkpoint.get("exists") is not True
                or not checkpoint.get("path")
                or not _hex_string(checkpoint.get("sha256"), 64)
                for checkpoint in checkpoints
            )
        ):
            raise ValueError("a full manifest requires hashed existing checkpoints")
        split = manifest["split"]
        if (
            not isinstance(split, dict)
            or split.get("mode") != "checkpoint"
            or not _positive_int(split.get("n_train"))
            or not _positive_int(split.get("n_val"))
            or split.get("shared_across_encoder_and_bc_seeds") is not True
            or len(split.get("episodes", []))
            != int(split.get("n_train", 0)) + int(split.get("n_val", 0))
        ):
            raise ValueError("a full manifest requires a materialized train/val split")
        labels_by_environment_key: dict[tuple[int, int], set[int]] = {}
        folds_by_checkpoint: dict[int, set[str]] = {}
        for episode in split["episodes"]:
            if (
                not isinstance(episode, dict)
                or episode.get("fold") not in {"train", "validation"}
                or not _positive_int(episode.get("valid_length"))
                or not isinstance(episode.get("environment_key"), list)
                or len(episode["environment_key"]) != 2
            ):
                raise ValueError("full split has malformed episode provenance")
            environment_key = tuple(episode["environment_key"])
            labels_by_environment_key.setdefault(environment_key, set()).add(
                episode.get("strategy_label")
            )
            folds_by_checkpoint.setdefault(
                episode.get("checkpoint_seed"), set()
            ).add(episode["fold"])
        if any(
            labels != {0, 1, 2}
            for labels in labels_by_environment_key.values()
        ):
            raise ValueError("full split must match environment keys across strategies")
        if set(folds_by_checkpoint) != set(config["ckpt_seeds"]) or any(
            len(folds) != 1 for folds in folds_by_checkpoint.values()
        ):
            raise ValueError("full split leaks a checkpoint across train and validation")
        required_metrics = {
            "environment",
            "representations",
            "anytime",
            "controls",
            "sequential_belief_mixture",
            "bc",
        }
        if not isinstance(manifest["metrics"], dict) or required_metrics - set(
            manifest["metrics"]
        ):
            raise ValueError("a full manifest is missing required Part 1 metrics")
        required_stages = {
            "data",
            "environment_evaluation",
            "representations",
            "anytime_and_calibration",
            "sequential_belief_mixture",
            "behaviour_cloning",
        }
        if any(
            manifest["stages"].get(name, {}).get("status")
            != "full_run_finished"
            for name in required_stages
        ):
            raise ValueError("a full manifest is missing a finished required stage")
    for name, stage in manifest["stages"].items():
        if not isinstance(stage, dict) or stage.get("status") not in STAGE_STATES:
            raise ValueError(
                f"stage {name!r} needs status in {sorted(STAGE_STATES)}"
            )
        if (
            stage["status"] == "full_run_finished"
            and manifest["run_kind"] != "full"
        ):
            raise ValueError(
                "only run_kind='full' may mark a stage full_run_finished"
            )


def write_manifest(path: Path | str, manifest: dict[str, Any]) -> Path:
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
