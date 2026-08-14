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

SCHEMA_VERSION = 1


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


def package_versions(names: tuple[str, ...] = ("numpy", "jax", "flax", "optax", "scikit-learn", "opponent-modeling")) -> dict[str, str]:
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
    notes: list[str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Assemble a versioned Part 1 JSON artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(cwd),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_pins": package_versions(),
        "config": config,
        "checkpoints": checkpoints,
        "split": split or {},
        "metrics": metrics,
        "notes": notes
        or [
            "Logistic probe is evaluation-only; not a training-time belief.",
            "Latent-conditioned BC is a point-z baseline, not mixture-over-belief.",
        ],
    }


def write_manifest(path: Path | str, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
