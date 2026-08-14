"""mopa — Modeling OPponent Agents (Part 1 scaffold).

Uncertainty-aware opponent modeling on the objective-typed predator env:
GRU-JEPA / VAE encoders, soft strategy beliefs, latent-conditioned BC.
"""
from __future__ import annotations

__version__ = "0.1.0"

from mopa.belief import Belief, categorical_entropy, fit_latent_belief
from mopa.encoders import (
    Enc,
    EncVAE,
    GRUEnc,
    Pred,
    collapse_diagnostics,
    encode_jepa_gru,
    evaluate_encoders,
    train_jepa,
    train_jepa_gru,
    train_jepa_gru_with_params,
    train_vae,
)
from mopa.metrics import (
    expected_calibration_error,
    metrics,
    oracle_acc,
    probe_acc,
    survival_time_probe_acc,
    train_only_metrics,
)
from mopa.splits import checkpoint_validation_mask, episode_validation_mask
from mopa.types import BCRunStats, CheckpointRef, ObjectiveDataset

__all__ = [
    "BCRunStats",
    "Belief",
    "CheckpointRef",
    "Enc",
    "EncVAE",
    "GRUEnc",
    "ObjectiveDataset",
    "Pred",
    "__version__",
    "categorical_entropy",
    "checkpoint_validation_mask",
    "collapse_diagnostics",
    "encode_jepa_gru",
    "episode_validation_mask",
    "evaluate_encoders",
    "expected_calibration_error",
    "fit_latent_belief",
    "metrics",
    "oracle_acc",
    "probe_acc",
    "survival_time_probe_acc",
    "train_jepa",
    "train_jepa_gru",
    "train_jepa_gru_with_params",
    "train_only_metrics",
    "train_vae",
]
