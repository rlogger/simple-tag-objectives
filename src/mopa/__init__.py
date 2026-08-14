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
    evaluate_encoders,
    train_jepa,
    train_jepa_gru,
    train_vae,
)
from mopa.metrics import expected_calibration_error, metrics, oracle_acc, probe_acc
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
    "evaluate_encoders",
    "expected_calibration_error",
    "fit_latent_belief",
    "metrics",
    "oracle_acc",
    "probe_acc",
    "train_jepa",
    "train_jepa_gru",
    "train_vae",
]
