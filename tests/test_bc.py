"""Causal split and metric checks for behaviour cloning."""

import numpy as np
import pytest

from mopa.bc import train_eval_bc_metrics


def test_bc_uses_explicit_manifest_validation_mask():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(30, 4)).astype(np.float32)
    actions = (states[:, 0] > 0).astype(np.int32)
    episodes = np.repeat(np.arange(10), 3).astype(np.int32)
    validation = episodes >= 8

    result = train_eval_bc_metrics(
        states,
        actions,
        episodes,
        rng_seed=7,
        steps=4,
        validation_mask=validation,
    )

    assert result["n_train"] == 24
    assert result["n_val"] == 6
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["nll"] >= 0.0


def test_bc_rejects_empty_or_misaligned_manifest_split():
    states = np.zeros((6, 2), dtype=np.float32)
    actions = np.zeros(6, dtype=np.int32)
    episodes = np.arange(6, dtype=np.int32)

    with pytest.raises(ValueError, match="align"):
        train_eval_bc_metrics(
            states,
            actions,
            episodes,
            rng_seed=0,
            steps=1,
            validation_mask=np.zeros(5, dtype=bool),
        )
    with pytest.raises(ValueError, match="both train and validation"):
        train_eval_bc_metrics(
            states,
            actions,
            episodes,
            rng_seed=0,
            steps=1,
            validation_mask=np.zeros(6, dtype=bool),
        )
