"""Numerical checks for held-out evaluation and calibration utilities."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from mopa.metrics import (
    calibration_metrics,
    classwise_ece,
    multiclass_brier,
    multiclass_nll,
    open_set_auroc,
    reliability_bins,
    softmax_with_temperature,
    temperature_scale_logits,
)


def test_calibration_metrics_perfect_predictions():
    probs = np.eye(3, dtype=np.float32)
    labels = np.arange(3)
    summary = calibration_metrics(probs, labels, n_bins=5)

    assert summary["nll"] == pytest.approx(0.0, abs=1e-8)
    assert summary["brier"] == pytest.approx(0.0, abs=1e-8)
    assert summary["ece"] == pytest.approx(0.0, abs=1e-8)
    assert summary["classwise_ece"] == pytest.approx(0.0, abs=1e-8)
    assert sum(summary["reliability"]["count"]) == 3


def test_nll_brier_and_reliability_known_example():
    probs = np.array([[0.75, 0.25], [0.40, 0.60]], dtype=np.float32)
    labels = np.array([0, 0])

    assert multiclass_nll(probs, labels) == pytest.approx(
        -(np.log(0.75) + np.log(0.40)) / 2
    )
    expected_brier = ((0.25**2 + 0.25**2) + (0.60**2 + 0.60**2)) / 2
    assert multiclass_brier(probs, labels) == pytest.approx(expected_brier)
    bins = reliability_bins(probs, labels, n_bins=2)
    assert sum(bins["count"]) == 2
    assert 0.0 <= classwise_ece(probs, labels, n_bins=2) <= 1.0


def test_temperature_grid_improves_overconfident_wrong_logits():
    logits = np.array(
        [[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 1, 0])
    before = multiclass_nll(softmax_with_temperature(logits, 1.0), labels)
    temperature = temperature_scale_logits(logits, labels)
    after = multiclass_nll(
        softmax_with_temperature(logits, temperature), labels
    )

    assert temperature > 1.0
    assert after < before


def test_open_set_auroc_and_probability_validation():
    score = np.array([0.95, 0.8, 0.2, 0.05])
    known = np.array([True, True, False, False])
    assert open_set_auroc(score, known) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        multiclass_nll(np.array([[0.0, 0.0]]), np.array([0]))
    with pytest.raises(ValueError, match="at least one"):
        calibration_metrics(np.empty((0, 3)), np.empty((0,), dtype=np.int32))
    with pytest.raises(ValueError, match="n_bins"):
        classwise_ece(np.eye(2), np.array([0, 1]), n_bins=0)


def test_adjusted_rand_is_label_permutation_invariant():
    truth = np.array([0, 0, 1, 1, 2, 2])
    permuted_clusters = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_score(truth, permuted_clusters) == pytest.approx(1.0)
