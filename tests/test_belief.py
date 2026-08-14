"""Smoke tests for the Part 1 belief API."""
import numpy as np

from mopa.belief import (
    Belief,
    categorical_entropy,
    fit_latent_belief,
    softmax_belief_from_logits,
)
from mopa.metrics import expected_calibration_error


def test_softmax_belief_from_logits_normalized():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    b = softmax_belief_from_logits(logits)
    assert isinstance(b, Belief)
    np.testing.assert_allclose(b.probs.sum(-1), 1.0, atol=1e-5)
    assert b.probs[0].argmax() == 0
    assert b.entropy[1] > b.entropy[0]


def test_fit_latent_belief_separates_clusters():
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(40, 2)).astype(np.float32) + np.array([3.0, 0.0])
    z1 = rng.normal(size=(40, 2)).astype(np.float32) + np.array([-3.0, 0.0])
    z = np.concatenate([z0, z1])
    y = np.array([0] * 40 + [1] * 40)
    b = fit_latent_belief(z, y)
    assert b.probs.shape == (80, 2)
    assert (b.hard == y).mean() > 0.9
    assert b.mean_entropy() < 0.5


def test_categorical_entropy_uniform_is_log_k():
    p = np.full((1, 4), 0.25, dtype=np.float32)
    ent = categorical_entropy(p)
    np.testing.assert_allclose(ent[0], np.log(4.0), atol=1e-5)


def test_ece_perfectly_confident_correct():
    probs = np.eye(3, dtype=np.float32)
    y = np.arange(3)
    assert expected_calibration_error(probs, y) < 1e-6
