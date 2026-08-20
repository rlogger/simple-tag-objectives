"""Tests for sequential, uncertainty-aware opponent strategy inference."""
import inspect

import numpy as np

from mopa.strategy import (
    BayesianStrategyFilter,
    SequentialOpponentModel,
    StrategyPolicy,
    mixture_policy_metrics,
    mixture_policy_probs,
)


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two strategies, including one with only a single action class."""
    constant_states = np.linspace(-2.0, 2.0, 20)[:, None]
    responsive_states = np.linspace(-4.0, 4.0, 40)[:, None]
    states = np.concatenate([constant_states, responsive_states])
    actions = np.concatenate(
        [
            np.full(20, 7),
            np.where(responsive_states[:, 0] < 0.0, 3, 7),
        ]
    )
    strategies = np.concatenate([np.full(20, 5), np.full(40, 9)])
    return states, actions, strategies


def test_strategy_policy_aligns_actions_and_handles_one_class_strategy():
    states, actions, strategies = _training_data()
    policy = StrategyPolicy.fit(states, actions, strategies)

    probabilities = policy.action_probs(np.array([[-4.0], [4.0]]))

    assert probabilities.shape == (2, 2, 2)
    np.testing.assert_array_equal(policy.strategy_classes, [5, 9])
    np.testing.assert_array_equal(policy.action_classes, [3, 7])
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    # Strategy 5 saw only action 7, while strategy 9 responds to the state.
    assert np.all(probabilities[:, 0, 1] > 0.999)
    assert probabilities[0, 1, 0] > 0.9
    assert probabilities[1, 1, 1] > 0.9


def test_strategy_policy_assigns_floor_mass_to_declared_unseen_actions():
    states, actions, strategies = _training_data()
    policy = StrategyPolicy.fit(
        states,
        actions,
        strategies,
        action_classes=np.array([3, 7, 11]),
    )

    probabilities = policy.action_probs(np.array([[0.0]]))
    np.testing.assert_array_equal(policy.action_classes, [3, 7, 11])
    assert probabilities.shape == (1, 2, 3)
    assert np.all(probabilities[..., 2] > 0.0)
    likelihood = policy.action_likelihoods(
        np.array([[0.0]]), np.array([11])
    )
    assert np.all(likelihood > 0.0)


def test_bayesian_filter_normalizes_and_adapts_after_switch():
    likelihoods = np.array(
        [[0.98, 0.02]] * 7 + [[0.02, 0.98]] * 7,
        dtype=np.float64,
    )
    strategy_filter = BayesianStrategyFilter(
        n_strategies=2,
        prior=np.array([3.0, 1.0]),
        switch_probability=0.05,
    )

    trace = strategy_filter.filter(likelihoods)

    np.testing.assert_allclose(trace.predictive[0], [0.75, 0.25])
    np.testing.assert_allclose(trace.predictive.sum(axis=-1), 1.0)
    np.testing.assert_allclose(trace.probs.sum(axis=-1), 1.0)
    assert trace.probs[5, 0] > 0.99
    assert trace.probs[8, 1] > 0.99
    assert trace.hard[-1] == 1
    assert np.all(trace.entropy >= 0.0)


def test_custom_transition_rows_are_normalized():
    strategy_filter = BayesianStrategyFilter(
        n_strategies=2,
        transition=np.array([[9.0, 1.0], [2.0, 8.0]]),
    )
    np.testing.assert_allclose(
        strategy_filter.transition,
        [[0.9, 0.1], [0.2, 0.8]],
    )


def test_batched_inference_ignores_masked_padding_values():
    states = np.zeros((12, 1), dtype=np.float64)
    actions = np.array([0] * 6 + [1] * 6)
    strategies = np.array([10] * 6 + [20] * 6)
    model = SequentialOpponentModel.fit(
        states,
        actions,
        strategies,
        switch_probability=0.1,
    )

    padded_states = np.zeros((2, 5, 1), dtype=np.float64)
    padded_actions = np.array(
        [
            [0, 0, 0, 999, 999],
            [1, 1, 999, 999, 999],
        ]
    )
    padded_states[0, 3:] = np.nan
    padded_states[1, 2:] = np.nan
    batch_trace = model.infer(
        padded_states,
        padded_actions,
        lengths=np.array([3, 2]),
    )
    explicit_mask = np.array(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )
    mask_trace = model.infer(padded_states, padded_actions, mask=explicit_mask)
    first_trace = model.infer(np.zeros((3, 1)), np.array([0, 0, 0]))
    second_trace = model.infer(np.zeros((2, 1)), np.array([1, 1]))

    np.testing.assert_allclose(batch_trace.probs, mask_trace.probs)
    np.testing.assert_allclose(batch_trace.probs[0, :3], first_trace.probs)
    np.testing.assert_allclose(batch_trace.probs[1, :2], second_trace.probs)
    np.testing.assert_allclose(
        batch_trace.probs[0, 3:],
        np.repeat(first_trace.probs[-1][None, :], 2, axis=0),
    )
    np.testing.assert_allclose(
        batch_trace.probs[1, 2:],
        np.repeat(second_trace.probs[-1][None, :], 3, axis=0),
    )
    np.testing.assert_array_equal(
        batch_trace.mask,
        explicit_mask,
    )


def test_mixture_policy_and_distributional_metrics():
    beliefs = np.array([[0.75, 0.25], [0.20, 0.80]])
    per_strategy = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.6, 0.4], [0.1, 0.9]],
        ]
    )

    mixture = mixture_policy_probs(beliefs, per_strategy)

    expected = np.array([[0.725, 0.275], [0.20, 0.80]])
    np.testing.assert_allclose(mixture, expected)
    np.testing.assert_allclose(mixture.sum(axis=-1), 1.0)
    metrics = mixture_policy_metrics(
        mixture,
        np.array([4, 9]),
        action_classes=np.array([4, 9]),
    )
    assert metrics["top1"] == 1.0
    np.testing.assert_allclose(
        metrics["nll"],
        -np.mean(np.log([0.725, 0.8])),
    )


def test_inference_api_has_no_strategy_label_argument():
    parameters = inspect.signature(SequentialOpponentModel.infer).parameters
    assert "strategy_labels" not in parameters
    assert "labels" not in parameters
