"""Simple sequential opponent strategy inference.

The module has three small pieces:

* :class:`StrategyPolicy` fits one discrete action model ``pi_k(a | s)`` per
  strategy using strategy labels only during fitting.
* :class:`BayesianStrategyFilter` updates a soft strategy belief from action
  likelihoods and can allow strategies to switch over time.
* :class:`SequentialOpponentModel` joins the two without accepting strategy
  labels at inference time.

All public probabilities use the class order stored on the fitted policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mopa.belief import categorical_entropy

__all__ = [
    "BayesianStrategyFilter",
    "BeliefTrace",
    "SequentialOpponentModel",
    "StrategyPolicy",
    "mixture_policy_metrics",
    "mixture_policy_probs",
]


def _normalized_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return a finite, non-negative vector normalized to sum to one."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def _normalized_last_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Normalize a probability tensor on its final axis."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = array.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError(f"every {name} row must have positive mass")
    return array / total


def _label_indices(labels: np.ndarray, classes: np.ndarray, name: str) -> np.ndarray:
    """Map arbitrary discrete labels to indices in ``classes``."""
    indices = np.empty(labels.size, dtype=np.int64)
    for i, label in enumerate(labels.reshape(-1)):
        matches = np.flatnonzero(classes == label)
        if matches.size != 1:
            raise ValueError(f"unknown {name} label: {label!r}")
        indices[i] = int(matches[0])
    return indices.reshape(labels.shape)


def _sequence_mask(
    batch_size: int,
    time_steps: int,
    *,
    lengths: int | Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine optional sequence lengths and a Boolean validity mask."""
    valid = np.ones((batch_size, time_steps), dtype=bool)
    if lengths is not None:
        length_array = np.asarray(lengths)
        if length_array.ndim == 0 and batch_size == 1:
            length_array = length_array.reshape(1)
        if length_array.shape != (batch_size,):
            raise ValueError("lengths must have shape (batch,)")
        if not np.issubdtype(length_array.dtype, np.integer):
            if not np.all(np.equal(length_array, np.floor(length_array))):
                raise ValueError("lengths must contain integers")
        length_array = length_array.astype(np.int64)
        if np.any(length_array < 0) or np.any(length_array > time_steps):
            raise ValueError("lengths must be between zero and the padded length")
        valid &= np.arange(time_steps)[None, :] < length_array[:, None]
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == (time_steps,) and batch_size == 1:
            mask_array = mask_array[None, :]
        if mask_array.shape != (batch_size, time_steps):
            raise ValueError("mask must have shape (batch, time)")
        valid &= mask_array
    return valid


@dataclass(frozen=True)
class StrategyPolicy:
    """Fitted per-strategy discrete policies with aligned output classes.

    Use :meth:`fit` to create the policy. A strategy with only one observed
    action is represented by a constant ``DummyClassifier`` rather than asking
    logistic regression to fit an invalid one-class problem.
    """

    strategy_classes: np.ndarray
    action_classes: np.ndarray
    estimators: tuple[Any, ...]
    probability_floor: float = 1e-6

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        strategy_labels: np.ndarray,
        *,
        action_classes: np.ndarray | Sequence[Any] | None = None,
        probability_floor: float = 1e-6,
        max_iter: int = 1000,
    ) -> "StrategyPolicy":
        """Fit ``pi_k(a | s)`` for each observed strategy label.

        ``strategy_labels`` are used only here to partition the training rows.
        They are not accepted by any inference method.
        """
        state_array = np.asarray(states, dtype=np.float64)
        action_array = np.asarray(actions)
        strategy_array = np.asarray(strategy_labels)
        if state_array.ndim != 2:
            raise ValueError("states must have shape (samples, features)")
        if action_array.ndim != 1 or strategy_array.ndim != 1:
            raise ValueError("actions and strategy_labels must be one-dimensional")
        if not (
            len(state_array) == len(action_array) == len(strategy_array)
        ):
            raise ValueError("states, actions, and strategy_labels must align")
        if len(state_array) == 0:
            raise ValueError("at least one training sample is required")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("states must be finite")
        if not 0.0 <= probability_floor < 1.0:
            raise ValueError("probability_floor must be in [0, 1)")

        strategy_classes = np.unique(strategy_array)
        if action_classes is None:
            fitted_action_classes = np.unique(action_array)
        else:
            fitted_action_classes = np.asarray(action_classes)
            if fitted_action_classes.ndim != 1 or fitted_action_classes.size == 0:
                raise ValueError("action_classes must be a non-empty vector")
            if np.unique(fitted_action_classes).size != fitted_action_classes.size:
                raise ValueError("action_classes must be unique")
            for action in np.unique(action_array):
                if not np.any(fitted_action_classes == action):
                    raise ValueError(
                        f"observed action {action!r} is absent from action_classes"
                    )
        estimators: list[Any] = []
        for strategy in strategy_classes:
            rows = strategy_array == strategy
            local_actions = action_array[rows]
            if np.unique(local_actions).size == 1:
                estimator: Any = DummyClassifier(strategy="prior")
            else:
                estimator = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=max_iter),
                )
            estimator.fit(state_array[rows], local_actions)
            estimators.append(estimator)

        return cls(
            strategy_classes=np.asarray(strategy_classes),
            action_classes=np.asarray(fitted_action_classes),
            estimators=tuple(estimators),
            probability_floor=float(probability_floor),
        )

    @property
    def n_strategies(self) -> int:
        """Number of fitted strategy types."""
        return int(self.strategy_classes.size)

    @property
    def n_actions(self) -> int:
        """Number of globally aligned action classes."""
        return int(self.action_classes.size)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """Return aligned per-strategy probabilities with shape ``(N, K, A)``."""
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim != 2:
            raise ValueError("states must have shape (samples, features)")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("states must be finite")

        output = np.zeros(
            (len(state_array), self.n_strategies, self.n_actions),
            dtype=np.float64,
        )
        for strategy_index, estimator in enumerate(self.estimators):
            local_probs = np.asarray(
                estimator.predict_proba(state_array), dtype=np.float64
            )
            local_classes = np.asarray(estimator.classes_)
            for local_index, action in enumerate(local_classes):
                action_index = int(
                    _label_indices(
                        np.asarray([action]), self.action_classes, "action"
                    )[0]
                )
                output[:, strategy_index, action_index] = local_probs[:, local_index]

        output = np.maximum(output, self.probability_floor)
        return output / output.sum(axis=-1, keepdims=True)

    def action_likelihoods(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Return ``pi_k(a_i | s_i)`` with shape ``(N, K)``."""
        action_array = np.asarray(actions)
        if action_array.ndim != 1:
            raise ValueError("actions must be one-dimensional")
        probabilities = self.action_probs(states)
        if len(action_array) != len(probabilities):
            raise ValueError("states and actions must align")
        action_indices = _label_indices(action_array, self.action_classes, "action")
        rows = np.arange(len(action_array))[:, None]
        strategies = np.arange(self.n_strategies)[None, :]
        return probabilities[rows, strategies, action_indices[:, None]]


@dataclass(frozen=True)
class BeliefTrace:
    """Predictive and posterior strategy beliefs along padded sequences.

    ``predictive[..., t, :]`` is the belief before observing action ``t``;
    ``probs[..., t, :]`` is the posterior after that action. Masked positions
    leave the previous posterior unchanged.
    """

    predictive: np.ndarray
    probs: np.ndarray
    entropy: np.ndarray
    classes: np.ndarray
    mask: np.ndarray

    @property
    def hard(self) -> np.ndarray:
        """Most likely strategy label at each step."""
        return self.classes[np.argmax(self.probs, axis=-1)]


class BayesianStrategyFilter:
    """Hidden-Markov Bayesian filter over a finite set of strategies.

    ``switch_probability`` is the total probability of leaving the current
    strategy on a valid step and is spread uniformly over the other types.
    Supply ``transition`` instead for a custom row-wise transition matrix.
    The initial prior and transition rows are normalized on construction.
    """

    def __init__(
        self,
        classes: np.ndarray | Sequence[Any] | None = None,
        *,
        n_strategies: int | None = None,
        prior: np.ndarray | Sequence[float] | None = None,
        switch_probability: float = 0.0,
        transition: np.ndarray | None = None,
    ) -> None:
        if classes is None:
            if n_strategies is None or n_strategies < 1:
                raise ValueError("provide classes or a positive n_strategies")
            class_array = np.arange(n_strategies, dtype=np.int64)
        else:
            class_array = np.asarray(classes)
            if class_array.ndim != 1 or class_array.size == 0:
                raise ValueError("classes must be a non-empty vector")
            if np.unique(class_array).size != class_array.size:
                raise ValueError("classes must be unique")
            if n_strategies is not None and n_strategies != class_array.size:
                raise ValueError("n_strategies does not match classes")

        self.classes = class_array
        self.n_strategies = int(class_array.size)
        default_prior = np.ones(self.n_strategies, dtype=np.float64)
        self.prior = _normalized_vector(
            default_prior if prior is None else np.asarray(prior), "prior"
        )
        if self.prior.shape != (self.n_strategies,):
            raise ValueError("prior must have one value per strategy")

        if transition is not None:
            if switch_probability != 0.0:
                raise ValueError("provide transition or switch_probability, not both")
            transition_array = np.asarray(transition, dtype=np.float64)
            expected_shape = (self.n_strategies, self.n_strategies)
            if transition_array.shape != expected_shape:
                raise ValueError(f"transition must have shape {expected_shape}")
            if (
                not np.all(np.isfinite(transition_array))
                or np.any(transition_array < 0.0)
            ):
                raise ValueError("transition must be finite and non-negative")
            row_sums = transition_array.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0.0):
                raise ValueError("every transition row must have positive mass")
            self.transition = transition_array / row_sums
        else:
            if not 0.0 <= switch_probability <= 1.0:
                raise ValueError("switch_probability must be in [0, 1]")
            if self.n_strategies == 1:
                self.transition = np.ones((1, 1), dtype=np.float64)
            else:
                off_diagonal = switch_probability / (self.n_strategies - 1)
                self.transition = np.full(
                    (self.n_strategies, self.n_strategies),
                    off_diagonal,
                    dtype=np.float64,
                )
                np.fill_diagonal(self.transition, 1.0 - switch_probability)

    def filter(
        self,
        likelihoods: np.ndarray,
        *,
        prior: np.ndarray | Sequence[float] | None = None,
        lengths: int | None = None,
        mask: np.ndarray | None = None,
    ) -> BeliefTrace:
        """Filter one likelihood sequence with shape ``(T, K)``."""
        likelihood_array = np.asarray(likelihoods, dtype=np.float64)
        if likelihood_array.ndim != 2:
            raise ValueError("likelihoods must have shape (time, strategies)")
        trace = self.filter_batch(
            likelihood_array[None, ...],
            prior=prior,
            lengths=lengths,
            mask=mask,
        )
        return BeliefTrace(
            predictive=trace.predictive[0],
            probs=trace.probs[0],
            entropy=trace.entropy[0],
            classes=trace.classes,
            mask=trace.mask[0],
        )

    def filter_batch(
        self,
        likelihoods: np.ndarray,
        *,
        prior: np.ndarray | Sequence[float] | None = None,
        lengths: int | Sequence[int] | np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> BeliefTrace:
        """Filter padded likelihoods with shape ``(batch, time, K)``.

        ``lengths`` and ``mask`` may be supplied together; a position is valid
        only when both mark it valid. Values in masked positions are ignored.
        """
        likelihood_array = np.asarray(likelihoods, dtype=np.float64)
        if likelihood_array.ndim != 3:
            raise ValueError("likelihoods must have shape (batch, time, strategies)")
        batch_size, time_steps, n_strategies = likelihood_array.shape
        if n_strategies != self.n_strategies:
            raise ValueError("likelihood strategy axis does not match the filter")
        valid = _sequence_mask(
            batch_size,
            time_steps,
            lengths=lengths,
            mask=mask,
        )
        active_likelihoods = likelihood_array[valid]
        if active_likelihoods.size and (
            not np.all(np.isfinite(active_likelihoods))
            or np.any(active_likelihoods < 0.0)
        ):
            raise ValueError("valid likelihoods must be finite and non-negative")

        if prior is None:
            current = np.broadcast_to(self.prior, (batch_size, n_strategies)).copy()
        else:
            prior_array = np.asarray(prior, dtype=np.float64)
            if prior_array.shape == (n_strategies,):
                current = np.broadcast_to(
                    _normalized_vector(prior_array, "prior"),
                    (batch_size, n_strategies),
                ).copy()
            elif prior_array.shape == (batch_size, n_strategies):
                current = np.stack(
                    [_normalized_vector(row, "prior") for row in prior_array]
                )
            else:
                raise ValueError("prior must have shape (K,) or (batch, K)")

        predictive = np.empty_like(likelihood_array)
        posterior = np.empty_like(likelihood_array)
        seen_observation = np.zeros(batch_size, dtype=bool)
        for time_index in range(time_steps):
            active = valid[:, time_index]
            prediction = current.copy()
            continuing = active & seen_observation
            prediction[continuing] = current[continuing] @ self.transition

            if np.any(active):
                weighted = (
                    prediction[active] * likelihood_array[active, time_index, :]
                )
                evidence = weighted.sum(axis=-1, keepdims=True)
                informative = evidence[:, 0] > 0.0
                active_posteriors = prediction[active].copy()
                active_posteriors[informative] = (
                    weighted[informative] / evidence[informative]
                )
                current[active] = active_posteriors
                seen_observation[active] = True

            predictive[:, time_index, :] = np.where(
                active[:, None], prediction, current
            )
            posterior[:, time_index, :] = current

        return BeliefTrace(
            predictive=predictive,
            probs=posterior,
            entropy=categorical_entropy(posterior),
            classes=self.classes.copy(),
            mask=valid,
        )


def mixture_policy_probs(
    beliefs: np.ndarray,
    per_strategy_probs: np.ndarray,
) -> np.ndarray:
    """Compute ``sum_k b_k pi_k`` over matching leading dimensions.

    ``beliefs`` has shape ``(..., K)`` and ``per_strategy_probs`` has shape
    ``(..., K, A)``. The returned action probabilities have shape ``(..., A)``.
    """
    belief_array = np.asarray(beliefs, dtype=np.float64)
    policy_array = np.asarray(per_strategy_probs, dtype=np.float64)
    if belief_array.ndim < 1 or policy_array.ndim < 2:
        raise ValueError("belief and policy arrays have too few dimensions")
    if policy_array.shape[:-1] != belief_array.shape:
        raise ValueError("policy leading dimensions must match beliefs")
    normalized_beliefs = _normalized_last_axis(belief_array, "belief")
    normalized_policies = _normalized_last_axis(policy_array, "policy")
    mixture = np.sum(normalized_beliefs[..., :, None] * normalized_policies, axis=-2)
    return mixture / mixture.sum(axis=-1, keepdims=True)


def mixture_policy_metrics(
    probabilities: np.ndarray,
    actions: np.ndarray,
    *,
    action_classes: np.ndarray | Sequence[Any] | None = None,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return top-1 accuracy and mean action NLL for a mixture policy."""
    probability_array = _normalized_last_axis(probabilities, "probability")
    action_array = np.asarray(actions)
    if action_array.shape != probability_array.shape[:-1]:
        raise ValueError("actions must match the probability leading dimensions")
    valid = np.ones(action_array.shape, dtype=bool)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != action_array.shape:
            raise ValueError("mask must match actions")
        valid &= mask_array
    if not np.any(valid):
        raise ValueError("at least one valid action is required")

    classes = (
        np.arange(probability_array.shape[-1])
        if action_classes is None
        else np.asarray(action_classes)
    )
    if classes.shape != (probability_array.shape[-1],):
        raise ValueError("action_classes must align with the final probability axis")
    valid_actions = action_array[valid]
    action_indices = _label_indices(valid_actions, classes, "action")
    valid_probs = probability_array[valid]
    selected = valid_probs[np.arange(len(valid_actions)), action_indices]
    predicted_actions = classes[np.argmax(valid_probs, axis=-1)]
    return {
        "top1": float(np.mean(predicted_actions == valid_actions)),
        "nll": float(-np.mean(np.log(np.clip(selected, eps, 1.0)))),
    }


@dataclass(frozen=True)
class SequentialOpponentModel:
    """Per-strategy action policies plus an online Bayesian strategy filter."""

    policy: StrategyPolicy
    belief_filter: BayesianStrategyFilter

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        strategy_labels: np.ndarray,
        *,
        action_classes: np.ndarray | Sequence[Any] | None = None,
        prior: np.ndarray | Sequence[float] | None = None,
        switch_probability: float = 0.0,
        transition: np.ndarray | None = None,
        probability_floor: float = 1e-6,
        max_iter: int = 1000,
    ) -> "SequentialOpponentModel":
        """Fit action models and configure the label-free online filter."""
        policy = StrategyPolicy.fit(
            states,
            actions,
            strategy_labels,
            action_classes=action_classes,
            probability_floor=probability_floor,
            max_iter=max_iter,
        )
        belief_filter = BayesianStrategyFilter(
            classes=policy.strategy_classes,
            prior=prior,
            switch_probability=switch_probability,
            transition=transition,
        )
        return cls(policy=policy, belief_filter=belief_filter)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """Return fitted ``pi_k(a | s)`` values with shape ``(N, K, A)``."""
        return self.policy.action_probs(states)

    def infer(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        prior: np.ndarray | Sequence[float] | None = None,
        lengths: int | Sequence[int] | np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> BeliefTrace:
        """Infer online beliefs from states and actions, never strategy labels.

        Inputs may be one sequence ``(T, F), (T,)`` or a padded batch
        ``(B, T, F), (B, T)``. Masked states/actions are never evaluated, so
        arbitrary padding values cannot affect valid beliefs.
        """
        state_array = np.asarray(states)
        action_array = np.asarray(actions)
        unbatched = state_array.ndim == 2
        if unbatched:
            state_batch = state_array[None, ...]
            action_batch = action_array[None, ...]
        elif state_array.ndim == 3:
            state_batch = state_array
            action_batch = action_array
        else:
            raise ValueError("states must have shape (T, F) or (batch, T, F)")
        if action_batch.shape != state_batch.shape[:2]:
            raise ValueError("actions must match the state sequence dimensions")

        batch_size, time_steps, _ = state_batch.shape
        valid = _sequence_mask(
            batch_size,
            time_steps,
            lengths=lengths,
            mask=mask,
        )
        likelihoods = np.ones(
            (batch_size, time_steps, self.policy.n_strategies),
            dtype=np.float64,
        )
        if np.any(valid):
            likelihoods[valid] = self.policy.action_likelihoods(
                state_batch[valid], action_batch[valid]
            )
        trace = self.belief_filter.filter_batch(
            likelihoods,
            prior=prior,
            mask=valid,
        )
        if not unbatched:
            return trace
        return BeliefTrace(
            predictive=trace.predictive[0],
            probs=trace.probs[0],
            entropy=trace.entropy[0],
            classes=trace.classes,
            mask=trace.mask[0],
        )

    def mixture_probs(
        self,
        states: np.ndarray,
        beliefs: np.ndarray,
    ) -> np.ndarray:
        """Return belief-mixture action probabilities for arbitrary leading axes."""
        state_array = np.asarray(states)
        belief_array = np.asarray(beliefs)
        if state_array.ndim < 2:
            raise ValueError("states must end in a feature axis")
        if state_array.shape[:-1] != belief_array.shape[:-1]:
            raise ValueError("states and beliefs must share leading dimensions")
        if belief_array.shape[-1] != self.policy.n_strategies:
            raise ValueError("belief strategy axis does not match the policy")
        per_strategy = self.policy.action_probs(
            state_array.reshape(-1, state_array.shape[-1])
        ).reshape(
            *belief_array.shape,
            self.policy.n_actions,
        )
        return mixture_policy_probs(belief_array, per_strategy)

    def mixture_metrics(
        self,
        states: np.ndarray,
        beliefs: np.ndarray,
        actions: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Score belief-mixture top-1 accuracy and NLL."""
        probabilities = self.mixture_probs(states, beliefs)
        return mixture_policy_metrics(
            probabilities,
            actions,
            action_classes=self.policy.action_classes,
            mask=mask,
        )
