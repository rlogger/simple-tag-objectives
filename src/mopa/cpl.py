"""Contrastive preference learning and counterfactual rollouts.

The generator is deliberately model-agnostic. The red policy chooses the next
red action and the blue planner returns both its action and the modeled next
transition. This keeps environment/model details out of the CPL foundation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

from mopa.replay import PlannerProvenance, Trajectory, TrajectorySchema

__all__ = [
    "BluePlanner",
    "BluePlannerRequest",
    "OpponentPolicy",
    "OpponentPolicyRequest",
    "PlannedStep",
    "PreferencePair",
    "bradley_terry_cpl_loss",
    "cpl_loss",
    "generate_counterfactual",
    "trajectory_log_probability",
]


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _same_trajectory(left: Trajectory, right: Trajectory) -> bool:
    return all(
        _same_array(getattr(left, name), getattr(right, name))
        for name in (
            "observations",
            "blue_actions",
            "red_actions",
            "rewards",
            "dones",
            "valid_mask",
            "strategy_context",
        )
    )


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """An original preferred rollout and a counterfactual dispreferred rollout."""

    preferred: Trajectory
    dispreferred: Trajectory
    deviation_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.preferred, Trajectory) or not isinstance(
            self.dispreferred, Trajectory
        ):
            raise TypeError("preferred and dispreferred must be Trajectory instances")
        if isinstance(self.deviation_step, bool) or not isinstance(
            self.deviation_step, int
        ):
            raise TypeError("deviation_step must be an integer")

        TrajectorySchema.from_trajectory(self.preferred).validate(self.dispreferred)
        if self.preferred.provenance != self.dispreferred.provenance:
            raise ValueError(
                "preferred and dispreferred trajectories must use identical "
                "planner checkpoint and version"
            )
        if not _same_array(
            self.preferred.strategy_context, self.dispreferred.strategy_context
        ):
            raise ValueError("preference trajectories must share strategy_context")

        max_deviation = min(
            self.preferred.valid_length, self.dispreferred.valid_length
        )
        if not 0 <= self.deviation_step < max_deviation:
            raise ValueError(
                "deviation_step must identify a valid transition in both trajectories"
            )

        step = self.deviation_step
        prefix_arrays = (
            ("observations", step + 1),
            ("blue_actions", step),
            ("red_actions", step),
            ("rewards", step),
            ("dones", step),
            ("valid_mask", step),
        )
        for name, stop in prefix_arrays:
            preferred_prefix = getattr(self.preferred, name)[:stop]
            dispreferred_prefix = getattr(self.dispreferred, name)[:stop]
            if not _same_array(preferred_prefix, dispreferred_prefix):
                raise ValueError(
                    f"preference trajectories do not share their original {name} prefix"
                )
        if _same_trajectory(self.preferred, self.dispreferred):
            raise ValueError(
                "preferred and dispreferred trajectories must be non-identical"
            )

    @property
    def prefix_steps(self) -> int:
        """Number of transitions copied unchanged before the deviation."""
        return self.deviation_step


def _mask_for(log_probs, mask, name: str):  # noqa: ANN001, ANN202
    if mask is None:
        return jnp.ones_like(log_probs, dtype=bool)
    result = jnp.asarray(mask, dtype=bool)
    if result.shape != log_probs.shape:
        raise ValueError(
            f"{name} must have shape {log_probs.shape}, got {result.shape}"
        )
    return result


def trajectory_log_probability(log_probs, valid_mask=None):  # noqa: ANN001, ANN201
    """Sum per-step policy log probabilities over the final (time) axis."""
    values = jnp.asarray(log_probs)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("log_probs must have a non-empty time axis")
    mask = _mask_for(values, valid_mask, "valid_mask")
    return jnp.sum(jnp.where(mask, values, 0.0), axis=-1)


def bradley_terry_cpl_loss(
    preferred_log_probs,
    dispreferred_log_probs,
    beta: float = 1.0,
    *,
    preferred_mask=None,
    dispreferred_mask=None,
    reduction: Literal["none", "mean", "sum"] = "mean",
):  # noqa: ANN001, ANN201
    """Bradley--Terry CPL loss from summed trajectory log probabilities.

    For each pair the margin is
    ``beta * (sum(log pi_preferred) - sum(log pi_dispreferred))`` and the loss
    is ``softplus(-margin)``. Masks exclude padded transitions from each sum.
    The implementation uses JAX arrays so it remains differentiable.
    """
    preferred = jnp.asarray(preferred_log_probs)
    dispreferred = jnp.asarray(dispreferred_log_probs)
    if preferred.shape != dispreferred.shape:
        raise ValueError(
            "preferred and dispreferred log probabilities must have the same shape"
        )
    if preferred.ndim < 1 or preferred.shape[-1] == 0:
        raise ValueError("log probabilities must have a non-empty time axis")
    if isinstance(beta, jax_core.Tracer):
        raise TypeError("beta must be a concrete scalar hyperparameter")
    beta_value = np.asarray(beta)
    if beta_value.shape != () or not np.isfinite(beta_value) or beta_value <= 0:
        raise ValueError("beta must be finite and positive")

    preferred_score = trajectory_log_probability(preferred, preferred_mask)
    dispreferred_score = trajectory_log_probability(
        dispreferred, dispreferred_mask
    )
    margin = jnp.asarray(beta_value) * (preferred_score - dispreferred_score)
    losses = jnp.logaddexp(0.0, -margin)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return jnp.mean(losses)
    if reduction == "sum":
        return jnp.sum(losses)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def cpl_loss(
    preferred_log_probs,
    dispreferred_log_probs,
    beta: float = 1.0,
    **kwargs,
):  # noqa: ANN001, ANN201
    """Short alias for :func:`bradley_terry_cpl_loss`."""
    return bradley_terry_cpl_loss(
        preferred_log_probs, dispreferred_log_probs, beta, **kwargs
    )


@dataclass(frozen=True, slots=True)
class OpponentPolicyRequest:
    """Immutable trajectory prefix passed to the learned red policy."""

    step: int
    observation: np.ndarray
    observations: np.ndarray
    blue_actions: np.ndarray
    red_actions: np.ndarray
    rewards: np.ndarray
    strategy_context: np.ndarray
    provenance: PlannerProvenance
    rng: np.random.Generator


@dataclass(frozen=True, slots=True)
class BluePlannerRequest(OpponentPolicyRequest):
    """The same prefix plus the newly sampled red action."""

    red_action: np.ndarray


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """One blue plan/model transition and the provenance actually used."""

    blue_action: Any
    next_observation: Any
    reward: Any
    done: bool
    planner_checkpoint: str
    planner_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.done, (bool, np.bool_)):
            raise TypeError("done must be boolean")
        provenance = PlannerProvenance(
            self.planner_checkpoint, self.planner_version
        )
        object.__setattr__(self, "blue_action", np.asarray(self.blue_action))
        object.__setattr__(
            self, "next_observation", np.asarray(self.next_observation)
        )
        object.__setattr__(self, "reward", np.asarray(self.reward))
        object.__setattr__(self, "done", bool(self.done))
        object.__setattr__(self, "planner_checkpoint", provenance.checkpoint)
        object.__setattr__(self, "planner_version", provenance.version)

    @property
    def provenance(self) -> PlannerProvenance:
        return PlannerProvenance(self.planner_checkpoint, self.planner_version)


class OpponentPolicy(Protocol):
    def __call__(self, request: OpponentPolicyRequest) -> Any: ...


class BluePlanner(Protocol):
    def __call__(self, request: BluePlannerRequest) -> PlannedStep: ...


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _coerce_step_value(name: str, value: Any, template: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != template.shape:
        raise ValueError(
            f"{name} must have shape {template.shape}, got {result.shape}"
        )
    if template.dtype.kind in "iu" and result.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer values")
    if template.dtype.kind == "b" and result.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    if template.dtype.kind == "f" and result.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if template.dtype.kind in "iu" and result.size:
        bounds = np.iinfo(template.dtype)
        if result.min() < bounds.min or result.max() > bounds.max:
            raise ValueError(f"{name} cannot be represented as {template.dtype}")
    return result.astype(template.dtype, copy=False)


def _opponent_request(
    step: int,
    observations: np.ndarray,
    blue_actions: np.ndarray,
    red_actions: np.ndarray,
    rewards: np.ndarray,
    original: Trajectory,
    rng: np.random.Generator,
) -> OpponentPolicyRequest:
    return OpponentPolicyRequest(
        step=step,
        observation=_readonly_copy(observations[step]),
        observations=_readonly_copy(observations[: step + 1]),
        blue_actions=_readonly_copy(blue_actions[:step]),
        red_actions=_readonly_copy(red_actions[:step]),
        rewards=_readonly_copy(rewards[:step]),
        strategy_context=original.strategy_context,
        provenance=original.provenance,
        rng=rng,
    )


def generate_counterfactual(
    original: Trajectory,
    deviation_step: int,
    opponent_policy: OpponentPolicy,
    blue_planner: BluePlanner,
    *,
    rng: np.random.Generator | int | None = None,
) -> Trajectory:
    """Regenerate a trajectory suffix from ``deviation_step`` onward.

    Red is resampled before blue replans at every regenerated step. The blue
    callback also supplies the modeled next observation/reward/done. Every
    returned step must attest to the original planner checkpoint and version;
    a mismatch is rejected instead of silently creating an invalid CPL pair.
    """
    if not isinstance(original, Trajectory):
        raise TypeError("original must be a Trajectory")
    if isinstance(deviation_step, bool) or not isinstance(deviation_step, int):
        raise TypeError("deviation_step must be an integer")
    if not 0 <= deviation_step < original.valid_length:
        raise ValueError("deviation_step must identify a valid original transition")
    if not callable(opponent_policy) or not callable(blue_planner):
        raise TypeError("opponent_policy and blue_planner must be callable")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    observations = np.array(original.observations, copy=True)
    blue_actions = np.array(original.blue_actions, copy=True)
    red_actions = np.array(original.red_actions, copy=True)
    rewards = np.array(original.rewards, copy=True)
    dones = np.array(original.dones, copy=True)
    valid = np.array(original.valid_mask, copy=True)

    # Everything at/after the deviation is regenerated, including padded time.
    dones[deviation_step:] = False
    valid[deviation_step:] = False
    for step in range(deviation_step, original.horizon):
        opponent_request = _opponent_request(
            step,
            observations,
            blue_actions,
            red_actions,
            rewards,
            original,
            generator,
        )
        sampled_red = _coerce_step_value(
            "red_action",
            opponent_policy(opponent_request),
            original.red_actions[step],
        )
        planner_request = BluePlannerRequest(
            step=opponent_request.step,
            observation=opponent_request.observation,
            observations=opponent_request.observations,
            blue_actions=opponent_request.blue_actions,
            red_actions=opponent_request.red_actions,
            rewards=opponent_request.rewards,
            strategy_context=opponent_request.strategy_context,
            provenance=opponent_request.provenance,
            rng=opponent_request.rng,
            red_action=_readonly_copy(sampled_red),
        )
        planned = blue_planner(planner_request)
        if not isinstance(planned, PlannedStep):
            raise TypeError("blue_planner must return PlannedStep")
        if planned.provenance != original.provenance:
            raise ValueError(
                "counterfactual planner provenance must exactly match the "
                "original checkpoint and version"
            )

        red_actions[step] = sampled_red
        blue_actions[step] = _coerce_step_value(
            "blue_action", planned.blue_action, original.blue_actions[step]
        )
        observations[step + 1] = _coerce_step_value(
            "next_observation",
            planned.next_observation,
            original.observations[step + 1],
        )
        rewards[step] = _coerce_step_value(
            "reward", planned.reward, original.rewards[step]
        )
        dones[step] = planned.done
        valid[step] = True

        if planned.done:
            # Canonical padding: frozen terminal observation and zero transitions.
            if step + 1 < original.horizon:
                observations[step + 2 :] = observations[step + 1]
                blue_actions[step + 1 :] = 0
                red_actions[step + 1 :] = 0
                rewards[step + 1 :] = 0
                dones[step + 1 :] = False
                valid[step + 1 :] = False
            break

    return Trajectory(
        observations=observations,
        blue_actions=blue_actions,
        red_actions=red_actions,
        rewards=rewards,
        dones=dones,
        valid_mask=valid,
        strategy_context=original.strategy_context,
        planner_checkpoint=original.planner_checkpoint,
        planner_version=original.planner_version,
    )
