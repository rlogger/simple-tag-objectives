"""Small, validated trajectory and replay-buffer types.

The counterfactual/CPL pipeline relies on padded, fixed-horizon trajectories.
``valid_mask`` separates real transitions from padding, while planner
provenance makes it possible to check that an original and its counterfactual
were produced with the same planning code and weights.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

__all__ = [
    "PlannerProvenance",
    "ReplayBuffer",
    "Trajectory",
    "TrajectorySchema",
]


def _provenance_value(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _readonly_array(name: str, value: Any, *, kinds: str) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in kinds:
        allowed = {
            "observations": "numeric",
            "blue_actions": "numeric",
            "red_actions": "numeric",
            "rewards": "numeric",
            "strategy_context": "numeric or boolean",
        }.get(name, kinds)
        raise ValueError(f"{name} must have a {allowed} dtype, got {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _readonly_bool_vector(name: str, value: Any, horizon: int) -> Array:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError(f"{name} must have boolean dtype")
    if array.shape != (horizon,):
        raise ValueError(f"{name} must have shape ({horizon},), got {array.shape}")
    copy = np.array(array, dtype=bool, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True, slots=True)
class PlannerProvenance:
    """Exact planner weights and implementation used for a trajectory."""

    checkpoint: str
    version: str

    def __post_init__(self) -> None:
        _provenance_value("checkpoint", self.checkpoint)
        _provenance_value("version", self.version)


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One padded joint trajectory with immutable arrays.

    ``observations`` has one more time step than transition arrays. Blue and
    red action shapes may differ after the time dimension. ``dones`` is the
    global episode-done flag and ``valid_mask`` must be a contiguous prefix.
    A terminal transition, when present, must be the final valid transition.
    """

    observations: Array
    blue_actions: Array
    red_actions: Array
    rewards: Array
    dones: Array
    valid_mask: Array
    strategy_context: Array
    planner_checkpoint: str
    planner_version: str

    def __post_init__(self) -> None:
        observations = _readonly_array(
            "observations", self.observations, kinds="iuf"
        )
        blue_actions = _readonly_array(
            "blue_actions", self.blue_actions, kinds="biuf"
        )
        red_actions = _readonly_array(
            "red_actions", self.red_actions, kinds="biuf"
        )
        rewards = _readonly_array("rewards", self.rewards, kinds="iuf")
        context = _readonly_array(
            "strategy_context", self.strategy_context, kinds="biuf"
        )

        if blue_actions.ndim < 1:
            raise ValueError("blue_actions must have a time dimension")
        horizon = int(blue_actions.shape[0])
        if horizon <= 0:
            raise ValueError("trajectory horizon must be positive")
        if observations.ndim < 1 or observations.shape[0] != horizon + 1:
            raise ValueError(
                "observations must have shape (horizon + 1, ...); "
                f"got {observations.shape} for horizon {horizon}"
            )
        for name, array in (
            ("red_actions", red_actions),
            ("rewards", rewards),
        ):
            if array.ndim < 1 or array.shape[0] != horizon:
                raise ValueError(
                    f"{name} must have horizon {horizon} on axis 0, "
                    f"got {array.shape}"
                )

        dones = _readonly_bool_vector("dones", self.dones, horizon)
        valid = _readonly_bool_vector("valid_mask", self.valid_mask, horizon)
        valid_length = int(valid.sum())
        if valid_length == 0:
            raise ValueError("valid_mask must contain at least one valid transition")
        expected_valid = np.arange(horizon) < valid_length
        if not np.array_equal(valid, expected_valid):
            raise ValueError("valid_mask must be one contiguous prefix of True values")
        if np.any(dones[~valid]):
            raise ValueError("dones cannot be True in padded transitions")
        terminal_steps = np.flatnonzero(dones)
        if len(terminal_steps) > 1:
            raise ValueError("dones may contain at most one terminal transition")
        if len(terminal_steps) == 1 and terminal_steps[0] != valid_length - 1:
            raise ValueError("done must be the final valid transition")

        checkpoint = _provenance_value(
            "planner_checkpoint", self.planner_checkpoint
        )
        version = _provenance_value("planner_version", self.planner_version)

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "blue_actions", blue_actions)
        object.__setattr__(self, "red_actions", red_actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "dones", dones)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "strategy_context", context)
        object.__setattr__(self, "planner_checkpoint", checkpoint)
        object.__setattr__(self, "planner_version", version)

    @property
    def horizon(self) -> int:
        return int(self.blue_actions.shape[0])

    @property
    def valid_length(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def provenance(self) -> PlannerProvenance:
        return PlannerProvenance(self.planner_checkpoint, self.planner_version)


@dataclass(frozen=True, slots=True)
class TrajectorySchema:
    """Array shapes and dtypes shared by trajectories in one replay buffer."""

    observations_shape: tuple[int, ...]
    blue_actions_shape: tuple[int, ...]
    red_actions_shape: tuple[int, ...]
    rewards_shape: tuple[int, ...]
    strategy_context_shape: tuple[int, ...]
    observations_dtype: str
    blue_actions_dtype: str
    red_actions_dtype: str
    rewards_dtype: str
    strategy_context_dtype: str

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> "TrajectorySchema":
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory")
        return cls(
            observations_shape=trajectory.observations.shape,
            blue_actions_shape=trajectory.blue_actions.shape,
            red_actions_shape=trajectory.red_actions.shape,
            rewards_shape=trajectory.rewards.shape,
            strategy_context_shape=trajectory.strategy_context.shape,
            observations_dtype=trajectory.observations.dtype.str,
            blue_actions_dtype=trajectory.blue_actions.dtype.str,
            red_actions_dtype=trajectory.red_actions.dtype.str,
            rewards_dtype=trajectory.rewards.dtype.str,
            strategy_context_dtype=trajectory.strategy_context.dtype.str,
        )

    def validate(self, trajectory: Trajectory) -> None:
        """Raise when ``trajectory`` cannot be batched with this schema."""
        actual = type(self).from_trajectory(trajectory)
        if actual != self:
            mismatches = [
                name
                for name in self.__dataclass_fields__
                if getattr(actual, name) != getattr(self, name)
            ]
            raise ValueError(
                "trajectory does not match replay schema: " + ", ".join(mismatches)
            )


class ReplayBuffer(Sequence[Trajectory]):
    """A fixed-capacity FIFO replay buffer with one strict array schema."""

    def __init__(
        self,
        capacity: int,
        trajectories: Iterable[Trajectory] = (),
        *,
        provenance: PlannerProvenance | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if provenance is not None and not isinstance(provenance, PlannerProvenance):
            raise TypeError("provenance must be PlannerProvenance or None")
        self.capacity = capacity
        self.expected_provenance = provenance
        self._items: list[Trajectory] = []
        self._schema: TrajectorySchema | None = None
        self.extend(trajectories)

    @property
    def schema(self) -> TrajectorySchema | None:
        return self._schema

    def add(self, trajectory: Trajectory) -> None:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("replay entries must be Trajectory instances")
        if (
            self.expected_provenance is not None
            and trajectory.provenance != self.expected_provenance
        ):
            raise ValueError(
                "trajectory planner provenance does not match the replay buffer"
            )
        if self._schema is None:
            self._schema = TrajectorySchema.from_trajectory(trajectory)
        else:
            self._schema.validate(trajectory)
        self._items.append(trajectory)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    append = add

    def extend(self, trajectories: Iterable[Trajectory]) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | int | None = None,
        replace: bool = False,
    ) -> tuple[Trajectory, ...]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not replace and batch_size > len(self):
            raise ValueError("batch_size exceeds replay size without replacement")
        if len(self) == 0:
            raise ValueError("cannot sample an empty replay buffer")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        indices = generator.choice(len(self), size=batch_size, replace=replace)
        return tuple(self._items[int(index)] for index in np.atleast_1d(indices))

    def clear(self) -> None:
        self._items.clear()
        self._schema = None

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):  # noqa: ANN001, ANN204
        return self._items[index]

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._items)
