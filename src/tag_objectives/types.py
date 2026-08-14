"""Typed result containers for environment evaluation."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

Arrayf = NDArray[np.floating[Any]]


class _ArrayMapping:
    """Mixin: dataclass of arrays that also behaves like ``Mapping[str, ndarray]``."""

    def as_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def keys(self) -> Iterator[str]:
        return iter(self)

    def values(self) -> Iterator[np.ndarray]:
        return (getattr(self, f.name) for f in fields(self))

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        return ((f.name, getattr(self, f.name)) for f in fields(self))


@dataclass
class EpisodeMetrics(_ArrayMapping):
    """Per-episode behavior panel (one row per episode)."""

    capture_rate: Arrayf
    survival_time: Arrayf
    resources_collected: Arrayf
    near_lava_collected: Arrayf
    pred_lava_steps: Arrayf
    prey_lava_steps: Arrayf
    pred_coverage: Arrayf

    def means(self) -> dict[str, float]:
        return {k: float(np.asarray(v).mean()) for k, v in self.as_dict().items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, np.ndarray]) -> EpisodeMetrics:
        return cls(**{f.name: np.asarray(d[f.name]) for f in fields(cls)})
