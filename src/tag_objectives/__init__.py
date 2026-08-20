"""Objective-typed predator-prey environment on JaxMARL simple-tag."""

from __future__ import annotations

__version__ = "0.2.0"

from tag_objectives.api import (
    evaluate_policy,
    list_objectives,
    make_env,
    random_policy,
)
from tag_objectives.objectives import (
    OBJECTIVES,
    ObjectiveSpec,
    ObjectiveState,
    SimpleTagObjectivesMPE,
    register_objective,
)
from tag_objectives.resources import ResourceState, SimpleTagResourcesMPE
from tag_objectives.types import EpisodeMetrics

__all__ = [
    "OBJECTIVES",
    "EpisodeMetrics",
    "ObjectiveSpec",
    "ObjectiveState",
    "ResourceState",
    "SimpleTagObjectivesMPE",
    "SimpleTagResourcesMPE",
    "__version__",
    "evaluate_policy",
    "list_objectives",
    "make_env",
    "random_policy",
    "register_objective",
]
