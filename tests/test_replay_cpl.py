from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.cpl import (
    PlannedStep,
    PreferencePair,
    bradley_terry_cpl_loss,
    generate_counterfactual,
)
from mopa.replay import PlannerProvenance, ReplayBuffer, Trajectory


def trajectory(*, checkpoint="weights-abc", version="planner-v1", horizon=4):
    return Trajectory(
        observations=np.arange((horizon + 1) * 2, dtype=np.float32).reshape(
            horizon + 1, 2
        ),
        blue_actions=np.arange(horizon, dtype=np.int32),
        red_actions=np.arange(horizon, dtype=np.int32) + 10,
        rewards=np.arange(horizon, dtype=np.float32),
        dones=np.array([False] * horizon),
        valid_mask=np.array([True] * horizon),
        strategy_context=np.array([0.25, -0.5], dtype=np.float32),
        planner_checkpoint=checkpoint,
        planner_version=version,
    )


def changed_suffix(original, deviation=2):
    red_actions = np.array(original.red_actions, copy=True)
    red_actions[deviation:] += 20
    return replace(original, red_actions=red_actions)


def test_trajectory_and_replay_validate_shapes_masks_and_provenance():
    original = trajectory()
    assert original.valid_length == 4
    assert original.provenance == PlannerProvenance("weights-abc", "planner-v1")
    assert not original.observations.flags.writeable

    with pytest.raises(ValueError, match=r"horizon \+ 1"):
        replace(original, observations=original.observations[:-1])
    with pytest.raises(ValueError, match="contiguous prefix"):
        replace(original, valid_mask=np.array([True, False, True, False]))
    with pytest.raises(ValueError, match="non-empty string"):
        replace(original, planner_checkpoint="")

    replay = ReplayBuffer(
        2, provenance=PlannerProvenance("weights-abc", "planner-v1")
    )
    replay.add(original)
    replay.add(changed_suffix(original))
    assert replay.sample(2, rng=0) == (replay[0], replay[1]) or replay.sample(
        2, rng=0
    ) == (replay[1], replay[0])
    with pytest.raises(ValueError, match="replay schema"):
        replay.add(
            replace(
                original,
                strategy_context=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            )
        )
    with pytest.raises(ValueError, match="provenance"):
        replay.add(replace(original, planner_version="planner-v2"))


def test_preference_pair_requires_integrity_and_a_shared_prefix():
    preferred = trajectory()
    dispreferred = changed_suffix(preferred, deviation=2)
    pair = PreferencePair(preferred, dispreferred, deviation_step=2)
    assert pair.prefix_steps == 2
    np.testing.assert_array_equal(
        pair.preferred.observations[:3], pair.dispreferred.observations[:3]
    )

    with pytest.raises(ValueError, match="non-identical"):
        PreferencePair(preferred, preferred, deviation_step=2)

    bad_prefix = np.array(dispreferred.red_actions, copy=True)
    bad_prefix[1] += 1
    with pytest.raises(ValueError, match="red_actions prefix"):
        PreferencePair(
            preferred, replace(dispreferred, red_actions=bad_prefix), deviation_step=2
        )

    with pytest.raises(ValueError, match="identical planner"):
        PreferencePair(
            preferred,
            replace(dispreferred, planner_version="planner-v2"),
            deviation_step=2,
        )


def test_bradley_terry_loss_has_expected_numerics_and_direction():
    equal = bradley_terry_cpl_loss(
        np.array([-1.0, -1.0]), np.array([-1.0, -1.0]), beta=2.0
    )
    np.testing.assert_allclose(equal, np.log(2.0), rtol=1e-6)

    preferred = np.array([[-0.1, -0.2], [-2.0, -2.0]], dtype=np.float32)
    dispreferred = np.array([[-1.0, -1.0], [-0.2, -0.2]], dtype=np.float32)
    losses = bradley_terry_cpl_loss(
        preferred, dispreferred, beta=0.5, reduction="none"
    )
    expected = np.logaddexp(0.0, -0.5 * np.array([1.7, -3.6]))
    np.testing.assert_allclose(losses, expected, rtol=1e-6)
    assert losses[0] < losses[1]

    masked = bradley_terry_cpl_loss(
        np.array([-0.1, -0.2, -100.0]),
        np.array([-1.0, -1.0, 100.0]),
        preferred_mask=np.array([True, True, False]),
        dispreferred_mask=np.array([True, True, False]),
    )
    np.testing.assert_allclose(masked, np.logaddexp(0.0, -1.7), rtol=1e-6)

    # The foundation remains usable as a differentiable training objective.
    grad = jax.grad(lambda x: bradley_terry_cpl_loss(x, jnp.array([-1.0])))(
        jnp.array([-0.2])
    )
    assert grad[0] < 0

    with pytest.raises(ValueError, match="finite and positive"):
        bradley_terry_cpl_loss(
            jnp.array([-0.2]), jnp.array([-1.0]), beta=jnp.array(-1.0)
        )
    with pytest.raises(ValueError, match="finite and positive"):
        bradley_terry_cpl_loss(
            jnp.array([-0.2]), jnp.array([-1.0]), beta=jnp.array(jnp.nan)
        )


def test_counterfactual_preserves_prefix_and_uses_both_callbacks_each_step():
    original = trajectory()
    red_calls = []
    blue_calls = []

    def red_policy(request):
        red_calls.append(request.step)
        # The second request sees the first generated red action in its history.
        if request.step == 3:
            assert request.red_actions[-1] == 102
        return 100 + request.step

    def blue_planner(request):
        blue_calls.append(request.step)
        assert request.red_action == 100 + request.step
        return PlannedStep(
            blue_action=200 + request.step,
            next_observation=request.observation + request.red_action,
            reward=float(request.step),
            done=False,
            planner_checkpoint=request.provenance.checkpoint,
            planner_version=request.provenance.version,
        )

    counterfactual = generate_counterfactual(
        original, 2, red_policy, blue_planner, rng=7
    )

    assert red_calls == [2, 3]
    assert blue_calls == [2, 3]
    np.testing.assert_array_equal(
        counterfactual.observations[:3], original.observations[:3]
    )
    np.testing.assert_array_equal(
        counterfactual.blue_actions[:2], original.blue_actions[:2]
    )
    np.testing.assert_array_equal(
        counterfactual.red_actions[:2], original.red_actions[:2]
    )
    np.testing.assert_array_equal(counterfactual.red_actions[2:], [102, 103])
    np.testing.assert_array_equal(counterfactual.blue_actions[2:], [202, 203])
    assert counterfactual.provenance == original.provenance
    PreferencePair(original, counterfactual, deviation_step=2)


def test_counterfactual_stops_on_done_and_canonicalizes_padding():
    original = trajectory(horizon=5)
    calls = []

    def red_policy(request):
        calls.append(("red", request.step))
        return 50 + request.step

    def planner(request):
        calls.append(("blue", request.step))
        return PlannedStep(
            blue_action=60 + request.step,
            next_observation=request.observation + 1,
            reward=-1.0,
            done=request.step == 2,
            planner_checkpoint=request.provenance.checkpoint,
            planner_version=request.provenance.version,
        )

    counterfactual = generate_counterfactual(original, 1, red_policy, planner)
    assert calls == [("red", 1), ("blue", 1), ("red", 2), ("blue", 2)]
    np.testing.assert_array_equal(
        counterfactual.valid_mask, [True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        counterfactual.dones, [False, False, True, False, False]
    )
    np.testing.assert_array_equal(counterfactual.blue_actions[3:], 0)
    np.testing.assert_array_equal(counterfactual.red_actions[3:], 0)


def test_counterfactual_rejects_planner_provenance_drift():
    original = trajectory()

    def red_policy(request):
        return 99

    def wrong_planner(request):
        return PlannedStep(
            blue_action=1,
            next_observation=request.observation,
            reward=0.0,
            done=False,
            planner_checkpoint="different-weights",
            planner_version=request.provenance.version,
        )

    with pytest.raises(ValueError, match="provenance must exactly match"):
        generate_counterfactual(original, 2, red_policy, wrong_planner)
