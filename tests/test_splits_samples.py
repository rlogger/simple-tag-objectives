import numpy as np

from mopa.samples import build_predator_samples, predator_state_features, valid_bc_steps
from mopa.splits import episode_validation_mask


def toy_dataset(n=2, horizon=6, n_pred=3):
    prey = np.zeros((n, horizon + 1, 2), dtype=np.float32)
    preds = np.zeros((n, horizon + 1, n_pred, 2), dtype=np.float32)
    acts = np.zeros((n, horizon, n_pred), dtype=np.int32)
    for e in range(n):
        for t in range(horizon + 1):
            prey[e, t] = [10 * e + t, 10 * e - t]
            for p in range(n_pred):
                preds[e, t, p] = [100 * p + t, 10 * e + p]
        for t in range(horizon):
            for p in range(n_pred):
                acts[e, t, p] = (e + t + p) % 5
    if horizon > 4:
        acts[:, 4, :] = 99
    return {"prey_pos": prey, "pred_pos": preds, "pred_act": acts}


def test_valid_bc_steps_skip_zero_and_reset_boundaries():
    assert valid_bc_steps(0, 7, ep_len=3) == (1, 2, 3, 5, 6)
    assert valid_bc_steps(0, 30, ep_len=25) == tuple(
        t for t in range(1, 30) if t % 26 != 0
    )


def test_predator_state_feature_order_and_identity():
    ds = toy_dataset(n=1, horizon=2, n_pred=3)
    feats = predator_state_features(
        ds["pred_pos"][:, 1],
        ds["prey_pos"][:, 1],
        ds["pred_pos"][:, 0],
        ds["prey_pos"][:, 0],
    )

    assert feats.shape == (1, 3, 19)
    np.testing.assert_array_equal(feats[0, 0, :8], [1, 0, 101, 1, 201, 2, 1, -1])
    np.testing.assert_array_equal(feats[0, 0, 8:16], [1, 0, 1, 0, 1, 0, 1, -1])
    np.testing.assert_array_equal(feats[0, 0, 16:], [1, 0, 0])
    np.testing.assert_array_equal(feats[0, 2, 16:], [0, 0, 1])


def test_build_predator_samples_excludes_reset_leakage():
    ds = toy_dataset()

    S, A, ep = build_predator_samples(ds, 1, 6, ep_len=3)

    assert S.shape == (24, 19)
    assert A.shape == (24,)
    assert ep.shape == (24,)
    assert 99 not in A
    np.testing.assert_array_equal(ep[:2], [0, 1])


def test_build_samples_supports_single_predator():
    ds = toy_dataset(n_pred=1)
    S, A, ep = build_predator_samples(ds, 1, 6, ep_len=3)
    # steps (1,2,3,5) × 2 eps × 1 pred; feat = pos(4)+vel(4)+id(1)
    assert S.shape == (8, 9)
    assert A.shape == (8,)
    assert ep.shape == (8,)


def test_episode_validation_mask_keeps_whole_episodes_together():
    ep = np.repeat(np.arange(10), 3)

    mask = episode_validation_mask(ep, rng_seed=7, val_frac=0.2)

    val_eps = set(ep[mask])
    train_eps = set(ep[~mask])
    assert len(val_eps) == 2
    assert val_eps.isdisjoint(train_eps)
    for e in val_eps:
        assert mask[ep == e].all()
