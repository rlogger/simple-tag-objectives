"""Tests for the 7/31-review JEPA training procedure (GRU, L2, LayerNorm)."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp

from mopa.encoders import Enc, GRUEnc, Pred, train_jepa, train_jepa_gru


def _param_names(tree):
    names = set()

    def visit(node, prefix=""):
        for k, v in node.items():
            if isinstance(v, dict):
                visit(v, prefix + k + "/")
            else:
                names.add(prefix + k)

    visit(jax.tree_util.tree_map(lambda x: None, tree)["params"])
    return names


@pytest.mark.parametrize(
    "module,x",
    [
        (Enc(lat=2), jnp.zeros((1, 8))),
        (Pred(lat=2), jnp.zeros((1, 2))),
        (GRUEnc(lat=2, hid=16), jnp.zeros((1, 4, 3))),
    ],
)
def test_encoders_use_layernorm_not_batchnorm(module, x):
    params = module.init(jax.random.PRNGKey(0), x)
    names = _param_names(params)
    assert any("LayerNorm" in n for n in names)
    assert not any("BatchNorm" in n for n in names)


def test_gru_prefix_latent_ignores_future_steps():
    enc = GRUEnc(lat=3, hid=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, 10, 4))
    params = enc.init(rng, x)
    z = enc.apply(params, x)
    assert z.shape == (5, 10, 3)

    x_mut = x.at[:, 6:].set(99.0)
    z_mut = enc.apply(params, x_mut)
    np.testing.assert_allclose(z[:, :6], z_mut[:, :6], atol=1e-5)
    assert not np.allclose(z[:, 9], z_mut[:, 9])


def test_train_jepa_gru_handles_variable_lengths():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 12, 4)).astype(np.float32)
    lengths = rng.integers(5, 13, size=32)
    z = train_jepa_gru(
        x, lengths, jax.random.PRNGKey(0), lat=2, hid=16, steps=5
    )
    assert z.shape == (32, 12, 2)
    assert np.isfinite(z).all()


def test_train_jepa_smoke():
    rng = np.random.default_rng(0)
    xc = rng.normal(size=(32, 6)).astype(np.float32)
    xt = rng.normal(size=(32, 6)).astype(np.float32)
    z = train_jepa(xc, xt, jax.random.PRNGKey(0), lat=2, steps=5)
    assert z.shape == (32, 2)
    assert np.isfinite(z).all()
