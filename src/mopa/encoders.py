"""Self-supervised opponent-trajectory encoders: VAE (reconstruct) vs JEPA
(predict the future's representation).

Generalised from the validated hidden-intent study
(src/jepa_vs_vae_encoder.py) to any window sizes and label cardinality, with
the two protocol upgrades the audit called for:

  * multi-seed: every reported number is mean +/- std over ENCODER training
    seeds (default 3), not a single PRNGKey(0) point estimate;
  * the probe is cross-validated at the trajectory level (one sample per
    episode), so there is no pooled-timestep leakage.

The JEPA training procedure follows the 2026-07-31 "Thoughts on Latent
Strategy Modeling" review:

  1. GRU-based encoder (``GRUEnc`` / ``train_jepa_gru``) for variable-length
     sequences, so an episode can be encoded "so far" at any step and full
     100-step episodes are no longer truncated to a fixed window;
  2. L2 norm (not L1) for the latent prediction loss;
  3. no VICReg variance penalty -- it pushes latents apart and conflicts with
     JEPA on correlated RL batches;
  4. LayerNorm (never BatchNorm) inside the encoders to prevent collapse:
     LayerNorm normalizes per trajectory, so context and target windows with
     different distributions do not contaminate each other's statistics.

The old study scripts under ``src/`` keep their own frozen copies of the
pre-review loss so their published numbers stay reproducible.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
import optax

from mopa.metrics import metrics, oracle_acc, probe_acc  # noqa: F401 — re-export

HID = 64
STEPS = 5000
BATCH = 128
EMA = 0.996


class Enc(nn.Module):
    """MLP context encoder with LayerNorm on the hidden layers (anti-collapse:
    if activations head toward a constant, the per-trajectory variance in the
    norm's denominator shrinks and gradients grow, pushing the network back)."""

    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x)))
        x = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x)))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(x)


class EncVAE(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.lat)(x), nn.Dense(self.lat)(x)   # mu, logvar


class Dec(nn.Module):
    out: int

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(self.out)(z)


class Pred(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z)))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(z)


class GRUEnc(nn.Module):
    """GRU sequence encoder for variable-length trajectories.

    Takes padded sequences ``x`` of shape ``(N, T, F)`` and returns per-prefix
    latents of shape ``(N, T, lat)``: output ``[:, k - 1]`` encodes the first
    ``k`` steps. LayerNorm on the input projection and recurrent states
    prevents collapse.

    Hand-rolled GRU (explicit kernels + ``lax.scan``) avoids Flax ``nn.RNN`` /
    ``nn.scan`` APIs that require newer JAX than JaxMARL currently pins.
    """

    lat: int = 2
    hid: int = HID

    @nn.compact
    def __call__(self, x):
        # x: (N, T, F)
        x = nn.relu(
            nn.LayerNorm()(
                nn.Dense(self.hid, kernel_init=orthogonal(np.sqrt(2)))(x)
            )
        )
        # Explicit kernels so the scan body is pure jnp (no Module.param inside).
        wz = self.param("wz", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        bz = self.param("bz", nn.initializers.zeros, (self.hid,))
        wr = self.param("wr", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        br = self.param("br", nn.initializers.zeros, (self.hid,))
        wh = self.param("wh", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        bh = self.param("bh", nn.initializers.zeros, (self.hid,))
        uz = self.param("uz", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        ur = self.param("ur", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        uh = self.param("uh", orthogonal(np.sqrt(2)), (self.hid, self.hid))

        def step(h, xt):
            z = jax.nn.sigmoid(xt @ wz + h @ uz + bz)
            r = jax.nn.sigmoid(xt @ wr + h @ ur + br)
            n = jnp.tanh(xt @ wh + (r * h) @ uh + bh)
            h_new = (1.0 - z) * n + z * h
            return h_new, h_new

        h0 = jnp.zeros((x.shape[0], self.hid), dtype=x.dtype)
        x_t = jnp.swapaxes(x, 0, 1)  # (T, N, H)
        _, h_t = jax.lax.scan(step, h0, x_t)
        h = jnp.swapaxes(h_t, 0, 1)  # (N, T, H)
        h = nn.LayerNorm()(h)
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(h)


def train_vae(Xc, rng, lat=2, steps=STEPS):
    """ELBO on the context window. Returns z (N, lat)."""
    enc, dec = EncVAE(lat=lat), Dec(out=Xc.shape[1])
    rng, ke, kd = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "d": dec.init(kd, jnp.zeros((1, lat)))}
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xj = jnp.asarray(Xc)
    n = len(Xc)

    def loss_fn(p, x, rng, beta):
        mu, lv = enc.apply(p["e"], x)
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(rng, mu.shape)
        rec = jnp.mean(jnp.sum((dec.apply(p["d"], z) - x) ** 2, -1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1))
        return rec + beta * kl

    @jax.jit
    def upd(params, opt, idx, rng, beta):
        g = jax.grad(loss_fn)(params, Xj[idx], rng, beta)
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for s in range(steps):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx, sk, float(min(1.0, s / 1500)))
    return np.asarray(enc.apply(params["e"], Xj)[0])


def train_jepa(Xc, Xt, rng, lat=2, steps=STEPS):
    """Predict the EMA-target representation of the target window from the
    context window. L2 prediction loss; collapse is prevented by the LayerNorm
    encoders + EMA target (no VICReg term). Returns z (N, lat)."""
    enc, pred = Enc(lat=lat), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt)
    n = len(Xc)

    def loss_fn(p, tgt, xc, xt):
        z = enc.apply(p["e"], xc)
        pz = pred.apply(p["p"], z)
        t = jax.lax.stop_gradient(enc.apply(tgt, xt))
        return jnp.mean((pz - t) ** 2)

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx])
        u, opt = tx.update(g, opt)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(lambda a, b: EMA * a + (1 - EMA) * b,
                                        target, params["e"])
        return params, target, opt

    for s in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx)
    return np.asarray(enc.apply(params["e"], Xcj))


def train_jepa_gru(X, lengths, rng, lat=2, hid=HID, steps=STEPS, tmin=3):
    """Variable-length JEPA with a GRU encoder.

    ``X`` is a zero-padded batch of sequences ``(N, T, F)`` and ``lengths``
    gives each episode's true number of valid steps (``1..T``). Every training
    step samples a per-episode context length ``t in [tmin, len)``; the
    predictor must map the online encoding of the first ``t`` steps to the
    EMA-target encoding of the full episode (all ``len`` steps), so it has to
    anticipate the representation of the not-yet-observed future. L2 loss, no
    VICReg; LayerNorm inside ``GRUEnc`` guards against collapse.

    Returns per-prefix latents ``z`` of shape ``(N, T, lat)`` from the trained
    online encoder; ``z[:, k - 1]`` encodes the first ``k`` steps, and
    ``z[arange(N), lengths - 1]`` is the full-episode embedding.
    """
    enc, pred = GRUEnc(lat=lat, hid=hid), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, X[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xj = jnp.asarray(X)
    lens = jnp.clip(jnp.asarray(lengths, jnp.int32), tmin + 1, X.shape[1])
    n = len(X)

    def loss_fn(p, tgt, x, t_ctx, t_full):
        zs = enc.apply(p["e"], x)
        z_ctx = zs[jnp.arange(len(x)), t_ctx - 1]
        pz = pred.apply(p["p"], z_ctx)
        zs_t = jax.lax.stop_gradient(enc.apply(tgt, x))
        z_full = zs_t[jnp.arange(len(x)), t_full - 1]
        return jnp.mean((pz - z_full) ** 2)

    @jax.jit
    def upd(params, target, opt, idx, tk):
        t_full = lens[idx]
        t_ctx = jax.random.randint(tk, t_full.shape, tmin, t_full)
        g = jax.grad(loss_fn)(params, target, Xj[idx], t_ctx, t_full)
        u, opt = tx.update(g, opt)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(lambda a, b: EMA * a + (1 - EMA) * b,
                                        target, params["e"])
        return params, target, opt

    for s in range(steps):
        rng, bk, tk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx, tk)
    return np.asarray(enc.apply(params["e"], Xj))


def supervised_contrastive_loss(z, labels, temperature=0.20):
    """Identity-grounded contrastive loss for an episode-level latent.

    The labels identify positive pairs only. They are never passed to the
    encoder as inputs and no classifier head is optimized. Every batch needs
    at least two examples of each identity, which the objective-typed dataset
    satisfies by sampling across many rollout episodes.
    """

    z = unit_normalize(z)
    logits = z @ z.T / temperature
    n = logits.shape[0]
    eye = jnp.eye(n, dtype=bool)
    same = labels[:, None] == labels[None, :]
    positives = same & (~eye)
    log_prob = jax.nn.log_softmax(jnp.where(eye, -1e9, logits), axis=1)
    per_row = -jnp.sum(positives * log_prob, axis=1) / jnp.maximum(
        positives.sum(axis=1), 1
    )
    return jnp.mean(per_row)


def unit_normalize(z):
    """Map trajectory embeddings to the unit sphere for stable JEPA training."""

    return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)


def train_identity_jepa_with_params(
    Xc,
    Xt,
    identity,
    rng,
    lat=2,
    steps=STEPS,
    contrastive_weight=1.0,
):
    """Train a stable JEPA encoder with an identity-grounded contrastive term.

    JEPA still predicts the target-window representation. The additional
    contrastive objective makes behavior type an explicit representative target
    through positive/negative episode pairs. Returns the context latent and
    the encoder parameters needed for rollout-time inference.
    """

    enc, pred = Enc(lat=lat), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    # A clipped, mildly regularized optimizer keeps the representative space
    # stable when a held-out checkpoint activates an otherwise rare feature.
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3, weight_decay=1e-5),
    )
    opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt)
    yj = jnp.asarray(identity, dtype=jnp.int32)
    n = len(Xc)

    def loss_fn(p, tgt, xc, xt, label):
        z = unit_normalize(enc.apply(p["e"], xc))
        pz = unit_normalize(pred.apply(p["p"], z))
        target_z = jax.lax.stop_gradient(unit_normalize(enc.apply(tgt, xt)))
        # Unit-normalized cosine prediction prevents norm growth from carrying
        # the predictive objective.  Strategy identity is represented by angle.
        jepa = jnp.mean(1.0 - jnp.sum(pz * target_z, axis=-1))
        contrastive = supervised_contrastive_loss(z, label)
        return jepa + contrastive_weight * contrastive

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx], yj[idx])
        u, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(
            lambda a, b: EMA * a + (1 - EMA) * b, target, params["e"]
        )
        return params, target, opt

    for _ in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx)
    return np.asarray(unit_normalize(enc.apply(params["e"], Xcj))), params["e"]


def train_identity_jepa(Xc, Xt, identity, rng, lat=2, steps=STEPS, contrastive_weight=1.0):
    """Return the ID-JEPA latent when rollout-time encoder parameters are not needed."""

    z, _ = train_identity_jepa_with_params(
        Xc, Xt, identity, rng, lat=lat, steps=steps,
        contrastive_weight=contrastive_weight,
    )
    return z


def encode_identity_jepa(params, X):
    """Run the trained ID-JEPA encoder on standardized context features."""

    z = Enc(lat=params["params"]["Dense_2"]["kernel"].shape[-1]).apply(params, jnp.asarray(X))
    return np.asarray(unit_normalize(z))


def evaluate_encoders(Xc, Xt, y, n_classes, seeds=(0, 1, 2), lat=2,
                      steps=STEPS, keep_z_seed=0):
    """Train VAE and JEPA over several encoder seeds; return mean/std metrics
    and one representative latent per encoder (from keep_z_seed) for scatter
    plots. Xc = standardized context windows, Xt = standardized target windows."""
    out = {"vae": {"probe": [], "ari": []}, "jepa": {"probe": [], "ari": []}}
    z_keep = {}
    for s in seeds:
        zv = train_vae(Xc, jax.random.PRNGKey(s), lat=lat, steps=steps)
        zj = train_jepa(Xc, Xt, jax.random.PRNGKey(s), lat=lat, steps=steps)
        for name, z in (("vae", zv), ("jepa", zj)):
            p, a = metrics(z, y, n_classes)
            out[name]["probe"].append(p)
            out[name]["ari"].append(a)
            if s == keep_z_seed:
                z_keep[name] = z
    summary = {}
    for name in out:
        for m in ("probe", "ari"):
            v = np.asarray(out[name][m])
            summary[f"{name}_{m}_mean"] = float(v.mean())
            summary[f"{name}_{m}_std"] = float(v.std())
            summary[f"{name}_{m}_all"] = v
    return summary, z_keep
