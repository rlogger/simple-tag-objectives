"""Part 1 representation metrics: linear probe, supervised oracle, GMM ARI."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score


def probe_acc(z, y, cv=5):
    """Linear probe: tests how usable the information in the latent space is."""
    return float(
        cross_val_score(LogisticRegression(max_iter=1000), z, y, cv=cv).mean()
    )


def oracle_acc(X, y, cv=5, hidden=128, seed=0):
    """Supervised MLP baseline predicting ground-truth labels from trajectories."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier((hidden,), max_iter=800, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def survival_time_probe_acc(survival_time, y, cv=5):
    """Shortcut control: linear probe on episode length / survival time alone."""
    x = np.asarray(survival_time, dtype=np.float32).reshape(-1, 1)
    return probe_acc(x, y, cv=cv)


def metrics(z, y, n_classes):
    """Return ``(probe_accuracy, GMM_ARI)`` on centered latents.

    Prefer :func:`train_only_metrics` for published numbers (avoids full-pool
    centering leakage).
    """
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    probe = probe_acc(zc, y)
    gm = GaussianMixture(n_classes, n_init=8, random_state=0).fit(zc)
    return probe, float(adjusted_rand_score(y, gm.predict(zc)))


def train_only_standardize(
    z_train: np.ndarray,
    z_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center/scale with train statistics only."""
    mu = z_train.mean(0)
    sd = z_train.std(0) + 1e-6
    return ((z_train - mu) / sd).astype(np.float32), (
        (z_query - mu) / sd
    ).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def train_only_metrics(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray,
    y_query: np.ndarray,
    n_classes: int,
    seed: int = 0,
) -> dict[str, float]:
    """Fit probe + GMM on train latents; score held-out query."""
    zt, zq, _, _ = train_only_standardize(z_train, z_query)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(zt, y_train)
    probe = float(accuracy_score(y_query, clf.predict(zq)))
    gm = GaussianMixture(n_classes, n_init=8, random_state=seed).fit(zt)
    ari = float(adjusted_rand_score(y_query, gm.predict(zq)))
    return {"probe": probe, "ari": ari}


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE for multiclass soft predictions ``probs`` of shape ``(N, K)``."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1]
        )
        if not np.any(m):
            continue
        ece += float(m.mean()) * abs(
            float(correct[m].mean()) - float(conf[m].mean())
        )
    return float(ece)
