"""Part 1 representation metrics: linear probe, supervised oracle, GMM ARI."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
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


def metrics(z, y, n_classes):
    """Return ``(probe_accuracy, GMM_ARI)`` on centered latents."""
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    probe = probe_acc(zc, y)
    gm = GaussianMixture(n_classes, n_init=8, random_state=0).fit(zc)
    return probe, float(adjusted_rand_score(y, gm.predict(zc)))


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE for multiclass soft predictions ``probs`` of shape ``(N, K)``."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)
