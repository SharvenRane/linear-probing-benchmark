import numpy as np
from sklearn.datasets import make_classification

from src.probe import LinearProbe, evaluate_probe


def _separable_blobs(n_per_class=80, dim=20, n_classes=3, seed=0):
    """Linearly separable features with well separated class means."""
    rng = np.random.default_rng(seed)
    feats = []
    labels = []
    centers = rng.normal(scale=8.0, size=(n_classes, dim))
    for c in range(n_classes):
        pts = centers[c] + rng.normal(scale=1.0, size=(n_per_class, dim))
        feats.append(pts)
        labels.append(np.full(n_per_class, c))
    x = np.concatenate(feats)
    y = np.concatenate(labels)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_probe_high_accuracy_on_separable_features():
    x, y = _separable_blobs(seed=1)
    split = len(y) // 2
    out = evaluate_probe(x[:split], y[:split], x[split:], y[split:])
    assert out["accuracy"] >= 0.95


def test_probe_beats_chance_on_make_classification():
    x, y = make_classification(
        n_samples=400,
        n_features=30,
        n_informative=10,
        n_classes=4,
        random_state=3,
    )
    split = 300
    out = evaluate_probe(x[:split], y[:split], x[split:], y[split:])
    # Four balanced classes give a chance level of 0.25.
    assert out["accuracy"] > 0.5


def test_probe_predict_requires_fit():
    probe = LinearProbe()
    try:
        probe.predict(np.zeros((2, 3)))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_probe_score_matches_manual_accuracy():
    x, y = _separable_blobs(seed=5)
    probe = LinearProbe().fit(x, y)
    preds = probe.predict(x)
    manual = float(np.mean(preds == y))
    assert abs(probe.score(x, y) - manual) < 1e-12
