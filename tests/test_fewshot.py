import numpy as np

from src.fewshot import sample_k_shot, evaluate_k_shot


def _blobs(n_per_class=60, dim=15, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    centers = rng.normal(scale=7.0, size=(n_classes, dim))
    for c in range(n_classes):
        feats.append(centers[c] + rng.normal(scale=1.0, size=(n_per_class, dim)))
        labels.append(np.full(n_per_class, c))
    x = np.concatenate(feats)
    y = np.concatenate(labels)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_sample_k_shot_uses_exactly_k_per_class():
    _, y = _blobs(n_per_class=30, n_classes=5, seed=2)
    k = 4
    idx = sample_k_shot(y, k, random_state=1)
    classes, counts = np.unique(y[idx], return_counts=True)
    assert len(classes) == 5
    assert np.all(counts == k)
    assert len(idx) == k * 5


def test_sample_k_shot_indices_unique():
    _, y = _blobs(seed=3)
    idx = sample_k_shot(y, 5, random_state=0)
    assert len(np.unique(idx)) == len(idx)


def test_sample_k_shot_raises_when_class_too_small():
    y = np.array([0, 0, 1, 1, 1])
    try:
        sample_k_shot(y, 3)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_evaluate_k_shot_support_size_and_accuracy():
    x, y = _blobs(seed=7)
    k = 5
    out = evaluate_k_shot(x, y, k, random_state=0)
    # Support is exactly k per class.
    _, counts = np.unique(y[out["support_indices"]], return_counts=True)
    assert np.all(counts == k)
    assert out["k"] == k
    # Well separated blobs are easy even from few shots.
    assert out["accuracy"] >= 0.9


def test_evaluate_k_shot_with_explicit_test_set():
    x, y = _blobs(seed=11)
    split = len(y) // 2
    out = evaluate_k_shot(
        x[:split], y[:split], k=5,
        test_features=x[split:], test_labels=y[split:],
        random_state=0,
    )
    assert out["accuracy"] >= 0.9
    assert len(out["support_indices"]) == 5 * len(np.unique(y))
