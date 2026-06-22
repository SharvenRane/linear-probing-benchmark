"""Few shot evaluation: fit a probe using exactly k examples per class.

The k shot path draws a balanced support set of k samples for every
class, fits a linear probe on just those samples, and scores on the rest.
This mirrors how a frozen encoder is judged in a low label budget regime.
"""

from __future__ import annotations

import numpy as np

from .probe import LinearProbe


def sample_k_shot(
    labels: np.ndarray,
    k: int,
    random_state: int = 0,
) -> np.ndarray:
    """Pick exactly k indices for each class.

    Parameters
    ----------
    labels
        Integer label array of shape (n,).
    k
        Number of examples to draw per class.
    random_state
        Seed for reproducible sampling.

    Returns
    -------
    support_indices
        A 1D int array of length k times the number of classes. Each
        class contributes exactly k indices. A class with fewer than k
        examples raises a ValueError.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    classes = np.unique(labels)

    chosen: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.shape[0] < k:
            raise ValueError(
                f"class {cls} has {cls_idx.shape[0]} examples, fewer than k={k}"
            )
        picked = rng.choice(cls_idx, size=k, replace=False)
        chosen.append(picked)

    support_indices = np.concatenate(chosen)
    return support_indices


def evaluate_k_shot(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    test_features: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    C: float = 1.0,
    random_state: int = 0,
) -> dict:
    """Run a k shot probe.

    A balanced support set of k samples per class is drawn from
    ``features``/``labels`` and a linear probe is fit on it. The probe is
    then scored. When ``test_features`` is given the score is computed on
    that held out set; otherwise it is computed on the samples that were
    not selected as support.

    Returns a dict with the accuracy, the support indices that were used,
    and the value of k.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)

    support_idx = sample_k_shot(labels, k, random_state=random_state)
    support_features = features[support_idx]
    support_labels = labels[support_idx]

    probe = LinearProbe(C=C, random_state=random_state)
    probe.fit(support_features, support_labels)

    if test_features is not None and test_labels is not None:
        accuracy = probe.score(test_features, np.asarray(test_labels))
    else:
        mask = np.ones(features.shape[0], dtype=bool)
        mask[support_idx] = False
        accuracy = probe.score(features[mask], labels[mask])

    return {
        "accuracy": accuracy,
        "support_indices": support_idx,
        "k": k,
        "probe": probe,
    }
