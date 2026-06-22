"""Linear probe on top of frozen features.

A linear probe is a plain logistic regression fit on features that were
read out from a frozen encoder. It measures how linearly separable the
representation is without touching the encoder weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearProbe:
    """A standardizing logistic regression probe.

    Features are standardized to zero mean and unit variance before the
    linear classifier is fit. The fitted scaler is reused at predict time
    so that train and test go through the same transform.
    """

    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 0
    _scaler: StandardScaler = field(default=None, repr=False)
    _clf: LogisticRegression = field(default=None, repr=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self._scaler = StandardScaler().fit(features)
        x = self._scaler.transform(features)
        self._clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(x, labels)
        return self

    def _check_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("probe must be fit before use")

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._scaler.transform(np.asarray(features, dtype=np.float64))
        return self._clf.predict(x)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict(features)
        return float(np.mean(preds == np.asarray(labels)))


def evaluate_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 0,
) -> dict:
    """Fit a linear probe on train data and score it on test data.

    Returns a dict with the held out accuracy and the fitted probe.
    """
    probe = LinearProbe(C=C, max_iter=max_iter, random_state=random_state)
    probe.fit(train_features, train_labels)
    accuracy = probe.score(test_features, test_labels)
    return {"accuracy": accuracy, "probe": probe}
