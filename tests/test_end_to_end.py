import numpy as np
import torch
from torch import nn

from src.features import extract_features
from src.probe import evaluate_probe
from src.fewshot import evaluate_k_shot


def test_frozen_encoder_full_pipeline():
    """Freeze an encoder, read features, probe full and k shot."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Build inputs whose class structure survives a frozen random encoder:
    # widely separated input means stay separated after a linear map.
    n_classes, n_per = 3, 60
    in_dim = 12
    centers = rng.normal(scale=10.0, size=(n_classes, in_dim))
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(centers[c] + rng.normal(scale=0.5, size=(n_per, in_dim)))
        ys.append(np.full(n_per, c))
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    perm = rng.permutation(len(y_all))
    x = torch.tensor(x_all[perm], dtype=torch.float32)
    y = y_all[perm]

    encoder = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU())
    feats, labels = extract_features(encoder, x, y)
    assert feats.shape == (n_classes * n_per, 16)

    split = len(labels) // 2
    full = evaluate_probe(feats[:split], labels[:split], feats[split:], labels[split:])
    assert full["accuracy"] >= 0.9

    shot = evaluate_k_shot(feats, labels, k=5, random_state=0)
    _, counts = np.unique(labels[shot["support_indices"]], return_counts=True)
    assert np.all(counts == 5)
    assert shot["accuracy"] >= 0.8
