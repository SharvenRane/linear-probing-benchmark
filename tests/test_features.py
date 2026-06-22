import numpy as np
import torch
from torch import nn

from src.features import extract_features, freeze_encoder


def test_freeze_disables_grad_and_eval():
    enc = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    enc.train()
    freeze_encoder(enc)
    assert not enc.training
    assert all(not p.requires_grad for p in enc.parameters())


def test_extract_features_shapes_flat_encoder():
    torch.manual_seed(0)
    enc = nn.Linear(6, 5)
    x = torch.randn(20, 6)
    y = np.arange(20) % 3
    feats, labels = extract_features(enc, x, y, batch_size=7)
    assert feats.shape == (20, 5)
    assert feats.dtype == np.float32
    assert labels.shape == (20,)
    assert np.array_equal(labels, y)


def test_extract_features_global_pools_spatial_output():
    # A conv backbone returns (n, c, h, w); features must collapse to (n, c).
    enc = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    x = torch.randn(5, 3, 8, 8)
    feats, labels = extract_features(enc, x)
    assert feats.shape == (5, 4)
    assert labels is None


def test_extract_features_no_grad():
    enc = nn.Linear(4, 4)
    x = torch.randn(3, 4, requires_grad=True)
    feats, _ = extract_features(enc, x)
    # Output is detached numpy, encoder params untouched.
    assert isinstance(feats, np.ndarray)
    assert all(not p.requires_grad for p in enc.parameters())
