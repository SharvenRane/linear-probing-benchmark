"""Feature extraction from a frozen encoder.

The encoder is run in eval mode with gradients disabled so that no
parameters move while we read out representations. Outputs are pooled to
a flat feature vector per input sample.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    """Put an encoder into a frozen state.

    Every parameter has requires_grad set to False and the module is put
    in eval mode so that batch norm and dropout stay fixed.
    """
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _flatten_features(out: torch.Tensor) -> torch.Tensor:
    """Reduce an encoder output to a 2D tensor of shape (batch, dim).

    If the output has spatial dimensions they are averaged away so that a
    convolutional backbone and a flat backbone both yield one vector per
    sample.
    """
    if out.dim() == 2:
        return out
    # Average over every axis past the channel axis (global average pool).
    reduce_axes = tuple(range(2, out.dim()))
    if reduce_axes:
        out = out.mean(dim=reduce_axes)
    return out


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    inputs: torch.Tensor | Iterable,
    labels: torch.Tensor | np.ndarray | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract frozen features for a batch of inputs.

    Parameters
    ----------
    encoder
        Any module that maps a batch of inputs to a tensor. It is frozen
        before the forward pass.
    inputs
        A tensor of shape (n, ...) or an iterable that yields such a
        tensor when stacked. When a tensor is given it is split into mini
        batches internally.
    labels
        Optional labels. When given they are returned aligned with the
        features as an integer array.
    batch_size
        Mini batch size for the forward pass.
    device
        Device to run the encoder on.

    Returns
    -------
    features
        A float32 array of shape (n, dim).
    labels_out
        An int64 array of shape (n,) or None when no labels were given.
    """
    device = torch.device(device)
    encoder = freeze_encoder(encoder).to(device)

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.as_tensor(np.asarray(inputs))
    inputs = inputs.float()

    if labels is not None:
        labels_t = torch.as_tensor(np.asarray(labels)).long()
        if labels_t.shape[0] != inputs.shape[0]:
            raise ValueError("inputs and labels must have the same length")
        dataset = TensorDataset(inputs, labels_t)
    else:
        dataset = TensorDataset(inputs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device)
        out = encoder(x)
        out = _flatten_features(out)
        feats.append(out.detach().cpu().numpy().astype(np.float32))
        if labels is not None:
            labs.append(batch[1].numpy().astype(np.int64))

    features = np.concatenate(feats, axis=0)
    labels_out = np.concatenate(labs, axis=0) if labels is not None else None
    return features, labels_out
