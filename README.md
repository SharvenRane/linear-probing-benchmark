# linear-probing-benchmark

A small suite for evaluating a frozen encoder by linear probing and by few shot probing. The idea is simple. You take a model, freeze its weights, read out the features it produces, and fit a plain linear classifier on top. If the features are good, a linear classifier already separates the classes well. This is the standard way to judge a representation without fine tuning the encoder.

## What is in here

The code lives under `src/` and breaks into three pieces.

`features.py` freezes an encoder and extracts features. Freezing puts every parameter into a no gradient state and switches the module to eval mode so that batch norm and dropout stay still. The forward pass runs under `torch.no_grad`. When the encoder returns a spatial map the output is collapsed with a global average pool, so a convolutional backbone and a flat backbone both hand back one vector per sample.

`probe.py` holds the linear probe itself. Features are standardized to zero mean and unit variance, then a logistic regression is fit. The fitted scaler is kept and reused at predict time so train and test go through the same transform. `evaluate_probe` is the one call convenience: fit on a train split, score on a held out split.

`fewshot.py` is the low label budget path. `sample_k_shot` draws a balanced support set of exactly k examples for every class. `evaluate_k_shot` fits a probe on that support set and scores it, either against an explicit test set you pass in or against the samples that were not chosen as support.

## Quick use

```python
import torch
from torch import nn
from src.features import extract_features
from src.probe import evaluate_probe
from src.fewshot import evaluate_k_shot

encoder = nn.Sequential(nn.Linear(12, 16), nn.ReLU())

# inputs is a tensor of shape (n, ...), labels is an int array of shape (n,)
feats, labels = extract_features(encoder, inputs, labels)

# full linear probe on a train/test split
split = len(labels) // 2
result = evaluate_probe(feats[:split], labels[:split], feats[split:], labels[split:])
print(result["accuracy"])

# k shot probe using exactly 5 labelled examples per class
shot = evaluate_k_shot(feats, labels, k=5)
print(shot["accuracy"], len(shot["support_indices"]))
```

The encoder you pass can be anything that maps a batch of inputs to a tensor. It is frozen for you before the forward pass, so you do not need to call `eval` or set `requires_grad` yourself.

## Tests

The test suite uses tiny synthetic tensors and runs on CPU with no downloads. The checks are behavioural rather than fixed numbers:

* a probe reaches high accuracy on linearly separable synthetic features and beats chance on a harder `make_classification` set
* the scaler and classifier round trip so that `score` matches a manual accuracy count
* the k shot sampler returns exactly k indices per class, with no repeats, and raises when a class has fewer than k examples
* feature extraction collapses a spatial encoder output to one vector per sample and leaves the encoder frozen
* an end to end pass freezes a small encoder, reads features, and runs both the full probe and the k shot probe

Run them with:

```
python -m pytest tests/ -q
```

On the reference machine all 14 tests pass.

## Requirements

See `requirements.txt`. The core dependencies are numpy, torch, and scikit-learn, with pytest for the tests.

## A note on results

This repo does not ship benchmark numbers for any real encoder. It is the harness. Point it at a frozen model and a dataset of your own and it will report the accuracy that run actually produces.
