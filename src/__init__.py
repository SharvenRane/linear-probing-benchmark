"""Linear probing and few shot evaluation suite for frozen encoders."""

from .features import extract_features
from .probe import LinearProbe, evaluate_probe
from .fewshot import sample_k_shot, evaluate_k_shot

__all__ = [
    "extract_features",
    "LinearProbe",
    "evaluate_probe",
    "sample_k_shot",
    "evaluate_k_shot",
]
