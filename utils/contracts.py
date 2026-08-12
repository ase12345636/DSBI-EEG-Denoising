"""Small data contracts shared by tasks, denoisers, and classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalDataset:
    signals: np.ndarray
    labels: np.ndarray
    sampling_rate: float
    class_names: tuple[str, ...]
    primary_metric: str
    fixed_train: np.ndarray | None = None
    groups: np.ndarray | None = None
    sample_ids: np.ndarray | None = None

    def validate(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"signals must have shape (epochs, channels, samples), got {self.signals.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be one-dimensional, got {self.labels.shape}")
        if not len(self.signals):
            raise ValueError("dataset contains no epochs")
        if len(self.signals) != len(self.labels):
            raise ValueError("signals and labels have different lengths")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if len(self.class_names) < 2:
            raise ValueError("at least two class names are required")
        labels = np.unique(self.labels)
        if labels[0] < 0 or labels[-1] >= len(self.class_names):
            raise ValueError("labels must be zero-based indices into class_names")
        if self.fixed_train is not None and len(self.fixed_train) != len(self.labels):
            raise ValueError("fixed_train mask has the wrong length")
        if self.groups is not None and len(self.groups) != len(self.labels):
            raise ValueError("groups and labels have different lengths")


@dataclass
class Prediction:
    labels: np.ndarray
    scores: np.ndarray
