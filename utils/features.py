"""Feature extraction shared by the three downstream tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from utils.progress import progress


def stft_mean_power(
    signals: np.ndarray,
    sampling_rate: float,
    config: dict[str, Any] | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-channel STFT features using the explicitly configured interpretation.

    The report does not specify every STFT detail. ``mean_power_over_time``
    records the current interpretation: squared magnitude, averaged over the
    STFT time-frame axis, retaining one value per frequency bin and channel.
    """
    from scipy.signal import stft

    settings = dict(config or {})
    requested_nperseg = int(settings.get("nperseg", 256))
    overlap_fraction = float(settings.get("overlap_fraction", 0.5))
    nperseg = min(requested_nperseg, int(signals.shape[-1]))
    noverlap = min(int(round(nperseg * overlap_fraction)), nperseg - 1)

    outputs = []
    for start in progress(
        range(0, len(signals), batch_size),
        total=(len(signals) + batch_size - 1) // batch_size,
        desc="STFT features",
        unit="batch",
        leave=False,
    ):
        batch = np.asarray(signals[start : start + batch_size], dtype=np.float64)
        _, _, spectrum = stft(
            batch,
            fs=float(sampling_rate),
            window=settings.get("window", "hann"),
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=settings.get("boundary", "zeros"),
            padded=bool(settings.get("padded", True)),
            axis=-1,
        )
        mean_power = np.mean(np.abs(spectrum) ** 2, axis=-1)
        outputs.append(mean_power.reshape(len(batch), -1).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def bci_xdawn_tangent(
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ML-classifier.ipynb: XdawnCovariances(nfilter=5) then TangentSpace."""
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace

    xdawn = XdawnCovariances(nfilter=5)
    tangent = TangentSpace(metric="riemann")

    train_cov = xdawn.fit_transform(np.asarray(train), np.asarray(y_train))
    test_cov = xdawn.transform(np.asarray(test))
    return tangent.fit_transform(train_cov), tangent.transform(test_cov)
