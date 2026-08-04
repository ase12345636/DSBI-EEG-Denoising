"""Feature extraction used by the report's classical classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.progress import progress


STFT_CACHE_VERSION = "stft-v2-explicit"


def stft_mean_power(
    signals: np.ndarray,
    sampling_rate: float,
    config: dict[str, Any] | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-channel mean STFT power with explicit stable parameters.

    The output shape is ``(epochs, channels * frequency_bins)``.  Power is
    averaged across STFT time frames, leaving one feature per frequency bin and
    channel.  Parameters are explicit so SciPy-version defaults cannot silently
    change the experiment.
    """
    from scipy.signal import stft

    settings = dict(config or {})
    requested_nperseg = int(settings.get("nperseg", 256))
    overlap_fraction = float(settings.get("overlap_fraction", 0.5))
    window = settings.get("window", "hann")
    boundary = settings.get("boundary", "zeros")
    padded = bool(settings.get("padded", True))
    use_power = bool(settings.get("power", True))

    signals = np.asarray(signals)
    if signals.ndim != 3:
        raise ValueError(
            "STFT input must have shape (epochs, channels, samples), "
            f"got {signals.shape}"
        )
    if not np.isfinite(signals).all():
        raise ValueError("STFT input contains NaN or infinity")

    nperseg = min(requested_nperseg, int(signals.shape[-1]))
    if nperseg < 2:
        raise ValueError("STFT needs at least two samples per epoch")
    noverlap = min(int(round(nperseg * overlap_fraction)), nperseg - 1)

    batches: list[np.ndarray] = []
    starts = range(0, len(signals), batch_size)
    for start in progress(
        starts,
        total=(len(signals) + batch_size - 1) // batch_size,
        desc="STFT features",
        unit="batch",
        leave=False,
    ):
        batch = np.asarray(signals[start : start + batch_size], dtype=np.float64)
        _, _, spectrum = stft(
            batch,
            fs=float(sampling_rate),
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=boundary,
            padded=padded,
            axis=-1,
        )
        magnitude = np.abs(spectrum)
        values = magnitude**2 if use_power else magnitude
        mean_values = np.mean(values, axis=-1)
        batches.append(
            mean_values.reshape(len(batch), -1).astype(np.float32, copy=False)
        )
    return np.concatenate(batches, axis=0)


def bci_xdawn_tangent(
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit xDAWN and tangent space on training data; transform test data.

    This is the leakage-free path recovered from the author's
    ``ML-classifier.ipynb``.
    """

    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace

    xdawn = XdawnCovariances(nfilter=5)
    train_cov = xdawn.fit_transform(train, y_train)
    tangent = TangentSpace(metric="riemann")
    x_train = tangent.fit_transform(train_cov)
    test_cov = xdawn.transform(test)
    x_test = tangent.transform(test_cov)
    return np.asarray(x_train), np.asarray(x_test)
