"""Butterworth filter extracted from BCI total.ipynb (order=5, SOS, filtfilt)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass_filter(
    raw_data: np.ndarray,
    sampling_rate: float,
    lowcut: float = 1.0,
    highcut: float = 50.0,
    order: int = 5,
) -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    if highcut >= nyquist:
        highcut = np.nextafter(nyquist, 0.0)
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, raw_data, axis=-1)


class BandpassDenoiser:
    name = "bandpass"

    def transform(self, signals: np.ndarray, sampling_rate: float, **_: object) -> np.ndarray:
        return butter_bandpass_filter(np.asarray(signals, dtype=np.float64), sampling_rate)


DENOISER = BandpassDenoiser()
