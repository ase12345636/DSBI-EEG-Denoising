"""Raw EEG: intentionally no denoising."""

from __future__ import annotations

import numpy as np


class RawDenoiser:
    name = "raw"

    def transform(self, signals: np.ndarray, sampling_rate: float, **_: object) -> np.ndarray:
        return np.asarray(signals)


DENOISER = RawDenoiser()
