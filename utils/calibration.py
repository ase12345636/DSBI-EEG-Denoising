"""Deterministic, label-free calibration windows for adaptive denoisers."""

from __future__ import annotations

import numpy as np


def calibration_slices(
    sample_count: int,
    sampling_rate: float,
    *,
    max_seconds: float = 300.0,
    block_count: int = 10,
) -> list[tuple[int, int]]:
    """Use all of a short input or uniform blocks from a long recording."""
    sample_count = int(sample_count)
    budget = int(round(max_seconds * sampling_rate))
    if sample_count <= budget:
        return [(0, sample_count)]

    block_samples = budget // block_count
    starts = np.linspace(
        0,
        sample_count - block_samples,
        block_count,
        dtype=np.int64,
    )
    return [(int(start), int(start + block_samples)) for start in starts]


def calibration_array(signal: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([signal[:, start:end] for start, end in slices], axis=-1)
