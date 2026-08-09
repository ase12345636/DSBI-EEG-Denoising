"""Artifact Subspace Reconstruction used by the report (cutoff k=5)."""

from __future__ import annotations

import numpy as np

from utils.progress import progress


class ASRDenoiser:
    name = "asr"
    cutoff = 5

    @staticmethod
    def _asr_functions():
        try:
            from asrpy import asr_calibrate, asr_process
        except ImportError as exc:
            raise RuntimeError("ASR requires asrpy") from exc
        return asr_calibrate, asr_process

    def transform_recording(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Calibrate and process one recording/epoch exactly as in the source code."""
        calibrate, process = self._asr_functions()
        signal = np.asarray(signal, dtype=np.float64)
        matrix, threshold = calibrate(signal, sampling_rate, cutoff=self.cutoff)
        return np.asarray(process(signal, sampling_rate, matrix, threshold))

    def transform(self, signals: np.ndarray, sampling_rate: float, task_name=None, **_) -> np.ndarray:
        """Apply ASR independently to each supplied epoch."""
        signals = np.asarray(signals, dtype=np.float64)
        output = np.empty_like(signals)
        for index, epoch in progress(
            enumerate(signals),
            total=len(signals),
            desc=f"ASR {task_name or 'EEG'}",
            unit="epoch",
            leave=False,
        ):
            output[index] = self.transform_recording(epoch, sampling_rate)
        return output


DENOISER = ASRDenoiser()
