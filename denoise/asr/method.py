"""Artifact Subspace Reconstruction with cutoff 5."""

from __future__ import annotations

import numpy as np

from utils.progress import progress


class ASRDenoiser:
    name = "asr"
    cutoff = 5

    @staticmethod
    def _functions():
        try:
            from asrpy import asr_calibrate, asr_process
        except ImportError as exc:
            raise RuntimeError("ASR needs asrpy; install requirements.txt") from exc
        return asr_calibrate, asr_process

    def transform_recording(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        asr_calibrate, asr_process = self._functions()
        matrix, threshold = asr_calibrate(signal, sampling_rate, cutoff=self.cutoff)
        return np.asarray(asr_process(signal, sampling_rate, matrix, threshold))

    def transform(
        self,
        signals: np.ndarray,
        sampling_rate: float,
        task_name: str | None = None,
        **_: object,
    ) -> np.ndarray:
        # BCI epochs are filtered to 1--40 Hz before ASR.
        if task_name == "bci_errp":
            from denoise.bandpass.method import butter_bandpass_filter

            signals = butter_bandpass_filter(signals, sampling_rate, highcut=40.0)
        output = np.empty_like(signals, dtype=np.float64)
        epochs = progress(
            enumerate(signals),
            total=len(signals),
            desc=f"ASR {task_name or 'EEG'}",
            unit="epoch",
            leave=False,
        )
        for index, epoch in epochs:
            output[index] = self.transform_recording(epoch, sampling_rate)
        return output


DENOISER = ASRDenoiser()
