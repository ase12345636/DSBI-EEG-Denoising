"""Artifact Subspace Reconstruction used by the report (cutoff k=5)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from utils.calibration import calibration_array, calibration_slices


class ASRDenoiser:
    name = "asr"
    cutoff = 5

    def __init__(self) -> None:
        self.cutoff = 5
        self.max_calibration_seconds = 300.0
        self.calibration_block_count = 10

    def configure(self, settings: dict) -> None:
        self.cutoff = int(settings.get("cutoff", 5))
        self.max_calibration_seconds = float(settings.get("max_seconds", 300.0))
        self.calibration_block_count = int(settings.get("block_count", 10))

    @staticmethod
    def _asr_functions():
        try:
            from asrpy import ASR, asr_calibrate, clean_windows
        except ImportError as exc:
            raise RuntimeError("ASR requires asrpy") from exc
        return ASR, asr_calibrate, clean_windows

    @staticmethod
    def _highpass(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        sos = butter(4, 1.0 / (sampling_rate / 2.0), btype="highpass", output="sos")
        return sosfiltfilt(sos, signal, axis=-1)

    def transform_recording(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Fit on a high-pass copy and transfer only the aligned ASR correction."""
        import mne

        ASR, asr_calibrate, clean_windows = self._asr_functions()
        signal = np.asarray(signal, dtype=np.float64)
        prepared = self._highpass(signal, sampling_rate)
        slices = calibration_slices(
            prepared.shape[-1],
            sampling_rate,
            max_seconds=self.max_calibration_seconds,
            block_count=self.calibration_block_count,
        )
        calibration = calibration_array(prepared, slices)
        info = mne.create_info(
            [f"EEG{index:03d}" for index in range(signal.shape[0])],
            sampling_rate,
            ch_types="eeg",
        )
        asr = ASR(sampling_rate, cutoff=self.cutoff)
        clean, _ = clean_windows(
            calibration,
            sfreq=sampling_rate,
            win_len=asr.win_len,
            win_overlap=asr.win_overlap,
            max_bad_chans=asr.max_bad_chans,
            min_clean_fraction=asr.min_clean_fraction,
            max_dropout_fraction=asr.max_dropout_fraction,
        )
        # asrpy 0.0.7 block_covariance has an off-by-one error for this
        # exact remainder. Dropping one calibration sample avoids that bug.
        if (clean.shape[-1] - 2) % asr.blocksize == 0:
            clean = clean[:, :-1]
        asr.M, asr.T = asr_calibrate(
            clean,
            sfreq=sampling_rate,
            cutoff=asr.cutoff,
            blocksize=asr.blocksize,
            win_len=asr.win_len,
            win_overlap=asr.win_overlap,
            max_dropout_fraction=asr.max_dropout_fraction,
            min_clean_fraction=asr.min_clean_fraction,
            ab=(asr.A, asr.B),
            method=asr.method,
        )
        asr._fitted = True
        processing_raw = mne.io.RawArray(prepared, info, verbose=False)
        cleaned = asr.transform(processing_raw).get_data()

        # The 1-Hz fitting/processing copy supplies ASR with valid zero-mean
        # input, while transferring only its reconstruction avoids adding a
        # band-pass advantage over the other denoising methods.
        return signal + (cleaned - prepared)


DENOISER = ASRDenoiser()
