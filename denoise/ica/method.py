"""Automatic ICA denoising for the benchmark.

ICA is fitted on continuous EEG because a 0.7-s/1-s epoch is not a meaningful
ICA calibration unit. The fit uses the same explicit label-free calibration
windows as ASR, plus Extended Infomax, common-average reference, a 1-Hz
high-pass fitting copy, and ICLabel. No task labels are used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from utils.calibration import calibration_slices


KEEP_LABELS = {"brain", "other"}
ICLABEL_CLASSES = (
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
)
ALIASES = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz", "FZ": "Fz",
    "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}


class ICADenoiser:
    name = "ica"

    def __init__(self) -> None:
        self.random_state = 97
        self.max_calibration_seconds = 300.0
        self.calibration_block_count = 10
        self._reports: list[dict] = []

    def configure(self, settings: dict) -> None:
        self.random_state = int(settings.get("random_state", 97))
        self.max_calibration_seconds = float(settings.get("max_seconds", 300.0))
        self.calibration_block_count = int(settings.get("block_count", 10))
        self._reports = []

    @staticmethod
    def _imports():
        try:
            import mne
            from mne.preprocessing import ICA
            from mne_icalabel import label_components
            from mne_icalabel.iclabel import iclabel_label_components
        except ImportError as exc:
            raise RuntimeError("ICA requires mne and mne-icalabel") from exc
        return mne, ICA, label_components, iclabel_label_components

    @staticmethod
    def normalise_channel_name(name: str) -> str:
        value = str(name).strip()
        value = re.sub(r"^EEG\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"-(REF|LE|RE)$", "", value, flags=re.IGNORECASE)
        value = value.replace(" ", "")
        return ALIASES.get(value.upper(), value)

    @classmethod
    def normalise_channel_names(cls, names: Sequence[str]) -> list[str]:
        return [cls.normalise_channel_name(name) for name in names]

    def _fit_input(self, filtered_raw):
        """Use the calibration ranges shared with ASR."""
        mne, _, _, _ = self._imports()
        sfreq = float(filtered_raw.info["sfreq"])
        n_times = int(filtered_raw.n_times)
        slices = calibration_slices(
            n_times,
            sfreq,
            max_seconds=self.max_calibration_seconds,
            block_count=self.calibration_block_count,
        )
        if slices == [(0, n_times)]:
            return filtered_raw, n_times / sfreq

        data = filtered_raw.get_data(picks="eeg")
        blocks = np.stack([data[:, start:end] for start, end in slices])
        epochs = mne.EpochsArray(
            blocks, filtered_raw.info.copy(), tmin=0.0, baseline=None, verbose=False
        )
        return epochs, blocks.shape[0] * blocks.shape[2] / sfreq

    def transform_recording(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        *,
        channel_names: Sequence[str],
        unit_scale_to_volts: float,
        task_name: str | None = None,
        recording_id: str | None = None,
    ) -> np.ndarray:
        mne, ICA, label_components, full_probabilities = self._imports()
        values = np.asarray(signal, dtype=np.float64)
        names = self.normalise_channel_names(channel_names)
        if values.ndim != 2 or values.shape[0] != len(names):
            raise ValueError("ICA expects (channels, samples) matching channel_names")

        info = mne.create_info(names, float(sampling_rate), ch_types="eeg")
        raw = mne.io.RawArray(values * float(unit_scale_to_volts), info, verbose=False)
        raw.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            match_case=False,
            on_missing="raise",
        )
        raw.set_eeg_reference("average", projection=False, verbose=False)

        fit_raw = raw.copy()
        nyquist = sampling_rate / 2.0
        fit_raw.filter(
            l_freq=1.0,
            h_freq=100.0 if nyquist > 100.0 else None,
            picks="eeg",
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False,
        )
        fit_input, fit_seconds = self._fit_input(fit_raw)

        rank = int(mne.compute_rank(fit_input, rank=None, verbose=False)["eeg"])
        n_components = min(len(names) - 1, rank)
        if n_components < 2:
            raise ValueError("ICA input has insufficient EEG rank")

        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params={"extended": True},
            random_state=self.random_state,
            max_iter="auto",
        )
        ica.fit(fit_input, picks="eeg", reject_by_annotation=True, verbose=False)

        result = label_components(fit_input, ica, method="iclabel")
        labels = [str(label) for label in result["labels"]]
        exclude = [i for i, label in enumerate(labels) if label not in KEEP_LABELS]

        # This case occurred in CHB-MIT during preflight. Removing every IC would
        # collapse the signal, so retain the component with the largest combined
        # brain+other probability and record that the fallback was used.
        fallback = None
        if len(exclude) == len(labels):
            probs = np.asarray(full_probabilities(fit_input, ica, inplace=True), dtype=float)
            keep_score = probs[:, [0, 6]].sum(axis=1)
            fallback = int(np.argmax(keep_score))
            exclude.remove(fallback)

        cleaned = raw.copy()
        ica.apply(cleaned, exclude=exclude, verbose=False)

        # Keep ICA as the only intervention. Common-average reference is needed
        # for ICA/ICLabel fitting, but it must not become an extra preprocessing
        # advantage/disadvantage versus Raw/Band-pass/ASR/IC-U-Net. Apply only
        # the ICA-derived correction to the original task-native signal, so
        # exclude=[] is exactly an identity transform (up to float precision).
        scale = float(unit_scale_to_volts)
        correction = (cleaned.get_data() - raw.get_data()) / scale
        output = values + correction

        self._reports.append({
            "task": task_name,
            "recording": recording_id,
            "fit_seconds": float(fit_seconds),
            "n_components": int(ica.n_components_),
            "labels": labels,
            "excluded_components": exclude,
            "all_artifact_fallback_component": fallback,
        })
        return output.astype(np.float32)

    def save_reports(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._reports, indent=2, ensure_ascii=False), encoding="utf-8")


DENOISER = ICADenoiser()
