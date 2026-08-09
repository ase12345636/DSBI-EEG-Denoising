"""ICA denoising with ICLabel-guided automatic component rejection.

The generic ICA denoiser is intentionally task-agnostic. Dataset-specific
representations (for example CHB-MIT bipolar derivations) are adapted outside
this module before calling :meth:`transform_recording`.

Processing sequence
-------------------
1. create an MNE Raw object with standard scalp locations;
2. apply common-average reference;
3. create a 1--100 Hz fitting copy (bounded by Nyquist);
4. fit Extended Infomax ICA from continuous time blocks distributed across the whole recording;
5. label those same fitted components with ICLabel;
6. keep components labelled ``brain`` or ``other`` and reject artifact classes;
7. apply the fitted ICA solution to the average-referenced continuous recording.

The fitting budget replaces the old "first 300 seconds" crop. When a recording
is longer than the budget, v4 builds continuous fitting blocks distributed
across the entire recording. The same ``Epochs`` object is used for ICA fitting
and ICLabel, preserving the documented requirement that ICLabel receives the
data instance used to fit the decomposition.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np


KEEP_LABELS = frozenset({"brain", "other"})
ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)

# Older temporal labels used by some EEG datasets are mapped to modern 10-10 names.
_CHANNEL_ALIASES = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FPZ": "Fpz",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


class ICADenoiser:
    """Extended-Infomax ICA with ICLabel-guided artifact rejection."""

    name = "ica"
    cache_tag = "ica-v4"
    random_state = 97
    fit_sample_budget_seconds = 300.0

    def __init__(self) -> None:
        self.keep_labels = KEEP_LABELS
        self.component_labeler = "iclabel"
        self._reports: list[dict[str, object]] = []

    def configure(self, settings: dict[str, object]) -> None:
        """Apply fixed ICA settings from ``configs/default.json``."""
        if not settings:
            self._reports = []
            return

        decomposition = str(settings.get("decomposition", "extended_infomax")).lower()
        if decomposition != "extended_infomax":
            raise ValueError(
                "ICA denoising is fixed to Extended Infomax; "
                f"received decomposition={decomposition!r}"
            )

        fit_reference = str(settings.get("fit_reference", "common_average")).lower()
        if fit_reference != "common_average":
            raise ValueError(
                "ICLabel fitting requires common-average reference; "
                f"received fit_reference={fit_reference!r}"
            )

        fit_band = tuple(float(value) for value in settings.get("fit_band_hz", (1.0, 100.0)))
        if fit_band != (1.0, 100.0):
            raise ValueError(
                "ICA/ICLabel fitting band is fixed at [1, 100] Hz before Nyquist limiting; "
                f"received {fit_band!r}"
            )

        self.random_state = int(settings.get("random_state", self.random_state))

        # New v4 name. Fall back to the old key so older custom configs still run,
        # but the meaning is now a whole-recording sample budget, not a front crop.
        budget = settings.get(
            "fit_sample_budget_seconds",
            settings.get("max_fit_seconds", self.fit_sample_budget_seconds),
        )
        self.fit_sample_budget_seconds = float(budget)
        if self.fit_sample_budget_seconds <= 0:
            raise ValueError("fit_sample_budget_seconds must be positive")

        labeler = str(settings.get("component_labeler", self.component_labeler)).lower()
        if labeler != "iclabel":
            raise ValueError(
                'This ICA implementation currently supports only component_labeler="iclabel"'
            )
        self.component_labeler = labeler

        keep = settings.get("keep_labels")
        if keep is not None:
            selected = frozenset(str(label) for label in keep)
            unknown = selected - set(ICLABEL_CLASSES)
            if unknown:
                raise ValueError("Unknown ICLabel keep_labels: " + ", ".join(sorted(unknown)))
            if not selected:
                raise ValueError("keep_labels cannot be empty")
            self.keep_labels = selected

        self._reports = []

    @staticmethod
    def _imports():
        try:
            import mne
            from mne.preprocessing import ICA
            from mne_icalabel import label_components
            from mne_icalabel.iclabel import iclabel_label_components
        except ImportError as exc:
            raise RuntimeError(
                "ICA component labeling needs mne and mne-icalabel. "
                "Install requirements.txt before running this method."
            ) from exc
        return mne, ICA, label_components, iclabel_label_components

    @staticmethod
    def normalise_channel_name(name: str) -> str:
        """Normalise common EEG channel-name variants to standard montage names."""
        value = str(name).strip()
        value = re.sub(r"^EEG\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"-(REF|LE|RE)$", "", value, flags=re.IGNORECASE)
        value = value.replace(" ", "")
        return _CHANNEL_ALIASES.get(value.upper(), value)

    @classmethod
    def normalise_channel_names(cls, channel_names: Sequence[str]) -> list[str]:
        result = [cls.normalise_channel_name(name) for name in channel_names]
        if len(set(name.casefold() for name in result)) != len(result):
            raise ValueError(f"Channel names are not unique after normalization: {result}")
        return result

    @staticmethod
    def _fit_band(sampling_rate: float) -> tuple[float | None, float]:
        nyquist = float(sampling_rate) / 2.0
        if nyquist <= 1.0:
            raise ValueError(
                f"Sampling rate {sampling_rate} Hz is too low for 1-Hz high-pass ICA"
            )
        h_freq = 100.0 if nyquist > 100.0 else None
        return h_freq, min(100.0, nyquist)

    @staticmethod
    def _set_standard_montage(raw) -> None:
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="raise")

    def _distributed_fit_instance(self, filtered_raw):
        """Build ICA/ICLabel fitting data distributed across the recording.

        For recordings within the budget, use the complete filtered ``Raw``.
        Longer recordings are represented by ten continuous time blocks spread
        from beginning to end, with total duration approximately equal to the
        configured fitting budget. The resulting ``Epochs`` object is used by
        both ``ICA.fit`` and ICLabel.
        """
        mne, _, _, _ = self._imports()
        sfreq = float(filtered_raw.info["sfreq"])
        budget_samples = max(2, int(round(self.fit_sample_budget_seconds * sfreq)))
        n_times = int(filtered_raw.n_times)

        if n_times <= budget_samples:
            return filtered_raw, {
                "fit_sampling_strategy": "full_recording",
                "fit_samples": n_times,
                "fit_equivalent_seconds": n_times / sfreq,
                "recording_seconds": n_times / sfreq,
                "fit_blocks": 1,
            }

        block_count = 10
        block_samples = max(2, budget_samples // block_count)
        block_samples = min(block_samples, n_times)
        max_start = n_times - block_samples
        starts = np.linspace(0, max_start, num=block_count, dtype=np.int64)
        starts = np.unique(starts)
        data = filtered_raw.get_data(picks="eeg")
        blocks = np.stack(
            [data[:, int(start) : int(start) + block_samples] for start in starts]
        )
        fit_epochs = mne.EpochsArray(
            blocks,
            filtered_raw.info.copy(),
            tmin=0.0,
            baseline=None,
            verbose=False,
        )
        return fit_epochs, {
            "fit_sampling_strategy": "uniform_time_blocks",
            "fit_samples": int(blocks.shape[0] * blocks.shape[2]),
            "fit_equivalent_seconds": float(blocks.shape[0] * blocks.shape[2]) / sfreq,
            "recording_seconds": n_times / sfreq,
            "fit_blocks": int(blocks.shape[0]),
            "fit_block_seconds": float(blocks.shape[2]) / sfreq,
            "fit_block_start_seconds": (starts / sfreq).astype(float).tolist(),
        }

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
        """Clean one continuous monopolar EEG recording."""
        mne, ICA, label_components, iclabel_label_components = self._imports()

        values = np.asarray(signal, dtype=np.float64)
        scale = float(unit_scale_to_volts)
        if values.ndim != 2:
            raise ValueError(f"Expected (channels, samples), got {values.shape}")
        if values.shape[0] != len(channel_names):
            raise ValueError(
                f"Channel-name count mismatch: {values.shape[0]} signals vs "
                f"{len(channel_names)} names"
            )
        if values.shape[0] < 4:
            raise ValueError("ICLabel needs at least four positioned EEG channels")
        if not np.isfinite(values).all():
            raise ValueError("ICA input contains NaN or infinite values")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"unit_scale_to_volts must be finite and positive, got {scale}"
            )

        names = self.normalise_channel_names(channel_names)
        info = mne.create_info(
            ch_names=names,
            sfreq=float(sampling_rate),
            ch_types=["eeg"] * len(names),
        )
        target = mne.io.RawArray(values * scale, info, verbose=False)
        self._set_standard_montage(target)

        # Keep the v3 ICLabel-compatible reference convention unchanged.
        target.set_eeg_reference("average", projection=False, verbose=False)

        # Fit/label on a filtered copy; apply the solution to the unfiltered,
        # average-referenced target recording.
        fit_raw = target.copy()
        h_freq, fit_upper = self._fit_band(sampling_rate)
        fit_raw.filter(
            l_freq=1.0,
            h_freq=h_freq,
            picks="eeg",
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False,
        )

        fit_input, sampling_report = self._distributed_fit_instance(fit_raw)
        rank = int(mne.compute_rank(fit_input, rank=None, verbose=False)["eeg"])
        n_components = min(len(names) - 1, rank)
        if n_components < 2:
            raise ValueError(f"ICA fitting data have insufficient EEG rank: {n_components}")

        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params={"extended": True},
            random_state=self.random_state,
            max_iter="auto",
        )
        ica.fit(
            fit_input,
            picks="eeg",
            reject_by_annotation=True,
            verbose=False,
        )

        # ICLabel receives the same Raw/Epochs instance used to fit ICA.
        component_info = label_components(fit_input, ica, method=self.component_labeler)
        labels = [str(label) for label in component_info["labels"]]
        confidence = np.asarray(component_info["y_pred_proba"], dtype=float)

        if len(labels) != int(ica.n_components_):
            raise RuntimeError(
                "ICLabel returned a different number of labels than ICA components"
            )
        if confidence.shape != (len(labels),):
            raise RuntimeError(
                "Unexpected ICLabel confidence shape: "
                f"{confidence.shape}; expected {(len(labels),)}"
            )
        unknown_labels = set(labels) - set(ICLABEL_CLASSES)
        if unknown_labels:
            raise RuntimeError(
                "ICLabel returned unknown labels: " + ", ".join(sorted(unknown_labels))
            )
        if not np.isfinite(confidence).all():
            raise RuntimeError("ICLabel returned non-finite confidence values")

        exclude = [
            index for index, label in enumerate(labels) if label not in self.keep_labels
        ]

        # Rare safety case: ICLabel can assign an artifact argmax label to every
        # component.  Removing all components would make the reconstruction
        # meaningless, so do not crash the entire benchmark.  Only in this edge
        # case, inspect the full ICLabel probability matrix and retain the IC with
        # the largest combined probability of the pre-registered keep classes
        # (brain + other by default).  Normal recordings use the exact same v4
        # selection rule as before.
        guard_component = None
        guard_keep_probability = None
        if len(exclude) == len(labels):
            probabilities = np.asarray(
                iclabel_label_components(fit_input, ica, inplace=True),
                dtype=float,
            )
            expected_shape = (len(labels), len(ICLABEL_CLASSES))
            if probabilities.shape != expected_shape:
                raise RuntimeError(
                    "Unexpected ICLabel probability shape in all-artifact guard: "
                    f"{probabilities.shape}; expected {expected_shape}"
                )
            if not np.isfinite(probabilities).all():
                raise RuntimeError(
                    "ICLabel returned non-finite probabilities in all-artifact guard"
                )

            keep_class_indices = [
                index
                for index, class_name in enumerate(ICLABEL_CLASSES)
                if class_name in self.keep_labels
            ]
            keep_probability = probabilities[:, keep_class_indices].sum(axis=1)
            guard_component = int(np.argmax(keep_probability))
            guard_keep_probability = float(keep_probability[guard_component])
            exclude.remove(guard_component)

        cleaned = target.copy()
        ica.apply(cleaned, exclude=exclude, verbose=False)
        output = cleaned.get_data() / scale
        if output.shape != values.shape:
            raise RuntimeError(
                f"ICA output shape changed from {values.shape} to {output.shape}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError("ICA output contains NaN or infinite values")

        self._reports.append(
            {
                "ica_version": self.cache_tag,
                "task": task_name,
                "recording": recording_id,
                "sampling_rate": float(sampling_rate),
                "fit_band_hz": [1.0, fit_upper],
                "fit_sample_budget_seconds": self.fit_sample_budget_seconds,
                **sampling_report,
                "n_channels": len(names),
                "estimated_eeg_rank": rank,
                "n_components": int(ica.n_components_),
                "n_iterations": int(ica.n_iter_),
                "labels": labels,
                "label_confidence": confidence.tolist(),
                "all_artifact_guard_triggered": guard_component is not None,
                "guard_retained_component": guard_component,
                "guard_keep_probability": guard_keep_probability,
                "excluded_components": exclude,
                "excluded_labels": [labels[index] for index in exclude],
                "excluded_fraction": float(len(exclude) / len(labels)),
                "kept_labels": sorted(self.keep_labels),
                "iclabel_upper_band_limited_by_nyquist": bool(fit_upper < 100.0),
            }
        )
        return output.astype(np.float32)

    def update_last_report(self, metadata: dict[str, object]) -> None:
        """Attach dataset-adapter diagnostics to the most recent ICA report."""
        if not self._reports:
            raise RuntimeError("No ICA report exists to annotate")
        self._reports[-1].update(metadata)

    def save_reports(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._reports, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


DENOISER = ICADenoiser()
