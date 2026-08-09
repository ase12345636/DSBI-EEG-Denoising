"""CHB-MIT seizure detection task."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from utils.contracts import SignalDataset
from utils.features import stft_mean_power
from utils.progress import progress


FS = 256.0
SIGNAL_UNIT = "volt"
STFT_POWER_UNIT_SCALE = np.float32(1e12)  # V^2 -> microvolt^2
CACHE_VERSION = "seizure"
NON_SEIZURE_CAP_PER_PATIENT = 999
EXCLUDED_PATIENTS = {"chb15"}
EXCLUDED_FILES = {"chb12_27.edf", "chb12_28.edf", "chb12_29.edf"}


class SeizureDetectionTask:
    name = "seizure_detection"
    cache_version = CACHE_VERSION
    feature_kind = "stft_mean_power"
    validation_size = 0.20

    def prepare(
        self,
        data_dir: Path,
        cache_dir: Path,
        denoiser,
        checkpoint_path: Path | None,
        quick: bool,
    ) -> SignalDataset:
        # Quick ICA uses one recording and must use a separate cache.
        cache_method = denoiser.name
        if denoiser.name == "ica":
            cache_method = "ica-v4-quick" if quick else "ica-v4"
        cache = cache_dir / self.name / f"{CACHE_VERSION}-{cache_method}.npz"
        if cache.exists():
            return self._load(cache, quick)

        intervals = self._parse_summaries(data_dir)
        examples: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[str] = []
        non_seizure_per_patient: dict[str, int] = {}

        edf_files = [
            path
            for path in sorted(data_dir.rglob("*.edf"))
            if self._is_usable_recording(path)
        ]
        if not edf_files:
            raise FileNotFoundError(
                f"No CHB-MIT EDF files found below {data_dir}"
            )
        if quick:
            # Select one recording that actually contains a seizure interval so
            # the smoke test retains both classes after one-second segmentation.
            seizure_files = [
                path for path in edf_files if intervals.get(path.name)
            ]
            edf_files = (seizure_files or edf_files)[:1]

        import mne

        for path in progress(
            edf_files,
            total=len(edf_files),
            desc="CHB-MIT seizure recordings",
            unit="EDF",
            leave=False,
        ):
            patient = path.parent.name
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            try:
                # MNE returns EDF EEG values in volts.
                recording = raw.get_data().astype(np.float64, copy=False)
                channel_names = tuple(raw.ch_names)
            finally:
                raw.close()

            if denoiser.name == "ica":
                from dataset.seizure_detection.ica_adapter import clean_bipolar_recording

                recording = clean_bipolar_recording(
                    denoiser,
                    recording,
                    FS,
                    channel_names=channel_names,
                    task_name=self.name,
                    recording_id=str(path.relative_to(data_dir)),
                )
            elif denoiser.name == "asr":
                # ASR is applied to the full EDF before channel trimming and segmentation.
                recording = denoiser.transform_recording(recording, FS)
            elif denoiser.name == "bandpass":
                # Band-pass filtering is applied to the full EDF before segmentation.
                recording = denoiser.transform(recording[np.newaxis], FS)[0]

            recording = self._trim_channels(recording)
            if recording is None:
                continue

            seizure_ranges = intervals.get(path.name, [])
            duration_seconds = recording.shape[1] // int(FS)

            for second in range(duration_seconds):
                is_seizure = any(
                    start <= second < end for start, end in seizure_ranges
                )
                if (
                    not is_seizure
                    and non_seizure_per_patient.get(patient, 0)
                    >= NON_SEIZURE_CAP_PER_PATIENT
                ):
                    continue

                segment = recording[
                    :,
                    second * int(FS) : (second + 1) * int(FS),
                ]
                if segment.shape[-1] != int(FS):
                    continue

                if is_seizure:
                    label = 1
                else:
                    label = 0
                    non_seizure_per_patient[patient] = (
                        non_seizure_per_patient.get(patient, 0) + 1
                    )

                examples.append(segment.astype(np.float32, copy=False))
                labels.append(label)
                groups.append(patient)

        if not examples:
            raise ValueError("No usable CHB-MIT 1-second epochs were produced")

        signals = np.stack(examples)
        labels_array = np.asarray(labels, dtype=np.int8)
        groups_array = np.asarray(groups)

        # The report uses equal seizure and non-seizure sample counts.  Keep the
        # selection deterministic and identical for all denoising methods.
        rng = np.random.default_rng(42)
        positive = np.flatnonzero(labels_array == 1)
        negative = np.flatnonzero(labels_array == 0)
        count = min(len(positive), len(negative))
        if count == 0:
            raise ValueError("CHB-MIT preparation produced only one class")
        selected = np.sort(
            np.concatenate(
                [
                    rng.choice(positive, size=count, replace=False),
                    rng.choice(negative, size=count, replace=False),
                ]
            )
        )
        signals = signals[selected]
        labels_array = labels_array[selected]
        groups_array = groups_array[selected]

        if denoiser.name == "ic_unet":
            if checkpoint_path is None:
                raise ValueError("IC-U-Net checkpoint path is required")
            signals = denoiser.transform(
                signals,
                FS,
                checkpoint_path=checkpoint_path,
                task_name=self.name,
            )

        dataset = SignalDataset(
            signals=signals,
            labels=labels_array,
            sampling_rate=FS,
            class_names=("non_seizure", "seizure"),
            primary_metric="accuracy",
            groups=groups_array,
        )
        dataset.validate()
        self._save(cache, dataset)
        if denoiser.name == "ica":
            denoiser.save_reports(cache.with_suffix(".components.json"))
        return self._quick(dataset) if quick else dataset

    @staticmethod
    def split(data: SignalDataset, seed: int):
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(data.labels))
        return train_test_split(
            indices,
            test_size=0.2,
            stratify=data.labels,
            random_state=seed,
        )

    @staticmethod
    def features(train, test, y_train, y_test):
        del y_train, y_test
        return stft_mean_power(train, FS), stft_mean_power(test, FS)

    @staticmethod
    def transform_feature_matrix(features: np.ndarray) -> np.ndarray:
        """Convert shared STFT power from V^2 to microvolt^2.

        MNE returns the time-domain EDF signal in volts.  The cached STFT
        power therefore remains in V^2.  This positive global unit conversion
        is applied once in the common feature path, before balancing and before
        any classifier-specific standardization, so LR, SVM, RF, LightGBM, and
        MLP all receive the same Seizure feature representation.  EEGNet uses
        the time-domain signal and does not pass through this method.
        """
        values = np.asarray(features, dtype=np.float32)
        return values * STFT_POWER_UNIT_SCALE

    @staticmethod
    def balance_features(x_train, y_train, seed: int):
        del seed
        return x_train, y_train

    @staticmethod
    def standardize_features(x_train, x_test):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_test)

    @staticmethod
    def _is_usable_recording(path: Path) -> bool:
        patient = path.parent.name
        return patient not in EXCLUDED_PATIENTS and path.name not in EXCLUDED_FILES

    @staticmethod
    def _trim_channels(data: np.ndarray) -> np.ndarray | None:
        """Normalize recordings to the required 23-channel layout."""
        output = np.asarray(data, dtype=np.float64)
        channel_count = output.shape[0]

        if channel_count in (22, 25):
            return None
        if channel_count == 28:
            output = np.delete(output, [4, 9, 12, 17, 22], axis=0)
        elif channel_count == 24:
            output = np.delete(output, 23, axis=0)
        elif channel_count == 29:
            output = np.delete(output, [4, 9, 12, 17, 22, 28], axis=0)

        return output if output.shape[0] == 23 else None

    @staticmethod
    def _parse_summaries(root: Path) -> dict[str, list[tuple[int, int]]]:
        result: dict[str, list[tuple[int, int]]] = {}
        for summary in root.rglob("*-summary.txt"):
            current: str | None = None
            start: int | None = None
            for line in summary.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                match = re.match(r"File Name:\s+(.+\.edf)", line.strip())
                if match:
                    current = match.group(1)
                    result.setdefault(current, [])
                elif "Start Time" in line:
                    value = re.search(r"(\d+)\s*seconds", line)
                    start = int(value.group(1)) if value else None
                elif "End Time" in line and current and start is not None:
                    value = re.search(r"(\d+)\s*seconds", line)
                    if value:
                        result[current].append((start, int(value.group(1))))
                    start = None
        return result

    @staticmethod
    def _save(path: Path, data: SignalDataset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signals=data.signals,
            labels=data.labels,
            groups=data.groups,
            signal_unit=np.asarray(SIGNAL_UNIT),
        )

    def _load(self, path: Path, quick: bool) -> SignalDataset:
        with np.load(path, allow_pickle=False) as saved:
            data = SignalDataset(
                saved["signals"],
                saved["labels"],
                FS,
                ("non_seizure", "seizure"),
                "accuracy",
                groups=saved["groups"],
            )
        return self._quick(data) if quick else data

    @staticmethod
    def _quick(data: SignalDataset) -> SignalDataset:
        indices = np.concatenate(
            [np.flatnonzero(data.labels == label)[:100] for label in (0, 1)]
        )
        return SignalDataset(
            data.signals[indices],
            data.labels[indices],
            FS,
            data.class_names,
            data.primary_metric,
            groups=data.groups[indices],
        )


TASK = SeizureDetectionTask()
