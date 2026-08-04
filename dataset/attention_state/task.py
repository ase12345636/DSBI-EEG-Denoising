"""Mental-attention preprocessing reconstructed from report section 4.3.5."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from utils.contracts import SignalDataset
from utils.features import stft_mean_power
from utils.progress import progress


FS = 128.0
WINDOW = 640
STATE_SAMPLES = 10 * 60 * 128
EXPECTED_RECORDINGS = 34
SUBJECT_RECORDING_COUNTS = (7, 7, 7, 7, 6)
CACHE_VERSION = "attention-v3-recording-denoise"


class AttentionStateTask:
    name = "attention_state"
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
        cache = cache_dir / self.name / f"{CACHE_VERSION}-{denoiser.name}.npz"
        if cache.exists():
            return self._load(cache, quick)

        files = self._selected_files(data_dir)
        windows: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[str] = []

        for path in progress(
            files,
            total=len(files),
            desc="Attention recordings",
            unit="file",
            leave=False,
        ):
            signal = self._load_eeg(path)
            if signal.shape[1] < 3 * STATE_SAMPLES:
                raise ValueError(
                    f"{path.name} is shorter than the report's first 30 minutes: "
                    f"{signal.shape[1] / FS / 60:.2f} minutes"
                )
            signal = signal[:, : 3 * STATE_SAMPLES]

            # The report explicitly standardizes each recording before 5-second
            # segmentation.  Statistics are channel-wise and use only that
            # recording, avoiding cross-recording leakage.
            mean = np.mean(signal, axis=1, keepdims=True)
            standard_deviation = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / np.where(
                standard_deviation == 0, 1.0, standard_deviation
            )

            # Long-recording processing is preferable for ASR calibration and
            # avoids independent filter edge effects at every 5-second window.
            if denoiser.name == "raw":
                processed = np.asarray(signal, dtype=np.float32)
            elif denoiser.name == "asr":
                processed = denoiser.transform_recording(signal, FS).astype(np.float32)
            elif denoiser.name == "bandpass":
                processed = denoiser.transform(signal[np.newaxis], FS)[0].astype(np.float32)
            elif denoiser.name == "ic_unet":
                if checkpoint_path is None:
                    raise ValueError("IC-U-Net checkpoint path is required")
                processed = denoiser.transform_recording(
                    signal,
                    FS,
                    checkpoint_path,
                    task_name=self.name,
                    chunk_seconds=30,
                    overlap_seconds=2,
                )
            else:
                processed = denoiser.transform(
                    signal[np.newaxis], FS, task_name=self.name
                )[0].astype(np.float32)

            for label in range(3):
                state = processed[
                    :,
                    label * STATE_SAMPLES : (label + 1) * STATE_SAMPLES,
                ]
                for start in range(0, STATE_SAMPLES, WINDOW):
                    window = state[:, start : start + WINDOW]
                    if window.shape[-1] != WINDOW:
                        continue
                    windows.append(window.astype(np.float32, copy=False))
                    labels.append(label)
                    groups.append(path.stem)

        signals = np.stack(windows)
        dataset = SignalDataset(
            signals=signals,
            labels=np.asarray(labels, dtype=np.int8),
            sampling_rate=FS,
            class_names=("focused", "unfocused", "drowsy"),
            primary_metric="accuracy",
            groups=np.asarray(groups),
            metadata={
                "source": "report reconstruction",
                "window_seconds": 5,
                "recording_minutes": 30,
                "normalization": "per-recording per-channel z-score",
                "denoising_scope": "complete recording before windowing",
                "split": "stratified sample-level 80/20",
                "files": [path.name for path in files],
            },
        )
        dataset.validate()
        self._save(cache, dataset)
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
    def features(
        train,
        test,
        y_train,
        y_test,
    ):
        del y_train, y_test
        return (
            stft_mean_power(train, FS),
            stft_mean_power(test, FS),
        )

    @staticmethod
    def balance_features(x_train, y_train, seed: int):
        del seed
        return x_train, y_train

    @staticmethod
    def standardize_features(x_train, x_test):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_test)

    def _selected_files(self, data_dir: Path) -> list[Path]:
        all_files = sorted(
            data_dir.rglob("*.mat"),
            key=self._natural_path_key,
        )

        if len(all_files) < 23:
            raise FileNotFoundError(
                f"Expected at least 23 attention .mat files below {data_dir}"
            )

        selection_path = Path(__file__).with_name(
            "selection.json"
        )

        if selection_path.exists():
            payload = json.loads(
                selection_path.read_text(
                    encoding="utf-8"
                )
            )
            names = payload["files"]
            by_name = {
                path.name: path
                for path in all_files
            }
            missing = [
                name
                for name in names
                if name not in by_name
            ]

            if missing:
                raise FileNotFoundError(
                    "selection.json names not downloaded: "
                    f"{missing}"
                )

            selected = [
                by_name[name]
                for name in names
            ]

            invalid = [
                path.name
                for path in selected
                if self._load_eeg(path).shape[1]
                < 3 * STATE_SAMPLES
            ]

            if not invalid:
                return selected

            print(
                "[attention] Existing selection.json contains "
                "recordings shorter than 30 minutes; rebuilding it: "
                + ", ".join(invalid)
            )

        grouped: dict[str, list[Path]] = {}

        for path in all_files:
            key = re.sub(
                r"[_-]?\d+$",
                "",
                path.stem,
            )
            grouped.setdefault(
                key,
                [],
            ).append(path)

        if (
            len(grouped) != 5
            or any(
                len(group) < 6
                for group in grouped.values()
            )
        ):
            if len(all_files) != EXPECTED_RECORDINGS:
                raise ValueError(
                    "The official attention release has "
                    f"{EXPECTED_RECORDINGS} files; got "
                    f"{len(all_files)}"
                )

            grouped = {}
            cursor = 0

            for subject, size in enumerate(
                SUBJECT_RECORDING_COUNTS,
                start=1,
            ):
                grouped[str(subject)] = all_files[
                    cursor : cursor + size
                ]
                cursor += size

        candidates = [
            path
            for group in grouped.values()
            for path in sorted(
                group,
                key=self._natural_path_key,
            )[2:]
        ]

        if len(candidates) != 24:
            raise ValueError(
                "Could not reconstruct the report's "
                "post-habituation set: got "
                f"{len(candidates)} files"
            )

        quality: list[tuple[Path, int, float]] = []

        for path in progress(
            candidates,
            total=len(candidates),
            desc="Attention quality scan",
            unit="file",
            leave=False,
        ):
            eeg = self._load_eeg(path)
            quality.append(
                (
                    path,
                    eeg.shape[1],
                    self._outlier_score_from_eeg(eeg),
                )
            )

        required_samples = 3 * STATE_SAMPLES
        short_recordings = [
            item
            for item in quality
            if item[1] < required_samples
        ]

        if len(short_recordings) == 1:
            rejected = short_recordings[0][0]
            rejection_reason = (
                "shorter_than_report_30_minutes"
            )

        elif len(short_recordings) > 1:
            details = ", ".join(
                f"{path.name}="
                f"{sample_count / FS / 60:.2f}min"
                for path, sample_count, _ in short_recordings
            )
            raise ValueError(
                "More than one post-habituation Attention "
                "recording is shorter than 30 minutes: "
                f"{details}"
            )

        else:
            rejected = max(
                quality,
                key=lambda item: item[2],
            )[0]
            rejection_reason = (
                "strongest_robust_amplitude_outlier"
            )

        selected = [
            path
            for path in candidates
            if path != rejected
        ]

        selection_path.write_text(
            json.dumps(
                {
                    "files": [
                        path.name
                        for path in selected
                    ],
                    "rejected": rejected.name,
                    "rejection_reason": rejection_reason,
                    "note": (
                        "Auto-reconstructed because the "
                        "supplied report/code omitted the "
                        "exact rejected filename."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return selected

    def _outlier_score(self, path: Path) -> float:
        return self._outlier_score_from_eeg(
            self._load_eeg(path)
        )

    @staticmethod
    def _outlier_score_from_eeg(
        eeg: np.ndarray,
    ) -> float:
        centered = eeg - np.median(
            eeg,
            axis=1,
            keepdims=True,
        )
        return float(
            np.median(
                np.max(
                    np.abs(centered),
                    axis=1,
                )
            )
        )

    @staticmethod
    def _load_eeg(path: Path) -> np.ndarray:
        from scipy.io import loadmat

        content = loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
        )

        obj = content.get("o")
        data = (
            getattr(obj, "data", None)
            if obj is not None
            else None
        )

        if data is None:
            arrays = [
                value
                for key, value in content.items()
                if (
                    not key.startswith("__")
                    and isinstance(value, np.ndarray)
                    and value.ndim == 2
                )
            ]

            if not arrays:
                raise ValueError(
                    f"Could not find o.data in {path}"
                )

            data = max(
                arrays,
                key=lambda value: value.size,
            )

        data = np.asarray(
            data,
            dtype=np.float64,
        )

        if data.ndim != 2:
            raise ValueError(
                "Unexpected attention matrix dimensions "
                f"in {path}: {data.shape}"
            )

        if data.shape[1] == 25:
            eeg = data[:, 3:17].T

        elif data.shape[0] == 25:
            eeg = data[3:17, :]

        elif data.shape[1] == 14:
            eeg = data.T

        elif data.shape[0] == 14:
            eeg = data

        elif (
            data.shape[1] >= 17
            and data.shape[0] > data.shape[1]
        ):
            eeg = data[:, 3:17].T

        elif (
            data.shape[0] >= 17
            and data.shape[1] > data.shape[0]
        ):
            eeg = data[3:17, :]

        else:
            raise ValueError(
                f"Unexpected attention matrix shape in "
                f"{path}: {data.shape}"
            )

        if eeg.shape[0] != 14:
            raise ValueError(
                "Attention EEG extraction did not produce "
                f"14 channels for {path}: {eeg.shape}"
            )

        if not np.isfinite(eeg).all():
            raise ValueError(
                "Attention recording contains NaN or "
                f"infinity: {path}"
            )

        return np.asarray(
            eeg,
            dtype=np.float64,
        )

    @staticmethod
    def _natural_path_key(
        path: Path,
    ) -> tuple[object, ...]:
        return tuple(
            int(piece)
            if piece.isdigit()
            else piece.lower()
            for piece in re.split(
                r"(\d+)",
                path.name,
            )
        )

    @staticmethod
    def _save(
        path: Path,
        data: SignalDataset,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            signals=data.signals,
            labels=data.labels,
            groups=data.groups,
        )

    def _load(
        self,
        path: Path,
        quick: bool,
    ) -> SignalDataset:
        with np.load(
            path,
            allow_pickle=False,
        ) as saved:
            data = SignalDataset(
                saved["signals"],
                saved["labels"],
                FS,
                (
                    "focused",
                    "unfocused",
                    "drowsy",
                ),
                "accuracy",
                groups=saved["groups"],
            )

        return (
            self._quick(data)
            if quick
            else data
        )

    @staticmethod
    def _quick(
        data: SignalDataset,
    ) -> SignalDataset:
        indices = np.concatenate(
            [
                np.flatnonzero(
                    data.labels == label
                )[:64]
                for label in (0, 1, 2)
            ]
        )

        return SignalDataset(
            data.signals[indices],
            data.labels[indices],
            FS,
            data.class_names,
            data.primary_metric,
            groups=data.groups[indices],
            metadata=data.metadata,
        )


TASK = AttentionStateTask()
