"""Mental-attention downstream task with subject-level evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.contracts import SignalDataset
from utils.features import stft_mean_power
from utils.progress import progress


FS = 128.0
ATTENTION_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
WINDOW = 5 * 128
STATE_SAMPLES = 10 * 60 * 128


class AttentionStateTask:
    name = "attention_state"
    feature_kind = "stft_mean_power"
    split_cycle_size = 5

    def prepare(self, data_dir: Path, cache_dir: Path, denoiser, checkpoint_path: Path | None,
                quick: bool) -> SignalDataset:
        cache = cache_dir / self.name / f"{denoiser.name}.npz"
        if cache.exists() and not quick:
            data = self._load(cache)
            if data.sample_ids is None:
                data.sample_ids = self._sample_ids(self._selected_files(data_dir))
            data.validate()
            return data

        files = self._selected_files(data_dir)
        if quick:
            files = files[:1]

        windows, labels, groups, sample_ids = [], [], [], []
        for path in progress(files, total=len(files), desc="Attention recordings", unit="file", leave=False):
            signal = self._load_eeg(path)[:, : 3 * STATE_SAMPLES]
            if signal.shape[1] < 3 * STATE_SAMPLES:
                raise ValueError(f"{path.name} is shorter than 30 minutes")

            if denoiser.name == "ica":
                processed = denoiser.transform_recording(
                    signal,
                    FS,
                    channel_names=ATTENTION_CHANNELS,
                    unit_scale_to_volts=1e-6,
                    task_name=self.name,
                    recording_id=path.stem,
                )

            elif denoiser.name == "raw":
                processed = signal
            elif denoiser.name in {"asr", "asr20"}:
                processed = denoiser.transform_recording(signal, FS)
            elif denoiser.name == "bandpass":
                processed = denoiser.transform(signal[None], FS)[0]
            elif denoiser.name == "ic_unet":
                if checkpoint_path is None:
                    raise ValueError("IC-U-Net checkpoint path is required")
                processed = denoiser.transform_recording(
                    signal,
                    FS,
                    checkpoint_path,
                    task_name=self.name,
                )
            else:
                raise ValueError(f"Unsupported Attention denoiser: {denoiser.name}")

            # Apply the same recording-level Z-score after every preprocessing
            # condition so downstream classifiers receive consistently scaled data.
            mean = processed.mean(axis=1, keepdims=True)
            std = processed.std(axis=1, keepdims=True)
            processed = (processed - mean) / np.where(std == 0, 1.0, std)

            for label in range(3):
                state = processed[:, label * STATE_SAMPLES : (label + 1) * STATE_SAMPLES]
                for start in range(0, STATE_SAMPLES, WINDOW):
                    window = state[:, start : start + WINDOW]
                    if window.shape[-1] == WINDOW:
                        windows.append(window.astype(np.float32, copy=False))
                        labels.append(label)
                        groups.append(self._subject(path.stem))
                        sample_ids.append(f"{path.stem}:{label}:{start}")

        data = SignalDataset(
            np.stack(windows),
            np.asarray(labels, dtype=np.int8),
            FS,
            ("focused", "unfocused", "drowsy"),
            "balanced_accuracy",
            groups=np.asarray(groups),
            sample_ids=np.asarray(sample_ids),
        )
        data.validate()
        if not quick:
            self._save(cache, data)
            if denoiser.name == "ica":
                denoiser.save_reports(cache.with_suffix(".components.json"))
        return data

    @staticmethod
    def split(data: SignalDataset, seed: int, repeat: int = 0):
        from sklearn.model_selection import StratifiedGroupKFold, train_test_split
        indices = np.arange(len(data.labels))
        if len(np.unique(data.groups)) == 1:  # quick smoke run
            return train_test_split(
                indices, test_size=0.2, stratify=data.labels, random_state=seed
            )
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(folds.split(indices, data.labels, data.groups))[repeat % 5]

    @staticmethod
    def features(train, test, y_train, y_test):
        del y_train, y_test
        return stft_mean_power(train, FS), stft_mean_power(test, FS)

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
    def _selected_files(data_dir: Path) -> list[Path]:
        selection = json.loads(Path(__file__).with_name("selection.json").read_text(encoding="utf-8"))["files"]
        by_name = {path.name: path for path in data_dir.rglob("*.mat")}
        missing = [name for name in selection if name not in by_name]
        if missing:
            raise FileNotFoundError("Missing Attention files: " + ", ".join(missing))
        return [by_name[name] for name in selection]

    @staticmethod
    def _sample_ids(files: list[Path]) -> np.ndarray:
        return np.asarray([
            f"{path.stem}:{label}:{start}"
            for path in files
            for label in range(3)
            for start in range(0, STATE_SAMPLES, WINDOW)
        ])

    @staticmethod
    def _subject(value: str) -> str:
        record = int(Path(value.split(":", 1)[0]).stem.removeprefix("eeg_record"))
        return f"subject_{(record - 1) // 7 + 1}"

    @staticmethod
    def _load_eeg(path: Path) -> np.ndarray:
        from scipy.io import loadmat

        content = loadmat(path, squeeze_me=True, struct_as_record=False)
        obj = content.get("o")
        data = getattr(obj, "data", None) if obj is not None else None
        if data is None:
            arrays = [v for k, v in content.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2]
            if not arrays:
                raise ValueError(f"Could not find EEG matrix in {path}")
            data = max(arrays, key=lambda value: value.size)

        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] == 25:
            eeg = data[:, 3:17].T
        elif data.shape[0] == 25:
            eeg = data[3:17]
        elif data.shape[1] == 14:
            eeg = data.T
        elif data.shape[0] == 14:
            eeg = data
        elif data.shape[1] >= 17 and data.shape[0] > data.shape[1]:
            eeg = data[:, 3:17].T
        elif data.shape[0] >= 17 and data.shape[1] > data.shape[0]:
            eeg = data[3:17]
        else:
            raise ValueError(f"Unexpected Attention matrix shape in {path}: {data.shape}")
        if eeg.shape[0] != 14:
            raise ValueError(f"Attention extraction did not produce 14 channels: {path}")
        return np.asarray(eeg, dtype=np.float64)

    @staticmethod
    def _save(path: Path, data: SignalDataset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signals=data.signals,
            labels=data.labels,
            groups=data.groups,
            sample_ids=data.sample_ids,
        )

    @classmethod
    def _load(cls, path: Path) -> SignalDataset:
        with np.load(path, allow_pickle=False) as saved:
            sample_ids = saved["sample_ids"] if "sample_ids" in saved.files else None
            return SignalDataset(
                saved["signals"], saved["labels"], FS,
                ("focused", "unfocused", "drowsy"), "balanced_accuracy",
                groups=(
                    np.asarray([cls._subject(value) for value in sample_ids])
                    if sample_ids is not None else saved["groups"]
                ),
                sample_ids=sample_ids,
            )


TASK = AttentionStateTask()
