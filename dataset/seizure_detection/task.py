"""CHB-MIT seizure-detection downstream task."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from utils.contracts import SignalDataset
from utils.features import stft_mean_power
from utils.progress import progress


FS = 256.0
FEATURE_POWER_SCALE = np.float32(1e12)  # V^2 -> microvolt^2
NON_SEIZURE_CAP_PER_PATIENT = 999
EXCLUDED_PATIENTS = {"chb15"}
EXCLUDED_FILES = {"chb12_27.edf", "chb12_28.edf", "chb12_29.edf"}


class SeizureDetectionTask:
    name = "seizure_detection"
    feature_kind = "stft_mean_power"
    validation_size = 0.20
    feature_power_scale = FEATURE_POWER_SCALE

    def prepare(self, data_dir: Path, cache_dir: Path, denoiser, checkpoint_path: Path | None,
                quick: bool) -> SignalDataset:
        cache = cache_dir / self.name / f"{denoiser.name}.npz"
        if cache.exists() and not quick:
            return self._load(cache)

        intervals = self._parse_summaries(data_dir)
        files = [p for p in sorted(data_dir.rglob("*.edf")) if self._usable(p)]
        if not files:
            raise FileNotFoundError(f"No usable CHB-MIT EDF files below {data_dir}")
        if quick:
            seizure_files = [p for p in files if intervals.get(p.name)]
            files = (seizure_files or files)[:1]

        examples, labels = [], []
        non_seizure_per_patient: dict[str, int] = {}
        import mne

        for path in progress(files, total=len(files), desc="CHB-MIT recordings", unit="EDF", leave=False):
            patient = path.parent.name
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            try:
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
                # Retain the original integrated reproduction behavior: ASR is
                # calibrated on each continuous EDF with report cutoff k=5.
                recording = denoiser.transform_recording(recording, FS)
            elif denoiser.name == "bandpass":
                recording = denoiser.transform(recording[None], FS)[0]

            recording = self._trim_channels(recording)
            if recording is None:
                continue

            seizure_ranges = intervals.get(path.name, [])
            for second in range(recording.shape[1] // int(FS)):
                is_seizure = any(start <= second < end for start, end in seizure_ranges)
                if not is_seizure and non_seizure_per_patient.get(patient, 0) >= NON_SEIZURE_CAP_PER_PATIENT:
                    continue
                segment = recording[:, second * int(FS) : (second + 1) * int(FS)]
                if segment.shape[-1] != int(FS):
                    continue
                examples.append(segment.astype(np.float32, copy=False))
                labels.append(int(is_seizure))
                if not is_seizure:
                    non_seizure_per_patient[patient] = non_seizure_per_patient.get(patient, 0) + 1

        signals = np.stack(examples)
        y = np.asarray(labels, dtype=np.int8)

        # Match the report's balanced seizure/non-seizure dataset. The fixed
        # RNG makes every denoising method use the same sample subset.
        rng = np.random.default_rng(42)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        count = min(len(positive), len(negative))
        if count == 0:
            raise ValueError("CHB-MIT preparation produced only one class")
        selected = np.sort(np.concatenate([
            rng.choice(positive, count, replace=False),
            rng.choice(negative, count, replace=False),
        ]))
        signals, y = signals[selected], y[selected]

        if denoiser.name == "ic_unet":
            if checkpoint_path is None:
                raise ValueError("IC-U-Net checkpoint path is required")
            signals = denoiser.transform(
                signals,
                FS,
                checkpoint_path=checkpoint_path,
                task_name=self.name,
            )

        data = SignalDataset(
            signals, y, FS, ("non_seizure", "seizure"), "accuracy"
        )
        data.validate()
        if not quick:
            self._save(cache, data)
            if denoiser.name == "ica":
                denoiser.save_reports(cache.with_suffix(".components.json"))
        return data

    @staticmethod
    def split(data: SignalDataset, seed: int):
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(data.labels))
        return train_test_split(indices, test_size=0.2, stratify=data.labels, random_state=seed)

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
    def _usable(path: Path) -> bool:
        return path.parent.name not in EXCLUDED_PATIENTS and path.name not in EXCLUDED_FILES

    @staticmethod
    def _trim_channels(data: np.ndarray) -> np.ndarray | None:
        output = np.asarray(data, dtype=np.float64)
        if output.shape[0] in (22, 25):
            return None
        if output.shape[0] == 28:
            output = np.delete(output, [4, 9, 12, 17, 22], axis=0)
        elif output.shape[0] == 24:
            output = np.delete(output, 23, axis=0)
        elif output.shape[0] == 29:
            output = np.delete(output, [4, 9, 12, 17, 22, 28], axis=0)
        return output if output.shape[0] == 23 else None

    @staticmethod
    def _parse_summaries(root: Path) -> dict[str, list[tuple[int, int]]]:
        result: dict[str, list[tuple[int, int]]] = {}
        for summary in root.rglob("*-summary.txt"):
            current = None
            start = None
            for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
                file_match = re.match(r"File Name:\s+(.+\.edf)", line.strip())
                if file_match:
                    current = file_match.group(1)
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
        np.savez_compressed(path, signals=data.signals, labels=data.labels)

    @staticmethod
    def _load(path: Path) -> SignalDataset:
        with np.load(path, allow_pickle=False) as saved:
            return SignalDataset(
                saved["signals"], saved["labels"], FS,
                ("non_seizure", "seizure"), "accuracy",
            )


TASK = SeizureDetectionTask()
