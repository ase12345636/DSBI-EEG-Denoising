"""CHB-MIT seizure-detection downstream task."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.contracts import SignalDataset
from utils.features import stft_mean_power
from utils.progress import progress


FS = 256.0
FEATURE_POWER_SCALE = np.float32(1e12)  # V^2 -> microvolt^2
CANONICAL_BIPOLAR_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)


class SeizureDetectionTask:
    name = "seizure_detection"
    selection_seed = 0
    feature_kind = "stft_mean_power"
    feature_power_scale = FEATURE_POWER_SCALE
    split_cycle_size = 5

    def prepare(self, data_dir: Path, cache_dir: Path, denoiser, checkpoint_path: Path | None,
                quick: bool) -> SignalDataset:
        cache = cache_dir / self.name / f"{denoiser.name}.npz"
        if cache.exists() and not quick:
            return self._load(cache)

        intervals = self._parse_summaries(data_dir)
        files = sorted(data_dir.rglob("*.edf"))
        if not files:
            raise FileNotFoundError(f"No CHB-MIT EDF files below {data_dir}")

        if quick:
            seizure_files = [path for path in files if intervals.get(path.name)]
            files = (seizure_files or files)[:1]

        plan = self._sample_plan(
            data_dir,
            files,
            intervals,
        )
        selected_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for relative, second, label in plan:
            selected_by_file[relative].append((second, label))

        examples, labels, groups, sample_ids = [], [], [], []
        import mne

        selected_recordings = sorted(selected_by_file)
        for relative in progress(
            selected_recordings,
            total=len(selected_recordings),
            desc="CHB-MIT selected recordings",
            unit="EDF",
            leave=False,
        ):
            path = data_dir / relative
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            try:
                if not np.isclose(float(raw.info["sfreq"]), FS):
                    raise ValueError(
                        f"Unexpected sampling rate in {relative}: {raw.info['sfreq']}"
                    )
                recording = self._canonical_bipolar(
                    raw.get_data().astype(np.float64, copy=False),
                    tuple(raw.ch_names),
                )
            finally:
                raw.close()

            if denoiser.name == "ica":
                from dataset.seizure_detection.ica_adapter import clean_bipolar_recording
                recording = clean_bipolar_recording(
                    denoiser,
                    recording,
                    FS,
                    channel_names=CANONICAL_BIPOLAR_CHANNELS,
                    task_name=self.name,
                    recording_id=relative,
                )
            elif denoiser.name in {"asr", "asr20"}:
                recording = denoiser.transform_recording(recording, FS)
            elif denoiser.name == "bandpass":
                recording = denoiser.transform(recording[None], FS)[0]
            elif denoiser.name == "ic_unet":
                if checkpoint_path is None:
                    raise ValueError("IC-U-Net checkpoint path is required")
                recording = denoiser.transform_bipolar_recording(
                    recording,
                    FS,
                    checkpoint_path,
                    channel_names=CANONICAL_BIPOLAR_CHANNELS,
                    task_name=self.name,
                )
            elif denoiser.name != "raw":
                raise ValueError(f"Unsupported Seizure denoiser: {denoiser.name}")

            for second, label in selected_by_file[relative]:
                segment = recording[:, second * int(FS) : (second + 1) * int(FS)]
                examples.append(segment.astype(np.float32, copy=False))
                labels.append(label)
                groups.append(self._patient(relative))
                sample_ids.append(f"{relative}#{second:06d}")

        signals = np.stack(examples)
        labels = np.asarray(labels, dtype=np.int8)
        sample_ids = np.asarray(sample_ids)

        data = SignalDataset(
            signals,
            labels,
            FS,
            ("non_seizure", "seizure"),
            "balanced_accuracy",
            groups=np.asarray(groups),
            sample_ids=sample_ids,
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

    def _sample_plan(
        self,
        data_dir: Path,
        files: list[Path],
        intervals: dict[str, list[tuple[int, int]]],
    ) -> list[tuple[str, int, int]]:
        positives, negatives = [], []
        for path in files:
            relative = str(path.relative_to(data_dir))
            duration = self._edf_duration_seconds(path)
            seizure_ranges = intervals.get(Path(relative).name, [])
            for second in range(duration):
                if any(start <= second < end for start, end in seizure_ranges):
                    positives.append((relative, second))
                else:
                    negatives.append((relative, second))

        rng = np.random.default_rng(self.selection_seed)
        chosen = rng.choice(len(negatives), len(positives), replace=False)
        plan = [(*sample, 1) for sample in positives]
        plan.extend((*negatives[int(index)], 0) for index in chosen)
        return sorted(plan)

    @staticmethod
    def _edf_duration_seconds(path: Path) -> int:
        with path.open("rb") as handle:
            header = handle.read(256)
        if len(header) != 256:
            raise ValueError(f"Invalid EDF header: {path}")
        record_count = int(header[236:244].decode("ascii").strip())
        record_seconds = float(header[244:252].decode("ascii").strip())
        duration = int(round(record_count * record_seconds))
        if duration <= 0:
            raise ValueError(f"Invalid EDF duration: {path}")
        return duration

    @staticmethod
    def _normalise_edf_name(name: str) -> str:
        value = str(name).strip().upper().replace(" ", "")
        value = re.sub(r"^EEG", "", value)
        value = re.sub(r"-\d+$", "", value)  # MNE duplicate-label suffix
        value = re.sub(r"-(REF|LE|RE)$", "", value)
        return "O1" if value == "01" else value

    @staticmethod
    def _patient(value: str) -> str:
        patient = Path(value.split("#", 1)[0]).parts[0].lower()
        return "chb01" if patient == "chb21" else patient

    @classmethod
    def _canonical_bipolar(
        cls,
        data: np.ndarray,
        channel_names: tuple[str, ...],
    ) -> np.ndarray:
        """Return one common 18-derivation order for every CHB-MIT layout."""
        values = np.asarray(data, dtype=np.float64)
        names = [cls._normalise_edf_name(name) for name in channel_names]
        direct = {}
        for index, name in enumerate(names):
            if name and name not in {"-", "."}:
                direct.setdefault(name, values[index])

        if all(name in direct for name in CANONICAL_BIPOLAR_CHANNELS):
            return np.stack([direct[name] for name in CANONICAL_BIPOLAR_CHANNELS])

        electrodes = {}
        for index, name in enumerate(names):
            if not name or name in {"-", "."}:
                continue
            if "-" not in name:
                electrodes.setdefault(name, values[index])
                continue
            first, reference = name.split("-", 1)
            if reference in {"CS2", "REF"}:
                electrodes.setdefault(first, values[index])

        output = []
        for derivation in CANONICAL_BIPOLAR_CHANNELS:
            first, second = derivation.split("-", 1)
            if first not in electrodes or second not in electrodes:
                raise ValueError(
                    "EDF recording cannot be mapped to the common bipolar montage; "
                    f"missing {derivation} from {channel_names}"
                )
            output.append(electrodes[first] - electrodes[second])
        return np.stack(output)

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
            sample_ids = saved["sample_ids"]
            return SignalDataset(
                saved["signals"],
                saved["labels"],
                FS,
                ("non_seizure", "seizure"),
                "balanced_accuracy",
                groups=np.asarray([cls._patient(value) for value in sample_ids]),
                sample_ids=sample_ids,
            )


TASK = SeizureDetectionTask()
