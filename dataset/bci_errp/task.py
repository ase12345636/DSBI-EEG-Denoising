"""BCI ErrP preprocessing and evaluation task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.contracts import SignalDataset
from utils.features import bci_xdawn_tangent
from utils.progress import progress


CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
    "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz",
    "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "POz", "O2",
)
TEST_SUBJECTS = {1, 3, 4, 5, 8, 9, 10, 15, 19, 25}
FS = 200.0


class BCIErrPTask:
    name = "bci_errp"
    cache_version = "bci"
    feature_kind = "xdawn_tangent"
    validation_size = 0.25

    def prepare(self, data_dir: Path, cache_dir: Path, denoiser, checkpoint_path: Path | None,
                quick: bool) -> SignalDataset:
        # Quick ICA uses representative sessions and must use a separate cache.
        cache_method = denoiser.name
        if denoiser.name == "ica":
            cache_method = "ica-v4-quick" if quick else "ica-v4"
        variant_cache = cache_dir / self.name / f"{self.cache_version}-{cache_method}.npz"
        if variant_cache.exists():
            return self._load(variant_cache, quick)
        if denoiser.name == "ica":
            dataset = self._ica_epochs(data_dir, denoiser, quick=quick)
            self._save(variant_cache, dataset)
            denoiser.save_reports(variant_cache.with_suffix(".components.json"))
            return self._quick(dataset) if quick else dataset

        raw = self._raw_epochs(data_dir, cache_dir)
        kwargs = {"task_name": self.name}
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        if denoiser.name == "bandpass":
            # Filter each raw epoch, then subtract its baseline.
            signals = denoiser.transform(raw.signals, FS, **kwargs)
            signals -= np.mean(signals[:, :, :20], axis=2, keepdims=True)
        elif denoiser.name == "asr":
            # Apply 1--40 Hz filtering and baseline correction before epoch-level ASR.
            from denoise.bandpass.method import butter_bandpass_filter

            filtered = butter_bandpass_filter(raw.signals, FS, highcut=40.0)
            filtered -= np.mean(filtered[:, :, :20], axis=2, keepdims=True)
            signals = np.stack([
                denoiser.transform_recording(epoch, FS)
                for epoch in progress(
                    filtered, total=len(filtered), desc="BCI ASR", unit="epoch", leave=False
                )
            ])
        else:
            # Raw and IC-U-Net both receive the baseline-corrected epoch.
            baseline_corrected = raw.signals - np.mean(
                raw.signals[:, :, :20], axis=2, keepdims=True
            )
            signals = denoiser.transform(baseline_corrected, FS, **kwargs)
        dataset = SignalDataset(
            signals=signals, labels=raw.labels, sampling_rate=FS,
            class_names=("bad_feedback", "good_feedback"),
            primary_metric="balanced_accuracy", fixed_train=raw.fixed_train,
            groups=raw.groups,
        )
        self._save(variant_cache, dataset)
        return self._quick(dataset) if quick else dataset

    def split(self, data: SignalDataset, seed: int):
        del seed
        train = np.flatnonzero(data.fixed_train)
        test = np.flatnonzero(~data.fixed_train)
        return train, test

    def features(self, train, test, y_train, y_test):
        del y_test
        return bci_xdawn_tangent(train, test, y_train)

    @staticmethod
    def standardize_features(x_train, x_test):
        # Tangent-space features are passed directly to the classifiers.
        return np.asarray(x_train), np.asarray(x_test)

    def balance_features(self, x_train, y_train, seed: int):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError("BCI feature balancing needs imbalanced-learn") from exc
        return SMOTE(random_state=seed).fit_resample(x_train, y_train)

    def _ica_epochs(
        self,
        data_dir: Path,
        denoiser,
        *,
        quick: bool,
    ) -> SignalDataset:
        """Run ICA on each continuous BCI session before epoching."""
        csv_files = sorted(data_dir.rglob("Data_S*_Sess*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No BCI Data_S*_Sess*.csv files found below {data_dir}"
            )
        if quick:
            train_file = next(
                (
                    path
                    for path in csv_files
                    if int(path.name.split("_S", 1)[1].split("_", 1)[0])
                    not in TEST_SUBJECTS
                ),
                None,
            )
            test_file = next(
                (
                    path
                    for path in csv_files
                    if int(path.name.split("_S", 1)[1].split("_", 1)[0])
                    in TEST_SUBJECTS
                ),
                None,
            )
            if train_file is None or test_file is None:
                raise ValueError(
                    "BCI ICA quick mode needs at least one training and one test session"
                )
            csv_files = [train_file, test_file]

        train_labels = self._label_map(data_dir, "TrainLabels.csv")
        test_labels = self._test_label_map(data_dir)
        epochs: list[np.ndarray] = []
        labels: list[int] = []
        train_mask: list[bool] = []
        groups: list[str] = []

        for path in progress(
            csv_files,
            total=len(csv_files),
            desc="BCI ICA",
            unit="file",
            leave=False,
        ):
            subject = int(path.name.split("_S", 1)[1].split("_", 1)[0])
            is_train = subject not in TEST_SUBJECTS
            frame = pd.read_csv(path, usecols=[*CHANNELS, "FeedBackEvent"])
            markers = np.flatnonzero(frame["FeedBackEvent"].to_numpy() == 1)
            eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64).T
            cleaned = denoiser.transform_recording(
                eeg,
                FS,
                channel_names=CHANNELS,
                unit_scale_to_volts=1e-6,
                task_name=self.name,
                recording_id=path.stem,
            )
            prefix = path.stem.removeprefix("Data_")
            label_map = train_labels if is_train else test_labels
            for event_number, marker in enumerate(markers, start=1):
                segment = cleaned[:, marker - 20 : marker + 120]
                if segment.shape[-1] != 140:
                    continue
                segment = segment - np.mean(
                    segment[:, :20], axis=1, keepdims=True
                )
                event_id = f"{prefix}_FB{event_number:03d}"
                if event_id not in label_map:
                    raise KeyError(f"Missing BCI label for {event_id}")
                epochs.append(segment.astype(np.float32, copy=False))
                labels.append(label_map[event_id])
                train_mask.append(is_train)
                groups.append(prefix.split("_Sess", 1)[0])

        dataset = SignalDataset(
            signals=np.stack(epochs),
            labels=np.asarray(labels, dtype=np.int8),
            sampling_rate=FS,
            class_names=("bad_feedback", "good_feedback"),
            primary_metric="balanced_accuracy",
            fixed_train=np.asarray(train_mask, dtype=bool),
            groups=np.asarray(groups),
        )
        dataset.validate()
        return dataset

    def _raw_epochs(self, data_dir: Path, cache_dir: Path) -> SignalDataset:
        cache = cache_dir / self.name / f"{self.cache_version}-raw-unbaselined-epochs.npz"
        if cache.exists():
            return self._load(cache, quick=False)
        csv_files = sorted(data_dir.rglob("Data_S*_Sess*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No BCI Data_S*_Sess*.csv files found below {data_dir}")
        train_labels = self._label_map(data_dir, "TrainLabels.csv")
        test_labels = self._test_label_map(data_dir)
        epochs: list[np.ndarray] = []
        labels: list[int] = []
        train_mask: list[bool] = []
        groups: list[str] = []
        for path in progress(
            csv_files, total=len(csv_files), desc="BCI epoching", unit="file", leave=False
        ):
            subject = int(path.name.split("_S", 1)[1].split("_", 1)[0])
            is_train = subject not in TEST_SUBJECTS
            frame = pd.read_csv(path, usecols=[*CHANNELS, "FeedBackEvent"])
            markers = np.flatnonzero(frame["FeedBackEvent"].to_numpy() == 1)
            eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64).T
            prefix = path.stem.removeprefix("Data_")
            label_map = train_labels if is_train else test_labels
            for event_number, marker in enumerate(markers, start=1):
                segment = eeg[:, marker - 20:marker + 120]
                if segment.shape[-1] != 140:
                    continue
                event_id = f"{prefix}_FB{event_number:03d}"
                if event_id not in label_map:
                    raise KeyError(f"Missing BCI label for {event_id}")
                epochs.append(segment)
                labels.append(label_map[event_id])
                train_mask.append(is_train)
                groups.append(prefix.split("_Sess", 1)[0])
        dataset = SignalDataset(
            signals=np.stack(epochs), labels=np.asarray(labels, dtype=np.int8), sampling_rate=FS,
            class_names=("bad_feedback", "good_feedback"), primary_metric="balanced_accuracy",
            fixed_train=np.asarray(train_mask, dtype=bool), groups=np.asarray(groups),
        )
        dataset.validate()
        self._save(cache, dataset)
        return dataset

    @staticmethod
    def _label_map(data_dir: Path, filename: str) -> dict[str, int]:
        matches = list(data_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"{filename} was not downloaded below {data_dir}")
        frame = pd.read_csv(matches[0])
        return {str(row.IdFeedBack): int(row.Prediction) for row in frame.itertuples()}

    @staticmethod
    def _test_label_map(data_dir: Path) -> dict[str, int]:
        id_files = list(data_dir.rglob("benchmark.csv"))
        if not id_files:
            id_files = list(data_dir.rglob("SampleSubmission.csv"))
        bundled = Path(__file__).with_name("evaluation_labels.csv")
        downloaded = list(data_dir.rglob("true_labels.csv"))
        label_file = downloaded[0] if downloaded else bundled
        if not id_files or not label_file.exists():
            raise FileNotFoundError(
                "BCI evaluation needs benchmark.csv or SampleSubmission.csv plus "
                "dataset/bci_errp/evaluation_labels.csv"
            )
        ids = pd.read_csv(id_files[0])["IdFeedBack"].astype(str).to_numpy()
        labels = pd.read_csv(label_file, header=None).iloc[:, 0].astype(int).to_numpy()
        if len(ids) != len(labels):
            raise ValueError(f"BCI test ID/label length mismatch: {len(ids)} vs {len(labels)}")
        return dict(zip(ids, labels))

    @staticmethod
    def _save(path: Path, data: SignalDataset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, signals=data.signals, labels=data.labels,
            fixed_train=data.fixed_train, groups=data.groups)

    def _load(self, path: Path, quick: bool) -> SignalDataset:
        with np.load(path, allow_pickle=False) as saved:
            data = SignalDataset(saved["signals"], saved["labels"], FS,
                ("bad_feedback", "good_feedback"), "balanced_accuracy",
                saved["fixed_train"], saved["groups"])
        return self._quick(data) if quick else data

    @staticmethod
    def _quick(data: SignalDataset) -> SignalDataset:
        selected: list[int] = []
        for train_value in (True, False):
            for label in np.unique(data.labels):
                candidates = np.flatnonzero((data.fixed_train == train_value) & (data.labels == label))
                selected.extend(candidates[:64])
        indices = np.asarray(sorted(selected))
        return SignalDataset(data.signals[indices], data.labels[indices], data.sampling_rate,
            data.class_names, data.primary_metric, data.fixed_train[indices], data.groups[indices])


TASK = BCIErrPTask()
