"""BCI ErrP task reproduced from the author's BCI-task source."""

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
    feature_kind = "xdawn_tangent"
    validation_size = 0.25

    def prepare(self, data_dir: Path, cache_dir: Path, denoiser, checkpoint_path: Path | None,
                quick: bool) -> SignalDataset:
        cache = cache_dir / self.name / f"{denoiser.name}.npz"
        if cache.exists() and not quick:
            data = self._load(cache)
            if data.sample_ids is None:
                data.sample_ids = self._event_ids(data_dir)
            data.validate()
            return data

        if denoiser.name == "ica":
            data = self._ica_epochs(data_dir, denoiser, quick)
            if not quick:
                self._save(cache, data)
                denoiser.save_reports(cache.with_suffix(".components.json"))
            return data

        if denoiser.name == "asr":
            data = self._asr_epochs(data_dir, denoiser, quick)
            if not quick:
                self._save(cache, data)
            return data

        raw = self._raw_epochs(data_dir, quick)
        unbaselined = raw.signals

        # This ordering follows preprocessing_data.ipynb, not the alternate
        # preprocessing_data_copy.ipynb: epoch first, then method, then baseline.
        if denoiser.name == "bandpass":
            signals = denoiser.transform(unbaselined, FS, task_name=self.name)
            signals = self._baseline(signals)
        else:
            baseline_corrected = self._baseline(unbaselined)
            if denoiser.name == "raw":
                signals = baseline_corrected
            elif denoiser.name == "ic_unet":
                if checkpoint_path is None:
                    raise ValueError("IC-U-Net checkpoint path is required")
                signals = denoiser.transform(
                    baseline_corrected,
                    FS,
                    checkpoint_path=checkpoint_path,
                    task_name=self.name,
                )
            else:
                raise ValueError(f"Unsupported BCI denoiser: {denoiser.name}")

        data = SignalDataset(
            signals=np.asarray(signals, dtype=np.float32),
            labels=raw.labels,
            sampling_rate=FS,
            class_names=("bad_feedback", "good_feedback"),
            primary_metric="balanced_accuracy",
            fixed_train=raw.fixed_train,
            sample_ids=raw.sample_ids,
        )
        data.validate()
        if not quick:
            self._save(cache, data)
        return data

    @staticmethod
    def split(data: SignalDataset, seed: int):
        del seed
        return np.flatnonzero(data.fixed_train), np.flatnonzero(~data.fixed_train)

    @staticmethod
    def features(train, test, y_train, y_test):
        del y_test
        return bci_xdawn_tangent(train, test, y_train)

    @staticmethod
    def standardize_features(x_train, x_test):
        # The author's BCI ML notebook feeds tangent-space features directly
        # to the classifiers.
        return np.asarray(x_train), np.asarray(x_test)

    @staticmethod
    def balance_features(x_train, y_train, seed: int):
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=seed).fit_resample(x_train, y_train)

    @staticmethod
    def _baseline(signals: np.ndarray) -> np.ndarray:
        values = np.asarray(signals, dtype=np.float64).copy()
        values -= np.mean(values[:, :, :20], axis=2, keepdims=True)
        return values

    def _raw_epochs(self, data_dir: Path, quick: bool) -> SignalDataset:
        files = sorted(data_dir.rglob("Data_S*_Sess*.csv"))
        if not files:
            raise FileNotFoundError(f"No BCI session CSV files found below {data_dir}")
        if quick:
            train_file = next(p for p in files if self._subject(p) not in TEST_SUBJECTS)
            test_file = next(p for p in files if self._subject(p) in TEST_SUBJECTS)
            files = [train_file, test_file]

        train_labels = self._label_map(data_dir, "TrainLabels.csv")
        test_labels = self._test_label_map(data_dir)
        epochs, labels, train_mask, sample_ids = [], [], [], []

        for path in progress(files, total=len(files), desc="BCI epoching", unit="file", leave=False):
            subject = self._subject(path)
            is_train = subject not in TEST_SUBJECTS
            frame = pd.read_csv(path, usecols=[*CHANNELS, "FeedBackEvent"])
            markers = np.flatnonzero(frame["FeedBackEvent"].to_numpy() == 1)
            eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64).T
            label_map = train_labels if is_train else test_labels
            prefix = path.stem.removeprefix("Data_")

            for event_number, marker in enumerate(markers, start=1):
                epoch = eeg[:, marker - 20 : marker + 120]
                if epoch.shape[-1] != 140:
                    continue
                event_id = f"{prefix}_FB{event_number:03d}"
                epochs.append(epoch)
                labels.append(label_map[event_id])
                train_mask.append(is_train)
                sample_ids.append(event_id)

        return SignalDataset(
            np.stack(epochs),
            np.asarray(labels, dtype=np.int8),
            FS,
            ("bad_feedback", "good_feedback"),
            "balanced_accuracy",
            fixed_train=np.asarray(train_mask, dtype=bool),
            sample_ids=np.asarray(sample_ids),
        )

    def _ica_epochs(self, data_dir: Path, denoiser, quick: bool) -> SignalDataset:
        """ICA is the one added method and must be fitted on continuous EEG."""
        files = sorted(data_dir.rglob("Data_S*_Sess*.csv"))
        if not files:
            raise FileNotFoundError(f"No BCI session CSV files found below {data_dir}")
        if quick:
            train_file = next(p for p in files if self._subject(p) not in TEST_SUBJECTS)
            test_file = next(p for p in files if self._subject(p) in TEST_SUBJECTS)
            files = [train_file, test_file]

        train_labels = self._label_map(data_dir, "TrainLabels.csv")
        test_labels = self._test_label_map(data_dir)
        epochs, labels, train_mask, sample_ids = [], [], [], []

        for path in progress(files, total=len(files), desc="BCI ICA", unit="file", leave=False):
            subject = self._subject(path)
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
            label_map = train_labels if is_train else test_labels
            prefix = path.stem.removeprefix("Data_")

            for event_number, marker in enumerate(markers, start=1):
                epoch = cleaned[:, marker - 20 : marker + 120]
                if epoch.shape[-1] != 140:
                    continue
                epoch = epoch - np.mean(epoch[:, :20], axis=1, keepdims=True)
                event_id = f"{prefix}_FB{event_number:03d}"
                epochs.append(epoch)
                labels.append(label_map[event_id])
                train_mask.append(is_train)
                sample_ids.append(event_id)

        return SignalDataset(
            np.stack(epochs).astype(np.float32),
            np.asarray(labels, dtype=np.int8),
            FS,
            ("bad_feedback", "good_feedback"),
            "balanced_accuracy",
            fixed_train=np.asarray(train_mask, dtype=bool),
            sample_ids=np.asarray(sample_ids),
        )

    def _asr_epochs(self, data_dir: Path, denoiser, quick: bool) -> SignalDataset:
        """Apply ASR to each continuous session before event epoching."""
        files = sorted(data_dir.rglob("Data_S*_Sess*.csv"))
        if not files:
            raise FileNotFoundError(f"No BCI session CSV files found below {data_dir}")
        if quick:
            train_file = next(p for p in files if self._subject(p) not in TEST_SUBJECTS)
            test_file = next(p for p in files if self._subject(p) in TEST_SUBJECTS)
            files = [train_file, test_file]

        train_labels = self._label_map(data_dir, "TrainLabels.csv")
        test_labels = self._test_label_map(data_dir)
        epochs, labels, train_mask, sample_ids = [], [], [], []

        for path in progress(files, total=len(files), desc="BCI ASR", unit="file", leave=False):
            subject = self._subject(path)
            is_train = subject not in TEST_SUBJECTS
            frame = pd.read_csv(path, usecols=[*CHANNELS, "FeedBackEvent"])
            markers = np.flatnonzero(frame["FeedBackEvent"].to_numpy() == 1)
            eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64).T
            cleaned = denoiser.transform_recording(eeg, FS)
            label_map = train_labels if is_train else test_labels
            prefix = path.stem.removeprefix("Data_")

            for event_number, marker in enumerate(markers, start=1):
                epoch = cleaned[:, marker - 20 : marker + 120]
                if epoch.shape[-1] != 140:
                    continue
                epoch = epoch - np.mean(epoch[:, :20], axis=1, keepdims=True)
                event_id = f"{prefix}_FB{event_number:03d}"
                epochs.append(epoch)
                labels.append(label_map[event_id])
                train_mask.append(is_train)
                sample_ids.append(event_id)

        return SignalDataset(
            np.stack(epochs).astype(np.float32),
            np.asarray(labels, dtype=np.int8),
            FS,
            ("bad_feedback", "good_feedback"),
            "balanced_accuracy",
            fixed_train=np.asarray(train_mask, dtype=bool),
            sample_ids=np.asarray(sample_ids),
        )

    def _event_ids(self, data_dir: Path) -> np.ndarray:
        """Reconstruct IDs for caches created before sample manifests existed.

        Both official label tables contain exactly the retained event IDs. Their
        lexical order is the same session/event order used by ``_raw_epochs``,
        so migration does not need to reread every large EEG CSV.
        """
        train_ids = self._label_map(data_dir, "TrainLabels.csv")
        test_ids = self._test_label_map(data_dir)
        return np.asarray(sorted([*train_ids, *test_ids]))

    @staticmethod
    def _subject(path: Path) -> int:
        return int(path.name.split("_S", 1)[1].split("_", 1)[0])

    @staticmethod
    def _label_map(data_dir: Path, filename: str) -> dict[str, int]:
        path = next(iter(data_dir.rglob(filename)), None)
        if path is None:
            raise FileNotFoundError(f"{filename} not found below {data_dir}")
        frame = pd.read_csv(path)
        return {str(row.IdFeedBack): int(row.Prediction) for row in frame.itertuples()}

    @staticmethod
    def _test_label_map(data_dir: Path) -> dict[str, int]:
        id_file = next(iter(data_dir.rglob("benchmark.csv")), None)
        if id_file is None:
            id_file = next(iter(data_dir.rglob("SampleSubmission.csv")), None)
        label_file = next(iter(data_dir.rglob("true_labels.csv")), None)
        if label_file is None:
            label_file = Path(__file__).with_name("evaluation_labels.csv")
        if id_file is None or not label_file.exists():
            raise FileNotFoundError("BCI test IDs or evaluation labels are missing")
        ids = pd.read_csv(id_file)["IdFeedBack"].astype(str).to_numpy()
        values = pd.read_csv(label_file, header=None).iloc[:, 0].astype(int).to_numpy()
        if len(ids) != len(values):
            raise ValueError("BCI test ID/label length mismatch")
        return dict(zip(ids, values))

    @staticmethod
    def _save(path: Path, data: SignalDataset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signals=data.signals,
            labels=data.labels,
            fixed_train=data.fixed_train,
            sample_ids=data.sample_ids,
        )

    @staticmethod
    def _load(path: Path) -> SignalDataset:
        with np.load(path, allow_pickle=False) as saved:
            return SignalDataset(
                saved["signals"],
                saved["labels"],
                FS,
                ("bad_feedback", "good_feedback"),
                "balanced_accuracy",
                fixed_train=saved["fixed_train"],
                sample_ids=saved["sample_ids"] if "sample_ids" in saved.files else None,
            )


TASK = BCIErrPTask()
