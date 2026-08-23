"""End-to-end download, preprocessing, experiment, and reporting pipeline."""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.downloader import (
    DownloadError,
    describe_source,
    ensure_dataset,
    ensure_icunet_checkpoint,
    resolve_dataset_path,
    sources,
)
from utils.features import stft_mean_power
from utils.progress import progress, progress_write
from utils.registry import (
    CLASSIFIER_MODULES,
    DENOISE_MODULES,
    TASK_MODULES,
    load_classifier,
    load_denoiser,
    load_task,
)
from utils.reporting import checkpoint_results, write_outputs


_RESULT_KEY_COLUMNS = ("task", "method", "classifier", "repeat", "seed")
_RESULT_COLUMNS = (
    "task",
    "method",
    "classifier",
    "repeat",
    "seed",
    "primary_name",
    "primary",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "auc",
)

def _set_seed(seed: int) -> None:
    """Set Python, NumPy, TensorFlow, and PyTorch random states."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ReproductionPipeline:
    def __init__(self, config):
        self.config = config

    def print_registry(self) -> None:
        print("Tasks:       " + ", ".join(TASK_MODULES))
        print("Denoisers:   " + ", ".join(DENOISE_MODULES))
        print("Classifiers: " + ", ".join(CLASSIFIER_MODULES))

    def run(
        self,
        tasks=None,
        methods=None,
        classifiers=None,
        download_only=False,
        skip_download=False,
        force_download=False,
        quick=False,
        dry_run=False,
    ) -> int:
        task_names = tuple(tasks or self.config.tasks)
        method_names = tuple(methods or self.config.methods)
        classifier_names = tuple(classifiers or self.config.classifiers)

        self._validate(task_names, method_names, classifier_names)

        print(f"Tasks: {', '.join(task_names)}")
        print(f"Denoising: {', '.join(method_names)}")
        print(f"Classifiers: {', '.join(classifier_names)}")
        print("Mode: EEG denoising benchmark")

        output_dir = (
            self.config.output_dir / "quick_smoke"
            if quick
            else self.config.output_dir
        )

        if dry_run:
            for name in task_names:
                print("[dry-run] " + describe_source(sources(self.config.root)[name]))
            if "ic_unet" in method_names:
                print(
                    "[dry-run] IC-U-Net uses included "
                    "denoise/ic_unet/weights/BEST_checkpoint.pth.tar"
                )
            print(f"[dry-run] outputs -> {output_dir}")
            print(f"[dry-run] cache -> {self.config.cache_dir}")
            return 0

        if not quick:
            self._prepare_output_directory(output_dir)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            data_paths: dict[str, Path] = {}
            download_total = len(task_names) + int("ic_unet" in method_names)
            with progress(total=download_total, desc="Downloads", unit="item") as bar:
                for task_name in task_names:
                    bar.set_postfix_str(task_name, refresh=False)
                    data_paths[task_name] = (
                        resolve_dataset_path(self.config.root, task_name)
                        if skip_download
                        else ensure_dataset(
                            self.config.root,
                            task_name,
                            force_download,
                        )
                    )
                    bar.update()

                checkpoint = None
                if "ic_unet" in method_names:
                    bar.set_postfix_str("IC-U-Net BEST checkpoint", refresh=False)
                    checkpoint = ensure_icunet_checkpoint(
                        self.config.root,
                        force_download,
                    )
                    bar.update()
        except DownloadError as exc:
            print(f"Download stopped: {exc}", file=sys.stderr)
            return 2

        if download_only:
            print("All requested downloads are ready.")
            return 0

        repeats = 1 if quick else self.config.repeats

        seed_plan_path = output_dir / "repeat_seed_plan.json"
        seed_plan_existed = seed_plan_path.exists()
        if seed_plan_existed:
            try:
                seed_plan = json.loads(seed_plan_path.read_text(encoding="utf-8"))
                master_seed = int(seed_plan["master_seed"])
                print(f"[seed] resumed master seed: {master_seed}")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid repeat seed plan at {seed_plan_path}: {exc}"
                ) from exc
        else:
            master_seed = int(time.time())
            print(f"[seed] new master seed: {master_seed}")

        seed_rng = np.random.default_rng(master_seed)
        repeat_seeds = seed_rng.integers(
            low=0,
            high=2**31 - 1,
            size=repeats,
            dtype=np.int64,
        ).tolist()
        seed_plan_path.write_text(
            json.dumps(
                {
                    "master_seed": master_seed,
                    "repeat_seeds": repeat_seeds,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[seed] repeat seeds: {repeat_seeds}")

        if quick or not seed_plan_existed:
            # A result CSV without its seed plan cannot be resumed safely,
            # because the original time-derived repeat seeds are unknown.
            existing_frame = pd.DataFrame(columns=_RESULT_COLUMNS)
            existing_source = None
        else:
            existing_frame, existing_source = self._load_existing_results(output_dir)

        rows: list[dict] = existing_frame.to_dict(orient="records")
        completed_keys = self._result_keys(existing_frame)
        selected_keys = {
            (
                task_name,
                method_name,
                classifier_name,
                repeat + 1,
                repeat_seeds[repeat],
            )
            for task_name in task_names
            for method_name in method_names
            for classifier_name in classifier_names
            for repeat in range(repeats)
        }
        completed_selected = len(selected_keys & completed_keys)
        pending_selected = len(selected_keys) - completed_selected

        if existing_source is not None:
            print(
                f"[resume] loaded {len(existing_frame)} completed runs "
                f"from {existing_source}"
            )
        elif not quick:
            print("[resume] no compatible run-level CSV found; starting fresh")
        print(
            "[resume] selected experiment runs: "
            f"completed={completed_selected}, "
            f"pending={pending_selected}, total={len(selected_keys)}"
        )

        prepare_total = len(task_names) * len(method_names)
        experiment_total = len(selected_keys)

        with progress(
            total=prepare_total,
            desc="Prepare variants",
            unit="variant",
        ) as prepare_bar, progress(
            total=experiment_total,
            initial=completed_selected,
            desc="Experiments",
            unit="model run",
        ) as experiment_bar:
            for task_name in task_names:
                task = load_task(task_name)
                if hasattr(task, "selection_seed"):
                    task.selection_seed = master_seed

                for method_name in method_names:
                    prepare_bar.set_postfix_str(
                        f"{task_name}/{method_name}", refresh=False
                    )
                    pending_method_keys = {
                        (
                            task_name,
                            method_name,
                            classifier_name,
                            repeat + 1,
                            repeat_seeds[repeat],
                        )
                        for classifier_name in classifier_names
                        for repeat in range(repeats)
                    } - completed_keys
                    if not pending_method_keys:
                        progress_write(
                            f"[resume] skip completed variant {task_name}/{method_name}"
                        )
                        prepare_bar.update()
                        continue

                    denoiser = load_denoiser(method_name)
                    if hasattr(denoiser, "configure"):
                        method_settings = dict(self.config.calibration)
                        if method_name in {"asr", "asr20"}:
                            method_settings.update(self.config.asr)
                        elif method_name == "ica":
                            method_settings.update(self.config.ica)
                        denoiser.configure(method_settings)
                    try:
                        data = task.prepare(
                            data_paths[task_name],
                            self.config.cache_dir,
                            denoiser,
                            checkpoint if method_name == "ic_unet" else None,
                            quick,
                        )
                        data.validate()
                    finally:
                        del denoiser
                        self._cleanup_accelerators()

                    pending_classifier_names = {key[2] for key in pending_method_keys}
                    needs_features = any(
                        load_classifier(name).expects_features
                        for name in pending_classifier_names
                    )
                    fixed_feature_pair = None
                    if task.feature_kind == "xdawn_tangent" and needs_features:
                        fixed_train_index, fixed_test_index = task.split(
                            data, repeat_seeds[0], 0
                        )
                        fixed_feature_pair = task.features(
                            data.signals[fixed_train_index],
                            data.signals[fixed_test_index],
                            data.labels[fixed_train_index],
                            data.labels[fixed_test_index],
                        )

                    full_stft = None
                    if task.feature_kind == "stft_mean_power" and needs_features:
                        feature_cache = self._stft_cache_path(task, method_name)
                        if feature_cache.exists() and not quick:
                            full_stft = np.load(feature_cache, mmap_mode="r")
                            if len(full_stft) != len(data.signals):
                                full_stft = None
                                feature_cache.unlink(missing_ok=True)
                        if full_stft is None:
                            full_stft = stft_mean_power(
                                data.signals,
                                data.sampling_rate,
                                self.config.stft,
                            )
                            if not quick:
                                feature_cache.parent.mkdir(parents=True, exist_ok=True)
                                np.save(feature_cache, full_stft)

                    prepare_bar.update()

                    for repeat in range(repeats):
                        seed = repeat_seeds[repeat]
                        pending_classifiers = [
                            classifier_name
                            for classifier_name in classifier_names
                            if (
                                task_name,
                                method_name,
                                classifier_name,
                                repeat + 1,
                                seed,
                            )
                            not in completed_keys
                        ]
                        if not pending_classifiers:
                            continue

                        _set_seed(seed)
                        cycle_size = getattr(task, "split_cycle_size", 1)
                        split_seed = repeat_seeds[repeat - repeat % cycle_size]
                        train_index, test_index = task.split(data, split_seed, repeat)
                        train_signal = data.signals[train_index]
                        test_signal = data.signals[test_index]
                        y_train = data.labels[train_index]
                        y_test = data.labels[test_index]

                        feature_pair = None
                        balanced_x = None
                        balanced_y = None
                        scaled_feature_pair = None
                        repeat_needs_features = any(
                            load_classifier(name).expects_features
                            for name in pending_classifiers
                        )
                        if repeat_needs_features:
                            if fixed_feature_pair is not None:
                                feature_pair = fixed_feature_pair
                            elif full_stft is not None:
                                feature_pair = (
                                    np.asarray(full_stft[train_index]),
                                    np.asarray(full_stft[test_index]),
                                )
                            else:
                                feature_pair = task.features(
                                    train_signal,
                                    test_signal,
                                    y_train,
                                    y_test,
                                )

                            feature_scale = float(
                                getattr(task, "feature_power_scale", 1.0)
                            )
                            if feature_scale != 1.0:
                                feature_pair = (
                                    feature_pair[0] * feature_scale,
                                    feature_pair[1] * feature_scale,
                                )

                            balanced_x, balanced_y = task.balance_features(
                                feature_pair[0], y_train, seed
                            )

                        for classifier_name in pending_classifiers:
                            classifier = load_classifier(classifier_name)
                            if classifier.expects_features:
                                model_y_train = balanced_y
                                if getattr(classifier, "requires_standardization", False):
                                    if scaled_feature_pair is None:
                                        scaled_feature_pair = task.standardize_features(
                                            balanced_x, feature_pair[1]
                                        )
                                    x_train, x_test = scaled_feature_pair
                                else:
                                    x_train = balanced_x
                                    x_test = feature_pair[1]
                            else:
                                x_train = train_signal
                                model_y_train = y_train
                                x_test = test_signal

                            experiment_bar.set_postfix_str(
                                f"{task_name}/{method_name}/{classifier_name} "
                                f"repeat {repeat + 1}/{repeats}",
                                refresh=False,
                            )
                            prediction = None
                            try:
                                prediction = classifier.fit_predict(
                                    x_train,
                                    model_y_train,
                                    x_test,
                                    seed=seed,
                                    cache_dir=(
                                        self.config.cache_dir
                                        / task_name
                                        / method_name
                                        / classifier_name
                                    ),
                                    config=self.config.eegnet,
                                    quick=quick,
                                    validation_size=float(self.config.validation.get("fraction", 0.20)),
                                    task_name=task_name,
                                )
                                prediction_labels = self._copy_to_cpu(prediction.labels)
                                prediction_scores = self._copy_to_cpu(prediction.scores)
                            finally:
                                prediction = None
                                classifier = None
                                self._cleanup_accelerators()

                            metrics = self._metrics(
                                y_test,
                                prediction_labels,
                                prediction_scores,
                                len(data.class_names),
                            )
                            result_key = (
                                task_name,
                                method_name,
                                classifier_name,
                                repeat + 1,
                                seed,
                            )
                            rows.append(
                                {
                                    "task": task_name,
                                    "method": method_name,
                                    "classifier": classifier_name,
                                    "repeat": repeat + 1,
                                    "seed": seed,
                                    "primary_name": data.primary_metric,
                                    "primary": metrics[data.primary_metric],
                                    **metrics,
                                }
                            )
                            completed_keys.add(result_key)
                            rows = self._normalise_result_frame(
                                pd.DataFrame(rows)
                            ).to_dict(orient="records")
                            checkpoint_results(rows, output_dir)
                            experiment_bar.update()

                    fixed_feature_pair = None
                    full_stft = None
                    data = None
                    self._cleanup_accelerators()

        result_frame = self._normalise_result_frame(pd.DataFrame(rows))
        checkpoint_results(result_frame.to_dict(orient="records"), output_dir)
        manifest = {
            "tasks": list(task_names),
            "methods": list(method_names),
            "classifiers": list(classifier_names),
            "repeats": repeats,
            "master_seed": master_seed,
            "repeat_seeds": repeat_seeds,
            "stft": self.config.stft,
            "eegnet": self.config.eegnet,
            "validation": self.config.validation,
            "asr": self.config.asr,
            "asr20": {**self.config.asr, "cutoff": 20},
            "calibration": self.config.calibration,
            "ica": self.config.ica,
            "statistics": {
                "primary": "two-sided paired Wilcoxon (denoised != raw)",
                "supplementary": "one-sided paired Wilcoxon (denoised < raw)",
                "correction": "Holm within each task and metric",
            },
        }
        progress_write(
            "[report] Writing tables and figures from "
            f"{len(result_frame)} merged model runs"
        )
        write_outputs(result_frame, output_dir, manifest)
        print(f"Finished. Results and figures: {output_dir}")
        return 0

    @staticmethod
    def _prepare_output_directory(output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _stft_cache_path(self, task, method_name: str) -> Path:
        return (
            self.config.cache_dir
            / task.name
            / f"{method_name}-stft.npy"
        )

    @staticmethod
    def _load_existing_results(
        output_dir: Path,
    ) -> tuple[pd.DataFrame, Path | None]:
        for name in ("all_runs.partial.csv", "all_runs.csv"):
            path = output_dir / name
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
                return ReproductionPipeline._normalise_result_frame(frame), path
            except Exception as exc:
                print(f"[resume] ignored invalid {path}: {exc}")
        return pd.DataFrame(columns=_RESULT_COLUMNS), None

    @staticmethod
    def _normalise_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=_RESULT_COLUMNS)
        missing = set(_RESULT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                "Result data are missing required columns: "
                + ", ".join(sorted(missing))
            )
        result = frame.loc[:, _RESULT_COLUMNS].copy()
        for column in ("task", "method", "classifier", "primary_name"):
            result[column] = result[column].astype(str)
        result["repeat"] = pd.to_numeric(result["repeat"], errors="raise").astype(int)
        result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)
        for column in (
            "primary",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "auc",
        ):
            result[column] = pd.to_numeric(result[column], errors="raise")
        result = result.drop_duplicates(
            subset=list(_RESULT_KEY_COLUMNS), keep="last"
        )
        return result.sort_values(
            list(_RESULT_KEY_COLUMNS), kind="stable"
        ).reset_index(drop=True)

    @staticmethod
    def _result_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, int, int]]:
        if frame.empty:
            return set()
        frame = ReproductionPipeline._normalise_result_frame(frame)
        return {
            (
                str(row.task),
                str(row.method),
                str(row.classifier),
                int(row.repeat),
                int(row.seed),
            )
            for row in frame.itertuples(index=False)
        }

    @staticmethod
    def _copy_to_cpu(value) -> np.ndarray:
        if value is None:
            raise ValueError("Classifier returned a missing prediction value")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value).copy()

    @staticmethod
    def _cleanup_accelerators() -> None:
        keras_module = sys.modules.get("keras")
        tensorflow_module = sys.modules.get("tensorflow")
        try:
            if keras_module is not None:
                keras_module.backend.clear_session()
            elif tensorflow_module is not None:
                tensorflow_module.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        torch_module = sys.modules.get("torch")
        try:
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _metrics(y_true, y_pred, scores, class_count: int) -> dict[str, float]:
        average = "binary" if class_count == 2 else "macro"
        if class_count == 2:
            auc_scores = scores[:, 1] if np.ndim(scores) == 2 else scores
            auc = roc_auc_score(y_true, auc_scores)
        else:
            score_array = np.asarray(scores)
            if score_array.ndim != 2 or score_array.shape[1] != class_count:
                raise ValueError("Multiclass AUC needs one score column per class")
            auc = float(np.mean([
                roc_auc_score((np.asarray(y_true) == class_index).astype(int),
                              score_array[:, class_index])
                for class_index in range(class_count)
            ]))
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "auc": float(auc),
        }

    @staticmethod
    def _validate(tasks, methods, classifiers) -> None:
        for selected, registry, kind in (
            (tasks, TASK_MODULES, "task"),
            (methods, DENOISE_MODULES, "denoiser"),
            (classifiers, CLASSIFIER_MODULES, "classifier"),
        ):
            unknown = sorted(set(selected) - set(registry))
            if unknown:
                raise ValueError(f"Unknown {kind}(s): {', '.join(unknown)}")
