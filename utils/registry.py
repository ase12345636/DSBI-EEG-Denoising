"""Central registries; adding one module here makes it available to main.py."""

from __future__ import annotations

from importlib import import_module
from typing import Any


TASK_MODULES = {
    "bci_errp": "dataset.bci_errp.task",
    "seizure_detection": "dataset.seizure_detection.task",
    "attention_state": "dataset.attention_state.task",
}

DENOISE_MODULES = {
    "raw": "denoise.raw.method",
    "bandpass": "denoise.bandpass.method",
    "asr": "denoise.asr.method",
    "ic_unet": "denoise.ic_unet.method",
    "ica": "denoise.ica.method",
}

CLASSIFIER_MODULES = {
    "logistic_regression": "classifier.logistic_regression.model",
    "svm": "classifier.svm.model",
    "random_forest": "classifier.random_forest.model",
    "lightgbm": "classifier.lightgbm.model",
    "mlp": "classifier.mlp.model",
    "eegnet": "classifier.eegnet.model",
}


def _load(registry: dict[str, str], name: str, attribute: str) -> Any:
    try:
        module_name = registry[name]
    except KeyError as exc:
        raise KeyError(f"Unknown {attribute.lower()} {name!r}; choices: {', '.join(registry)}") from exc
    return getattr(import_module(module_name), attribute)


def load_task(name: str) -> Any:
    return _load(TASK_MODULES, name, "TASK")


def load_denoiser(name: str) -> Any:
    return _load(DENOISE_MODULES, name, "DENOISER")


def load_classifier(name: str) -> Any:
    return _load(CLASSIFIER_MODULES, name, "CLASSIFIER")
