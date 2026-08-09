"""Helpers shared by classifier modules."""

from __future__ import annotations

import numpy as np

from utils.contracts import Prediction


def sklearn_prediction(model, x_test: np.ndarray) -> Prediction:
    labels = np.asarray(model.predict(x_test))
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_test))
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_test))
    else:
        scores = labels
    return Prediction(labels=labels, scores=scores)


def _gpu_to_numpy(value) -> np.ndarray:
    """Copy a cuML/cuDF/CuPy result to a plain NumPy array."""
    if hasattr(value, "to_pandas"):
        return value.to_pandas().to_numpy()
    if hasattr(value, "to_numpy"):
        try:
            return np.asarray(value.to_numpy())
        except TypeError:
            pass
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


def fit_cuml_classifier(model, x_train, y_train, x_test) -> Prediction:
    """Match the author's BCI notebook: train cuML models from cuDF inputs."""
    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError(
            "BCI reproduction uses RAPIDS cuDF/cuML. Install requirements.txt "
            "on Linux with an NVIDIA GPU."
        ) from exc

    train_frame = cudf.DataFrame(np.asarray(x_train))
    train_labels = cudf.Series(np.asarray(y_train))
    model.fit(train_frame, train_labels)

    labels = _gpu_to_numpy(model.predict(x_test))
    if hasattr(model, "predict_proba"):
        scores = _gpu_to_numpy(model.predict_proba(x_test))
    elif hasattr(model, "decision_function"):
        scores = _gpu_to_numpy(model.decision_function(x_test))
    else:
        scores = labels
    return Prediction(labels=labels, scores=scores)
