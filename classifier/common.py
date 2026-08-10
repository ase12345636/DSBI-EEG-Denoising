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


def fit_cuml_classifier(
    model,
    x_train,
    y_train,
    x_test,
    *,
    score_method: str = "auto",
) -> Prediction:
    """Fit a RAPIDS cuML classifier from cuDF inputs and return CPU results."""
    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError(
            "RAPIDS cuDF/cuML is required for LR/SVM/RF in this experiment."
        ) from exc

    train_frame = cudf.DataFrame(np.asarray(x_train))
    train_labels = cudf.Series(np.asarray(y_train))
    model.fit(train_frame, train_labels)

    labels = _gpu_to_numpy(model.predict(x_test)).reshape(-1)

    if score_method == "decision_function":
        scores = _gpu_to_numpy(model.decision_function(x_test))
    elif score_method == "predict_proba":
        scores = _gpu_to_numpy(model.predict_proba(x_test))
    elif score_method == "auto":
        # For ordinary probabilistic cuML estimators (LR/RF), use probabilities.
        # SVM explicitly requests decision_function because cuML 25.08's
        # probability=True calibration path is unstable on this workload.
        try:
            scores = _gpu_to_numpy(model.predict_proba(x_test))
        except (AttributeError, RuntimeError, ValueError):
            if hasattr(model, "decision_function"):
                scores = _gpu_to_numpy(model.decision_function(x_test))
            else:
                scores = labels
    else:
        raise ValueError(f"Unknown cuML score_method: {score_method}")

    return Prediction(labels=labels, scores=np.asarray(scores))
