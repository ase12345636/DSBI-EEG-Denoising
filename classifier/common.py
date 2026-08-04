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
