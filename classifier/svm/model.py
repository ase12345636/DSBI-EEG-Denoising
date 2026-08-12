"""GPU linear SVM using RAPIDS cuML LinearSVC."""

import numpy as np
from cuml.svm import LinearSVC

from utils.contracts import Prediction


class SVMClassifier:
    name = "svm"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        del seed

        # Keep cuML LinearSVC defaults.
        # probability=True is required because the pipeline uses predict_proba for AUC.
        model = LinearSVC(
            probability=True,
            output_type="numpy",
        )

        model.fit(
            np.asarray(x_train, dtype=np.float32),
            np.asarray(y_train),
        )

        x_test = np.asarray(x_test, dtype=np.float32)

        return Prediction(
            labels=np.asarray(model.predict(x_test)),
            scores=np.asarray(model.predict_proba(x_test)),
        )


CLASSIFIER = SVMClassifier()
