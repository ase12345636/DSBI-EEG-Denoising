"""Linear GPU SVM using RAPIDS cuML for all downstream tasks."""

from classifier.common import fit_cuml_classifier


class SVMClassifier:
    name = "svm"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        try:
            from cuml.svm import SVC as cuSVC
        except ImportError as exc:
            raise RuntimeError("cuML is required for SVM") from exc
        model = cuSVC(
            kernel="linear",
            probability=True,
        )
        return fit_cuml_classifier(
            model,
            x_train,
            y_train,
            x_test,
            score_method="decision_function",
        )


CLASSIFIER = SVMClassifier()
