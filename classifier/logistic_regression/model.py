"""GPU logistic regression using RAPIDS cuML for all downstream tasks."""

from classifier.common import fit_cuml_classifier


class LogisticRegressionClassifier:
    name = "logistic_regression"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        try:
            from cuml.linear_model import LogisticRegression as cuLR
        except ImportError as exc:
            raise RuntimeError("cuML is required for logistic regression") from exc

        return fit_cuml_classifier(cuLR(), x_train, y_train, x_test)


CLASSIFIER = LogisticRegressionClassifier()
