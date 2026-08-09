"""Logistic regression; cuML for the author-provided BCI task."""

from sklearn.linear_model import LogisticRegression

from classifier.common import fit_cuml_classifier, sklearn_prediction


class LogisticRegressionClassifier:
    name = "logistic_regression"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, task_name=None, **_):
        try:
            from cuml.linear_model import LogisticRegression as cuLR
        except ImportError as exc:
            raise RuntimeError("cuML is required.") from exc
        return fit_cuml_classifier(cuLR(), x_train, y_train, x_test)


CLASSIFIER = LogisticRegressionClassifier()
