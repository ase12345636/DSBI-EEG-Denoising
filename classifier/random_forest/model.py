"""Random forest; cuML for the author-provided BCI task."""

from sklearn.ensemble import RandomForestClassifier

from classifier.common import fit_cuml_classifier, sklearn_prediction


class RandomForestModel:
    name = "random_forest"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, task_name=None, **_):
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
        except ImportError as exc:
            raise RuntimeError("cuML is required") from exc
        return fit_cuml_classifier(cuRF(), x_train, y_train, x_test)

CLASSIFIER = RandomForestModel()
