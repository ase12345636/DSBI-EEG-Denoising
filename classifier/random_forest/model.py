"""GPU random forest using RAPIDS cuML for all downstream tasks."""

from classifier.common import fit_cuml_classifier


class RandomForestModel:
    name = "random_forest"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
        except ImportError as exc:
            raise RuntimeError("cuML is required for random forest") from exc

        return fit_cuml_classifier(cuRF(), x_train, y_train, x_test)


CLASSIFIER = RandomForestModel()
