"""Random forest; cuML for the author-provided BCI task."""

from sklearn.ensemble import RandomForestClassifier

from classifier.common import fit_cuml_classifier, sklearn_prediction


class RandomForestModel:
    name = "random_forest"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, task_name=None, **_):
        if task_name == "bci_errp":
            try:
                from cuml.ensemble import RandomForestClassifier as cuRF
            except ImportError as exc:
                raise RuntimeError("cuML is required for the BCI reproduction") from exc
            return fit_cuml_classifier(cuRF(), x_train, y_train, x_test)

        model = RandomForestClassifier(random_state=seed, n_jobs=-1)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = RandomForestModel()
