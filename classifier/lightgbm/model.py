"""LightGBM classifier."""

from classifier.common import sklearn_prediction


class LightGBMModel:
    name = "lightgbm"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, task_name=None, **_):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is missing; install requirements.txt") from exc

        # The author's BCI notebook constructs LGBMClassifier() with defaults.
        model = (
            LGBMClassifier()
            if task_name == "bci_errp"
            else LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=-1)
        )
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = LightGBMModel()
