"""LightGBM classifier using stable seeded defaults."""

from classifier.common import sklearn_prediction


class LightGBMModel:
    name = "lightgbm"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is missing; install requirements.txt") from exc
        model = LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=-1)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = LightGBMModel()
