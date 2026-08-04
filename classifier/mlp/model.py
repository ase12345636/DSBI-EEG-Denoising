"""Shared MLP settings recovered from the BCI classifier notebook."""

from sklearn.neural_network import MLPClassifier

from classifier.common import sklearn_prediction


class MLPModel:
    name = "mlp"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        model = MLPClassifier(
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = MLPModel()
