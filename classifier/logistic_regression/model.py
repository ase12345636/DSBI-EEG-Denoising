"""Shared logistic-regression classifier."""

from sklearn.linear_model import LogisticRegression

from classifier.common import sklearn_prediction


class LogisticRegressionClassifier:
    name = "logistic_regression"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        model = LogisticRegression(random_state=seed, max_iter=1000)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = LogisticRegressionClassifier()
