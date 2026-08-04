"""Random forest classifier."""

from sklearn.ensemble import RandomForestClassifier

from classifier.common import sklearn_prediction


class RandomForestModel:
    name = "random_forest"
    expects_features = True
    requires_standardization = False

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        model = RandomForestClassifier(random_state=seed, n_jobs=-1)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = RandomForestModel()
