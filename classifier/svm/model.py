"""Shared linear probability SVM recovered from the BCI source notebook."""

from sklearn.svm import SVC

from classifier.common import sklearn_prediction


class SVMClassifier:
    name = "svm"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, **_):
        model = SVC(kernel="linear", probability=True, random_state=seed)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = SVMClassifier()
