"""Linear probability SVM; cuML for the author-provided BCI task."""

from sklearn.svm import SVC

from classifier.common import fit_cuml_classifier, sklearn_prediction


class SVMClassifier:
    name = "svm"
    expects_features = True
    requires_standardization = True

    def fit_predict(self, x_train, y_train, x_test, seed: int, task_name=None, **_):
        if task_name == "bci_errp":
            try:
                from cuml.svm import SVC as cuSVC
            except ImportError as exc:
                raise RuntimeError("cuML is required for the BCI reproduction") from exc
            # Exact model definition used in ML-classifier.ipynb.
            return fit_cuml_classifier(
                cuSVC(kernel="linear", probability=True),
                x_train,
                y_train,
                x_test,
            )

        model = SVC(kernel="linear", probability=True, random_state=seed)
        model.fit(x_train, y_train)
        return sklearn_prediction(model, x_test)


CLASSIFIER = SVMClassifier()
