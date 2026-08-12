"""MobileNet adapted to multichannel EEG time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.contracts import Prediction


def mobilenet(
    nb_classes: int,
    channels: int,
    samples: int,
    width_multiplier: float = 1.0,
    dropout: float = 0.2,
):
    """Use MobileNet depthwise-separable blocks along the time axis."""
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        DepthwiseConv1D,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        Permute,
        ReLU,
        Reshape,
    )
    from tensorflow.keras.models import Model

    def filters(value: int) -> int:
        return max(8, int(round(value * width_multiplier)))

    def depthwise_separable(x, output_filters: int, stride: int):
        x = DepthwiseConv1D(3, strides=stride, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6.0)(x)
        x = Conv1D(filters(output_filters), 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        return ReLU(max_value=6.0)(x)

    inputs = Input(shape=(channels, samples, 1))
    x = Permute((2, 1, 3))(inputs)
    x = Reshape((samples, channels))(x)
    x = Conv1D(filters(32), 7, strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    for output_filters, stride in (
        (64, 1), (128, 2), (128, 1), (256, 2),
        (256, 1), (512, 2), (512, 1),
    ):
        x = depthwise_separable(x, output_filters, stride)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="EEG_MobileNet")


class MobileNetModel:
    name = "mobilenet"
    expects_features = False
    requires_standardization = False

    def fit_predict(
        self,
        x_train,
        y_train,
        x_test,
        seed: int,
        cache_dir: Path,
        config: dict,
        quick: bool = False,
        validation_size: float = 0.2,
        **_,
    ) -> Prediction:
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras import backend as keras_backend
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        x_train = np.asarray(x_train, dtype=np.float32)[..., np.newaxis]
        x_test = np.asarray(x_test, dtype=np.float32)[..., np.newaxis]
        y_train = np.asarray(y_train, dtype=np.int64)

        train_x, valid_x, train_y, valid_y = train_test_split(
            x_train,
            y_train,
            test_size=validation_size,
            stratify=y_train,
            random_state=seed,
        )
        classes = np.unique(train_y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_y,
        )
        class_weights = {
            int(label): float(weight)
            for label, weight in zip(classes, weights)
        }

        model = mobilenet(
            int(np.max(y_train)) + 1,
            x_train.shape[1],
            x_train.shape[2],
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = cache_dir / f"mobilenet-{seed}.weights.h5"
        epochs = 2 if quick else int(config.get("epochs", 150))
        patience = 2 if quick else int(config.get("patience", 10))
        callbacks = [
            ModelCheckpoint(
                checkpoint,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                verbose=0,
            ),
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            batch_size=int(config.get("batch_size", 32)),
            epochs=epochs,
            verbose=int(config.get("verbose", 0)),
            callbacks=callbacks,
            class_weight=class_weights,
        )
        if checkpoint.exists():
            model.load_weights(checkpoint)

        scores = np.asarray(model.predict(x_test, verbose=0))
        labels = np.argmax(scores, axis=1)
        try:
            checkpoint.unlink(missing_ok=True)
        finally:
            keras_backend.clear_session()
        return Prediction(labels=labels, scores=scores)


CLASSIFIER = MobileNetModel()
