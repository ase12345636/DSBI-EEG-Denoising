"""Consistent channels-last EEGNet and deterministic training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.contracts import Prediction


def eegnet(
    nb_classes: int,
    channels: int,
    samples: int,
    dropout_rate: float = 0.5,
    kernel_length: int = 64,
    f1: int = 8,
    depth_multiplier: int = 2,
    f2: int = 16,
    norm_rate: float = 0.25,
):
    """Build the canonical EEGNet block with one data format throughout."""
    from tensorflow.keras.constraints import max_norm
    from tensorflow.keras.layers import (
        Activation,
        AveragePooling2D,
        BatchNormalization,
        Conv2D,
        Dense,
        DepthwiseConv2D,
        Dropout,
        Flatten,
        Input,
        SeparableConv2D,
    )
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(channels, samples, 1))
    block1 = Conv2D(
        f1,
        (1, kernel_length),
        padding="same",
        use_bias=False,
        data_format="channels_last",
    )(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        depth_multiplier=depth_multiplier,
        depthwise_constraint=max_norm(1.0),
        data_format="channels_last",
    )(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4), data_format="channels_last")(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = SeparableConv2D(
        f2,
        (1, 16),
        use_bias=False,
        padding="same",
        data_format="channels_last",
    )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8), data_format="channels_last")(block2)
    block2 = Dropout(dropout_rate)(block2)

    flattened = Flatten(name="flatten")(block2)
    dense = Dense(
        nb_classes,
        name="dense",
        kernel_constraint=max_norm(norm_rate),
    )(flattened)
    output = Activation("softmax", name="softmax")(dense)
    return Model(inputs=input_layer, outputs=output)


class EEGNetModel:
    name = "eegnet"
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

        nb_classes = int(np.max(y_train)) + 1
        model = eegnet(
            nb_classes,
            x_train.shape[1],
            x_train.shape[2],
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = cache_dir / f"eegnet-{seed}.weights.h5"
        epochs = 2 if quick else int(config.get("epochs", 150))
        patience = 2 if quick else int(config.get("patience", 10))
        # Match DL-classifer.ipynb: checkpoint by val_loss, but early stop by
        # val_accuracy.  The notebook creates ReduceLROnPlateau but does not
        # actually pass it to model.fit(), so it is intentionally absent here.
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


CLASSIFIER = EEGNetModel()
