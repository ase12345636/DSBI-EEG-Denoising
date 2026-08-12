"""Vision Transformer adapted to multichannel EEG time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.contracts import Prediction


def vit(
    nb_classes: int,
    channels: int,
    samples: int,
    patch_size: int = 16,
    projection_dim: int = 64,
    transformer_layers: int = 4,
    num_heads: int = 4,
    mlp_dim: int = 128,
    dropout: float = 0.1,
):
    """Tokenize temporal patches containing all EEG channels."""
    from tensorflow.keras.layers import (
        Add,
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        Layer,
        LayerNormalization,
        MultiHeadAttention,
        Permute,
        Reshape,
    )
    from tensorflow.keras.models import Model

    class PositionEmbedding(Layer):
        def build(self, input_shape):
            self.embeddings = self.add_weight(
                name="embeddings",
                shape=(1, int(input_shape[1]), int(input_shape[2])),
                initializer="uniform",
                trainable=True,
            )

        def call(self, inputs):
            return inputs + self.embeddings

    num_patches = (samples + patch_size - 1) // patch_size

    inputs = Input(shape=(channels, samples, 1))
    x = Permute((2, 1, 3))(inputs)
    x = Reshape((samples, channels))(x)
    x = Conv1D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="temporal_patches",
    )(x)
    x = BatchNormalization(name="patch_normalization")(x)
    x = PositionEmbedding(name="position_embedding")(x)

    for _ in range(transformer_layers):
        normalized = LayerNormalization(epsilon=1e-6)(x)
        attended = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout,
        )(normalized, normalized)
        x = Add()([x, Dropout(dropout)(attended)])

        normalized = LayerNormalization(epsilon=1e-6)(x)
        projected = Dense(mlp_dim, activation="gelu")(normalized)
        projected = Dropout(dropout)(projected)
        projected = Dense(projection_dim)(projected)
        x = Add()([x, Dropout(dropout)(projected)])

    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="EEG_ViT")


class ViTModel:
    name = "vit"
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

        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
        input_mean = float(np.mean(x_train, dtype=np.float64))
        input_std = float(np.std(x_train, dtype=np.float64))
        x_train = ((x_train - input_mean) / input_std)[..., np.newaxis]
        x_test = ((x_test - input_mean) / input_std)[..., np.newaxis]
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

        model = vit(
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
        checkpoint = cache_dir / f"vit-{seed}.weights.h5"
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


CLASSIFIER = ViTModel()
