"""Convolutional Neural Network model for fake image detection."""

from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers


class FakeImageCNN:
    """CNN model for detecting fake/tampered images."""
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        num_classes: int = 2,
        learning_rate: float = 0.0005,
    ):
        """Initialize the CNN model.

        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (2 for real/fake)
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model: Optional[keras.Model] = None

    def build_model(self) -> "keras.Model":
        """Build the CNN architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='valid',
                activation='relu'
            ),
            layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='valid',
                activation='relu'
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        optimizer = optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=0.9,
            epsilon=1e-08
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return self.model

    def get_callbacks(
        self,
        patience: int = 2,
        monitor: str = 'val_accuracy',
        min_delta: float = 0.0
    ) -> List[callbacks.Callback]:
        """Get training callbacks.

        Args:
            patience: Number of epochs with no improvement after which training stops
            monitor: Metric to monitor for early stopping
            min_delta: Minimum change to qualify as improvement

        Returns:
            List of Keras callbacks
        """
        return [
            callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=min_delta,
                patience=patience,
                verbose=1,
                mode='max' if 'acc' in monitor else 'min'
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=patience,
                min_lr=1e-7,
                verbose=1
            )
        ]

    def train(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_val: tf.Tensor,
        y_val: tf.Tensor,
        epochs: int = 30,
        batch_size: int = 100,
        callbacks_list: Optional[List[callbacks.Callback]] = None
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            x_train: Training images
            y_train: Training labels (one-hot encoded)
            x_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks_list: List of callbacks to use during training

        Returns:
            Training history object
        """
        if self.model is None:
            self.build_model()

        if callbacks_list is None:
            callbacks_list = self.get_callbacks()

        assert self.model is not None
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )

        return history

    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """Make predictions on input data.

        Args:
            x: Input images

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError(
                "Model must be built and trained before making predictions"
            )

        result = self.model.predict(x)
        return tf.convert_to_tensor(result)

    def evaluate(self, x_test: tf.Tensor, y_test: tf.Tensor) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            x_test: Test images
            y_test: Test labels (one-hot encoded)

        Returns:
            Dictionary containing loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")

        result = self.model.evaluate(x_test, y_test, verbose=0)
        if isinstance(result, list):
            loss, accuracy = result[0], result[1]
        else:
            loss, accuracy = result, 0.0
        return {"loss": float(loss), "accuracy": float(accuracy)}

    def save_model(self, filepath: str) -> None:
        """Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
