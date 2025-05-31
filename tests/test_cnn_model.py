"""Tests for CNN model."""

import numpy as np
import pytest
import tensorflow as tf

from fake_image_detector.models.cnn import FakeImageCNN


class TestFakeImageCNN:
    """Test cases for CNN model."""

    def test_init_default_params(self):
        """Test CNN model initialization with default parameters."""
        model = FakeImageCNN()
        assert model.input_shape == (128, 128, 3)
        assert model.num_classes == 2
        assert model.learning_rate == 0.0005
        assert model.model is None

    def test_init_custom_params(self):
        """Test CNN model initialization with custom parameters."""
        model = FakeImageCNN(
            input_shape=(256, 256, 3), num_classes=3, learning_rate=0.001
        )
        assert model.input_shape == (256, 256, 3)
        assert model.num_classes == 3
        assert model.learning_rate == 0.001

    def test_build_model(self, cnn_model):
        """Test building the CNN model."""
        model = cnn_model.build_model()
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 128, 128, 3)
        assert model.output_shape == (None, 2)
        assert cnn_model.model is not None

    def test_get_callbacks(self, cnn_model):
        """Test getting training callbacks."""
        callbacks = cnn_model.get_callbacks()
        assert len(callbacks) == 2
        assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)
        assert isinstance(callbacks[1], tf.keras.callbacks.ReduceLROnPlateau)

    def test_predict_without_model(self, cnn_model):
        """Test prediction without building model first."""
        x = np.random.random((1, 128, 128, 3))
        with pytest.raises(ValueError, match="Model must be built and trained"):
            cnn_model.predict(x)

    def test_evaluate_without_model(self, cnn_model):
        """Test evaluation without building model first."""
        x = np.random.random((1, 128, 128, 3))
        y = np.array([[1, 0]])
        with pytest.raises(ValueError, match="Model must be built and trained"):
            cnn_model.evaluate(x, y)

    def test_save_model_without_building(self, cnn_model, temp_dir):
        """Test saving model without building it first."""
        model_path = temp_dir / "model.h5"
        with pytest.raises(ValueError, match="Model must be built before saving"):
            cnn_model.save_model(str(model_path))

    def test_model_workflow(self, cnn_model, temp_dir):
        """Test complete model workflow: build, train, predict, save, load."""
        x_train = np.random.random((10, 128, 128, 3))
        y_train = np.random.randint(0, 2, (10, 2))
        x_val = np.random.random((5, 128, 128, 3))
        y_val = np.random.randint(0, 2, (5, 2))

        cnn_model.build_model()

        history = cnn_model.train(
            x_train, y_train, x_val, y_val, epochs=1, batch_size=5
        )
        assert hasattr(history, "history")

        predictions = cnn_model.predict(x_val)
        assert predictions.shape == (5, 2)

        metrics = cnn_model.evaluate(x_val, y_val)
        assert "loss" in metrics
        assert "accuracy" in metrics

        model_path = temp_dir / "test_model.h5"
        cnn_model.save_model(str(model_path))
        assert model_path.exists()

        new_model = FakeImageCNN()
        new_model.load_model(str(model_path))
        assert new_model.model is not None

        new_predictions = new_model.predict(x_val)
        assert new_predictions.shape == (5, 2)
