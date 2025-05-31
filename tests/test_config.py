"""Tests for configuration settings."""

import pytest
from pydantic import ValidationError

from fake_image_detector.config.settings import (
    APIConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Test cases for model configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.input_shape == (128, 128, 3)
        assert config.num_classes == 2
        assert config.learning_rate == 0.0005

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            input_shape=(256, 256, 3), num_classes=3, learning_rate=0.001
        )
        assert config.input_shape == (256, 256, 3)
        assert config.num_classes == 3
        assert config.learning_rate == 0.001


class TestDataConfig:
    """Test cases for data configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.ela_quality == 90
        assert config.target_size == (128, 128)
        assert config.test_size == 0.2
        assert config.random_state == 42

    def test_quality_validation(self):
        """Test ELA quality validation."""
        with pytest.raises(ValidationError):
            DataConfig(ela_quality=0)

        with pytest.raises(ValidationError):
            DataConfig(ela_quality=101)

    def test_test_size_validation(self):
        """Test test size validation."""
        with pytest.raises(ValidationError):
            DataConfig(test_size=0.05)

        with pytest.raises(ValidationError):
            DataConfig(test_size=0.6)


class TestTrainingConfig:
    """Test cases for training configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.epochs == 30
        assert config.batch_size == 100
        assert config.early_stopping_patience == 2
        assert config.monitor_metric == "val_accuracy"
        assert config.output_dir == "outputs"

    def test_epochs_validation(self):
        """Test epochs validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)


class TestAPIConfig:
    """Test cases for API configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.model_path is None
        assert config.max_file_size == 10 * 1024 * 1024

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValidationError):
            APIConfig(port=0)

        with pytest.raises(ValidationError):
            APIConfig(port=70000)


class TestConfig:
    """Test cases for main configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.api, APIConfig)

    def test_nested_config(self):
        """Test nested configuration."""
        config = Config(
            model=ModelConfig(num_classes=3),
            data=DataConfig(ela_quality=80),
            training=TrainingConfig(epochs=50),
            api=APIConfig(port=9000),
        )
        assert config.model.num_classes == 3
        assert config.data.ela_quality == 80
        assert config.training.epochs == 50
        assert config.api.port == 9000
