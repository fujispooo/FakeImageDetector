"""Configuration settings using Pydantic models."""

from typing import Optional, Tuple

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for CNN model."""

    input_shape: Tuple[int, int, int] = Field(
        default=(128, 128, 3), description="Input image shape (height, width, channels)"
    )
    num_classes: int = Field(default=2, description="Number of output classes")
    learning_rate: float = Field(
        default=0.0005, description="Learning rate for optimizer"
    )


class DataConfig(BaseModel):
    """Configuration for data processing."""

    ela_quality: int = Field(
        default=90, ge=1, le=100, description="JPEG quality for ELA processing"
    )
    target_size: Tuple[int, int] = Field(
        default=(128, 128), description="Target size for image resizing"
    )
    test_size: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Fraction of data for testing"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(default=30, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=100, ge=1, description="Batch size for training")
    early_stopping_patience: int = Field(
        default=2, ge=1, description="Early stopping patience"
    )
    monitor_metric: str = Field(
        default="val_accuracy", description="Metric to monitor for early stopping"
    )
    output_dir: str = Field(
        default="outputs", description="Directory to save training outputs"
    )


class APIConfig(BaseModel):
    """Configuration for API server."""

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    model_path: Optional[str] = Field(
        default=None, description="Path to trained model file"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum file size for uploads"  # 10MB
    )


class Config(BaseModel):
    """Main configuration class."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
