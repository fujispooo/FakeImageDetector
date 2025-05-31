"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator


import pytest
from PIL import Image

from fake_image_detector.data.pipeline import DataPipeline
from fake_image_detector.models.cnn import FakeImageCNN
from fake_image_detector.preprocessing.ela import ELAProcessor


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample test image."""
    image = Image.new("RGB", (256, 256), color="red")
    image_path = temp_dir / "test_image.jpg"
    image.save(image_path, "JPEG")
    return image_path


@pytest.fixture
def sample_images(temp_dir: Path) -> list[Path]:
    """Create multiple sample test images."""
    images = []
    for i in range(3):
        image = Image.new("RGB", (256, 256), color=["red", "green", "blue"][i])
        image_path = temp_dir / f"test_image_{i}.jpg"
        image.save(image_path, "JPEG")
        images.append(image_path)
    return images


@pytest.fixture
def ela_processor() -> ELAProcessor:
    """Create an ELA processor instance."""
    return ELAProcessor(quality=90, target_size=(128, 128))


@pytest.fixture
def cnn_model() -> FakeImageCNN:
    """Create a CNN model instance."""
    return FakeImageCNN(input_shape=(128, 128, 3), num_classes=2)


@pytest.fixture
def data_pipeline(ela_processor: ELAProcessor) -> DataPipeline:
    """Create a data pipeline instance."""
    return DataPipeline(ela_processor=ela_processor, test_size=0.2, random_state=42)


@pytest.fixture
def sample_dataset(temp_dir: Path) -> Path:
    """Create a sample dataset structure."""
    real_dir = temp_dir / "real"
    fake_dir = temp_dir / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    for i in range(5):
        real_image = Image.new("RGB", (256, 256), color="blue")
        real_image.save(real_dir / f"real_{i}.jpg", "JPEG")

        fake_image = Image.new("RGB", (256, 256), color="red")
        fake_image.save(fake_dir / f"fake_{i}.jpg", "JPEG")

    return temp_dir


@pytest.fixture
def sample_csv_dataset(temp_dir: Path, sample_dataset: Path) -> Path:
    """Create a sample CSV dataset file."""
    import pandas as pd

    real_dir = sample_dataset / "real"
    fake_dir = sample_dataset / "fake"

    data = []
    for img_path in real_dir.glob("*.jpg"):
        data.append([str(img_path), 0])
    for img_path in fake_dir.glob("*.jpg"):
        data.append([str(img_path), 1])

    df = pd.DataFrame(data, columns=["image_path", "label"])
    csv_path = temp_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    return csv_path
