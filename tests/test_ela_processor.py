"""Tests for ELA processor."""

import numpy as np
import pytest
from PIL import Image

from fake_image_detector.preprocessing.ela import ELAProcessor


class TestELAProcessor:
    """Test cases for ELA processor."""

    def test_init_default_params(self):
        """Test ELA processor initialization with default parameters."""
        processor = ELAProcessor()
        assert processor.quality == 90
        assert processor.target_size == (128, 128)

    def test_init_custom_params(self):
        """Test ELA processor initialization with custom parameters."""
        processor = ELAProcessor(quality=80, target_size=(256, 256))
        assert processor.quality == 80
        assert processor.target_size == (256, 256)

    def test_init_invalid_quality(self):
        """Test ELA processor initialization with invalid quality."""
        with pytest.raises(ValueError, match="Quality must be between 1 and 100"):
            ELAProcessor(quality=0)

        with pytest.raises(ValueError, match="Quality must be between 1 and 100"):
            ELAProcessor(quality=101)

    def test_process_image(self, ela_processor, sample_image):
        """Test processing a single image."""
        result = ela_processor.process_image(sample_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_process_image_nonexistent(self, ela_processor):
        """Test processing a non-existent image."""
        with pytest.raises(FileNotFoundError):
            ela_processor.process_image("nonexistent.jpg")

    def test_process_and_resize(self, ela_processor, sample_image):
        """Test processing and resizing an image."""
        result = ela_processor.process_and_resize(sample_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1

    def test_process_batch(self, ela_processor, sample_images):
        """Test processing multiple images."""
        result = ela_processor.process_batch(sample_images)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 128, 128, 3)
        assert result.dtype == np.float32

    def test_process_batch_empty_list(self, ela_processor):
        """Test processing empty list of images."""
        with pytest.raises(
            ValueError, match="No images could be processed successfully"
        ):
            ela_processor.process_batch([])
