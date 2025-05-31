"""Error Level Analysis (ELA) image processing module."""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


class ELAProcessor:
    """Error Level Analysis processor for detecting image tampering."""

    def __init__(self, quality: int = 90, target_size: Tuple[int, int] = (128, 128)):
        """Initialize ELA processor.

        Args:
            quality: JPEG compression quality for ELA analysis (1-100)
            target_size: Target size for resized images (width, height)
        """
        if not 1 <= quality <= 100:
            raise ValueError("Quality must be between 1 and 100")

        self.quality = quality
        self.target_size = target_size

    def process_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Process an image using Error Level Analysis.

        Args:
            image_path: Path to the input image

        Returns:
            PIL Image object containing the ELA result

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be processed
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot open image {image_path}: {e}")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            original_image.save(temp_path, "JPEG", quality=self.quality)
            resaved_image = Image.open(temp_path)

            ela_image = ImageChops.difference(original_image, resaved_image)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            if max_diff == 0:
                max_diff = 1

            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            return ela_image

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def process_and_resize(self, image_path: Union[str, Path]) -> np.ndarray:
        """Process image with ELA and resize to target dimensions.

        Args:
            image_path: Path to the input image

        Returns:
            Normalized numpy array of shape (height, width, 3)
        """
        ela_image = self.process_image(image_path)
        resized_image = ela_image.resize(self.target_size)

        image_array = np.array(resized_image, dtype=np.float32) / 255.0

        return image_array

    def process_batch(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """Process multiple images with ELA.

        Args:
            image_paths: List of paths to input images

        Returns:
            Numpy array of shape (batch_size, height, width, 3)
        """
        processed_images = []

        for image_path in image_paths:
            try:
                processed_image = self.process_and_resize(image_path)
                processed_images.append(processed_image)
            except Exception as e:
                print(f"Warning: Failed to process {image_path}: {e}")
                continue

        if not processed_images:
            raise ValueError("No images could be processed successfully")

        return np.array(processed_images)
