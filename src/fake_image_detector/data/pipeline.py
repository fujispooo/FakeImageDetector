"""Data pipeline for loading and preprocessing images."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from ..preprocessing.ela import ELAProcessor


class DataPipeline:
    """Data pipeline for fake image detection."""

    def __init__(
        self,
        ela_processor: Optional[ELAProcessor] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize data pipeline.

        Args:
            ela_processor: ELA processor instance
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.ela_processor = ela_processor or ELAProcessor()
        self.test_size = test_size
        self.random_state = random_state

    def load_from_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from CSV file.

        Args:
            csv_path: Path to CSV file containing image paths and labels

        Returns:
            DataFrame with image paths and labels
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if len(df.columns) < 2:
            raise ValueError("CSV must have at least 2 columns: image_path, label")

        df.columns = ["image_path", "label"]

        missing_files = []
        for idx, row in df.iterrows():
            if not Path(row["image_path"]).exists():
                missing_files.append(row["image_path"])

        if missing_files:
            print(f"Warning: {len(missing_files)} image files not found")
            df = df[df["image_path"].apply(lambda x: Path(x).exists())]

        return df

    def load_from_directory(
        self,
        data_dir: Union[str, Path],
        real_subdir: str = "real",
        fake_subdir: str = "fake",
    ) -> pd.DataFrame:
        """Load dataset from directory structure.

        Args:
            data_dir: Root directory containing subdirectories for real/fake images
            real_subdir: Subdirectory name for real images
            fake_subdir: Subdirectory name for fake images

        Returns:
            DataFrame with image paths and labels
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        real_dir = data_dir / real_subdir
        fake_dir = data_dir / fake_subdir

        image_paths = []
        labels = []

        if real_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in real_dir.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(0)  # 0 for real

        if fake_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in fake_dir.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(1)  # 1 for fake

        if not image_paths:
            raise ValueError(f"No images found in {data_dir}")

        return pd.DataFrame({"image_path": image_paths, "label": labels})

    def prepare_data(
        self, df: pd.DataFrame, num_classes: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training.

        Args:
            df: DataFrame with image paths and labels
            num_classes: Number of classes for one-hot encoding

        Returns:
            Tuple of (processed_images, one_hot_labels)
        """
        image_paths = df["image_path"].tolist()
        labels = df["label"].tolist()

        processed_images = self.ela_processor.process_batch(image_paths)
        one_hot_labels = to_categorical(labels, num_classes)

        return processed_images, one_hot_labels

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def load_and_prepare(
        self, data_source: Union[str, Path], source_type: str = "csv"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for training.

        Args:
            data_source: Path to data source (CSV file or directory)
            source_type: Type of data source ("csv" or "directory")

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if source_type == "csv":
            df = self.load_from_csv(data_source)
        elif source_type == "directory":
            df = self.load_from_directory(data_source)
        else:
            raise ValueError("source_type must be 'csv' or 'directory'")

        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        return X_train, X_test, y_train, y_test
