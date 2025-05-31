"""Tests for data pipeline."""

import pandas as pd
import pytest


class TestDataPipeline:
    """Test cases for data pipeline."""

    def test_init_default_params(self, data_pipeline):
        """Test data pipeline initialization with default parameters."""
        assert data_pipeline.test_size == 0.2
        assert data_pipeline.random_state == 42
        assert data_pipeline.ela_processor is not None

    def test_load_from_csv(self, data_pipeline, sample_csv_dataset):
        """Test loading data from CSV file."""
        df = data_pipeline.load_from_csv(sample_csv_dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["image_path", "label"]

    def test_load_from_csv_nonexistent(self, data_pipeline):
        """Test loading from non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            data_pipeline.load_from_csv("nonexistent.csv")

    def test_load_from_directory(self, data_pipeline, sample_dataset):
        """Test loading data from directory structure."""
        df = data_pipeline.load_from_directory(sample_dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["image_path", "label"]
        assert (df["label"] == 0).sum() == 5  # real images
        assert (df["label"] == 1).sum() == 5  # fake images

    def test_load_from_directory_nonexistent(self, data_pipeline):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            data_pipeline.load_from_directory("nonexistent_dir")

    def test_prepare_data(self, data_pipeline, sample_csv_dataset):
        """Test data preparation."""
        df = data_pipeline.load_from_csv(sample_csv_dataset)
        X, y = data_pipeline.prepare_data(df)

        assert X.shape[0] == len(df)
        assert X.shape[1:] == (128, 128, 3)
        assert y.shape == (len(df), 2)

    def test_train_test_split(self, data_pipeline):
        """Test train-test split functionality."""
        import numpy as np

        X = np.random.random((100, 128, 128, 3))
        y = np.random.randint(0, 2, (100, 2))

        X_train, X_test, y_train, y_test = data_pipeline.train_test_split(X, y)

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_load_and_prepare_csv(self, data_pipeline, sample_csv_dataset):
        """Test complete load and prepare workflow with CSV."""
        X_train, X_test, y_train, y_test = data_pipeline.load_and_prepare(
            sample_csv_dataset, source_type="csv"
        )

        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert X_train.shape[1:] == (128, 128, 3)
        assert y_train.shape == (8, 2)
        assert y_test.shape == (2, 2)

    def test_load_and_prepare_directory(self, data_pipeline, sample_dataset):
        """Test complete load and prepare workflow with directory."""
        X_train, X_test, y_train, y_test = data_pipeline.load_and_prepare(
            sample_dataset, source_type="directory"
        )

        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert X_train.shape[1:] == (128, 128, 3)
        assert y_train.shape == (8, 2)
        assert y_test.shape == (2, 2)

    def test_load_and_prepare_invalid_source_type(
        self, data_pipeline, sample_csv_dataset
    ):
        """Test load and prepare with invalid source type."""
        with pytest.raises(
            ValueError, match="source_type must be 'csv' or 'directory'"
        ):
            data_pipeline.load_and_prepare(sample_csv_dataset, source_type="invalid")
