"""Model training and evaluation utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ..data.pipeline import DataPipeline
from ..models.cnn import FakeImageCNN


class ModelTrainer:
    """Trainer class for fake image detection model."""

    def __init__(
        self,
        model: FakeImageCNN,
        data_pipeline: DataPipeline,
        output_dir: str = "outputs",
    ):
        """Initialize model trainer.

        Args:
            model: CNN model instance
            data_pipeline: Data pipeline instance
            output_dir: Directory to save outputs
        """
        self.model = model
        self.data_pipeline = data_pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train(
        self,
        data_source: str,
        source_type: str = "csv",
        epochs: int = 30,
        batch_size: int = 100,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            data_source: Path to data source
            source_type: Type of data source ("csv" or "directory")
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_model: Whether to save the trained model

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting model training...")

        X_train, X_test, y_train, y_test = self.data_pipeline.load_and_prepare(
            data_source, source_type
        )

        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Test data shape: {X_test.shape}")

        history = self.model.train(
            X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size
        )

        results = self.evaluate_model(X_test, y_test, history)

        if save_model:
            model_path = self.output_dir / "fake_image_detector_model.h5"
            self.model.save_model(str(model_path))
            self.logger.info(f"Model saved to {model_path}")

        self.save_training_results(results)

        return results

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: tf.keras.callbacks.History,
    ) -> Dict[str, Any]:
        """Evaluate trained model.

        Args:
            X_test: Test images
            y_test: Test labels
            history: Training history

        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Evaluating model...")

        test_metrics = self.model.evaluate(X_test, y_test)

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        report = classification_report(y_true, y_pred_classes, output_dict=True)

        results = {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "training_history": {
                "loss": history.history.get("loss", []),
                "accuracy": history.history.get("accuracy", []),
                "val_loss": history.history.get("val_loss", []),
                "val_accuracy": history.history.get("val_accuracy", []),
            },
        }

        self.logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test loss: {test_metrics['loss']:.4f}")

        self.plot_training_history(history)
        self.plot_confusion_matrix(cm)

        return results

    def plot_training_history(self, history: tf.keras.callbacks.History) -> None:
        """Plot training history.

        Args:
            history: Training history object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history["loss"], label="Training Loss")
        ax1.plot(history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(history.history["accuracy"], label="Training Accuracy")
        ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix.

        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = ["Real", "Fake"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()

    def save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to JSON file.

        Args:
            results: Training results dictionary
        """
        results_path = self.output_dir / "training_results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Training results saved to {results_path}")
