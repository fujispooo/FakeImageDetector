"""Fake Image Detector - Image Tampering Detection using ELA and CNN."""

__version__ = "0.1.0"
__author__ = "Agus Gunawan, Holy Lovenia, Adrian Hartanto Pramudita"

from .data.pipeline import DataPipeline
from .models.cnn import FakeImageCNN
from .preprocessing.ela import ELAProcessor
from .training.trainer import ModelTrainer

__all__ = [
    "FakeImageCNN",
    "ELAProcessor",
    "DataPipeline",
    "ModelTrainer",
]
