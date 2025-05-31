"""FastAPI endpoints for fake image detection."""

import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..config.settings import APIConfig
from ..models.cnn import FakeImageCNN
from ..preprocessing.ela import ELAProcessor

app = FastAPI(
    title="Fake Image Detector API",
    description="API for detecting tampered/fake images using ELA and CNN",
    version="0.1.0",
)

logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    is_fake: bool
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool


class APIState:
    """Global state for API."""

    def __init__(self) -> None:
        self.model: Optional[FakeImageCNN] = None
        self.ela_processor: Optional[ELAProcessor] = None
        self.config: APIConfig = APIConfig()
        self.model_loaded = False

    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        try:
            self.model = FakeImageCNN()
            self.model.load_model(model_path)
            self.ela_processor = ELAProcessor()
            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


api_state = APIState()


def get_api_state() -> APIState:
    """Dependency to get API state."""
    return api_state


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the API on startup."""
    config = APIConfig()

    if config.model_path and Path(config.model_path).exists():
        try:
            api_state.load_model(config.model_path)
        except Exception as e:
            logger.warning(f"Could not load model on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check(state: APIState = Depends(get_api_state)):
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=state.model_loaded)


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), state: APIState = Depends(get_api_state)
) -> PredictionResponse:
    """Predict if an uploaded image is fake or real."""
    if not state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server configuration.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        if len(contents) > state.config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large. Maximum size: "
                    f"{state.config.max_file_size} bytes"
                ),
            )

        image = Image.open(io.BytesIO(contents))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, "JPEG")
            temp_path = temp_file.name

        try:
            processed_image = state.ela_processor.process_and_resize(temp_path)
            processed_image = np.expand_dims(processed_image, axis=0)

            predictions = state.model.predict(processed_image)
            probabilities = predictions[0]

            real_prob = float(probabilities[0])
            fake_prob = float(probabilities[1])

            is_fake = fake_prob > real_prob
            confidence = max(real_prob, fake_prob)

            return PredictionResponse(
                is_fake=is_fake,
                confidence=confidence,
                probabilities={"real": real_prob, "fake": fake_prob},
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...), state: APIState = Depends(get_api_state)
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """Predict multiple images at once."""
    if not state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server configuration.",
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 files allowed per batch"
        )

    results = []

    for i, file in enumerate(files):
        try:
            if not file.content_type or not file.content_type.startswith("image/"):
                results.append(
                    {"filename": file.filename, "error": "File must be an image"}
                )
                continue

            contents = await file.read()

            if len(contents) > state.config.max_file_size:
                results.append(
                    {
                        "filename": file.filename,
                        "error": (
                            f"File too large. Maximum size: "
                            f"{state.config.max_file_size} bytes"
                        ),
                    }
                )
                continue

            image = Image.open(io.BytesIO(contents))

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, "JPEG")
                temp_path = temp_file.name

            try:
                processed_image = state.ela_processor.process_and_resize(temp_path)
                processed_image = np.expand_dims(processed_image, axis=0)

                predictions = state.model.predict(processed_image)
                probabilities = predictions[0]

                real_prob = float(probabilities[0])
                fake_prob = float(probabilities[1])

                is_fake = fake_prob > real_prob
                confidence = max(real_prob, fake_prob)

                results.append(
                    {
                        "filename": file.filename,
                        "is_fake": str(is_fake),
                        "confidence": str(confidence),
                        "probabilities": str({"real": real_prob, "fake": fake_prob}),
                    }
                )

            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}


@app.post("/load_model")
async def load_model(
    model_path: str, state: APIState = Depends(get_api_state)
) -> Dict[str, str]:
    """Load a model from the specified path."""
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404, detail=f"Model file not found: {model_path}"
        )

    try:
        state.load_model(model_path)
        return {"message": f"Model loaded successfully from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    config = APIConfig()
    uvicorn.run(
        "fake_image_detector.api.endpoints:app",
        host=config.host,
        port=config.port,
        reload=True,
    )
