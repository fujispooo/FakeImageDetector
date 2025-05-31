# Fake Image Detector

**Modern Image Tampering Detection using ELA and CNN**

A complete rewrite of the original FakeImageDetector project with modern development practices, proper package structure, and MLOps capabilities.

## Features

- **Error Level Analysis (ELA)** preprocessing for image tampering detection
- **Convolutional Neural Network** for binary classification (real/fake)
- **Modern Python package** structure with proper separation of concerns
- **Type hints** and comprehensive documentation
- **Configuration management** with Pydantic models
- **REST API** with FastAPI for model serving
- **Command-line interface** for training and inference
- **Comprehensive testing** with pytest
- **CI/CD pipeline** with GitHub Actions
- **Docker support** for easy deployment
- **Pre-commit hooks** for code quality

## Installation

### From Source

```bash
git clone https://github.com/fujispooo/FakeImageDetector.git
cd FakeImageDetector
pip install -e ".[dev]"
```

### Using Docker

```bash
docker build -t fake-image-detector .
docker run -p 8000:8000 fake-image-detector
```

## Quick Start

### Training a Model

```bash
# Using CSV dataset
fake-image-detector train --data-source datasets/dataset.csv --source-type csv

# Using directory structure
fake-image-detector train --data-source datasets/train --source-type directory
```

### Making Predictions

```bash
# Single image prediction
fake-image-detector predict --model-path outputs/fake_image_detector_model.h5 --image-path test_image.jpg

# Start API server
fake-image-detector serve --model-path outputs/fake_image_detector_model.h5
```

### Using the API

```bash
# Health check
curl http://localhost:8000/health

# Predict single image
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

## Project Structure

```
src/fake_image_detector/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   └── ela.py              # Error Level Analysis processing
├── models/
│   ├── __init__.py
│   └── cnn.py              # CNN model definition
├── data/
│   ├── __init__.py
│   └── pipeline.py         # Data loading and preprocessing
├── training/
│   ├── __init__.py
│   └── trainer.py          # Model training utilities
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration management
├── api/
│   ├── __init__.py
│   └── endpoints.py        # FastAPI endpoints
└── cli/
    ├── __init__.py
    └── commands.py         # Command-line interface
```

## Configuration

Create a configuration file using:

```bash
fake-image-detector init-config --output my_config.yaml
```

Then use it with any command:

```bash
fake-image-detector --config my_config.yaml train --data-source datasets/
```

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/ -v --cov=src/fake_image_detector
```

### Code Quality

```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers (32 filters, 5x5 kernel)
- MaxPooling and Dropout layers
- Dense layers (256 units) with final softmax output
- RMSprop optimizer with configurable learning rate

![Model Architecture](docs/model-architecture.jpg)

## Dataset Format

### CSV Format
```csv
image_path,label
path/to/real_image.jpg,0
path/to/fake_image.jpg,1
```

### Directory Structure
```
dataset/
├── real/
│   ├── image1.jpg
│   └── image2.jpg
└── fake/
    ├── image1.jpg
    └── image2.jpg
```

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

### Custom Model Path

```bash
docker run -v /path/to/models:/app/models \
           -p 8000:8000 \
           fake-image-detector serve --model-path /app/models/my_model.h5
```

## Performance

The modernized implementation maintains the original performance characteristics:
- **Convergence**: Typically around epoch 9
- **Best accuracy**: ~91.83% on validation set
- **Input size**: 128x128 RGB images
- **Processing**: ELA with JPEG quality 90

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Run the test suite and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Original Authors

- [Agus Gunawan](https://github.com/agusgun)
- [Holy Lovenia](https://github.com/holylovenia)
- [Adrian Hartanto Pramudita](https://github.com/adrianhp97)

## Modernization

This modernized version was created to bring the project up to current Python and MLOps standards while maintaining the core functionality and performance of the original implementation.
