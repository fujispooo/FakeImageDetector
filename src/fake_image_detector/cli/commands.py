"""Command-line interface for fake image detector."""

import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from ..config.settings import Config
from ..data.pipeline import DataPipeline
from ..models.cnn import FakeImageCNN
from ..preprocessing.ela import ELAProcessor
from ..training.trainer import ModelTrainer


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """Fake Image Detector CLI."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if config:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)
        ctx.obj = Config(**config_data)
    else:
        ctx.obj = Config()


@main.command()
@click.option(
    "--data-source", "-d", required=True, help="Path to data source (CSV or directory)"
)
@click.option(
    "--source-type",
    "-t",
    type=click.Choice(["csv", "directory"]),
    default="csv",
    help="Type of data source",
)
@click.option("--epochs", "-e", type=int, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, help="Batch size for training")
@click.option("--output-dir", "-o", help="Output directory for results")
@click.pass_obj
def train(
    config: Config,
    data_source: str,
    source_type: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    output_dir: Optional[str],
) -> None:
    """Train the fake image detection model."""
    if epochs:
        config.training.epochs = epochs
    if batch_size:
        config.training.batch_size = batch_size
    if output_dir:
        config.training.output_dir = output_dir

    ela_processor = ELAProcessor(
        quality=config.data.ela_quality, target_size=config.data.target_size
    )

    data_pipeline = DataPipeline(
        ela_processor=ela_processor,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    model = FakeImageCNN(
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
        learning_rate=config.model.learning_rate,
    )

    trainer = ModelTrainer(
        model=model, data_pipeline=data_pipeline, output_dir=config.training.output_dir
    )

    results = trainer.train(
        data_source=data_source,
        source_type=source_type,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
    )

    click.echo(
        f"Training completed! Test accuracy: {results['test_accuracy']:.4f}"
    )


@main.command()
@click.option("--model-path", "-m", required=True, help="Path to trained model")
@click.option("--image-path", "-i", required=True, help="Path to image to predict")
@click.pass_obj
def predict(config: Config, model_path: str, image_path: str) -> None:
    """Predict if an image is fake or real."""
    if not Path(model_path).exists():
        click.echo(f"Error: Model file not found: {model_path}", err=True)
        return

    if not Path(image_path).exists():
        click.echo(f"Error: Image file not found: {image_path}", err=True)
        return

    ela_processor = ELAProcessor(
        quality=config.data.ela_quality, target_size=config.data.target_size
    )

    model = FakeImageCNN()
    model.load_model(model_path)

    processed_image = ela_processor.process_and_resize(image_path)
    processed_image = processed_image.reshape(1, *processed_image.shape)

    predictions = model.predict(processed_image)
    probabilities = predictions[0]

    real_prob = probabilities[0]
    fake_prob = probabilities[1]

    is_fake = fake_prob > real_prob
    confidence = max(real_prob, fake_prob)

    click.echo(f"Image: {image_path}")
    click.echo(f"Prediction: {'FAKE' if is_fake else 'REAL'}")
    click.echo(f"Confidence: {confidence:.4f}")
    click.echo(
        f"Probabilities - Real: {real_prob:.4f}, Fake: {fake_prob:.4f}"
    )


@main.command()
@click.option("--model-path", "-m", help="Path to trained model")
@click.option("--host", "-h", default="0.0.0.0", help="API server host")
@click.option("--port", "-p", type=int, default=8000, help="API server port")
@click.pass_obj
def serve(
    config: Config, model_path: Optional[str], host: str, port: int
) -> None:
    """Start the API server."""
    import uvicorn

    if model_path:
        config.api.model_path = model_path

    config.api.host = host
    config.api.port = port

    click.echo(f"Starting API server on {host}:{port}")
    if config.api.model_path:
        click.echo(f"Model path: {config.api.model_path}")

    uvicorn.run(
        "fake_image_detector.api.endpoints:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


@main.command()
@click.option(
    "--output", "-o", default="config.yaml", help="Output configuration file path"
)
def init_config(output: str) -> None:
    """Generate a default configuration file."""
    config = Config()

    config_dict = {
        "model": config.model.dict(),
        "data": config.data.dict(),
        "training": config.training.dict(),
        "api": config.api.dict(),
    }

    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    click.echo(f"Configuration file created: {output}")


if __name__ == "__main__":
    main()
