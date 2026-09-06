"""
Training Pipeline for Churn Prediction.

This module provides a CLI interface to orchestrate the training pipeline,
including ingestion, schema validation, preprocessing of training and validation
datasets and training the xgboost model. It is designed to be integrated with DVC
(Data Version Control) for pipeline reproducibility.
"""

import argparse
from typing import Any

from churn_prediction.config.settings import LogLevel, Settings, get_settings
from churn_prediction.data.ingestion import get_dataset
from churn_prediction.data.schemas import PROCESSED_SCHEMA, RAW_SCHEMA
from churn_prediction.data.validation import validate_dataset
from churn_prediction.features.train_preprocessing import preprocess_train_dataset
from churn_prediction.model.training import train_xgb_model


def run_ingestion(settings: Settings) -> None:
    """Execute the dataset ingestion stage.

    Downloads or retrieves the configured raw dataset and prepares the
    expected local dataset artifacts.
    """
    # Resolve Base Directories
    raw_data_dir = settings.RAW_DATA_DIR.expanduser().resolve()

    # Execute Ingestion
    get_dataset(raw_data_dir)


def run_raw_validation(settings: Settings) -> None:
    """Execute raw dataset validation.

    Validates the ingested raw dataset against ``RAW_SCHEMA`` and writes
    validation diagnostics to the configured report location.
    """
    # Resolve Base Directories
    raw_data_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    validation_report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    # Derive Specific File Paths
    raw_data_file = raw_data_dir / "train.parquet"
    raw_validation_report = validation_report_dir / "raw.json"

    # Execute Validation
    validate_dataset(
        dataset_path=raw_data_file,
        report_path=raw_validation_report,
        schema=RAW_SCHEMA,
    )


def run_preprocessing(settings: Settings) -> None:
    """Execute the training preprocessing stage.

    Transforms the raw dataset into model-ready training and validation
    datasets and persists the fitted preprocessing artifact.
    """
    # Resolve Base Directories
    raw_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()

    # Derive Specific File Paths
    raw_file = raw_dir / "train.parquet"
    train_file = processed_dir / "processed_train.parquet"
    val_file = processed_dir / "processed_val.parquet"
    preprocessor_file = model_dir / "preprocessor.joblib"

    # Execute Preprocessing
    preprocess_train_dataset(
        raw_input_path=raw_file,
        train_output_path=train_file,
        val_output_path=val_file,
        preprocessor_path=preprocessor_file,
    )


def run_train_validation(settings: Settings) -> None:
    """Validate the processed training dataset.

    Checks the generated training dataset against ``PROCESSED_SCHEMA`` and
    records any validation failures in the configured validation report.
    """
    # Resolve Base Directories
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    validation_report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    # Derive Specific File Paths
    train_file = processed_dir / "processed_train.parquet"
    train_validation_report = validation_report_dir / "train.json"

    # Execute Validation
    validate_dataset(
        dataset_path=train_file,
        report_path=train_validation_report,
        schema=PROCESSED_SCHEMA,
    )


def run_val_validation(settings: Settings) -> None:
    """Validate the processed validation dataset.

    Checks the generated validation dataset against ``PROCESSED_SCHEMA``
    and records any validation failures in the configured validation report.
    """
    # Resolve Base Directories
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    validation_report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    # Derive Specific File Paths
    val_file = processed_dir / "processed_val.parquet"
    val_validation_report = validation_report_dir / "val.json"

    # Execute Validation
    validate_dataset(
        dataset_path=val_file,
        report_path=val_validation_report,
        schema=PROCESSED_SCHEMA,
    )


def run_training(settings: Settings) -> None:
    """Execute the model training stage.

    Trains the configured XGBoost model using the processed training and
    validation datasets and exports the resulting model, model card, and
    training report.
    """
    # Resolve Base Directories
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()
    report_dir = settings.TRAINING_REPORT_DIR.expanduser().resolve()

    # Derive Specific File Paths
    train_file = processed_dir / "processed_train.parquet"
    val_file = processed_dir / "processed_val.parquet"
    model_file = model_dir / "model.ubj"
    model_card_file = model_dir / "card.json"
    report_file = report_dir / "training.json"

    # Execute Training
    train_xgb_model(
        train_data_path=train_file,
        val_data_path=val_file,
        model_filepath=model_file,
        model_card_path=model_card_file,
        report_path=report_file,
    )


def run_all_stages(settings: Settings) -> None:
    """Execute the complete training pipeline.

    Runs ingestion, raw-data validation, preprocessing, processed-data
    validation, and model training sequentially using the supplied
    application settings.
    """
    run_ingestion(settings)
    run_raw_validation(settings)
    run_preprocessing(settings)
    run_train_validation(settings)
    run_val_validation(settings)
    run_training(settings)


STAGES: dict[str, Any] = {
    "ingest-data": run_ingestion,
    "validate-raw": run_raw_validation,
    "preprocess-raw": run_preprocessing,
    "validate-train": run_train_validation,
    "validate-val": run_val_validation,
    "train-model": run_training,
    "all": run_all_stages,
}


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Churn Prediction Training Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(
        dest="stage",
        required=True,
        help="Target stage to execute",
    )

    for stage_name, stage_fn in STAGES.items():
        subparser = subparsers.add_parser(
            stage_name,
            help=stage_fn.__doc__ or f"Execute {stage_name} stage",
        )
        subparser.set_defaults(func=stage_fn)

    return parser


if __name__ == "__main__":
    settings = get_settings()
    parser = _build_parser()
    args = parser.parse_args()

    # Set Logging Level Based on Verbosity
    if args.verbose:
        settings.LOG_LEVEL = LogLevel.DEBUG

    # Execute the selected pipeline stage
    args.func(settings)
