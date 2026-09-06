"""
Prediction Pipeline for Churn Prediction.

This module defines the end-to-end workflow for
generating predictions on new data, integrating
dataset validation, feature preprocessing, and
model inference.
"""

from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder
from xgboost import Booster

from churn_prediction.config.settings import get_settings
from churn_prediction.data.schemas import PROCESSED_BASE_SCHEMA, RAW_BASE_SCHEMA
from churn_prediction.data.validation import validate_dataset
from churn_prediction.features.inference_preprocessing import preprocess_inference_dataset
from churn_prediction.model.inference import run_inference


def run_prediction_pipeline(
    batch_id: str,
    raw_file_path: str | Path,
    loaded_encoder: OrdinalEncoder,
    loaded_booster: Booster,
) -> None:
    """Execute the complete batch prediction workflow.

    The pipeline validates the raw input dataset, transforms it using the
    pre-fitted encoder, validates the resulting model-ready dataset, and
    generates predictions using the trained XGBoost booster.

    Parameters
    ----------
    batch_id : str
        Unique identifier for the current prediction batch.
    raw_file_path : str or pathlib.Path
        Path to the raw customer dataset to be scored.
    loaded_encoder : sklearn.preprocessing.OrdinalEncoder
        Pre-fitted encoder used to transform categorical features.
    loaded_booster : xgboost.Booster
        Trained XGBoost model used to generate churn predictions.

    Raises
    ------
    FileNotFoundError
        If the input dataset or required output directories are unavailable.
    ValueError
        If raw or processed data fails validation or preprocessing fails.
    RuntimeError
        If model inference or artifact persistence fails.
    """
    # Initialize settings
    settings = get_settings()
    batch_id = batch_id.strip()

    # Resolve base directories
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    prediction_dir = settings.PREDICTION_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()
    validation_report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    # Derive specific file paths
    raw_file_path = Path(raw_file_path).expanduser().resolve()
    processed_file_path = processed_dir / f"processed_{batch_id}.parquet"
    prediction_file_path = prediction_dir / f"prediction_{batch_id}.csv"
    model_card_path = model_dir / "card.json"
    raw_report_path = validation_report_dir / f"raw_{batch_id}.json"
    processed_report_path = validation_report_dir / f"processed_{batch_id}.json"

    # 1. Validate the raw dataset
    validate_dataset(
        dataset_path=raw_file_path,
        report_path=raw_report_path,
        schema=RAW_BASE_SCHEMA,
    )

    # 2. Preprocess the raw dataset for inference
    preprocess_inference_dataset(
        raw_input_path=raw_file_path,
        processed_output_path=processed_file_path,
        fitted_encoder=loaded_encoder,
    )

    # 3. Validate the processed dataset
    validate_dataset(
        dataset_path=processed_file_path,
        report_path=processed_report_path,
        schema=PROCESSED_BASE_SCHEMA,
    )

    # 4. Run inference on the processed dataset
    run_inference(
        booster=loaded_booster,
        raw_filepath=raw_file_path,
        input_filepath=processed_file_path,
        output_filepath=prediction_file_path,
        model_card_path=model_card_path,
    )
