"""
XGBoost Inference Module.

This module provides the `run_inference` function, which takes
a preprocessed DataFrame, a trained XGBoost booster model, and
a path to a model card JSON file. It generates churn probabilities,
calibrates them based on class imbalance, and applies a decision
threshold to produce final predictions.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings
from churn_prediction.model.training import unscale_probabilities


def run_inference(
    booster: xgb.Booster,
    raw_filepath: str | Path,
    input_filepath: str | Path,
    output_filepath: str | Path,
    model_card_path: str | Path,
) -> None:
    """Generate churn predictions for a processed inference dataset.

    Loads model metadata from the model card, generates raw XGBoost
    probabilities, reverses class-imbalance probability scaling, applies
    the configured decision threshold, and persists the resulting
    predictions alongside the original customer records.

    Parameters
    ----------
    booster : xgboost.Booster
        Trained XGBoost model used to generate predictions.
    raw_filepath : str or pathlib.Path
        Path to the original raw input dataset.
    input_filepath : str or pathlib.Path
        Path to the processed inference dataset supplied to the model.
    output_filepath : str or pathlib.Path
        Destination path for the generated prediction dataset.
    model_card_path : str or pathlib.Path
        Path to the JSON model card containing the class-imbalance weight
        and classification threshold.

    Raises
    ------
    FileNotFoundError
        If the model card or required input files do not exist.
    ValueError
        If required model metadata is missing or the inference data cannot
        be converted into an XGBoost input matrix.
    RuntimeError
        If prediction generation or output persistence fails.
    """
    logger = get_logger()
    settings = get_settings()

    # Resolve paths to absolute paths
    raw_filepath = Path(raw_filepath).expanduser().resolve()
    input_filepath = Path(input_filepath).expanduser().resolve()
    output_filepath = Path(output_filepath).expanduser().resolve()
    model_card_path = Path(model_card_path).expanduser().resolve()

    if not model_card_path.is_file():
        logger.error("Model card file missing at: %s", model_card_path)
        raise FileNotFoundError(f"Model card not found at path: {model_card_path}")

    # 1. Load model card to retrieve threshold and scale_pos_weight
    logger.info("Loading model metadata from card: %s", model_card_path.name)
    with model_card_path.open("r", encoding="utf-8") as f:
        model_card: dict[str, Any] = json.load(f)

    try:
        scale_pos_weight = float(model_card["model_details"]["hyperparameters"]["scale_pos_weight"])
        decision_thresholds = model_card["metrics"]["decision_thresholds"]
        threshold = float(
            decision_thresholds.get(
                "optimal_threshold", decision_thresholds.get("default_threshold", 0.5)
            )
        )
        logger.debug(
            "Extracted model params - scale_pos_weight: %s, threshold: %s",
            scale_pos_weight,
            threshold,
        )
    except (KeyError, TypeError, ValueError) as err:
        logger.error("Model card schema mismatch or missing values: %s", err)
        raise ValueError(f"Failed to extract required parameters from model card: {err}") from err

    # 2. Load the dataset
    logger.info("Reading Parquet file: %s", input_filepath)
    df = pd.read_parquet(input_filepath)

    # 3. Create DMatrix for inference
    try:
        feature_cols = settings.PARAMS.schema_config.feature_columns
        logger.info("Creating DMatrix for %d features and %d rows", len(feature_cols), len(df))
        dtest = xgb.DMatrix(df[feature_cols], feature_names=feature_cols)
        logger.info("DMatrix created successfully. Shapes: dtest=%s", dtest.num_row())
    except Exception as err:
        logger.error("Failed to create DMatrix: %s", err)
        raise ValueError(f"Failed to create DMatrix: {err}") from err

    # 4. Run inference using the booster model
    logger.info("Executing model prediction via XGBoost booster")
    raw_probs: np.ndarray = booster.predict(dtest)
    logger.debug("Raw probabilities generated. Sample (first 3): %s", raw_probs[:3])

    # 5. Calibrate probabilities using scale_pos_weight
    logger.info("Calibrating raw probabilities using scale_pos_weight=%s", scale_pos_weight)
    calibrated_probs = unscale_probabilities(raw_probs, scale_pos_weight)
    logger.debug("Calibrated probabilities sample (first 3): %s", calibrated_probs[:3])

    # 6. Make predictions based on the calibrated probabilities and threshold
    logger.info("Applying decision threshold: %s", threshold)
    predictions = (calibrated_probs >= threshold).astype(int)

    # Log distribution of predictions for observability
    pos_count = np.sum(predictions)
    logger.info(
        "Inference complete. Predictions: %d churn, %d non-churn",
        pos_count,
        len(predictions) - pos_count,
    )

    # 7. Prepare results DataFrame with churn probabilities and predictions
    logger.debug("Merging predictions back into results DataFrame")
    results_df = pd.read_parquet(raw_filepath)
    results_df["churn_probability"] = np.round(calibrated_probs, 6)
    results_df["churn_prediction"] = predictions

    # 8. Save the results to the specified output file
    logger.info("Saving predictions to CSV file: %s", output_filepath)
    results_df.to_csv(output_filepath, index=False)
