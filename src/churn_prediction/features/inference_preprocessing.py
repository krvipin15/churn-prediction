"""
Inference data preprocessing module for churn prediction.

This module provides functionality to transform raw input datasets into
a format compatible with the trained churn prediction model, ensuring
consistency in binary mapping, ordinal encoding, and data types.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings
from churn_prediction.features.train_preprocessing import BINARY_MAPPING, DEFAULT_DTYPE_MAPPING


def preprocess_inference_dataset(
    raw_input_path: str | Path,
    processed_output_path: str | Path,
    fitted_encoder: OrdinalEncoder,
) -> None:
    """Transform a raw inference dataset into model-ready features.

    Applies the same binary mappings, categorical encoding, feature ordering,
    and data type conversions used during model training. The supplied
    pre-fitted encoder is reused without modification to ensure training and
    inference transformations remain consistent.

    Parameters
    ----------
    raw_input_path : str or pathlib.Path
        Path to the raw dataset that will be scored.
    processed_output_path : str or pathlib.Path
        Destination path for the processed inference dataset.
    fitted_encoder : sklearn.preprocessing.OrdinalEncoder
        Encoder fitted during the training preprocessing stage.

    Raises
    ------
    FileNotFoundError
        If the raw input dataset cannot be found.
    ValueError
        If the input schema, binary mappings, categorical values, feature
        alignment, or data type conversions are invalid.
    RuntimeError
        If the processed inference dataset cannot be persisted.
    """
    logger = get_logger()
    settings = get_settings()

    # Resolve the input and output paths to absolute paths
    raw_input_path = Path(raw_input_path).expanduser().resolve()
    processed_output_path = Path(processed_output_path).expanduser().resolve()

    # Extract schema specifications
    binary_columns = settings.PARAMS.schema_config.binary_columns
    ordinal_columns = settings.PARAMS.schema_config.ordinal_columns

    # 1. Load the raw dataset
    logger.info("Loading raw batch data from %s", raw_input_path)
    df = pd.read_parquet(raw_input_path)
    logger.debug("Loaded batch raw shape: %s", df.shape)

    # 2. Map binary features
    logger.info("Applying binary mapping to %d columns", len(binary_columns))
    for column in binary_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.debug("Applying binary mapping to column: %s", column)
            df[column] = df[column].map(BINARY_MAPPING)
        else:
            logger.debug("Skipping binary mapping for %s; already numeric", column)

    # 3. Transform ordinal features via pre-fitted encoder
    try:
        logger.info(
            "Transforming %d ordinal features using pre-fitted encoder", len(ordinal_columns)
        )
        df[ordinal_columns] = fitted_encoder.transform(df[ordinal_columns])
    except Exception as err:
        logger.error("Failed to execute ordinal encoder transformation: %s", err)
        raise ValueError(f"Ordinal encoding transformation failed: {err}") from err

    # 4. Enforce precise data types for model compatibility
    try:
        logger.info("Casting inference dataframe to optimized schema dtypes")
        df = df.astype(DEFAULT_DTYPE_MAPPING)
    except Exception as err:
        logger.error("Data type casting failed: %s", err)
        raise ValueError(f"Data type coercion failed on inference data: {err}") from err

    # 5. Save processed data to parquet
    try:
        df.to_parquet(processed_output_path, index=False, engine="pyarrow")
        logger.info("Saved preprocessed batch dataset to %s", processed_output_path)
    except Exception as err:
        logger.error("Failed to save preprocessed data to %s: %s", processed_output_path, err)
        raise ValueError(f"Failed to save preprocessed data: {err}") from err
