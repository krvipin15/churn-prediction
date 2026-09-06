"""Train Data Preprocessing Module.

Provides functionality to clean, encode, and split the raw churn
dataset into training and validation sets. It ensures consistent
feature scaling and type casting to prepare the data for model training.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings

# Immutable default constants
BINARY_MAPPING = {
    "Yes": 1,
    "No": 0,
    "Female": 0,
    "Male": 1,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
}

DEFAULT_DTYPE_MAPPING: dict[str, Any] = {
    "gender": np.uint8,
    "SeniorCitizen": np.uint8,
    "Partner": np.uint8,
    "Dependents": np.uint8,
    "tenure": np.uint8,
    "PhoneService": np.uint8,
    "MultipleLines": np.int16,
    "InternetService": np.int16,
    "OnlineSecurity": np.int16,
    "OnlineBackup": np.int16,
    "DeviceProtection": np.int16,
    "TechSupport": np.int16,
    "StreamingTV": np.int16,
    "StreamingMovies": np.int16,
    "Contract": np.int16,
    "PaperlessBilling": np.uint8,
    "PaymentMethod": np.int16,
    "MonthlyCharges": np.float32,
    "TotalCharges": np.float32,
}


def preprocess_train_dataset(
    raw_input_path: str | Path,
    train_output_path: str | Path,
    val_output_path: str | Path,
    preprocessor_path: str | Path,
) -> None:
    """Prepare the raw training dataset for model development.

    Loads the raw dataset, applies deterministic binary feature mappings,
    performs a stratified train-validation split, fits the ordinal encoder
    exclusively on the training partition, and transforms both partitions
    using the fitted encoder.

    The resulting processed datasets and fitted preprocessing artifact are
    persisted to disk for subsequent training and inference stages.

    Parameters
    ----------
    raw_input_path : str or pathlib.Path
        Path to the raw input dataset.
    train_output_path : str or pathlib.Path
        Destination path for the processed training dataset.
    val_output_path : str or pathlib.Path
        Destination path for the processed validation dataset.
    preprocessor_path : str or pathlib.Path
        Destination path for the serialized preprocessing artifact.

    Raises
    ------
    FileNotFoundError
        If the raw input dataset does not exist.
    ValueError
        If binary mapping, schema alignment, stratified splitting, encoding,
        or feature type conversion fails.
    RuntimeError
        If a processed dataset or preprocessing artifact cannot be persisted.
    """
    logger = get_logger()
    settings = get_settings()

    # Resolve the input and output paths to absolute paths
    raw_input_path = Path(raw_input_path).expanduser().resolve()
    train_output_path = Path(train_output_path).expanduser().resolve()
    val_output_path = Path(val_output_path).expanduser().resolve()
    preprocessor_path = Path(preprocessor_path).expanduser().resolve()

    # Load preprocessing parameters from settings
    binary_columns = settings.PARAMS.schema_config.binary_columns
    ordinal_columns = settings.PARAMS.schema_config.ordinal_columns
    target_col = settings.PARAMS.schema_config.target_column
    test_size = settings.PARAMS.preprocessing.test_size
    random_state = settings.PARAMS.preprocessing.random_state

    # 1. Load the raw dataset
    logger.info("Loading raw dataset from %s", raw_input_path)
    df = pd.read_parquet(raw_input_path)
    logger.debug("Raw dataset loaded. Shape: %s", df.shape)

    # 2. Perform binary value mapping
    for column in [*binary_columns, target_col]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.debug("Applying binary mapping to column: %s", column)
            df[column] = df[column].map(BINARY_MAPPING)

    # 3. Split the dataset into training and validation sets with stratification
    try:
        logger.info("Splitting dataset (test_size=%.2f, random_state=%d)", test_size, random_state)
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[target_col],
            random_state=random_state,
        )
        train_df: pd.DataFrame = train_df.copy()
        val_df: pd.DataFrame = val_df.copy()
        logger.debug(
            "Stratification check - Train target dist: %s | Val target dist: %s",
            train_df[target_col].value_counts(normalize=True).to_dict(),
            val_df[target_col].value_counts(normalize=True).to_dict(),
        )
    except ValueError as err:
        logger.error("Stratified split failed on target column '%s': %s", target_col, err)
        raise ValueError(
            f"Failed to perform stratified split on target '{target_col}': {err}"
        ) from err

    logger.info(
        "Train set shape: %s | Validation set shape: %s",
        train_df.shape,
        val_df.shape,
    )

    # 4. Fit and apply ordinal encoder
    try:
        logger.info(
            "Fitting OrdinalEncoder on training partition for %d features", len(ordinal_columns)
        )
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[ordinal_columns] = encoder.fit_transform(train_df[ordinal_columns])
        val_df[ordinal_columns] = encoder.transform(val_df[ordinal_columns])
        logger.debug("Ordinal encoding completed successfully.")
    except Exception as err:
        logger.error("Ordinal encoding failed: %s", err)
        raise ValueError(f"Failed during ordinal encoding transformation: {err}") from err

    # 5. Coerce data types for memory optimization
    dtype_mapping = {**DEFAULT_DTYPE_MAPPING, target_col: np.uint8}
    try:
        mem_before = train_df.memory_usage(deep=True).sum() / 1024**2
        logger.debug("Memory usage before dtype coercion on train_df: %.2f MB", mem_before)
        logger.info("Casting dataframe features to target schema dtypes")
        train_df = train_df.astype(dtype_mapping)
        val_df = val_df.astype(dtype_mapping)
        mem_after = train_df.memory_usage(deep=True).sum() / 1024**2
        logger.debug("Memory usage after dtype coercion on train_df: %.2f MB", mem_after)
    except Exception as err:
        logger.error("Data type coercion failed: %s", err)
        raise ValueError(f"Failed during data type coercion: {err}") from err

    # 6. Persist processed dataframes to parquet
    try:
        train_df.to_parquet(train_output_path, index=False, engine="pyarrow")
        val_df.to_parquet(val_output_path, index=False, engine="pyarrow")
        logger.info(
            "Persisted processed dataframes: train -> %s | validation -> %s",
            train_output_path,
            val_output_path,
        )
    except Exception as err:
        logger.error("Failed to persist Parquet dataframes: %s", err)
        raise RuntimeError(f"Parquet serialization failed: {err}") from err

    # 7. Serialize the fitted encoder and preprocessing mappings
    joblib.dump(encoder, preprocessor_path)
    logger.info("Fitted encoder serialized to %s", preprocessor_path)
