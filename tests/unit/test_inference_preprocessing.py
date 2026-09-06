"""Test the inference data preprocessing module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.features.inference_preprocessing import (
    preprocess_inference_dataset,
)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create settings with the schema required by inference preprocessing."""
    settings = MagicMock()

    settings.PARAMS = SimpleNamespace(
        schema_config=SimpleNamespace(
            binary_columns=[
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "PaperlessBilling",
            ],
            ordinal_columns=[
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaymentMethod",
            ],
        )
    )

    return settings


@pytest.fixture
def raw_inference_df() -> pd.DataFrame:
    """Create a raw inference dataframe matching the expected schema."""
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes"],
            "tenure": [1, 23, 40, 12],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": [
                "No",
                "Yes",
                "No phone service",
                "No",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "No",
                "DSL",
            ],
            "OnlineSecurity": [
                "No",
                "Yes",
                "No internet service",
                "No",
            ],
            "OnlineBackup": [
                "No",
                "No",
                "No internet service",
                "Yes",
            ],
            "DeviceProtection": [
                "No",
                "Yes",
                "No internet service",
                "No",
            ],
            "TechSupport": [
                "No",
                "No",
                "No internet service",
                "Yes",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No internet service",
                "No",
            ],
            "StreamingMovies": [
                "No",
                "No",
                "No internet service",
                "Yes",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": [29.85, 1889.5, 108.15, 1840.75],
        }
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """Provide temporary input and output paths."""
    return {
        "input": tmp_path / "raw_inference.parquet",
        "output": tmp_path / "processed_inference.parquet",
    }


@pytest.fixture
def fitted_encoder(mock_settings: MagicMock) -> OrdinalEncoder:
    """Create an OrdinalEncoder fitted on representative training categories."""
    ordinal_columns = mock_settings.PARAMS.schema_config.ordinal_columns

    training_data = pd.DataFrame(
        {
            "MultipleLines": [
                "No",
                "Yes",
                "No phone service",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "No",
            ],
            "OnlineSecurity": [
                "No",
                "Yes",
                "No internet service",
            ],
            "OnlineBackup": [
                "No",
                "Yes",
                "No internet service",
            ],
            "DeviceProtection": [
                "No",
                "Yes",
                "No internet service",
            ],
            "TechSupport": [
                "No",
                "Yes",
                "No internet service",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No internet service",
            ],
            "StreamingMovies": [
                "No",
                "Yes",
                "No internet service",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
            ],
        }
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    encoder.fit(training_data[ordinal_columns])

    return encoder


@patch("churn_prediction.features.inference_preprocessing.get_logger")
@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_success(
    mock_get_settings: MagicMock,
    mock_get_logger: MagicMock,
    mock_settings: MagicMock,
    *,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test successful inference preprocessing and persistence."""
    # Arrange: Set up the mock objects and data
    mock_get_logger.return_value = MagicMock()
    mock_get_settings.return_value = mock_settings

    raw_inference_df.to_parquet(paths["input"])

    # Act: Call the preprocessing function
    preprocess_inference_dataset(
        raw_input_path=paths["input"],
        processed_output_path=paths["output"],
        fitted_encoder=fitted_encoder,
    )

    # Assert: Check that the output file exists and the data is processed correctly
    assert paths["output"].exists()

    processed_df = pd.read_parquet(paths["output"])

    assert len(processed_df) == len(raw_inference_df)

    # Binary columns are converted to numeric values.
    assert processed_df["gender"].dtype == np.uint8
    assert processed_df["Partner"].dtype == np.uint8
    assert processed_df["Dependents"].dtype == np.uint8
    assert processed_df["PhoneService"].dtype == np.uint8
    assert processed_df["PaperlessBilling"].dtype == np.uint8

    assert set(processed_df["gender"].unique()).issubset({0, 1})
    assert set(processed_df["Partner"].unique()).issubset({0, 1})
    assert set(processed_df["Dependents"].unique()).issubset({0, 1})

    # Ordinal columns are encoded and cast to the expected dtype.
    assert processed_df["MultipleLines"].dtype == np.int16
    assert processed_df["InternetService"].dtype == np.int16
    assert processed_df["Contract"].dtype == np.int16
    assert processed_df["PaymentMethod"].dtype == np.int16

    # Numeric features retain their required dtypes.
    assert processed_df["SeniorCitizen"].dtype == np.uint8
    assert processed_df["tenure"].dtype == np.uint8
    assert processed_df["MonthlyCharges"].dtype == np.float32
    assert processed_df["TotalCharges"].dtype == np.float32


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_file_not_found(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that a missing raw input raises FileNotFoundError."""
    # Arrange: Set up the mock objects and paths
    mock_get_settings.return_value = mock_settings

    # Act & Assert: Call the preprocessing function with a non-existent input path and expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        preprocess_inference_dataset(
            raw_input_path=paths["input"],
            processed_output_path=paths["output"],
            fitted_encoder=fitted_encoder,
        )


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_unknown_category(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that unknown ordinal categories are encoded as -1."""
    # Arrange: Set up the mock objects and data
    mock_get_settings.return_value = mock_settings

    raw_inference_df["Contract"] = [
        "Unknown contract",
        "One year",
        "Two year",
        "Month-to-month",
    ]

    raw_inference_df.to_parquet(paths["input"])

    # Act: Call the preprocessing function
    preprocess_inference_dataset(
        raw_input_path=paths["input"],
        processed_output_path=paths["output"],
        fitted_encoder=fitted_encoder,
    )

    processed_df = pd.read_parquet(paths["output"])

    # Assert: Check that the unknown category is encoded as -1 and the dtype is correct
    assert processed_df.loc[0, "Contract"] == -1
    assert processed_df["Contract"].dtype == np.int16


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_encoding_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that encoder transformation errors are wrapped in ValueError."""
    # Arrange: Set up the mock objects and data
    mock_get_settings.return_value = mock_settings
    raw_inference_df.to_parquet(paths["input"])

    # Act & Assert: Patch the encoder's transform method to raise an exception and expect a ValueError
    with (
        patch.object(
            fitted_encoder,
            "transform",
            side_effect=Exception("Encoder failure"),
        ),
        pytest.raises(
            ValueError,
            match="Ordinal encoding transformation failed",
        ),
    ):
        preprocess_inference_dataset(
            raw_input_path=paths["input"],
            processed_output_path=paths["output"],
            fitted_encoder=fitted_encoder,
        )


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_dtype_coercion_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that invalid numeric data raises a dtype coercion ValueError."""
    # Arrange: Set up the mock objects and data
    mock_get_settings.return_value = mock_settings

    raw_inference_df["MonthlyCharges"] = "not_a_number"
    raw_inference_df.to_parquet(paths["input"])

    # Act & Assert: Call the preprocessing function and expect a ValueError due to dtype coercion failure
    with pytest.raises(
        ValueError,
        match="Data type coercion failed on inference data",
    ):
        preprocess_inference_dataset(
            raw_input_path=paths["input"],
            processed_output_path=paths["output"],
            fitted_encoder=fitted_encoder,
        )


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_save_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that output serialization errors are wrapped in ValueError."""
    # Arrange: Set up the mock objects and data
    mock_get_settings.return_value = mock_settings

    raw_inference_df.to_parquet(paths["input"])

    # Act & Assert: Patch the DataFrame's to_parquet method to raise an exception and expect a ValueError
    with (
        patch(
            "pandas.DataFrame.to_parquet",
            side_effect=Exception("Disk Full"),
        ),
        pytest.raises(
            ValueError,
            match="Failed to save preprocessed data",
        ),
    ):
        preprocess_inference_dataset(
            raw_input_path=paths["input"],
            processed_output_path=paths["output"],
            fitted_encoder=fitted_encoder,
        )


@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_dataset_numeric_binary_columns(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    paths: dict[str, Path],
):
    """Test that already-numeric binary columns skip binary mapping."""
    # Arrange: Set up the mock objects and data
    mock_get_settings.return_value = mock_settings

    raw_inference_df["gender"] = [1, 0, 1, 0]
    raw_inference_df["Partner"] = [1, 0, 1, 0]
    raw_inference_df["Dependents"] = [0, 1, 0, 1]
    raw_inference_df["PhoneService"] = [1, 1, 0, 1]
    raw_inference_df["PaperlessBilling"] = [1, 0, 1, 0]

    raw_inference_df.to_parquet(paths["input"])

    # Act: Call the preprocessing function
    preprocess_inference_dataset(
        raw_input_path=paths["input"],
        processed_output_path=paths["output"],
        fitted_encoder=fitted_encoder,
    )

    processed_df = pd.read_parquet(paths["output"])

    # Assert: Check that the numeric binary columns remain unchanged
    assert processed_df["gender"].tolist() == [1, 0, 1, 0]
    assert processed_df["Partner"].tolist() == [1, 0, 1, 0]
    assert processed_df["Dependents"].tolist() == [0, 1, 0, 1]
    assert processed_df["PhoneService"].tolist() == [1, 1, 0, 1]
    assert processed_df["PaperlessBilling"].tolist() == [1, 0, 1, 0]


@patch("churn_prediction.features.inference_preprocessing.get_logger")
@patch("churn_prediction.features.inference_preprocessing.get_settings")
def test_preprocess_inference_datasetpaths_are_resolved(
    mock_get_settings: MagicMock,
    mock_get_logger: MagicMock,
    mock_settings: MagicMock,
    *,
    raw_inference_df: pd.DataFrame,
    fitted_encoder: OrdinalEncoder,
    tmp_path: Path,
):
    """Test that relative input and output paths are resolved correctly."""
    # Arrange: Set up the mock objects and data
    mock_get_logger.return_value = MagicMock()
    mock_get_settings.return_value = mock_settings

    input_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "processed.parquet"

    raw_inference_df.to_parquet(input_path)

    # Act: Call the preprocessing function with Path objects
    preprocess_inference_dataset(
        raw_input_path=str(input_path),
        processed_output_path=str(output_path),
        fitted_encoder=fitted_encoder,
    )

    # Assert: Check that the output file exists
    assert output_path.exists()
