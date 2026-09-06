"""Test the train data preprocessing module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.features.train_preprocessing import preprocess_train_dataset


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create a mock settings object mirroring the expected structure."""
    settings = MagicMock()

    settings.PARAMS = SimpleNamespace(
        schema_config=SimpleNamespace(
            binary_columns=[
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "PaperlessBilling",
            ],
            ordinal_columns=[
                "InternetService",
                "Contract",
                "PaymentMethod",
            ],
            target_column="Churn",
        ),
        preprocessing=SimpleNamespace(
            test_size=0.2,
            random_state=42,
        ),
    )

    return settings


@pytest.fixture
def raw_data_df() -> pd.DataFrame:
    """Create a dummy dataframe that matches the expected schema."""
    data = {
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "Yes", "No"],
        "InternetService": [
            "DSL",
            "Fiber optic",
            "DSL",
            "No",
            "Fiber optic",
        ],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
        ],
        "Churn": ["No", "Yes", "No", "Yes", "No"],
        "SeniorCitizen": [0, 1, 0, 1, 0],
        "tenure": [1, 23, 40, 12, 5],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No", "No", "Yes"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes", "No"],
        "OnlineBackup": ["No", "No", "Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No", "No"],
        "TechSupport": ["No", "No", "Yes", "No", "No"],
        "StreamingTV": ["No", "Yes", "No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes", "No", "Yes"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 151.65],
    }

    df = pd.DataFrame(data)
    return pd.concat([df] * 4, ignore_index=True)


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """Provide temporary paths for input and output files."""
    return {
        "input": tmp_path / "raw.parquet",
        "train": tmp_path / "train.parquet",
        "val": tmp_path / "val.parquet",
        "preprocessor": tmp_path / "encoder.joblib",
    }


@patch("churn_prediction.features.train_preprocessing.get_logger")
@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_success(
    mock_get_settings: MagicMock,
    mock_get_logger: MagicMock,
    mock_settings: MagicMock,
    raw_data_df: pd.DataFrame,
    paths: dict[str, Path],
):
    """Test the full successful pipeline from raw parquet to processed artifacts."""
    # Arrange: Set up the necessary mocks and data.
    mock_get_logger.return_value = MagicMock()
    mock_get_settings.return_value = mock_settings

    raw_data_df.to_parquet(paths["input"])

    # Act: Call the preprocessing function
    preprocess_train_dataset(
        raw_input_path=paths["input"],
        train_output_path=paths["train"],
        val_output_path=paths["val"],
        preprocessor_path=paths["preprocessor"],
    )

    # Assert: Check that the output files exist
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["preprocessor"].exists()

    # Verify contents
    train_df = pd.read_parquet(paths["train"])
    val_df = pd.read_parquet(paths["val"])

    # Check binary mapping (e.g., Male -> 1, Female -> 0)
    assert train_df["gender"].dtype == np.uint8
    assert val_df["gender"].dtype == np.uint8
    assert set(train_df["gender"].unique()).issubset({0, 1})
    assert set(val_df["gender"].unique()).issubset({0, 1})

    # Check ordinal encoding
    assert train_df["InternetService"].dtype == np.int16
    assert train_df["Contract"].dtype == np.int16
    assert val_df["InternetService"].dtype == np.int16
    assert val_df["Contract"].dtype == np.int16

    # Check preprocessor is a valid OrdinalEncoder
    encoder = joblib.load(paths["preprocessor"])
    assert isinstance(encoder, OrdinalEncoder)


@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_file_not_found(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    paths: dict[str, Path],
):
    """Test that FileNotFoundError is raised when input path doesn't exist."""
    mock_get_settings.return_value = mock_settings

    with pytest.raises(FileNotFoundError):
        preprocess_train_dataset(
            raw_input_path="non_existent_file.parquet",
            train_output_path=paths["train"],
            val_output_path=paths["val"],
            preprocessor_path=paths["preprocessor"],
        )


@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_stratify_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    paths: dict[str, Path],
):
    """Test that ValueError is raised when stratification fails (e.g., too few classes)."""
    mock_get_settings.return_value = mock_settings

    # Create a dataframe where the target column has only one unique value
    df_fail = pd.DataFrame(
        {
            "gender": ["Male"] * 10,
            "Partner": ["Yes"] * 10,
            "Dependents": ["No"] * 10,
            "InternetService": ["DSL"] * 10,
            "Contract": ["Month-to-month"] * 10,
            "Churn": ["Yes"] + ["No"] * 9,
            "SeniorCitizen": [0] * 10,
            "tenure": [1] * 10,
            "PhoneService": ["Yes"] * 10,
            "MultipleLines": ["No"] * 10,
            "OnlineSecurity": ["No"] * 10,
            "OnlineBackup": ["No"] * 10,
            "DeviceProtection": ["No"] * 10,
            "TechSupport": ["No"] * 10,
            "StreamingTV": ["No"] * 10,
            "StreamingMovies": ["No"] * 10,
            "PaperlessBilling": ["Yes"] * 10,
            "PaymentMethod": ["CC"] * 10,
            "MonthlyCharges": [20.0] * 10,
            "TotalCharges": [20.0] * 10,
        }
    )
    df_fail.to_parquet(paths["input"])

    with pytest.raises(ValueError, match="stratified split"):
        preprocess_train_dataset(
            raw_input_path=paths["input"],
            train_output_path=paths["train"],
            val_output_path=paths["val"],
            preprocessor_path=paths["preprocessor"],
        )


@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_encoding_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_data_df: pd.DataFrame,
    paths: dict[str, Path],
):
    """Test that ValueError is raised when ordinal encoding fails (e.g., incompatible types)."""
    mock_get_settings.return_value = mock_settings

    # Corrupt the ordinal columns by putting in un-encodable objects (like lists)
    raw_data_df["InternetService"] = [[1, 2]] * len(raw_data_df)
    raw_data_df.to_parquet(paths["input"])

    with pytest.raises(ValueError, match="Failed during ordinal encoding transformation"):
        preprocess_train_dataset(
            raw_input_path=paths["input"],
            train_output_path=paths["train"],
            val_output_path=paths["val"],
            preprocessor_path=paths["preprocessor"],
        )


@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_dtype_coercion_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_data_df: pd.DataFrame,
    paths: dict[str, Path],
):
    """Test that ValueError is raised when dtype coercion fails."""
    mock_get_settings.return_value = mock_settings

    raw_data_df["MonthlyCharges"] = "not_a_number"
    raw_data_df.to_parquet(paths["input"])

    with pytest.raises(ValueError, match="Failed during data type coercion"):
        preprocess_train_dataset(
            raw_input_path=paths["input"],
            train_output_path=paths["train"],
            val_output_path=paths["val"],
            preprocessor_path=paths["preprocessor"],
        )


@patch("churn_prediction.features.train_preprocessing.get_settings")
def test_preprocess_train_dataset_serialization_failure(
    mock_get_settings: MagicMock,
    mock_settings: MagicMock,
    raw_data_df: pd.DataFrame,
    paths: dict[str, Path],
):
    """Test that RuntimeError is raised when writing parquet fails (e.g., read-only directory)."""
    mock_get_settings.return_value = mock_settings
    raw_data_df.to_parquet(paths["input"])

    # Mock to_parquet to raise an Exception
    with (
        patch(
            "pandas.DataFrame.to_parquet",
            side_effect=Exception("Disk Full"),
        ),
        pytest.raises(
            RuntimeError,
            match="Parquet serialization failed",
        ),
    ):
        preprocess_train_dataset(
            raw_input_path=paths["input"],
            train_output_path=paths["train"],
            val_output_path=paths["val"],
            preprocessor_path=paths["preprocessor"],
        )
