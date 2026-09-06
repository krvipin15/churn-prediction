"""Test the XGBoost inference module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from churn_prediction.model import inference


@pytest.fixture
def logger_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the application logger."""
    logger = MagicMock()
    monkeypatch.setattr(inference, "get_logger", lambda: logger)
    return logger


@pytest.fixture
def feature_columns() -> list[str]:
    """Return feature columns used by the inference configuration."""
    return ["feature_a", "feature_b"]


@pytest.fixture
def settings_mock(
    monkeypatch: pytest.MonkeyPatch,
    feature_columns: list[str],
) -> SimpleNamespace:
    """Mock the settings object consumed by run_inference."""
    settings = SimpleNamespace(
        PARAMS=SimpleNamespace(
            schema_config=SimpleNamespace(
                feature_columns=feature_columns,
            )
        )
    )

    monkeypatch.setattr(inference, "get_settings", lambda: settings)

    return settings


@pytest.fixture
def input_df() -> pd.DataFrame:
    """Return a small processed inference dataset."""
    return pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.8],
            "feature_b": [1.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Return the original raw customer dataset."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "feature_a": [10, 20, 30],
            "feature_b": ["A", "B", "C"],
        }
    )


@pytest.fixture
def model_card() -> dict[str, dict[str, dict[str, float]]]:
    """Return a valid model card containing inference metadata."""
    return {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 2.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "default_threshold": 0.5,
                "optimal_threshold": 0.4,
            }
        },
    }


@pytest.fixture
def model_card_path(
    tmp_path: Path,
    model_card: dict[str, dict[str, dict[str, float]]],
) -> Path:
    """Write a valid model card to a temporary JSON file."""
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(model_card), encoding="utf-8")
    return path


@pytest.fixture
def parquet_files(
    tmp_path: Path,
    input_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Create temporary processed and raw Parquet datasets."""
    input_path = tmp_path / "input.parquet"
    raw_path = tmp_path / "raw.parquet"

    input_df.to_parquet(input_path, index=False)
    raw_df.to_parquet(raw_path, index=False)

    return raw_path, input_path


def test_run_inference_success(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate calibrated predictions and save them to CSV."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.2, 0.6, 0.9])

    dmatrix = MagicMock(spec=xgb.DMatrix)
    dmatrix.num_row.return_value = 3

    dmatrix_constructor = MagicMock(return_value=dmatrix)
    monkeypatch.setattr(inference.xgb, "DMatrix", dmatrix_constructor)

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda _probabilities, _weight: np.array([0.2, 0.4, 0.8]),
    )

    # Act: Call the function under test
    inference.run_inference(
        booster=booster,
        raw_filepath=raw_path,
        input_filepath=input_path,
        output_filepath=output_path,
        model_card_path=model_card_path,
    )

    # Assert: Verify that the output file exists and contains the expected data
    assert output_path.exists()

    result = pd.read_csv(output_path)

    assert list(result.columns) == [
        "customer_id",
        "feature_a",
        "feature_b",
        "churn_probability",
        "churn_prediction",
    ]

    np.testing.assert_allclose(
        result["churn_probability"].to_numpy(),
        [0.2, 0.4, 0.8],
    )

    np.testing.assert_array_equal(
        result["churn_prediction"].to_numpy(),
        [0, 1, 1],
    )

    booster.predict.assert_called_once_with(dmatrix)

    dmatrix_constructor.assert_called_once()
    call_args = dmatrix_constructor.call_args
    actual_features = call_args.args[0]
    expected_features = pd.read_parquet(input_path)[
        settings_mock.PARAMS.schema_config.feature_columns
    ]
    pd.testing.assert_frame_equal(
        actual_features,
        expected_features,
    )
    assert call_args.kwargs["feature_names"] == (settings_mock.PARAMS.schema_config.feature_columns)


def test_run_inference_uses_default_threshold_when_optimal_missing(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to the default threshold when optimal threshold is absent."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 1.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "default_threshold": 0.7,
            }
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.6, 0.7, 0.9])

    monkeypatch.setattr(
        inference.xgb,
        "DMatrix",
        MagicMock(),
    )

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that the predictions are classified correctly based on the default threshold
    np.testing.assert_array_equal(
        result["churn_prediction"].to_numpy(),
        [0, 1, 1],
    )


def test_run_inference_uses_hardcoded_default_when_both_thresholds_missing(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use 0.5 when neither optimal nor default threshold is present."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 1.0,
            }
        },
        "metrics": {
            "decision_thresholds": {},
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.49, 0.5, 0.51])

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())
    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that the predictions are classified correctly based on the hardcoded default threshold of 0.5
    np.testing.assert_array_equal(
        result["churn_prediction"].to_numpy(),
        [0, 1, 1],
    )


def test_run_inference_missing_model_card_raises_file_not_found(
    tmp_path: Path,
    logger_mock: MagicMock,
    settings_mock: SimpleNamespace,
) -> None:
    """Raise FileNotFoundError when the model card does not exist."""
    # Arrange: Set up the path to a non-existent model card
    model_card_path = tmp_path / "missing_model_card.json"

    # Act & Assert: Verify that FileNotFoundError is raised with the expected message
    with pytest.raises(
        FileNotFoundError,
        match="Model card not found at path",
    ):
        inference.run_inference(
            MagicMock(),
            tmp_path / "raw.parquet",
            tmp_path / "input.parquet",
            tmp_path / "output.csv",
            model_card_path,
        )

    logger_mock.error.assert_called_once()


@pytest.mark.parametrize(
    ("model_card", "error_pattern"),
    [
        (
            {},
            "Failed to extract required parameters from model card",
        ),
        (
            {
                "model_details": {},
                "metrics": {
                    "decision_thresholds": {},
                },
            },
            "Failed to extract required parameters from model card",
        ),
        (
            {
                "model_details": {
                    "hyperparameters": {},
                },
                "metrics": {
                    "decision_thresholds": {},
                },
            },
            "Failed to extract required parameters from model card",
        ),
        (
            {
                "model_details": {
                    "hyperparameters": {
                        "scale_pos_weight": "invalid",
                    }
                },
                "metrics": {
                    "decision_thresholds": {},
                },
            },
            "Failed to extract required parameters from model card",
        ),
    ],
)
def test_run_inference_invalid_model_card_raises_value_error(
    tmp_path: Path,
    model_card: dict,
    error_pattern: str,
    logger_mock: MagicMock,
    settings_mock: SimpleNamespace,
) -> None:
    """Reject malformed or incomplete model-card metadata."""
    # Arrange: Write the invalid model card to a temporary JSON file
    card_path = tmp_path / "model_card.json"
    card_path.write_text(
        json.dumps(model_card),
        encoding="utf-8",
    )

    # Act & Assert: Verify that ValueError is raised with the expected message
    with pytest.raises(ValueError, match=error_pattern):
        inference.run_inference(
            MagicMock(),
            tmp_path / "raw.parquet",
            tmp_path / "input.parquet",
            tmp_path / "output.csv",
            card_path,
        )

    logger_mock.error.assert_called()


def test_run_inference_invalid_threshold_value_raises_value_error(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    logger_mock: MagicMock,
    settings_mock: SimpleNamespace,
) -> None:
    """Reject a model card containing a non-numeric threshold."""
    # Arrange: Write a model card with an invalid threshold value
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 2.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "optimal_threshold": "not-a-number",
                "default_threshold": 0.5,
            }
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    # Act & Assert: Verify that ValueError is raised with the expected message
    with pytest.raises(
        ValueError,
        match="Failed to extract required parameters from model card",
    ):
        inference.run_inference(
            MagicMock(),
            raw_path,
            input_path,
            tmp_path / "output.csv",
            card_path,
        )


def test_run_inference_reads_input_parquet(
    tmp_path: Path,
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the processed inference dataset from the supplied path."""
    # Arrange: Create temporary Parquet files for raw and processed data
    raw_path = tmp_path / "raw.parquet"
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "predictions.csv"

    raw_df = pd.DataFrame({"id": [1, 2]})
    input_df = pd.DataFrame(
        {
            "feature_a": [0.1, 0.9],
            "feature_b": [1.0, 0.0],
        }
    )

    raw_df.to_parquet(raw_path)
    input_df.to_parquet(input_path)

    original_read_parquet = pd.read_parquet
    read_paths: list[Path] = []

    def tracked_read_parquet(path, *args, **kwargs):
        read_paths.append(Path(path))
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(inference.pd, "read_parquet", tracked_read_parquet)
    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.9])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        model_card_path,
    )

    # Assert: Verify that the input Parquet files were read
    assert input_path.resolve() in read_paths
    assert raw_path.resolve() in read_paths


def test_run_inference_missing_feature_column_raises_value_error(
    tmp_path: Path,
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap missing configured feature columns in ValueError."""
    # Arrange: Create temporary Parquet files with missing feature columns
    raw_path = tmp_path / "raw.parquet"
    input_path = tmp_path / "input.parquet"

    pd.DataFrame({"id": [1]}).to_parquet(raw_path)
    pd.DataFrame({"feature_a": [1.0]}).to_parquet(input_path)

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    # Act & Assert: Verify that ValueError is raised when a feature column is missing
    with pytest.raises(ValueError, match="Failed to create DMatrix"):
        inference.run_inference(
            MagicMock(),
            raw_path,
            input_path,
            tmp_path / "output.csv",
            model_card_path,
        )

    logger_mock.error.assert_called()


def test_run_inference_dmatrix_creation_failure_is_wrapped(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap arbitrary DMatrix construction failures in ValueError."""
    # Arrange: Create temporary Parquet files for raw and processed data
    raw_path, input_path = parquet_files

    def fail_dmatrix(*args, **kwargs):
        raise RuntimeError("DMatrix construction failed")

    monkeypatch.setattr(inference.xgb, "DMatrix", fail_dmatrix)

    # Act & Assert: Verify that ValueError is raised when DMatrix creation fails
    with pytest.raises(
        ValueError,
        match="Failed to create DMatrix",
    ):
        inference.run_inference(
            MagicMock(),
            raw_path,
            input_path,
            tmp_path / "output.csv",
            model_card_path,
        )

    logger_mock.error.assert_called_once()


def test_run_inference_calls_unscale_probabilities_with_model_weight(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass raw model probabilities and model-card weight to calibration."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    raw_probabilities = np.array([0.1, 0.4, 0.9])
    booster.predict.return_value = raw_probabilities

    unscale_mock = MagicMock(return_value=np.array([0.05, 0.3, 0.8]))
    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        unscale_mock,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        model_card_path,
    )

    # Assert: Verify that the unscale_probabilities function was called with the correct arguments
    unscale_mock.assert_called_once()

    args = unscale_mock.call_args.args
    np.testing.assert_array_equal(args[0], raw_probabilities)
    assert args[1] == pytest.approx(2.0)


def test_run_inference_rounds_calibrated_probabilities_to_six_places(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round exported churn probabilities to six decimal places."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.123456789, 0.987654321, 0.555555555])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        model_card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that the churn probabilities are rounded to six decimal places
    np.testing.assert_allclose(
        result["churn_probability"].to_numpy(),
        [0.123457, 0.987654, 0.555556],
    )


def test_run_inference_threshold_is_inclusive(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify probabilities equal to the threshold as churn."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 1.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "optimal_threshold": 0.5,
            }
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.499999, 0.5, 0.500001])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that probabilities equal to the threshold are classified as churn
    np.testing.assert_array_equal(
        result["churn_prediction"].to_numpy(),
        [0, 1, 1],
    )


def test_run_inference_preserves_raw_customer_columns(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the original raw customer records in the prediction output."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.6, 0.8])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        model_card_path,
    )

    result = pd.read_csv(output_path)
    original = pd.read_parquet(raw_path)

    # Assert: Verify that the original raw customer columns are preserved in the output
    for column in original.columns:
        assert result[column].tolist() == original[column].tolist()


def test_run_inference_logs_prediction_distribution(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log the number of churn and non-churn predictions."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.6, 0.8])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        model_card_path,
    )

    # Assert: Verify that the logging is correct
    matching_calls = [
        call
        for call in logger_mock.info.call_args_list
        if call.args and isinstance(call.args[0], str) and "Predictions:" in call.args[0]
    ]
    assert matching_calls
    call = matching_calls[-1]
    assert call.args[1:] == (2, 1)


def test_run_inference_passes_configured_features_to_dmatrix(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exactly the configured feature columns for model input."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    dmatrix_mock = MagicMock()
    monkeypatch.setattr(
        inference.xgb,
        "DMatrix",
        dmatrix_mock,
    )

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.6, 0.8])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        tmp_path / "predictions.csv",
        model_card_path,
    )

    call_args = dmatrix_mock.call_args

    actual_features = call_args.args[0]

    expected_features = pd.read_parquet(input_path)[
        settings_mock.PARAMS.schema_config.feature_columns
    ]

    # Assert: Verify that the DMatrix was created with the expected feature columns
    pd.testing.assert_frame_equal(
        actual_features,
        expected_features,
    )

    assert call_args.kwargs["feature_names"] == [
        "feature_a",
        "feature_b",
    ]


def test_run_inference_resolves_paths(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    model_card_path: Path,
    settings_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept relative paths and resolve them before file operations."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files
    output_path = tmp_path / "predictions.csv"

    monkeypatch.chdir(tmp_path)

    relative_raw = Path(raw_path.name)
    relative_input = Path(input_path.name)
    relative_card = Path(model_card_path.name)

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.6, 0.8])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test with relative paths
    inference.run_inference(
        booster,
        relative_raw,
        relative_input,
        output_path,
        relative_card,
    )

    # Assert: Verify that the output file was created
    assert output_path.exists()


def test_run_inference_with_zero_churn_predictions(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle a batch where no customers are classified as churn."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 1.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "optimal_threshold": 0.99,
            }
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.1, 0.2, 0.3])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that no customers are classified as churn
    assert result["churn_prediction"].sum() == 0

    prediction_logs = [
        call
        for call in logger_mock.info.call_args_list
        if call.args and isinstance(call.args[0], str) and "Predictions:" in call.args[0]
    ]

    assert prediction_logs
    assert prediction_logs[-1].args[1:] == (0, 3)


def test_run_inference_with_all_churn_predictions(
    tmp_path: Path,
    parquet_files: tuple[Path, Path],
    settings_mock: SimpleNamespace,
    logger_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle a batch where every customer is classified as churn."""
    # Arrange: Set up the mock objects and data
    raw_path, input_path = parquet_files

    model_card = {
        "model_details": {
            "hyperparameters": {
                "scale_pos_weight": 1.0,
            }
        },
        "metrics": {
            "decision_thresholds": {
                "optimal_threshold": 0.1,
            }
        },
    }

    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(model_card), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"

    monkeypatch.setattr(inference.xgb, "DMatrix", MagicMock())

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.return_value = np.array([0.2, 0.3, 0.4])

    monkeypatch.setattr(
        inference,
        "unscale_probabilities",
        lambda probabilities, _weight: probabilities,
    )

    # Act: Call the function under test
    inference.run_inference(
        booster,
        raw_path,
        input_path,
        output_path,
        card_path,
    )

    result = pd.read_csv(output_path)

    # Assert: Verify that all customers are classified as churn
    assert result["churn_prediction"].sum() == 3

    prediction_logs = [
        call
        for call in logger_mock.info.call_args_list
        if call.args and isinstance(call.args[0], str) and "Predictions:" in call.args[0]
    ]

    assert prediction_logs
    assert prediction_logs[-1].args[1:] == (3, 0)
