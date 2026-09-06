"""Test the dataset validation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pandas as pd
import pandera.errors as pa_errors
import pytest

from churn_prediction.data.validation import (
    _extract_failures,
    _write_report,
    validate_dataset,
)


@pytest.fixture
def mock_schema() -> MagicMock:
    """Provide a mocked Pandera DataFrameSchema."""
    schema = MagicMock()
    schema.name = "TestSchema"
    return schema


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a basic sample DataFrame for validation tests."""
    return pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )


def make_schema_error(
    message: str = "validation failed",
    column: str = "target",
    check: str = "custom_check",
    failure_case: object = "bad",
) -> pa_errors.SchemaError:
    """Create a real Pandera SchemaError for testing."""
    schema = MagicMock()
    schema.name = column

    error = pa_errors.SchemaError(
        schema=schema,
        data=pd.DataFrame({column: [failure_case]}),
        message=message,
    )

    error.check = check
    error.failure_cases = failure_case

    return error


def test_extract_failures_schema_errors_with_values():
    """Extract multiple populated Pandera failure records."""
    # Simulate a Pandera SchemaErrors object containing a DataFrame of failures
    failure_cases = pd.DataFrame(
        [
            {
                "column": "age",
                "check": "greater_than(0)",
                "failure_case": -1,
                "index": 3,
            },
            {
                "column": "target",
                "check": "isin([0, 1])",
                "failure_case": 2,
                "index": 7,
            },
        ]
    )

    err = MagicMock(spec=pa_errors.SchemaErrors)
    err.failure_cases = failure_cases

    failures = _extract_failures(err)

    # Ensure numeric values are cast to strings and error messages are formatted correctly
    assert failures == [
        {
            "column": "age",
            "check": "greater_than(0)",
            "failure_case": "-1",
            "index": "3",
            "error": "Check 'greater_than(0)' failed for column 'age'",
        },
        {
            "column": "target",
            "check": "isin([0, 1])",
            "failure_case": "2",
            "index": "7",
            "error": "Check 'isin([0, 1])' failed for column 'target'",
        },
    ]


def test_extract_failures_schema_errors_with_null_values():
    """Convert null failure-case values to None."""
    # Test resilience when Pandera returns empty or null failure metadata
    failure_cases = pd.DataFrame(
        [
            {
                "column": None,
                "check": None,
                "failure_case": None,
                "index": None,
            }
        ]
    )

    err = MagicMock(spec=pa_errors.SchemaErrors)
    err.failure_cases = failure_cases

    failures = _extract_failures(err)

    assert failures == [
        {
            "column": None,
            "check": None,
            "failure_case": None,
            "index": None,
            "error": "Check 'None' failed for column 'None'",
        }
    ]


def test_extract_failures_empty_schema_errors():
    """Return no failures when SchemaErrors has empty failure cases."""
    err = MagicMock(spec=pa_errors.SchemaErrors)
    err.failure_cases = pd.DataFrame()

    assert _extract_failures(err) == []


def test_extract_failures_schema_error_with_attributes():
    """Extract a single Pandera SchemaError."""
    # Test handling of a single SchemaError instead of a collection (SchemaErrors)
    schema_obj = MagicMock()
    schema_obj.name = "age"

    err = MagicMock(spec=pa_errors.SchemaError)
    err.schema = schema_obj
    err.check = "greater_than(0)"
    err.failure_cases = pd.DataFrame({"failure_case": [-1]})
    err.__str__.return_value = "age validation failed"

    failures = _extract_failures(err)

    assert failures == [
        {
            "column": "age",
            "check": "greater_than(0)",
            "failure_case": str(err.failure_cases),
            "index": None,
            "error": "age validation failed",
        }
    ]


def test_extract_failures_schema_error_without_optional_attributes():
    """Handle SchemaError objects without schema/check/failure_cases."""
    # Ensure the extractor doesn't crash if optional Pandera attributes are missing
    err = MagicMock(spec=pa_errors.SchemaError)

    del err.schema
    del err.check
    del err.failure_cases

    err.__str__.return_value = "validation failed"

    failures = _extract_failures(err)

    assert failures == [
        {
            "column": None,
            "check": None,
            "failure_case": None,
            "index": None,
            "error": "validation failed",
        }
    ]


def test_write_report(tmp_path: Path):
    """Write a validation report successfully."""
    report_path = tmp_path / "report.json"

    report_data = {
        "status": "PASSED",
        "failure_count": 0,
        "failures": [],
    }

    _write_report(report_path, report_data)

    assert report_path.exists()
    assert orjson.loads(report_path.read_bytes()) == report_data


def test_write_report_serialization_error(tmp_path: Path):
    """Wrap report serialization failures in OSError."""
    report_path = tmp_path / "report.json"

    # Simulate orjson failing to serialize a non-serializable object
    with (
        patch(
            "churn_prediction.data.validation.orjson.dumps",
            side_effect=TypeError("serialization failed"),
        ),
        pytest.raises(
            OSError,
            match="Could not persist validation report",
        ),
    ):
        _write_report(report_path, {"status": "PASSED"})


def test_write_report_write_error(tmp_path: Path):
    """Wrap report write failures in OSError."""
    report_path = tmp_path / "report.json"

    # Simulate filesystem permission or disk errors
    with (
        patch.object(
            Path,
            "write_bytes",
            side_effect=OSError("permission denied"),
        ),
        pytest.raises(
            OSError,
            match="Could not persist validation report",
        ),
    ):
        _write_report(report_path, {"status": "PASSED"})


def test_validate_dataset_missing_file(tmp_path: Path, mock_schema):
    """Raise FileNotFoundError for a missing dataset."""
    dataset_path = tmp_path / "missing.parquet"
    report_path = tmp_path / "report.json"

    with pytest.raises(
        FileNotFoundError,
        match="Dataset file not found",
    ):
        validate_dataset(
            dataset_path,
            report_path,
            mock_schema,
        )


def test_validate_dataset_directory(tmp_path: Path, mock_schema):
    """Reject a dataset path that points to a directory."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    with pytest.raises(
        ValueError,
        match="directory or special file",
    ):
        validate_dataset(
            dataset_path,
            tmp_path / "report.json",
            mock_schema,
        )


def test_validate_dataset_read_failure(tmp_path: Path, mock_schema):
    """Wrap Parquet loading failures in ValueError."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"invalid parquet")

    # Simulate a corrupted parquet file causing read_parquet to fail
    with (
        patch(
            "churn_prediction.data.validation.pd.read_parquet",
            side_effect=OSError("invalid parquet"),
        ),
        pytest.raises(
            ValueError,
            match="Failed to load dataset",
        ),
    ):
        validate_dataset(
            dataset_path,
            tmp_path / "report.json",
            mock_schema,
        )


def test_validate_dataset_passes(tmp_path: Path, mock_schema, sample_df):
    """Validate a valid dataset."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    report_path = tmp_path / "report.json"

    with patch(
        "churn_prediction.data.validation.pd.read_parquet",
        return_value=sample_df,
    ):
        validate_dataset(
            dataset_path,
            report_path,
            mock_schema,
        )

    # Verify that Pandera's validate was called and no report was written for success
    mock_schema.validate.assert_called_once_with(sample_df, lazy=True)
    assert not report_path.exists()


def test_validate_dataset_lazy_false(tmp_path: Path, mock_schema, sample_df):
    """Pass lazy=False through to Pandera validation."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    with patch(
        "churn_prediction.data.validation.pd.read_parquet",
        return_value=sample_df,
    ):
        validate_dataset(
            dataset_path,
            tmp_path / "report.json",
            mock_schema,
            lazy=False,
        )

    mock_schema.validate.assert_called_once_with(
        sample_df,
        lazy=False,
    )


def test_validate_dataset_schema_error(tmp_path: Path, mock_schema):
    """Persist diagnostics and raise ValueError for SchemaError."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    report_path = tmp_path / "report.json"

    df = pd.DataFrame({"age": [-1]})

    error = make_schema_error(
        message="age must be greater than 0",
        column="age",
        check="greater_than(0)",
        failure_case=-1,
    )

    mock_schema.validate.side_effect = error

    # Validate that a SchemaError triggers a report write and a ValueError
    with (
        patch(
            "churn_prediction.data.validation.pd.read_parquet",
            return_value=df,
        ),
        pytest.raises(
            ValueError,
            match="Dataset validation failed with 1 violation",
        ),
    ):
        validate_dataset(
            dataset_path,
            report_path,
            mock_schema,
        )

    assert report_path.exists()

    report = orjson.loads(report_path.read_bytes())

    assert report["status"] == "FAILED"
    assert report["failure_count"] == 1
    assert report["failures"][0]["column"] == "age"
    assert report["failures"][0]["check"] == "greater_than(0)"


def test_validate_dataset_schema_errors(tmp_path: Path, mock_schema):
    """Persist all diagnostics for multiple validation failures."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    report_path = tmp_path / "report.json"

    df = pd.DataFrame(
        {
            "age": [-1],
            "target": [2],
        }
    )

    error = make_schema_error(
        message="multiple validation failures",
        column="age",
        check="multiple_checks",
        failure_case=-1,
    )

    mock_schema.validate.side_effect = error

    # Mock the extraction process to return a list of multiple failures
    failures = [
        {
            "column": "age",
            "check": "greater_than(0)",
            "failure_case": "-1",
            "index": "0",
            "error": "Check 'greater_than(0)' failed for column 'age'",
        },
        {
            "column": "target",
            "check": "isin([0, 1])",
            "failure_case": "2",
            "index": "0",
            "error": "Check 'isin([0, 1])' failed for column 'target'",
        },
    ]

    with (
        patch(
            "churn_prediction.data.validation.pd.read_parquet",
            return_value=df,
        ),
        patch(
            "churn_prediction.data.validation._extract_failures",
            return_value=failures,
        ),
        pytest.raises(
            ValueError,
            match="Dataset validation failed with 2 violation",
        ),
    ):
        validate_dataset(
            dataset_path,
            report_path,
            mock_schema,
        )

    report = orjson.loads(report_path.read_bytes())

    assert report["status"] == "FAILED"
    assert report["failure_count"] == 2
    assert len(report["failures"]) == 2

    assert report["failures"][0]["column"] == "age"
    assert report["failures"][1]["column"] == "target"


def test_validate_dataset_schema_errors_without_columns(tmp_path: Path, mock_schema):
    """Exercise the empty failed-column summary branch."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    report_path = tmp_path / "report.json"

    df = pd.DataFrame({"value": [1]})

    error = make_schema_error(
        message="validation failed",
        column="value",
        check="custom_check",
        failure_case="bad",
    )

    mock_schema.validate.side_effect = error

    # Simulate a failure where the column name cannot be determined
    failures = [
        {
            "column": None,
            "check": "custom_check",
            "failure_case": "bad",
            "index": None,
            "error": "validation failed",
        }
    ]

    with (
        patch(
            "churn_prediction.data.validation.pd.read_parquet",
            return_value=df,
        ),
        patch(
            "churn_prediction.data.validation._extract_failures",
            return_value=failures,
        ),
        pytest.raises(
            ValueError,
            match="Dataset validation failed",
        ),
    ):
        validate_dataset(
            dataset_path,
            report_path,
            mock_schema,
        )

    report = orjson.loads(report_path.read_bytes())

    assert report["status"] == "FAILED"
    assert report["failure_count"] == 1
    assert report["failures"][0]["column"] is None


def test_validate_dataset_failed_report_write_failure(tmp_path: Path, mock_schema):
    """Propagate report-writing errors after failed validation."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    df = pd.DataFrame({"target": [2]})

    error = make_schema_error(
        message="target failed",
        column="target",
        check="isin([0, 1])",
        failure_case=2,
    )

    mock_schema.validate.side_effect = error

    # Ensure that if the report cannot be written, the original I/O error is propagated
    with (
        patch(
            "churn_prediction.data.validation.pd.read_parquet",
            return_value=df,
        ),
        patch(
            "churn_prediction.data.validation._write_report",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        validate_dataset(
            dataset_path,
            tmp_path / "report.json",
            mock_schema,
        )
