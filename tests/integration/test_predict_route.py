"""Test the /predict route of the FastAPI application."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from churn_prediction.api.app import app
from churn_prediction.api.dependencies import get_encoder, get_model

# Create a TestClient for the FastAPI app
client = TestClient(app)


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    return MagicMock()


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MagicMock()


@pytest.fixture
def valid_csv_file(tmp_path):
    """Create a temporary valid CSV file for testing."""
    content = "id,feature1,feature2\n1,val1,val2\n2,val3,val4"
    with Path(tmp_path / "test_input.csv").open("w") as f:
        f.write(content)
    return tmp_path / "test_input.csv"


@pytest.fixture
def mock_prediction_file(tmp_path):
    """Create a fake prediction output file on disk."""
    prediction_file = tmp_path / "prediction_12345678T123456_abcd.csv"
    prediction_file.write_text("id,prediction\n1,0.5\n2,0.8")
    return prediction_file


def override_encoder():
    """Override the encoder dependency with a mock."""
    return MagicMock()


def override_model():
    """Override the model dependency with a mock."""
    return MagicMock()


app.dependency_overrides[get_encoder] = override_encoder
app.dependency_overrides[get_model] = override_model


def test_predict_batch_success(valid_csv_file):
    """Verify that a valid CSV upload triggers the pipeline and returns a batch_id."""
    with (
        patch("churn_prediction.api.routes.predict.run_prediction_pipeline") as mock_pipeline,
        patch("churn_prediction.api.routes.predict._to_parquet") as mock_to_parquet,
    ):
        mock_to_parquet.return_value = None
        mock_pipeline.return_value = None

        with Path(valid_csv_file).open("rb") as f:
            response = client.post(
                "/api/v1/predict/batch", files={"file": ("test.csv", f, "text/csv")}
            )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "batch_id" in data
    assert data["status"] == "SUCCESS"
    assert "download_url" in data
    assert data["filename"] == "test.csv"

    # Ensure the pipeline was actually called
    mock_pipeline.assert_called_once()


def test_predict_batch_invalid_extension(tmp_path):
    """Verify that uploading a non-CSV file returns a 400 error."""
    with Path(tmp_path / "test.txt").open("w") as f:
        f.write("not a csv")

    with Path(tmp_path / "test.txt").open("rb") as f:
        response = client.post(
            "/api/v1/predict/batch", files={"file": ("test.txt", f, "text/plain")}
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid file format" in response.json()["detail"]


def test_predict_batch_malformed_csv(tmp_path):
    """Verify that a CSV that cannot be parsed returns a 400 error."""
    # Create a file that would cause pandas to fail (e.g., totally random bytes)
    with Path(tmp_path / "corrupt.csv").open("wb") as f:
        f.write(b"\x00\xff\x00\xff")

    with Path(tmp_path / "corrupt.csv").open("rb") as f:
        response = client.post(
            "/api/v1/predict/batch", files={"file": ("corrupt.csv", f, "text/csv")}
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Failed to parse CSV payload" in response.json()["detail"]


def test_predict_batch_pipeline_failure(valid_csv_file):
    """Verify that if the prediction pipeline crashes, a 422 is returned."""
    with (
        patch("churn_prediction.api.routes.predict._to_parquet"),
        patch("churn_prediction.api.routes.predict.run_prediction_pipeline") as mock_pipeline,
    ):
        mock_pipeline.side_effect = Exception("Pipeline Crash")

        with Path(valid_csv_file).open("rb") as f:
            response = client.post(
                "/api/v1/predict/batch", files={"file": ("test.csv", f, "text/csv")}
            )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "Batch execution pipeline failed" in response.json()["detail"]


def test_download_predictions_success(mock_prediction_file):
    """Verify that a valid batch_id returns the CSV file."""
    batch_id = "12345678T123456_abcd"

    # Mock settings to point to the tmp_path created by the fixture
    with patch("churn_prediction.api.routes.predict.get_settings") as mock_settings:
        mock_settings.return_value.PREDICTION_DATA_DIR = str(mock_prediction_file.parent)

        response = client.get(f"/api/v1/predictions/download/{batch_id}")

    assert response.status_code == status.HTTP_200_OK
    assert "text/csv" in response.headers["content-type"]
    assert b"id,prediction" in response.content


def test_download_predictions_not_found():
    """Verify that a non-existent batch_id returns a 404."""
    batch_id = "00000000T000000_dead"
    response = client.get(f"/api/v1/predictions/download/{batch_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_download_predictions_invalid_id():
    """Verify that a batch_id not matching the regex pattern returns 422 (FastAPI validation)."""
    invalid_id = "not-a-batch-id"
    response = client.get(f"/api/v1/predictions/download/{invalid_id}")

    # FastAPI returns 422 for Path parameter validation failures (Regex)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
