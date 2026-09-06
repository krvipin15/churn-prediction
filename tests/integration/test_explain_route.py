"""Test the /explain route of the FastAPI application."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from churn_prediction.api.app import app
from churn_prediction.api.dependencies import get_model

# Create a TestClient for the FastAPI app
client = TestClient(app)

# Override dependencies to return mocks instead of loading real models from disk
app.dependency_overrides[get_model] = MagicMock()


@pytest.fixture
def mock_processed_file(tmp_path: Path) -> Path:
    """Create a fake processed parquet file on disk."""
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    processed_file = tmp_path / "processed_test_batch_123.parquet"
    df.to_parquet(processed_file)
    return processed_file


def test_generate_explanation_success(mock_processed_file: Path, tmp_path: Path):
    """Verify that a valid batch_id with existing data returns success and artifacts path."""
    batch_id = "test_batch_123"

    with (
        patch("churn_prediction.api.routes.explain.get_settings") as mock_settings,
        patch("churn_prediction.api.routes.explain.generate_shap_artifacts") as mock_shap,
    ):
        settings = MagicMock()
        settings.PROCESSED_DATA_DIR = str(mock_processed_file.parent)
        settings.SHAP_REPORT_DIR = str(tmp_path / "reports")
        mock_settings.return_value = settings

        mock_shap.return_value = None

        response = client.post(f"/explain/{batch_id}")

    assert response.status_code == status.HTTP_200_OK
    data: dict[str, str] = response.json()
    assert data["batch_id"] == batch_id
    assert data["status"] == "success"
    assert "artifacts_path" in data

    mock_shap.assert_called_once()


def test_generate_explanation_not_found(tmp_path: Path):
    """Verify that a batch_id with no corresponding processed file returns 404."""
    batch_id = "non_existent_batch"

    with patch("churn_prediction.api.routes.explain.get_settings") as mock_settings:
        settings = MagicMock()
        settings.PROCESSED_DATA_DIR = str(tmp_path)  # Empty dir
        settings.SHAP_REPORT_DIR = str(tmp_path / "reports")
        mock_settings.return_value = settings

        response = client.post(f"/explain/{batch_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_generate_explanation_internal_error(mock_processed_file: Path, tmp_path: Path):
    """Verify that a crash during SHAP generation returns a 500 error."""
    batch_id = "test_batch_123"

    with (
        patch("churn_prediction.api.routes.explain.get_settings") as mock_settings,
        patch("churn_prediction.api.routes.explain.generate_shap_artifacts") as mock_shap,
    ):
        settings = MagicMock()
        settings.PROCESSED_DATA_DIR = str(mock_processed_file.parent)
        settings.SHAP_REPORT_DIR = str(tmp_path / "reports")
        mock_settings.return_value = settings

        # Simulate a crash in the explainability logic
        mock_shap.side_effect = RuntimeError("SHAP computation failed")

        response = client.post(f"/explain/{batch_id}")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "An unexpected error occurred" in response.json()["detail"]


def test_generate_explanation_invalid_id():
    """Verify that a batch_id not matching the regex pattern returns 422."""
    invalid_id = "batch!id"
    response = client.post(f"/explain/{invalid_id}")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
