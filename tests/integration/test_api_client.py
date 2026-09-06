"""Test the API client functions in churn prediction."""

from pathlib import Path

import pytest
import requests

from churn_prediction.client.api_client import (
    check_health,
    download_predictions,
    explain_batch,
    predict_batch,
)

# Define the base URL and endpoints
BASE_URL = "http://0.0.0.0:8000"
URLS = {
    "PREDICT": f"{BASE_URL}/api/v1/predict/batch",
    "DOWNLOAD": f"{BASE_URL}/api/v1/predictions/download",
    "EXPLAIN": f"{BASE_URL}/explain",
    "HEALTH": f"{BASE_URL}/health",
}


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch: pytest.MonkeyPatch):
    """Mock the settings used in the API client to point to the test server."""

    class MockSettings:
        PREDICT_API_URL = URLS["PREDICT"]
        DOWNLOAD_API_URL = URLS["DOWNLOAD"]
        EXPLAIN_API_URL = URLS["EXPLAIN"]
        HEALTH_API_URL = URLS["HEALTH"]

    def get_mock_settings():
        """Return the mock settings."""
        return MockSettings()

    monkeypatch.setattr("churn_prediction.client.api_client.get_settings", get_mock_settings)


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text("customer_id,churn\n1,0\n2,1")
    return file_path


def test_predict_batch_success(requests_mock, temp_csv: Path):
    """Test the predict_batch function for a successful API call."""
    expected_response = {"batch_id": "12345", "status": "submitted"}
    requests_mock.post(URLS["PREDICT"], json=expected_response, status_code=200)

    result = predict_batch(temp_csv)
    assert result == expected_response


def test_predict_batch_file_not_found():
    """Test the predict_batch function when the input file does not exist."""
    with pytest.raises(FileNotFoundError):
        predict_batch("non_existent_file.csv")


def test_predict_batch_http_error(requests_mock, temp_csv: Path):
    """Test the predict_batch function for an HTTP error response."""
    requests_mock.post(URLS["PREDICT"], status_code=500)
    with pytest.raises(requests.HTTPError):
        predict_batch(temp_csv)


def test_download_predictions_success(requests_mock, tmp_path: Path):
    """Test the download_predictions function for a successful API call."""
    batch_id = "12345"
    save_path = tmp_path / "results.csv"
    mock_content = b"id,prediction\n1,0.85"
    # Note the trailing slash and ID construction
    requests_mock.get(f"{URLS['DOWNLOAD']}/{batch_id}", content=mock_content, status_code=200)

    result_path = download_predictions(batch_id, str(save_path))
    assert Path(result_path).read_bytes() == mock_content


def test_download_predictions_http_error(requests_mock, tmp_path: Path):
    """Test the download_predictions function for an HTTP error response."""
    batch_id = "invalid_id"
    save_path = tmp_path / "results.csv"
    requests_mock.get(f"{URLS['DOWNLOAD']}/{batch_id}", status_code=404)

    with pytest.raises(requests.HTTPError):
        download_predictions(batch_id, str(save_path))


def test_explain_batch_success(requests_mock):
    """Test the explain_batch function for a successful API call."""
    batch_id = "12345"
    expected_response = {"status": "complete"}
    requests_mock.post(f"{URLS['EXPLAIN']}/{batch_id}", json=expected_response, status_code=200)

    result = explain_batch(batch_id)
    assert result == expected_response


def test_explain_batch_http_error(requests_mock):
    """Test the explain_batch function for an HTTP error response."""
    batch_id = "12345"
    requests_mock.post(f"{URLS['EXPLAIN']}/{batch_id}", status_code=400)

    with pytest.raises(requests.HTTPError):
        explain_batch(batch_id)


def test_check_health_success(requests_mock):
    """Test the check_health function for a successful API call."""
    expected_response = {"status": "healthy"}
    requests_mock.get(URLS["HEALTH"], json=expected_response, status_code=200)

    result = check_health()
    assert result == expected_response


def test_check_health_http_error(requests_mock):
    """Test the check_health function for an HTTP error response."""
    requests_mock.get(URLS["HEALTH"], status_code=503)
    with pytest.raises(requests.HTTPError):
        check_health()


def test_request_exception(requests_mock, temp_csv: Path):
    """Test the API client functions for a request exception."""
    requests_mock.post(URLS["PREDICT"], exc=requests.RequestException)
    with pytest.raises(requests.RequestException):
        predict_batch(temp_csv)
