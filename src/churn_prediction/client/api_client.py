"""HTTP wrappers around the churn-prediction FastAPI endpoints.

This module provides functions to interact with the churn-prediction
API endpoints for batch prediction, downloading predictions, and
explaining predictions. It uses the `requests` library to send HTTP
requests and handle responses.
"""

from pathlib import Path

import requests

from churn_prediction.config.settings import get_settings


def predict_batch(csv_path: str | Path) -> dict[str, str]:
    """Submit a CSV dataset to the batch prediction API.

    Opens the supplied CSV file and sends it as a multipart upload to the
    configured prediction endpoint. The API response is returned after
    validating the HTTP status.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file containing customer records to score.

    Returns
    -------
    dict[str, str]
        JSON response returned by the prediction API.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    requests.HTTPError
        If the API returns an unsuccessful HTTP status.
    requests.RequestException
        If the HTTP request fails.
    """
    settings = get_settings()
    csv_path = Path(csv_path).expanduser().resolve()

    with csv_path.open("rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        response = requests.post(settings.PREDICT_API_URL, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def download_predictions(batch_id: str, save_path: str) -> str:
    """Download prediction results for a completed batch.

    Requests the prediction artifact associated with ``batch_id`` from the
    API and writes the returned file contents to ``save_path``.

    Parameters
    ----------
    batch_id : str
        Unique identifier of the prediction batch.
    save_path : str
        Local destination path for the downloaded prediction file.

    Returns
    -------
    str
        Path to the saved prediction file.

    Raises
    ------
    requests.HTTPError
        If the API returns an unsuccessful HTTP status.
    requests.RequestException
        If the download request fails.
    OSError
        If the prediction file cannot be written to disk.
    """
    settings = get_settings()

    response = requests.get(f"{settings.DOWNLOAD_API_URL}/{batch_id}", timeout=30)
    response.raise_for_status()
    with Path(save_path).open("wb") as f:
        f.write(response.content)
    return save_path


def explain_batch(batch_id: str) -> dict[str, str]:
    """Request explainability artifacts for a prediction batch.

    Sends an explainability request for the specified batch and returns the
    API response containing information about the generated SHAP artifacts.

    Parameters
    ----------
    batch_id : str
        Unique identifier of the prediction batch to explain.

    Returns
    -------
    dict[str, str]
        JSON response returned by the explainability endpoint.

    Raises
    ------
    requests.HTTPError
        If the API returns an unsuccessful HTTP status.
    requests.RequestException
        If the HTTP request fails.
    """
    settings = get_settings()

    response = requests.post(f"{settings.EXPLAIN_API_URL}/{batch_id}", timeout=120)
    response.raise_for_status()
    return response.json()


def check_health() -> dict[str, str]:
    """Check the availability of the churn prediction API.

    Sends a request to the configured health endpoint and returns the
    service health information.

    Returns
    -------
    dict[str, str]
        JSON health status returned by the API.

    Raises
    ------
    requests.HTTPError
        If the API returns an unsuccessful HTTP status.
    requests.RequestException
        If the health-check request fails.
    """
    settings = get_settings()

    response = requests.get(settings.HEALTH_API_URL, timeout=10)
    response.raise_for_status()
    return response.json()
