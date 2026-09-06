"""Test the dependencies module of the churn_prediction API."""

from unittest.mock import MagicMock

import pytest
import xgboost as xgb
from fastapi import HTTPException, Request, status
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.api.dependencies import get_encoder, get_model


@pytest.fixture
def mock_request() -> Request:
    """Create a mock FastAPI request object with an app state."""
    request = MagicMock(spec=Request)
    request.app = MagicMock()

    class State:
        pass

    request.app.state = State()
    return request


def test_get_encoder_success(mock_request: Request):
    """Test that get_encoder returns the encoder when present in state."""
    mock_encoder = MagicMock(spec=OrdinalEncoder)
    mock_request.app.state.encoder = mock_encoder

    result = get_encoder(mock_request)

    assert result == mock_encoder
    assert isinstance(result, OrdinalEncoder)


def test_get_encoder_missing(mock_request: Request):
    """Test that get_encoder raises HTTP 503 when encoder is missing."""
    if hasattr(mock_request.app.state, "encoder"):
        del mock_request.app.state.encoder

    with pytest.raises(HTTPException) as excinfo:
        get_encoder(mock_request)

    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Encoder artifact is not loaded" in excinfo.value.detail


def test_get_model_success(mock_request: Request):
    """Test that get_model returns the booster when present in state."""
    mock_booster = MagicMock(spec=xgb.Booster)
    mock_request.app.state.booster = mock_booster

    result = get_model(mock_request)

    assert result == mock_booster
    assert isinstance(result, xgb.Booster)


def test_get_model_missing(mock_request: Request):
    """Test that get_model raises HTTP 503 when booster is missing."""
    if hasattr(mock_request.app.state, "booster"):
        del mock_request.app.state.booster

    with pytest.raises(HTTPException) as excinfo:
        get_model(mock_request)

    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "XGBoost model artifact is not loaded" in excinfo.value.detail
