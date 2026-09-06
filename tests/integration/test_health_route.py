"""Test the /health route in the FastAPI application."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from churn_prediction.api.routes.health import router

# Setup a minimal FastAPI app to host the router for testing
app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_check_health_healthy():
    """Test that the health check returns HEALTHY when both the encoder and booster are present."""
    app.state.encoder = MagicMock()
    app.state.booster = MagicMock()

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["encoder_loaded"] is True
    assert data["model_loaded"] is True
    assert "timestamp" in data
    assert 0 <= data["memory_usage_percent"] <= 100


def test_check_health_degraded_no_model():
    """Test that the health check returns DEGRADED when the booster model is missing."""
    # Arrange: Encoder exists, but booster is missing
    app.state.encoder = MagicMock()
    if hasattr(app.state, "booster"):
        del app.state.booster

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DEGRADED"
    assert data["encoder_loaded"] is True
    assert data["model_loaded"] is False


def test_check_health_degraded_no_encoder():
    """Test that the health check returns DEGRADED when the encoder is missing."""
    # Arrange: Booster exists, but encoder is missing
    app.state.booster = MagicMock()
    if hasattr(app.state, "encoder"):
        del app.state.encoder

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DEGRADED"
    assert data["encoder_loaded"] is False
    assert data["model_loaded"] is True


def test_check_health_fully_degraded():
    """Test that the health check returns DEGRADED when both dependencies are missing."""
    # Arrange: Wipe both from app state
    if hasattr(app.state, "encoder"):
        del app.state.encoder
    if hasattr(app.state, "booster"):
        del app.state.booster

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DEGRADED"
    assert data["encoder_loaded"] is False
    assert data["model_loaded"] is False
    assert "timestamp" in data
    assert 0 <= data["memory_usage_percent"] <= 100
