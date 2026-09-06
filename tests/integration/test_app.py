"""Test the FastAPI application setup, middleware, and lifespan management."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from churn_prediction.api.app import create_app, lifespan

# Constants
APP_MODULE_PATH = "churn_prediction.api.app"


@pytest.fixture
def mock_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Create temporary artifact files and patch application settings."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "preprocessor.joblib").write_bytes(b"dummy joblib content")
    (model_dir / "model.ubj").write_bytes(b"dummy xgboost content")
    (data_dir / "test.csv").write_text("customer_id,feature_1\n1,0.5")

    mock_settings = MagicMock()
    mock_settings.MODEL_DIR = model_dir
    mock_settings.RAW_DATA_DIR = data_dir

    monkeypatch.setattr(f"{APP_MODULE_PATH}.get_settings", lambda: mock_settings)

    return {"model_dir": model_dir, "data_dir": data_dir}


def test_create_app_metadata_and_routes():
    """Verify that create_app initializes metadata and registers operational routes."""
    app = create_app()

    # Metadata assertions
    assert app.title == "Churn Prediction Batch Inference API"
    assert isinstance(app.version, str)
    assert len(app.version) > 0

    # Extract registered paths
    registered_paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            registered_paths.add(path)
        elif hasattr(route, "original_router"):
            for sub_route in route.original_router.routes:  # ty: ignore[unresolved-attribute]
                sub_path = getattr(sub_route, "path", None)
                if isinstance(sub_path, str):
                    registered_paths.add(sub_path)

    # Verify at minimum that /health is registered
    assert "/health" in registered_paths, (
        f"Expected '/health' in registered routes. Found: {registered_paths}"
    )


def test_cors_middleware_config():
    """Verify that CORSMiddleware is configured with expected parameters."""
    app = create_app()

    cors_middlewares = [
        middleware for middleware in app.user_middleware if middleware.cls is CORSMiddleware
    ]

    assert len(cors_middlewares) == 1, "CORSMiddleware is not registered"

    cors_kwargs: dict[str, Any] = cors_middlewares[0].kwargs
    assert cors_kwargs.get("allow_origins") == ["*"]
    assert cors_kwargs.get("allow_methods") == ["*"]
    assert cors_kwargs.get("allow_headers") == ["*"]


@pytest.mark.asyncio
async def test_lifespan_success(mock_artifacts: dict[str, Path]):
    """Test full lifespan startup and shutdown cycle with valid artifacts."""
    app = FastAPI()

    mock_encoder = MagicMock()
    mock_booster_inst = MagicMock()

    with (
        patch(f"{APP_MODULE_PATH}.joblib.load", return_value=mock_encoder) as mock_joblib,
        patch(f"{APP_MODULE_PATH}.xgb.Booster", return_value=mock_booster_inst) as mock_xgb,
    ):
        async with lifespan(app):
            assert app.state.encoder is mock_encoder
            assert app.state.booster is mock_booster_inst
            mock_joblib.assert_called_once()
            mock_xgb.return_value.load_model.assert_called_once()

        assert app.state.encoder is None
        assert app.state.booster is None


@pytest.mark.asyncio
async def test_lifespan_missing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that lifespan raises FileNotFoundError when artifacts do not exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    mock_settings = MagicMock()
    mock_settings.MODEL_DIR = empty_dir
    mock_settings.RAW_DATA_DIR = empty_dir

    monkeypatch.setattr(f"{APP_MODULE_PATH}.get_settings", lambda: mock_settings)

    app = FastAPI()
    with pytest.raises(FileNotFoundError, match="Required artifacts doesn't exists"):
        async with lifespan(app):
            pass


@pytest.mark.asyncio
async def test_lifespan_loading_error(mock_artifacts: dict[str, Path]):
    """Test that lifespan raises exception if artifact loading fails."""
    app = FastAPI()

    with (
        patch(f"{APP_MODULE_PATH}.joblib.load", side_effect=RuntimeError("Corrupt artifact")),
        pytest.raises(RuntimeError, match="Corrupt artifact"),
    ):
        async with lifespan(app):
            pass


def test_health_check_endpoint_functional():
    """Functional test verifying the /health endpoint via TestClient."""
    # Arrange: Create a new app instance for this test
    app = create_app()
    client = TestClient(app)

    # Act: Call the /health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Assert: Check the response
    payload: dict[str, Any] = response.json()
    assert payload["status"] == "DEGRADED"
    assert payload["encoder_loaded"] is False
    assert payload["model_loaded"] is False
