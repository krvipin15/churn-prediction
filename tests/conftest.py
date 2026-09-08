"""Pytest configuration file for the churn_prediction package."""

import pytest

from churn_prediction.config.settings import get_settings


@pytest.fixture(autouse=True)
def override_settings_for_tests(monkeypatch: pytest.MonkeyPatch):
    """Ensure tests run under the testing environment and clear settings cache."""
    monkeypatch.setenv("ENVIRONMENT", "testing")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
