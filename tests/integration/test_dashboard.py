"""Test the dashboard module functions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

import churn_prediction.client.dashboard as exp


def test_recommend_for_feature_exact_match():
    """Test that the function returns the expected recommendation for a known feature."""
    assert "High early-tenure risk" in exp.recommend_for_feature("tenure")


def test_recommend_for_feature_normalization():
    """Test that the function normalizes the feature name before looking it up."""
    assert "High early-tenure risk" in exp.recommend_for_feature(" TENURE ")
    assert "Price sensitivity" in exp.recommend_for_feature("Monthly Charges")


def test_recommend_for_feature_unknown():
    """Test that the function returns a default recommendation for an unknown feature."""
    result = exp.recommend_for_feature("unknown_feature_123")
    assert "Review overall account engagement" in result


def test_sigmoid_math():
    """Test the mathematical correctness of the sigmoid function."""
    inputs: np.ndarray = np.array([-100, 0, 100])
    expected: np.ndarray = np.array([0.0, 0.5, 1.0])

    np.testing.assert_allclose(exp._sigmoid(inputs), expected, atol=1e-5)


def test_decode_features_non_ordinal(monkeypatch: pytest.MonkeyPatch):
    """Test that decode_features returns the string representation for non-ordinal features."""
    monkeypatch.setattr(exp, "_load_encoder", lambda: (None, ["some_ordinal_col"]))
    assert exp.decode_features("numeric_col", 12.5) == "12.5"


def test_decode_features_ordinal_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that decode_features correctly decodes an ordinal feature."""
    encoder = OrdinalEncoder()
    encoder.fit([["Fiber"], ["DSL"], ["None"]])

    class MockSettings:
        MODEL_DIR = str(tmp_path)
        PARAMS = type(
            "Params",
            (),
            {"schema_config": type("Config", (), {"ordinal_columns": ["internet_service"]})},
        )

    monkeypatch.setattr(exp, "get_settings", None)

    def mock_settings():
        """Return a mock settings object."""
        return MockSettings()

    monkeypatch.setattr("churn_prediction.client.dashboard.get_settings", mock_settings)

    joblib.dump(encoder, tmp_path / "preprocessor.joblib")
    exp._load_encoder.cache_clear()

    assert exp.decode_features("internet_service", 0) == "DSL"
    assert exp.decode_features("internet_service", 1) == "Fiber"


@pytest.fixture
def mock_artifact_dir(tmp_path: Path):
    """Create a temporary directory with mock artifacts for testing."""
    metadata = {
        "global_feature_importance": [
            {"feature": "tenure", "mean_abs_shap": 0.5},
            {"feature": "contract", "mean_abs_shap": 0.3},
        ]
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    df = pd.DataFrame(
        {
            "predicted_margin": [2.0, -2.0],
            "top_churn_drivers": [
                json.dumps([{"feature": "tenure", "shap_value": 0.8, "feature_value": 1.0}]),
                json.dumps([]),
            ],
            "top_retention_factors": [
                json.dumps([{"feature": "contract", "shap_value": -0.2, "feature_value": 2.0}]),
                json.dumps([]),
            ],
        },
        index=[101, 102],
    )
    df.index.name = "customer_id"
    df.to_parquet(tmp_path / "customer_risk_profiles.parquet")

    return tmp_path


def test_load_artifacts_success(mock_artifact_dir: Path):
    """Test that load_artifacts successfully loads the artifacts from the given directory."""
    data = exp.load_artifacts(mock_artifact_dir)
    assert "churn_probability" in data["risk_profiles"].columns
    assert data["risk_profiles"].iloc[0]["customer_id"] == 101


def test_load_artifacts_missing_dir(tmp_path: Path):
    """Test that load_artifacts raises FileNotFoundError for a non-existent directory."""
    # Create a path within the secure temporary directory that doesn't exist
    non_existent_dir = tmp_path / "non_existent_dir_123"

    with pytest.raises(FileNotFoundError):
        exp.load_artifacts(str(non_existent_dir))


def test_customer_shap_breakdown_success(mock_artifact_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that customer_shap_breakdown returns a non-empty DataFrame for a valid customer ID."""
    artifacts = exp.load_artifacts(mock_artifact_dir)
    risk_profiles = artifacts["risk_profiles"]

    monkeypatch.setattr(exp, "decode_features", lambda f, _v: f"Decoded_{f}")

    df_breakdown = exp.customer_shap_breakdown(risk_profiles, 101)
    assert not df_breakdown.empty
    assert "Decoded_tenure" in df_breakdown["display_value"].values


def test_customer_shap_breakdown_empty(mock_artifact_dir: Path):
    """Test that customer_shap_breakdown returns an empty DataFrame for a non-existent customer ID."""
    artifacts = exp.load_artifacts(mock_artifact_dir)
    risk_profiles = artifacts["risk_profiles"]

    df_empty = exp.customer_shap_breakdown(risk_profiles, 999)
    assert df_empty.empty
    assert "display_value" in df_empty.columns
