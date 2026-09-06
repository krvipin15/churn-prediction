"""Test the SHAP artifact export module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import churn_prediction.model.explainability as module


@pytest.fixture
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a mocked application logger."""
    logger = MagicMock()
    monkeypatch.setattr(module, "get_logger", lambda: logger)
    return logger


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide mocked application settings."""
    settings = MagicMock()
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def exporter(
    monkeypatch: pytest.MonkeyPatch,
    mock_logger: MagicMock,
    mock_settings: MagicMock,
    tmp_path: Path,
) -> module.SHAPArtifactExporter:
    """Create an exporter with a mocked SHAP explainer."""
    tree_explainer = MagicMock()
    tree_explainer.expected_value = 0.25
    monkeypatch.setattr(module.shap, "TreeExplainer", MagicMock(return_value=tree_explainer))

    df = pd.DataFrame(
        {
            "tenure": [1.0, 2.0, 3.0],
            "monthly_charges": [10.0, 20.0, 30.0],
            "support_calls": [5.0, 1.0, 2.0],
        },
        index=pd.Index([101, 102, 103], name="customer_id"),
    )

    return module.SHAPArtifactExporter(
        batch_id="batch-001",
        df=df,
        booster=MagicMock(),
        output_dir=tmp_path,
        max_samples=10,
        random_state=42,
    )


def test_init_sets_attributes_and_creates_tree_explainer(
    monkeypatch: pytest.MonkeyPatch,
    mock_logger: MagicMock,
    mock_settings: MagicMock,
    tmp_path: Path,
):
    """The constructor stores configuration and initializes TreeExplainer."""
    # Arrange: Mock TreeExplainer to return a known expected value
    tree_explainer = MagicMock()
    tree_explainer.expected_value = 0.5
    tree_mock = MagicMock(return_value=tree_explainer)
    monkeypatch.setattr(module.shap, "TreeExplainer", tree_mock)
    df = pd.DataFrame({"feature": [1, 2]}, index=[11, 12])
    booster = MagicMock()

    # Act: Create the exporter
    result = module.SHAPArtifactExporter(
        "batch-123",
        df,
        booster,
        tmp_path,
        max_samples=1,
        random_state=7,
    )

    # Assert: Attributes are set correctly and TreeExplainer is called with the booster
    assert result.batch_id == "batch-123"
    assert result.output_dir == tmp_path
    assert result.max_samples == 1
    assert result.random_state == 7
    pd.testing.assert_frame_equal(result.features_df, df)
    np.testing.assert_array_equal(result.customer_ids, [11, 12])
    assert result.booster is booster
    assert result.explainer is tree_explainer
    tree_mock.assert_called_once_with(booster)


def test_init_reraises_tree_explainer_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_logger: MagicMock,
    mock_settings: MagicMock,
    tmp_path: Path,
):
    """Constructor failures from TreeExplainer are logged and re-raised."""
    # Arrange: Mock TreeExplainer to raise an error
    error = RuntimeError("cannot initialize explainer")
    tree_mock = MagicMock(side_effect=error)
    monkeypatch.setattr(module.shap, "TreeExplainer", tree_mock)

    # Act & Assert: Expect the constructor to raise the same error and log it
    with pytest.raises(RuntimeError, match="cannot initialize explainer"):
        module.SHAPArtifactExporter(
            "batch-001",
            pd.DataFrame({"x": [1]}),
            MagicMock(),
            tmp_path,
        )

    mock_logger.exception.assert_called_once_with("Failed to initialize SHAPArtifactExporter")


def test_sample_dataset_returns_full_dataset_when_within_limit(
    exporter: module.SHAPArtifactExporter,
):
    """Datasets at or below max_samples are returned unchanged."""
    result = exporter._sample_dataset()

    pd.testing.assert_frame_equal(result, exporter.features_df)
    np.testing.assert_array_equal(exporter.sampled_customer_ids, [101, 102, 103])


def test_sample_dataset_samples_large_dataset_deterministically(
    exporter: module.SHAPArtifactExporter,
):
    """Datasets above max_samples are sampled with the configured seed."""
    # Arrange: Set max_samples to 2 and random_state to a known value
    exporter.max_samples = 2
    exporter.random_state = 42

    # Act: Sample the dataset
    result = exporter._sample_dataset()
    expected = exporter.features_df.sample(n=2, random_state=42)

    # Assert: The sampled dataset matches the expected subset and customer IDs are aligned
    pd.testing.assert_frame_equal(result, expected)
    np.testing.assert_array_equal(exporter.sampled_customer_ids, expected.index.to_numpy())


def test_compute_shap_explanation_with_scalar_expected_value(
    monkeypatch: pytest.MonkeyPatch,
    exporter: module.SHAPArtifactExporter,
):
    """SHAP values are wrapped into an Explanation using a scalar baseline."""
    # Arrange: Prepare a small evaluation dataset and mock the DMatrix and shap_values calls
    eval_df = exporter.features_df.iloc[:2].copy()
    shap_values = np.array(
        [
            [1.0, -2.0, 0.5],
            [-0.5, 2.0, 1.5],
        ]
    )

    dmatrix = MagicMock()
    dmatrix_mock = MagicMock(return_value=dmatrix)
    monkeypatch.setattr(
        module.xgb,
        "DMatrix",
        dmatrix_mock,
    )

    shap_values_mock = MagicMock(return_value=shap_values)
    monkeypatch.setattr(
        exporter.explainer,
        "shap_values",
        shap_values_mock,
    )

    exporter.explainer.expected_value = 0.75

    # Act: Compute the SHAP explanation
    result = exporter._compute_shap_explanation(eval_df)

    # Assert: DMatrix and shap_values were called with the correct arguments
    dmatrix_mock.assert_called_once_with(
        eval_df,
        feature_names=[
            "tenure",
            "monthly_charges",
            "support_calls",
        ],
    )
    shap_values_mock.assert_called_once_with(dmatrix)

    np.testing.assert_array_equal(
        result.values,
        shap_values,
    )
    assert result.base_values == 0.75

    np.testing.assert_array_equal(
        result.data,
        eval_df.values,
    )
    assert result.feature_names == [
        "tenure",
        "monthly_charges",
        "support_calls",
    ]


def test_compute_shap_explanation_with_array_expected_value(
    monkeypatch: pytest.MonkeyPatch,
    exporter: module.SHAPArtifactExporter,
):
    """The first element of an ndarray baseline is converted to float."""
    # Arrange: Prepare a small evaluation dataset and mock the DMatrix and shap_values calls
    eval_df = exporter.features_df.iloc[:1].copy()

    dmatrix = MagicMock()
    monkeypatch.setattr(
        module.xgb,
        "DMatrix",
        MagicMock(return_value=dmatrix),
    )

    shap_values_mock = MagicMock(return_value=np.array([[1.0, 2.0, 3.0]]))
    monkeypatch.setattr(
        exporter.explainer,
        "shap_values",
        shap_values_mock,
    )

    exporter.explainer.expected_value = np.array([0.625])

    # Act: Compute the SHAP explanation
    result = exporter._compute_shap_explanation(eval_df)

    # Assert: The base_values is converted to a scalar float
    assert result.base_values == 0.625
    shap_values_mock.assert_called_once_with(dmatrix)


def test_compute_shap_explanation_reraises_dmatrix_error(
    monkeypatch: pytest.MonkeyPatch,
    exporter: module.SHAPArtifactExporter,
    mock_logger: MagicMock,
):
    """DMatrix or SHAP errors are logged and propagated."""
    # Arrange: Mock DMatrix to raise a ValueError
    error = ValueError("bad feature matrix")
    monkeypatch.setattr(module.xgb, "DMatrix", MagicMock(side_effect=error))

    # Act & Assert: Expect the _compute_shap_explanation to raise the same error and log it
    with pytest.raises(ValueError, match="bad feature matrix"):
        exporter._compute_shap_explanation(exporter.features_df)

    mock_logger.exception.assert_called_with("Failed to compute SHAP values.")


def test_compute_shap_explanation_reraises_shap_error(
    monkeypatch: pytest.MonkeyPatch,
    exporter: module.SHAPArtifactExporter,
    mock_logger: MagicMock,
):
    """SHAP calculation failures are logged and propagated."""
    # Arrange: Mock DMatrix to return a MagicMock and shap_values to raise a RuntimeError
    dmatrix = MagicMock()

    monkeypatch.setattr(
        module.xgb,
        "DMatrix",
        MagicMock(return_value=dmatrix),
    )

    shap_values_mock = MagicMock(side_effect=RuntimeError("SHAP failed"))
    monkeypatch.setattr(
        exporter.explainer,
        "shap_values",
        shap_values_mock,
    )

    # Act & Assert: Expect the _compute_shap_explanation to raise the same error and log it
    with pytest.raises(RuntimeError, match="SHAP failed"):
        exporter._compute_shap_explanation(exporter.features_df)

    shap_values_mock.assert_called_once_with(dmatrix)
    mock_logger.exception.assert_called_once_with("Failed to compute SHAP values.")


def test_build_customer_risk_profiles_selects_top_positive_and_negative(
    exporter: module.SHAPArtifactExporter,
):
    """Customer profiles contain the strongest churn and retention drivers."""
    # Arrange: Create a mock SHAP Explanation with known values
    explanation = module.shap.Explanation(
        values=np.array(
            [
                [3.0, -4.0, 1.0, -0.5],
                [-2.0, 0.0, 4.0, -1.0],
            ]
        ),
        base_values=0.5,
        data=np.array(
            [
                [30.0, 40.0, 10.0, 5.0],
                [20.0, 50.0, 15.0, 8.0],
            ]
        ),
        feature_names=["a", "b", "c", "d"],
    )

    # Act: Build the customer risk profiles for two customers with top_k=2
    result = exporter._build_customer_risk_profiles(
        explanation,
        np.array([101, 102]),
        top_k=2,
    )

    # Assert: The resulting DataFrame has the expected structure and values
    assert list(result.index) == [101, 102]
    assert result.loc[101, "base_value"] == 0.5
    assert result.loc[101, "predicted_margin"] == pytest.approx(0.0)
    assert result.loc[102, "predicted_margin"] == pytest.approx(1.5)

    customer_101_pos = json.loads(result.loc[101, "top_churn_drivers"])
    customer_101_neg = json.loads(result.loc[101, "top_retention_factors"])
    assert customer_101_pos == [
        {"feature": "a", "shap_value": 3.0, "feature_value": 30.0},
        {"feature": "c", "shap_value": 1.0, "feature_value": 10.0},
    ]
    assert customer_101_neg == [
        {"feature": "b", "shap_value": -4.0, "feature_value": 40.0},
        {"feature": "d", "shap_value": -0.5, "feature_value": 5.0},
    ]

    customer_102_pos = json.loads(result.loc[102, "top_churn_drivers"])
    customer_102_neg = json.loads(result.loc[102, "top_retention_factors"])
    assert customer_102_pos == [
        {"feature": "c", "shap_value": 4.0, "feature_value": 15.0},
    ]
    assert customer_102_neg == [
        {"feature": "a", "shap_value": -2.0, "feature_value": 20.0},
        {"feature": "d", "shap_value": -1.0, "feature_value": 8.0},
    ]


def test_build_customer_risk_profiles_with_zero_top_k(exporter: module.SHAPArtifactExporter):
    """A zero top-k follows the implementation's Python slice behavior."""
    # Arrange: Create a mock SHAP Explanation with known values
    explanation = module.shap.Explanation(
        values=np.array([[2.0, -1.0]]),
        base_values=0.25,
        data=np.array([[10.0, 20.0]]),
        feature_names=["a", "b"],
    )

    # Act: Build the customer risk profiles with top_k=0
    result = exporter._build_customer_risk_profiles(
        explanation,
        np.array([999]),
        top_k=0,
    )

    # Assert: The resulting DataFrame has the expected structure and values
    assert json.loads(result.loc[999, "top_churn_drivers"]) == [
        {
            "feature": "a",
            "shap_value": 2.0,
            "feature_value": 10.0,
        }
    ]
    assert json.loads(result.loc[999, "top_retention_factors"]) == []


def test_export_metadata_writes_ranked_json(
    exporter: module.SHAPArtifactExporter,
    tmp_path: Path,
):
    """Metadata contains sample information and descending feature importance."""
    # Arrange: Create a mock SHAP Explanation with known values
    explanation = module.shap.Explanation(
        values=np.array(
            [
                [1.0, -4.0, 2.0],
                [-3.0, 2.0, 0.0],
            ]
        ),
        base_values=0.25,
        data=np.ones((2, 3)),
        feature_names=["a", "b", "c"],
    )
    output = tmp_path / "metadata.json"

    # Act: Export the metadata
    exporter._export_metadata(explanation, output, total_samples=2)

    # Assert: The metadata file exists and contains the expected information
    assert output.exists()
    metadata = json.loads(output.read_text())
    assert metadata["batch_id"] == exporter.batch_id
    assert metadata["base_value"] == 0.25
    assert metadata["sample_count"] == 2
    assert metadata["feature_count"] == 3
    assert metadata["global_feature_importance"] == [
        {"rank": 1, "feature": "b", "mean_abs_shap": 3.0},
        {"rank": 2, "feature": "a", "mean_abs_shap": 2.0},
        {"rank": 3, "feature": "c", "mean_abs_shap": 1.0},
    ]


def test_export_metadata_propagates_write_error(
    exporter: module.SHAPArtifactExporter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Metadata write errors are propagated to the caller."""
    # Arrange: Create a mock SHAP Explanation with known values
    explanation = module.shap.Explanation(
        values=np.array([[1.0]]),
        base_values=0.0,
        data=np.array([[1.0]]),
        feature_names=["feature"],
    )
    output = tmp_path / "metadata.json"

    class FailingFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def write(self, data):
            _ = data
            raise OSError("disk full")

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: FailingFile())  # noqa: ARG005

    # Act & Assert: Expect the _export_metadata to raise an OSError
    with pytest.raises(OSError, match="disk full"):
        exporter._export_metadata(explanation, output, total_samples=1)


def test_export_artifacts_end_to_end(
    exporter: module.SHAPArtifactExporter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """All four expected artifacts are written with aligned customer IDs."""
    # Arrange: Prepare a small evaluation dataset and mock the SHAP explanation
    eval_df = exporter.features_df.iloc[[2, 0]].copy()

    explanation = module.shap.Explanation(
        values=np.array(
            [
                [1.0, -2.0, 0.5],
                [-1.0, 3.0, -0.25],
            ]
        ),
        base_values=0.25,
        data=eval_df.values,
        feature_names=eval_df.columns.tolist(),
    )

    def sample_dataset():
        """Mock sampling while preserving the exporter contract."""
        exporter.sampled_customer_ids = eval_df.index.to_numpy()
        return eval_df.copy()

    monkeypatch.setattr(exporter, "_sample_dataset", sample_dataset)
    monkeypatch.setattr(exporter, "_compute_shap_explanation", lambda _: explanation)

    # Act: Export the artifacts
    result = exporter.export_artifacts(top_k_drivers=2)

    # Assert: The output directory is returned and contains the expected files with correct content
    assert result == tmp_path
    expected_files = {
        "raw_shap_values.parquet",
        "input_features.parquet",
        "customer_risk_profiles.parquet",
        "metadata.json",
    }
    assert {p.name for p in tmp_path.iterdir()} == expected_files

    shap_df = pd.read_parquet(tmp_path / "raw_shap_values.parquet")
    feature_df = pd.read_parquet(tmp_path / "input_features.parquet")
    profiles_df = pd.read_parquet(tmp_path / "customer_risk_profiles.parquet")

    assert list(shap_df.columns) == ["shap_tenure", "shap_monthly_charges", "shap_support_calls"]
    assert list(shap_df.index) == [103, 101]
    assert list(feature_df.index) == [103, 101]
    assert list(profiles_df.index) == [103, 101]

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["sample_count"] == 2
    assert metadata["feature_count"] == 3


def test_export_artifacts_reraises_internal_error(
    exporter: module.SHAPArtifactExporter,
    mock_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Failures during artifact generation are logged and propagated."""
    # Arrange: Mock the _sample_dataset method to raise a RuntimeError
    error = RuntimeError("export failed")
    monkeypatch.setattr(exporter, "_sample_dataset", MagicMock(side_effect=error))

    # Act & Assert: Expect the export_artifacts method to raise the same error and log it
    with pytest.raises(RuntimeError, match="export failed"):
        exporter.export_artifacts()

    mock_logger.exception.assert_called_with(
        "Export run failed for output directory: %s",
        exporter.output_dir,
    )


def test_generate_shap_artifacts_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The public function constructs an exporter and returns its output."""
    # Arrange: Mock the SHAPArtifactExporter to return a known output path
    exporter = MagicMock()
    exporter.export_artifacts.return_value = tmp_path

    constructor = MagicMock(return_value=exporter)
    monkeypatch.setattr(module, "SHAPArtifactExporter", constructor)

    logger = MagicMock()
    monkeypatch.setattr(module, "get_logger", lambda: logger)

    df = pd.DataFrame({"x": [1.0]})
    booster = MagicMock()

    # Act: Call the public function to generate SHAP artifacts
    result = module.generate_shap_artifacts("batch-x", df, booster, tmp_path)

    # Assert: The result matches the expected output and the constructor was called with correct arguments
    assert result == tmp_path
    constructor.assert_called_once_with(
        batch_id="batch-x",
        df=df,
        booster=booster,
        output_dir=tmp_path,
    )
    exporter.export_artifacts.assert_called_once_with()


def test_generate_shap_artifacts_reraises_constructor_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Public workflow logs and re-raises exporter construction failures."""
    # Arrange: Mock the SHAPArtifactExporter constructor to raise a RuntimeError
    logger = MagicMock()
    monkeypatch.setattr(module, "get_logger", lambda: logger)

    error = RuntimeError("constructor failed")
    monkeypatch.setattr(
        module,
        "SHAPArtifactExporter",
        MagicMock(side_effect=error),
    )

    # Act & Assert: Expect the public function to raise the same error and log it
    with pytest.raises(RuntimeError, match="constructor failed"):
        module.generate_shap_artifacts(
            "batch-x",
            pd.DataFrame({"x": [1.0]}),
            MagicMock(),
            tmp_path,
        )

    logger.exception.assert_called_once_with(
        "Failed to complete SHAP artifact generation workflow."
    )


def test_generate_shap_artifacts_reraises_export_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Public workflow logs and re-raises exporter export failures."""
    # Arrange: Mock the SHAPArtifactExporter to raise a RuntimeError during export
    logger = MagicMock()
    monkeypatch.setattr(module, "get_logger", lambda: logger)

    exporter = MagicMock()
    exporter.export_artifacts.side_effect = RuntimeError("export failed")
    monkeypatch.setattr(module, "SHAPArtifactExporter", MagicMock(return_value=exporter))

    # Act & Assert: Expect the public function to raise the same error and log it
    with pytest.raises(RuntimeError, match="export failed"):
        module.generate_shap_artifacts(
            "batch-x",
            pd.DataFrame({"x": [1.0]}),
            MagicMock(),
            tmp_path,
        )

    logger.exception.assert_called_once_with(
        "Failed to complete SHAP artifact generation workflow."
    )
