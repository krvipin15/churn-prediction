"""Test the XGBoost training module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from churn_prediction.model import training


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Return a small balanced training dataset."""
    return pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 1.0, 0.0, 0.0],
            "target": [0, 0, 1, 1],
        }
    )


@pytest.fixture
def val_df() -> pd.DataFrame:
    """Return a small validation dataset."""
    return pd.DataFrame(
        {
            "feature_a": [0.2, 1.2, 2.2, 2.8],
            "feature_b": [1.0, 0.8, 0.2, 0.0],
            "target": [0, 1, 1, 0],
        }
    )


@pytest.fixture
def patch_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the module logger factory with a mock logger."""
    logger = MagicMock()
    monkeypatch.setattr(training, "get_logger", lambda: logger)
    return logger


def test_build_dmatrices_returns_expected_matrices_and_weight(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    patch_logger: MagicMock,
) -> None:
    """Build DMatrices and calculate the negative/positive class ratio."""
    dtrain, dval, weight = training._build_dmatrices(
        train_df,
        val_df,
        "target",
        ["feature_a", "feature_b"],
    )

    assert isinstance(dtrain, xgb.DMatrix)
    assert isinstance(dval, xgb.DMatrix)
    assert dtrain.num_row() == 4
    assert dval.num_row() == 4
    np.testing.assert_array_equal(dtrain.get_label(), [0, 0, 1, 1])
    np.testing.assert_array_equal(dval.get_label(), [0, 1, 1, 0])
    assert weight == pytest.approx(1.0)
    assert dtrain.feature_names == ["feature_a", "feature_b"]
    patch_logger.info.assert_called()


@pytest.mark.parametrize(
    ("target", "features"),
    [
        ("missing_target", ["feature_a", "feature_b"]),
        ("target", ["missing_feature"]),
    ],
)
def test_build_dmatrices_invalid_columns_raise_value_error(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    features: list[str],
    patch_logger: MagicMock,
) -> None:
    """Wrap DMatrix construction failures in ValueError."""
    with pytest.raises(ValueError, match="Failed to create DMatrix"):
        training._build_dmatrices(train_df, val_df, target, features)

    patch_logger.error.assert_called_once()


def test_build_dmatrices_zero_positive_raises(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    patch_logger: MagicMock,
) -> None:
    """Reject training data containing no positive examples."""
    train_df["target"] = 0

    with pytest.raises(ValueError, match="0 positive samples"):
        training._build_dmatrices(train_df, val_df, "target", ["feature_a", "feature_b"])


def test_build_dmatrices_zero_negative_raises(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    patch_logger: MagicMock,
) -> None:
    """Reject training data containing no negative examples."""
    train_df["target"] = 1

    with pytest.raises(ValueError, match="0 negative samples"):
        training._build_dmatrices(train_df, val_df, "target", ["feature_a", "feature_b"])


def test_train_model_adds_scale_weight_and_returns_training_outputs(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass scale_pos_weight into xgb.train and return its outputs."""
    dtrain = MagicMock(spec=xgb.DMatrix)
    dval = MagicMock(spec=xgb.DMatrix)
    booster = MagicMock()
    booster.best_iteration = 3
    booster.best_score = 0.25

    evals_result: dict[str, dict[str, list[float]]] = {
        "train": {"logloss": [0.5]},
        "validation": {"logloss": [0.6]},
    }

    def fake_train(params, dtrain_arg, **kwargs):
        assert params["scale_pos_weight"] == 2.5
        assert dtrain_arg is dtrain
        assert kwargs["num_boost_round"] == 10
        assert kwargs["early_stopping_rounds"] == 3
        assert kwargs["verbose_eval"] == 2
        assert kwargs["evals"][0][0] is dtrain
        assert kwargs["evals"][1][0] is dval

        kwargs["evals_result"].update(evals_result)
        return booster

    monkeypatch.setattr(training.xgb, "train", fake_train)

    result_booster, result_eval, result_params = training._train_model(
        dtrain,
        dval,
        xgb_params={"objective": "binary:logistic"},
        scale_pos_weight=2.5,
        n_estimators=10,
        early_stopping_rounds=3,
        verbose_eval=2,
    )

    assert result_booster is booster
    assert result_eval == evals_result
    assert result_params == {
        "objective": "binary:logistic",
        "scale_pos_weight": 2.5,
    }


def test_train_model_wraps_xgboost_error(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap failures from xgb.train in ValueError."""

    def fail_train(params, dtrain_arg, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(training.xgb, "train", fail_train)

    with pytest.raises(ValueError, match=r"Model training failed: boom"):
        training._train_model(
            MagicMock(),
            MagicMock(),
            xgb_params={},
            scale_pos_weight=1.0,
            n_estimators=5,
            early_stopping_rounds=2,
            verbose_eval=False,
        )

    patch_logger.error.assert_called_once()


def test_train_model_uses_fallback_booster_attributes(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use fallback values when the booster lacks best-iteration metadata."""
    booster = object()

    def fake_train(params, dtrain_arg, **kwargs):
        return booster

    monkeypatch.setattr(training.xgb, "train", fake_train)

    result_booster, result_eval, params = training._train_model(
        MagicMock(),
        MagicMock(),
        xgb_params={"max_depth": 2},
        scale_pos_weight=1.5,
        n_estimators=7,
        early_stopping_rounds=1,
        verbose_eval=0,
    )

    assert result_booster is booster
    assert result_eval == {}
    assert params["max_depth"] == 2
    assert params["scale_pos_weight"] == 1.5
    patch_logger.info.assert_called()


@pytest.mark.parametrize("weight", [1.0, 0.0, -1.0])
def test_unscale_probabilities_returns_original_for_nonpositive_or_one_weight(
    weight: float,
    patch_logger: MagicMock,
) -> None:
    """Skip unscaling when no effective positive-class weighting is present."""
    probabilities = np.array([0.0, 0.25, 1.0])

    result = training.unscale_probabilities(probabilities, weight)

    assert result is probabilities
    np.testing.assert_array_equal(result, probabilities)


def test_unscale_probabilities_calibrates_and_clips_extremes(
    patch_logger: MagicMock,
) -> None:
    """Convert weighted probabilities back to the original class distribution."""
    probabilities = np.array([0.0, 0.25, 1.0])

    result = training.unscale_probabilities(probabilities, 3.0)

    clipped = np.clip(
        probabilities,
        training.EPSILON,
        1.0 - training.EPSILON,
    )

    expected = clipped / (clipped + 3.0 * (1.0 - clipped))

    np.testing.assert_allclose(result, expected)

    # Verify the transformation itself rather than incorrectly requiring
    # the calibrated values to remain within the original clipping bounds.
    assert result[0] > 0.0
    assert result[0] < probabilities[1]
    assert result[1] == pytest.approx(0.1)
    assert result[2] > probabilities[1]
    assert np.all(np.isfinite(result))


def test_find_optimal_threshold_returns_default_when_no_thresholds(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the supplied default when precision-recall yields no thresholds."""
    monkeypatch.setattr(
        training,
        "precision_recall_curve",
        lambda _y_true, _y_prob: (
            np.array([1.0]),
            np.array([1.0]),
            np.array([]),
        ),
    )

    threshold, score = training._find_optimal_threshold(
        np.array([0]),
        np.array([0.2]),
        0.42,
    )

    assert threshold == 0.42
    assert score == 0.0


def test_find_optimal_threshold_handles_zero_denominators(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore candidates whose precision plus recall is zero."""
    monkeypatch.setattr(
        training,
        "precision_recall_curve",
        lambda _y_true, _y_prob: (
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.1, 0.8]),
        ),
    )

    threshold, score = training._find_optimal_threshold(
        np.array([0, 1]),
        np.array([0.1, 0.8]),
        0.5,
    )

    assert threshold == pytest.approx(0.8)
    assert score == pytest.approx(0.5)


def test_find_optimal_threshold_selects_highest_f1(
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select the threshold corresponding to the maximum candidate F1."""
    monkeypatch.setattr(
        training,
        "precision_recall_curve",
        lambda _y_true, _y_prob: (
            np.array([0.5, 0.8, 0.6]),
            np.array([0.8, 0.8, 0.1]),
            np.array([0.2, 0.7]),
        ),
    )

    threshold, score = training._find_optimal_threshold(
        np.array([0, 1]),
        np.array([0.2, 0.7]),
        0.5,
    )

    # F1 candidates are 2*.5*.8/1.3 and 2*.8*.8/1.6.
    assert threshold == pytest.approx(0.7)
    assert score == pytest.approx(0.8)


@pytest.fixture
def evaluation_dmatrix() -> xgb.DMatrix:
    """Return a validation DMatrix with both target classes."""
    return xgb.DMatrix(
        pd.DataFrame({"feature_a": [0, 1, 2, 3]}),
        label=np.array([0, 1, 1, 0]),
        feature_names=["feature_a"],
    )


def test_evaluate_model_uses_best_iteration_and_returns_metrics(
    evaluation_dmatrix: xgb.DMatrix,
    patch_logger: MagicMock,
) -> None:
    """Evaluate using the booster best iteration and return all report artifacts."""
    booster = MagicMock()
    booster.best_iteration = 2
    booster.predict.return_value = np.array([0.1, 0.7, 0.9, 0.2])
    booster.get_score.return_value = {"feature_a": 1.5}

    metrics, report, cm, importances = training._evaluate_model(
        booster,
        evaluation_dmatrix,
        scale_pos_weight=1.0,
        classification_threshold=0.5,
    )

    booster.predict.assert_called_once_with(evaluation_dmatrix, iteration_range=(0, 3))
    assert {
        "val_roc_auc",
        "val_pr_auc",
        "val_logloss",
        "val_brier_score",
        "default_threshold",
        "val_precision_default",
        "val_recall_default",
        "val_f1_default",
        "cm_default",
        "optimal_threshold",
        "val_precision_optimal",
        "val_recall_optimal",
        "val_f1_optimal",
        "cm_optimal",
    }.issubset(metrics)
    assert metrics["default_threshold"] == 0.5
    assert isinstance(report, dict)
    assert cm == [[2, 0], [0, 2]]
    assert importances == {"feature_a": 1.5}


def test_evaluate_model_uses_all_iterations_without_best_iteration(
    evaluation_dmatrix: xgb.DMatrix,
    patch_logger: MagicMock,
) -> None:
    """Evaluate with all boosting iterations when best_iteration is absent."""
    booster = MagicMock()
    booster.best_iteration = None
    booster.predict.return_value = np.array([0.1, 0.7, 0.9, 0.2])
    booster.get_score.return_value = {}

    training._evaluate_model(
        booster,
        evaluation_dmatrix,
        scale_pos_weight=2.0,
        classification_threshold=0.5,
    )

    booster.predict.assert_called_once_with(evaluation_dmatrix)


def test_export_model_card_writes_expected_json(tmp_path: Path, patch_logger: MagicMock) -> None:
    """Serialize and persist the expected model-card structure."""
    output = tmp_path / "model_card.json"

    training._export_model_card(
        output,
        params={"objective": "binary:logistic"},
        metrics={
            "val_roc_auc": 0.9,
            "val_pr_auc": 0.8,
            "val_logloss": 0.3,
            "val_brier_score": 0.1,
            "default_threshold": 0.5,
            "optimal_threshold": 0.42,
            "val_f1_default": 0.7,
            "val_f1_optimal": 0.75,
            "cm_default": {"tn": 2, "fp": 1, "fn": 1, "tp": 2},
            "cm_optimal": {"tn": 3, "fp": 0, "fn": 1, "tp": 2},
        },
        clf_report={"1": {"precision": 0.75}},
        feature_importances={"feature_a": 2.0},
        feature_columns=["feature_a", "feature_b"],
        target_column="target",
        train_shape=(100, 3),
        val_shape=(20, 3),
        train_data_path=Path("/data/train.parquet"),
        val_data_path=Path("/data/val.parquet"),
    )

    assert output.exists()
    data = json.loads(output.read_text())
    assert data["model_details"]["developer"] == "ML Engineering Team"
    assert data["model_details"]["model_type"] == "XGBoost Classifier"
    assert data["model_details"]["hyperparameters"]["objective"] == "binary:logistic"
    assert data["metrics"]["performance_measures"]["val_roc_auc"] == 0.9
    assert data["evaluation_data"]["datasets"][0]["num_rows"] == 20
    assert data["training_data"]["datasets"][0]["num_rows"] == 100
    assert data["quantitative_analyses"]["feature_importance_gain"] == {"feature_a": 2.0}


def test_export_model_card_wraps_serialization_error(
    tmp_path: Path,
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap serialization failures in RuntimeError."""
    monkeypatch.setattr(
        training.orjson,
        "dumps",
        MagicMock(side_effect=TypeError("cannot serialize")),
    )

    with pytest.raises(RuntimeError, match="Failed to save model card"):
        training._export_model_card(
            tmp_path / "model_card.json",
            params={},
            metrics={},
            clf_report={},
            feature_importances={},
            feature_columns=[],
            target_column="target",
            train_shape=(1, 1),
            val_shape=(1, 1),
            train_data_path=Path("train.parquet"),
            val_data_path=Path("val.parquet"),
        )

    patch_logger.error.assert_called_once()


def test_export_model_card_wraps_write_error(
    tmp_path: Path,
    patch_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap filesystem write failures in RuntimeError."""
    output = MagicMock(spec=Path)
    output.__str__.return_value = str(tmp_path / "model_card.json")
    monkeypatch.setattr(
        training.orjson,
        "dumps",
        lambda *_args, **_kwargs: b"{}",
    )
    output.write_bytes.side_effect = OSError("disk full")

    with pytest.raises(RuntimeError, match="Failed to save model card"):
        training._export_model_card(
            output,
            params={},
            metrics={},
            clf_report={},
            feature_importances={},
            feature_columns=[],
            target_column="target",
            train_shape=(1, 1),
            val_shape=(1, 1),
            train_data_path=Path("train.parquet"),
            val_data_path=Path("val.parquet"),
        )


@pytest.fixture
def fake_settings() -> SimpleNamespace:
    """Return the settings shape consumed by train_xgb_model."""
    return SimpleNamespace(
        PARAMS=SimpleNamespace(
            schema_config=SimpleNamespace(
                target_column="target",
                feature_columns=["feature_a", "feature_b"],
            ),
            train=SimpleNamespace(
                verbose_eval=0,
                num_boost_round=10,
                early_stopping_rounds=2,
                xgb_params={"objective": "binary:logistic"},
            ),
            evaluate=SimpleNamespace(classification_threshold=0.5),
        )
    )


def _patch_training_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    *,
    best_iteration: int = 4,
    best_score: float = 0.12,
) -> dict[str, Any]:
    """Patch train_xgb_model dependencies with deterministic unit-test doubles."""
    train_data = pd.DataFrame(
        {
            "feature_a": [0, 1, 2, 3],
            "feature_b": [1, 1, 0, 0],
            "target": [0, 0, 1, 1],
        }
    )
    val_data = pd.DataFrame(
        {
            "feature_a": [0, 1, 2, 3],
            "feature_b": [1, 0, 1, 0],
            "target": [0, 1, 1, 0],
        }
    )

    monkeypatch.setattr(training, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(training.pd, "read_parquet", MagicMock(side_effect=[train_data, val_data]))

    dtrain = MagicMock(spec=xgb.DMatrix)
    dval = MagicMock(spec=xgb.DMatrix)
    booster = MagicMock()
    booster.best_iteration = best_iteration
    booster.best_score = best_score
    booster.save_model = MagicMock()

    build = MagicMock(return_value=(dtrain, dval, 2.0))
    train = MagicMock(
        return_value=(
            booster,
            {"train": {"logloss": [0.4]}, "validation": {"logloss": [0.5]}},
            {"objective": "binary:logistic", "scale_pos_weight": 2.0},
        )
    )
    evaluate = MagicMock(
        return_value=(
            {
                "val_roc_auc": 0.9,
                "val_pr_auc": 0.8,
                "val_logloss": 0.3,
                "val_brier_score": 0.1,
                "default_threshold": 0.5,
                "optimal_threshold": 0.4,
                "val_f1_default": 0.7,
                "val_f1_optimal": 0.8,
                "cm_default": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
                "cm_optimal": {"tn": 2, "fp": 0, "fn": 1, "tp": 1},
            },
            {"0": {"precision": 1.0}},
            [[2, 0], [1, 1]],
            {"feature_b": 1.0, "feature_a": 2.0},
        )
    )
    export_card = MagicMock()

    monkeypatch.setattr(training, "_build_dmatrices", build)
    monkeypatch.setattr(training, "_train_model", train)
    monkeypatch.setattr(training, "_evaluate_model", evaluate)
    monkeypatch.setattr(training, "_export_model_card", export_card)

    return {
        "build": build,
        "train": train,
        "evaluate": evaluate,
        "export_card": export_card,
        "booster": booster,
        "read_parquet": training.pd.read_parquet,
        "train_data": train_data,
        "val_data": val_data,
        "model": tmp_path / "model.json",
        "card": tmp_path / "model_card.json",
        "report": tmp_path / "report.json",
    }


def test_train_xgb_model_success_writes_report(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Run the complete orchestration path and validate the exported report."""
    mocks = _patch_training_pipeline(monkeypatch, tmp_path, fake_settings)

    # save_model is responsible for producing a non-empty model artifact.
    mocks["booster"].save_model.side_effect = lambda path: Path(path).write_bytes(b"model")

    training.train_xgb_model(
        tmp_path / "train.parquet",
        tmp_path / "val.parquet",
        mocks["model"],
        mocks["card"],
        mocks["report"],
    )

    assert mocks["model"].exists()
    assert mocks["model"].read_bytes() == b"model"
    assert mocks["report"].exists()

    report = json.loads(mocks["report"].read_text())
    assert report["metadata"]["target_column"] == "target"
    assert report["metadata"]["feature_count"] == 2
    assert report["metadata"]["class_imbalance"]["scale_pos_weight"] == 2.0
    assert report["training_summary"]["best_iteration"] == 4
    assert report["training_summary"]["best_score"] == pytest.approx(0.12)
    assert report["performance_evaluation"]["confusion_matrix_unpacked"] == {
        "true_negatives": 2,
        "false_positives": 0,
        "false_negatives": 1,
        "true_positives": 1,
    }
    assert report["performance_evaluation"]["feature_importance_gain"] == {
        "feature_a": 2.0,
        "feature_b": 1.0,
    }

    mocks["build"].assert_called_once()
    mocks["train"].assert_called_once()
    mocks["evaluate"].assert_called_once()
    mocks["export_card"].assert_called_once()


def test_train_xgb_model_model_save_failure_is_wrapped(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Wrap booster serialization failures in RuntimeError."""
    mocks = _patch_training_pipeline(monkeypatch, tmp_path, fake_settings)
    mocks["booster"].save_model.side_effect = OSError("permission denied")

    with pytest.raises(RuntimeError, match="Failed to save the trained model"):
        training.train_xgb_model(
            tmp_path / "train.parquet",
            tmp_path / "val.parquet",
            mocks["model"],
            mocks["card"],
            mocks["report"],
        )

    patch_logger.error.assert_called()


def test_train_xgb_model_rejects_empty_model_file(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Raise when save_model returns without producing a valid artifact."""
    mocks = _patch_training_pipeline(monkeypatch, tmp_path, fake_settings)
    mocks["booster"].save_model.side_effect = lambda path: Path(path).touch()

    with pytest.raises(RuntimeError, match="Model file is empty or does not exist"):
        training.train_xgb_model(
            tmp_path / "train.parquet",
            tmp_path / "val.parquet",
            mocks["model"],
            mocks["card"],
            mocks["report"],
        )

    patch_logger.error.assert_called()


def test_train_xgb_model_report_export_failure_is_wrapped(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Wrap failures while serializing or writing the training report."""
    mocks = _patch_training_pipeline(monkeypatch, tmp_path, fake_settings)
    mocks["booster"].save_model.side_effect = lambda path: Path(path).write_bytes(b"model")
    monkeypatch.setattr(
        training.orjson,
        "dumps",
        MagicMock(side_effect=TypeError("report serialization failed")),
    )

    with pytest.raises(RuntimeError, match="Failed to export training report"):
        training.train_xgb_model(
            tmp_path / "train.parquet",
            tmp_path / "val.parquet",
            mocks["model"],
            mocks["card"],
            mocks["report"],
        )

    patch_logger.error.assert_called()


def test_train_xgb_model_handles_nan_best_score(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Convert a NaN best score to null in the training report."""
    mocks = _patch_training_pipeline(
        monkeypatch,
        tmp_path,
        fake_settings,
        best_iteration=6,
        best_score=float("nan"),
    )
    mocks["booster"].save_model.side_effect = lambda path: Path(path).write_bytes(b"model")

    training.train_xgb_model(
        tmp_path / "train.parquet",
        tmp_path / "val.parquet",
        mocks["model"],
        mocks["card"],
        mocks["report"],
    )

    report = json.loads(mocks["report"].read_text())
    assert report["training_summary"]["best_iteration"] == 6
    assert report["training_summary"]["best_score"] is None


def test_train_xgb_model_uses_booster_attribute_fallbacks(
    tmp_path: Path,
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    patch_logger: MagicMock,
) -> None:
    """Exercise the fallback best-iteration and best-score paths."""
    mocks = _patch_training_pipeline(monkeypatch, tmp_path, fake_settings)
    booster = mocks["booster"]
    del booster.best_iteration
    del booster.best_score
    booster.save_model.side_effect = lambda path: Path(path).write_bytes(b"model")

    training.train_xgb_model(
        tmp_path / "train.parquet",
        tmp_path / "val.parquet",
        mocks["model"],
        mocks["card"],
        mocks["report"],
    )

    report = json.loads(mocks["report"].read_text())
    assert report["training_summary"]["best_iteration"] == 9
    assert report["training_summary"]["best_score"] is None
