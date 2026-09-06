"""XGBoost Training Module.

This module provides functionality to train an XGBoost model
for churn prediction, evaluate its performance, and generate
a comprehensive model card in JSON format.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings

# Constants
EPSILON = 1e-15


def _build_dmatrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[xgb.DMatrix, xgb.DMatrix, float]:
    """Construct XGBoost training and validation matrices.

    Selects the configured feature columns, attaches the target labels, and
    creates XGBoost ``DMatrix`` objects for both training and validation data.
    The function also calculates the class-imbalance weight used for the
    positive class.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Processed training dataset containing feature and target columns.
    val_df : pandas.DataFrame
        Processed validation dataset containing feature and target columns.
    target_column : str
        Name of the binary target column.
    feature_columns : list of str
        Ordered feature columns supplied to the XGBoost model.

    Returns
    -------
    dtrain : xgboost.DMatrix
        Training matrix containing the selected features and target labels.
    dval : xgboost.DMatrix
        Validation matrix containing the selected features and target labels.
    scale_pos_weight : float
        Ratio of negative to positive training samples used to compensate for
        class imbalance.

    Raises
    ------
    ValueError
        If the target or feature columns are unavailable, DMatrix creation
        fails, or either training class is absent.
    """
    logger = get_logger()

    logger.debug(
        "Building DMatrices. Target: %s, Features count: %d", target_column, len(feature_columns)
    )

    # Create DMatrix for training and validation datasets
    try:
        dtrain = xgb.DMatrix(
            train_df[feature_columns],
            label=train_df[target_column],
            feature_names=feature_columns,
        )
        dval = xgb.DMatrix(
            val_df[feature_columns],
            label=val_df[target_column],
            feature_names=feature_columns,
        )
    except Exception as err:
        logger.error("Failed to create DMatrix: %s", err)
        raise ValueError(f"Failed to create DMatrix: {err}") from err

    logger.info(
        "DMatrix created successfully. Shapes: dtrain=%s, dval=%s",
        dtrain.num_row(),
        dval.num_row(),
    )

    # Calculate scale_pos_weight for handling class imbalance
    num_pos: float = train_df[target_column].sum()
    logger.debug("Calculating scale_pos_weight. Positive samples: %.0f", num_pos)

    if num_pos == 0:
        raise ValueError("Training data contains 0 positive samples.")

    num_neg: float = float(train_df.shape[0] - num_pos)
    logger.debug("Negative samples: %.0f", num_neg)

    if num_neg == 0:
        raise ValueError("Training data contains 0 negative samples.")

    scale_pos_weight = num_neg / num_pos

    logger.info(
        "Class balance — Negatives (0): %d, Positives (1): %d. Computed scale_pos_weight: %.4f",
        int(num_neg),
        int(num_pos),
        scale_pos_weight,
    )

    return dtrain, dval, scale_pos_weight


def _train_model(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    *,
    xgb_params: dict[str, Any],
    scale_pos_weight: float,
    n_estimators: int,
    early_stopping_rounds: int,
    verbose_eval: int,
) -> tuple[xgb.Booster, dict[str, dict[str, list[float]]], dict[str, Any]]:
    """Train an XGBoost booster with validation-based early stopping.

    Adds the calculated class-imbalance weight to the supplied XGBoost
    parameters and trains the booster against both the training and
    validation matrices.

    Parameters
    ----------
    dtrain : xgboost.DMatrix
        Training data and labels used to fit the booster.
    dval : xgboost.DMatrix
        Validation data and labels used for evaluation and early stopping.
    xgb_params : dict
        XGBoost training parameters.
    scale_pos_weight : float
        Positive-class weighting factor used to compensate for class imbalance.
    n_estimators : int
        Maximum number of boosting rounds.
    early_stopping_rounds : int
        Number of consecutive rounds without validation improvement allowed
        before training stops.
    verbose_eval : int
        Frequency at which training evaluation metrics are logged.

    Returns
    -------
    booster : xgboost.Booster
        Trained XGBoost booster.
    evals_result : dict
        Evaluation metrics recorded for the training and validation datasets
        during boosting.
    params : dict
        Final XGBoost parameters used to train the booster.

    Raises
    ------
    ValueError
        If XGBoost training fails.
    """
    logger = get_logger()

    # Ensure scale_pos_weight is set
    params = dict(xgb_params)
    params["scale_pos_weight"] = scale_pos_weight
    logger.debug("Final XGBoost parameters: %s", params)

    # Train the model with early stopping
    evals_result: dict[str, Any] = {}
    try:
        logger.debug(
            "Starting xgb.train with n_estimators=%d, early_stopping=%d",
            n_estimators,
            early_stopping_rounds,
        )
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "validation")],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
    except Exception as err:
        logger.error("Failed to train the model: %s", err)
        raise ValueError(f"Model training failed: {err}") from err

    best_iter = getattr(booster, "best_iteration", n_estimators - 1)
    best_score = getattr(booster, "best_score", float("nan"))

    logger.info(
        "Model training completed. Best Iteration: %d, Best Score: %.4f",
        best_iter,
        best_score,
    )

    return booster, evals_result, params


def unscale_probabilities(
    y_prob_scaled: np.ndarray,
    scale_pos_weight: float,
) -> np.ndarray:
    """Convert class-weighted probabilities to calibrated likelihoods.

    Reverses the probability distortion introduced by ``scale_pos_weight``
    during model training. When no effective class weighting is present, the
    original probabilities are returned unchanged.

    Parameters
    ----------
    y_prob_scaled : numpy.ndarray
        Probabilities predicted by the class-weighted model.
    scale_pos_weight : float
        Positive-class weighting factor used during model training.

    Returns
    -------
    numpy.ndarray
        Probabilities adjusted to represent the estimated likelihood under
        the original class distribution.
    """
    logger = get_logger()

    if scale_pos_weight == 1.0 or scale_pos_weight <= 0:
        logger.debug("scale_pos_weight is %.4f; skipping probability unscaling", scale_pos_weight)
        return y_prob_scaled

    logger.debug("Unscaling probabilities using scale_pos_weight: %.4f", scale_pos_weight)
    p: np.ndarray = np.clip(y_prob_scaled, EPSILON, 1.0 - EPSILON)
    calibrated_p = p / (p + scale_pos_weight * (1.0 - p))

    logger.debug(
        "Probability range - Scaled: [%.4f, %.4f], Unscaled: [%.4f, %.4f]",
        y_prob_scaled.min(),
        y_prob_scaled.max(),
        calibrated_p.min(),
        calibrated_p.max(),
    )

    return calibrated_p


def _find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float,
) -> tuple[float, float]:
    """Find the probability threshold that maximizes validation F1 score.

    Evaluates the thresholds generated from the precision-recall curve and
    selects the threshold producing the highest F1 score. If no candidate
    threshold is available, the supplied default threshold is returned.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth binary labels for the validation dataset.
    y_prob : numpy.ndarray
        Predicted positive-class probabilities.
    default_threshold : float
        Fallback classification threshold used when no candidate threshold
        can be evaluated.

    Returns
    -------
    threshold : float
        Probability threshold producing the highest validation F1 score.
    f1_score : float
        F1 score achieved at the selected threshold.
    """
    logger = get_logger()

    logger.debug("Finding optimal threshold. Sample size: %d", len(y_true))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        logger.debug(
            "No thresholds found by precision_recall_curve; using default: %.4f",
            default_threshold,
        )
        return default_threshold, 0.0

    candidate_precisions = precisions[:-1]
    candidate_recalls = recalls[:-1]

    denom = candidate_precisions + candidate_recalls
    f1_scores = np.zeros_like(denom, dtype=float)

    mask = denom > 0
    f1_scores[mask] = 2.0 * candidate_precisions[mask] * candidate_recalls[mask] / denom[mask]

    best_idx = int(np.argmax(f1_scores))
    opt_threshold = float(thresholds[best_idx])
    opt_f1 = float(f1_scores[best_idx])

    logger.debug(
        "Optimal threshold search complete. Best Threshold: %.4f, Max F1: %.4f",
        opt_threshold,
        opt_f1,
    )

    return opt_threshold, opt_f1


def _evaluate_model(
    booster: xgb.Booster,
    dval: xgb.DMatrix,
    scale_pos_weight: float,
    classification_threshold: float,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    list[list[int]],
    dict[str, Any],
]:
    """Evaluate model performance on the validation dataset.

    Generates probability predictions, converts class-weighted probabilities
    into calibrated likelihoods, determines an F1-optimized threshold, and
    calculates classification, calibration, ranking, and feature-importance
    metrics.

    Parameters
    ----------
    booster : xgboost.Booster
        Trained XGBoost model to evaluate.
    dval : xgboost.DMatrix
        Validation dataset containing features and ground-truth labels.
    scale_pos_weight : float
        Class-imbalance weighting factor used during training.
    classification_threshold : float
        Default probability threshold used for binary classification.

    Returns
    -------
    metrics : dict
        Aggregate model performance metrics, including ROC-AUC, PR-AUC,
        log loss, Brier score, F1 scores, and decision thresholds.
    classification_report : dict
        Precision, recall, F1, and support metrics returned by
        ``sklearn.metrics.classification_report``.
    confusion_matrix : list of list of int
        Two-by-two confusion matrix evaluated at the selected classification
        threshold.
    feature_importances : dict
        Feature importance values derived from XGBoost gain statistics.
    """
    logger = get_logger()

    y_true: np.ndarray = dval.get_label().astype(np.int32)

    # 1. Predict using the best iteration strictly
    best_iteration: int | None = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        logger.debug("Using best_iteration=%d for predictions", best_iteration)
        y_prob_scaled: np.ndarray = booster.predict(dval, iteration_range=(0, best_iteration + 1))
    else:
        logger.debug("best_iteration not found; using all iterations for predictions")
        y_prob_scaled: np.ndarray = booster.predict(dval)

    # 2. Convert scaled probabilities back to true unweighted probabilities
    y_prob_calibrated: np.ndarray = unscale_probabilities(y_prob_scaled, scale_pos_weight)

    # 3. Dynamic Threshold Optimization for Imbalanced Churn
    logger.debug("Optimizing threshold based on y_true and y_prob_calibrated")
    optimal_threshold, best_val_f1 = _find_optimal_threshold(
        y_true, y_prob_calibrated, classification_threshold
    )

    # 4. Generate predictions using specified vs optimal threshold
    y_pred_default: np.ndarray = (y_prob_calibrated >= classification_threshold).astype(np.int32)
    y_pred_optimal: np.ndarray = (y_prob_calibrated >= optimal_threshold).astype(np.int32)

    # 5. Core Metric Calculations
    roc_auc = float(roc_auc_score(y_true, y_prob_calibrated))
    pr_auc = float(average_precision_score(y_true, y_prob_calibrated))
    brier_loss = float(brier_score_loss(y_true, y_prob_calibrated))
    logloss_val = float(log_loss(y_true, y_prob_calibrated))

    # Confusion matrix & metrics at default threshold
    cm_default: list[list[int]] = confusion_matrix(y_true, y_pred_default).tolist()
    tn_def, fp_def, fn_def, tp_def = np.array(cm_default).ravel()

    # Confusion matrix & metrics at optimal threshold
    cm_optimal: list[list[int]] = confusion_matrix(y_true, y_pred_optimal).tolist()
    tn_opt, fp_opt, fn_opt, tp_opt = np.array(cm_optimal).ravel()

    # Compile all evaluation metrics
    metrics: dict[str, Any] = {
        # Probabilistic Calibration Metrics
        "val_roc_auc": roc_auc,
        "val_pr_auc": pr_auc,
        "val_logloss": logloss_val,
        "val_brier_score": brier_loss,
        # Default Threshold Metrics
        "default_threshold": float(classification_threshold),
        "val_precision_default": float(precision_score(y_true, y_pred_default, zero_division=0)),
        "val_recall_default": float(recall_score(y_true, y_pred_default, zero_division=0)),
        "val_f1_default": float(f1_score(y_true, y_pred_default, zero_division=0)),
        "cm_default": {"tn": int(tn_def), "fp": int(fp_def), "fn": int(fn_def), "tp": int(tp_def)},
        # Optimal Threshold Metrics
        "optimal_threshold": optimal_threshold,
        "val_precision_optimal": float(precision_score(y_true, y_pred_optimal, zero_division=0)),
        "val_recall_optimal": float(recall_score(y_true, y_pred_optimal, zero_division=0)),
        "val_f1_optimal": float(f1_score(y_true, y_pred_optimal, zero_division=0)),
        "cm_optimal": {"tn": int(tn_opt), "fp": int(fp_opt), "fn": int(fn_opt), "tp": int(tp_opt)},
    }

    clf_report: dict[str, float] = classification_report(
        y_true,
        y_pred_optimal,
        output_dict=True,
        zero_division=0,
    )

    # 6. Extract Feature Importances by Gain
    feature_importances = booster.get_score(importance_type="gain")
    logger.debug("Extracted %d feature importance scores by gain", len(feature_importances))

    logger.info(
        "Evaluation Complete — ROC-AUC: %.4f | PR-AUC: %.4f | LogLoss: %.4f | Brier: %.4f",
        roc_auc,
        pr_auc,
        logloss_val,
        brier_loss,
    )
    logger.info(
        "Threshold Shift — Default (0.50) F1: %.4f | Optimal (%.4f) F1: %.4f",
        metrics["val_f1_default"],
        optimal_threshold,
        best_val_f1,
    )

    return metrics, clf_report, cm_optimal, feature_importances


def _export_model_card(
    output_path: Path,
    *,
    model_type: str = "XGBoost Classifier",
    version: str = "0.1.0",
    developer: str = "ML Engineering Team",
    params: dict[str, Any],
    metrics: dict[str, Any],
    clf_report: dict[str, Any],
    feature_importances: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
    train_shape: tuple[int, int],
    val_shape: tuple[int, int],
    train_data_path: Path,
    val_data_path: Path,
) -> None:
    """Generate and persist a machine-readable model card.

    Collects model metadata, training configuration, evaluation metrics,
    feature importance, dataset dimensions, and model limitations into a
    structured JSON document.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination path for the generated model card.
    params : dict
        Final model training parameters.
    metrics : dict
        Model evaluation metrics and decision thresholds.
    clf_report : dict
        Classification report generated from validation predictions.
    feature_importances : dict
        Mapping of feature names to XGBoost gain-based importance values.
    feature_columns : list of str
        Ordered feature names used by the model.
    target_column : str
        Name of the binary target column.
    train_shape : tuple of int
        Number of rows and columns in the training dataset.
    val_shape : tuple of int
        Number of rows and columns in the validation dataset.
    train_data_path : pathlib.Path
        Source path of the training dataset.
    val_data_path : pathlib.Path
        Source path of the validation dataset.

    Raises
    ------
    RuntimeError
        If the model card cannot be serialized or written to disk.
    """
    logger = get_logger()

    logger.debug("Constructing model card schema for version %s", version)
    # Construct schema compliant with Mitchell et al. Model Card standard
    model_card: dict[str, Any] = {
        "model_details": {
            "developer": developer,
            "model_date": datetime.now(UTC).isoformat(),
            "model_version": version,
            "model_type": model_type,
            "training_algorithm": "XGBoost Gradient Boosted Trees",
            "hyperparameters": params,
            "features_used": feature_columns,
            "target_column": target_column,
            "license": "MIT License",
        },
        "intended_use": {
            "primary_intended_uses": [
                "Predict customer churn probability on tabular features.",
                "Batch offline inference and real-time ONNX scoring service.",
            ],
            "primary_intended_users": [
                "Customer Retention Teams",
                "Automated Lifecycle Marketing Pipelines",
            ],
            "out_of_scope_use_cases": [
                "Real-time decision making without probability threshold calibration.",
                "Inference on un-preprocessed raw transactional telemetry.",
            ],
        },
        "factors": {
            "relevant_factors": [
                "Account tenure",
                "Historical usage frequency",
                "Support ticket volume",
            ],
            "evaluation_factors": [
                "Class balance skewness",
                "Decision threshold variation impact on precision/recall trade-off",
            ],
        },
        "metrics": {
            "performance_measures": {
                "val_roc_auc": metrics.get("val_roc_auc"),
                "val_pr_auc": metrics.get("val_pr_auc"),
                "val_logloss": metrics.get("val_logloss"),
                "val_brier_score": metrics.get("val_brier_score"),
            },
            "decision_thresholds": {
                "default_threshold": metrics.get("default_threshold"),
                "optimal_threshold": metrics.get("optimal_threshold"),
                "default_threshold_f1": metrics.get("val_f1_default"),
                "optimal_threshold_f1": metrics.get("val_f1_optimal"),
            },
            "variation_approaches": "Evaluated against class-imbalance re-weighted thresholds.",
        },
        "evaluation_data": {
            "datasets": [
                {
                    "name": "Validation Set",
                    "path": str(val_data_path),
                    "num_rows": val_shape[0],
                    "num_columns": val_shape[1],
                }
            ],
            "preprocessing": "Standardized and normalized according to exported preprocessor artifact.",
        },
        "training_data": {
            "datasets": [
                {
                    "name": "Training Set",
                    "path": str(train_data_path),
                    "num_rows": train_shape[0],
                    "num_columns": train_shape[1],
                }
            ],
            "preprocessing": "Standardized and normalized according to exported preprocessor artifact.",
        },
        "quantitative_analyses": {
            "unitary_results": {
                "classification_report": clf_report,
                "confusion_matrix_default": metrics.get("cm_default"),
                "confusion_matrix_optimal": metrics.get("cm_optimal"),
            },
            "feature_importance_gain": feature_importances,
        },
        "ethical_considerations": {
            "risks": [
                "False positive predictions may result in unwarranted discount offers.",
                "False negative predictions lead to unmitigated churn.",
            ],
            "mitigations": [
                "Probability unscaling applied post-training to eliminate artificially elevated positive probabilities.",
                "Optimal decision threshold selected to balance recall against business margin impact.",
            ],
        },
        "caveats_and_recommendations": [
            "Model probabilities must be unscaled using scale_pos_weight if calibrated true likelihoods are required.",
            "Recalibrate the optimal prediction threshold periodically as underlying churn baselines drift.",
        ],
    }

    try:
        logger.debug("Serializing model card to JSON bytes")
        # Serialize to binary JSON using orjson with native NumPy support
        json_bytes = orjson.dumps(
            model_card,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        )

        logger.debug("Writing model card to: %s", output_path)
        output_path.write_bytes(json_bytes)
        logger.info("Model card exported successfully to: %s", output_path)
    except Exception as err:
        logger.error("Failed to generate or save model card: %s", err)
        raise RuntimeError(f"Failed to save model card to '{output_path}': {err}") from err


def train_xgb_model(
    train_data_path: str | Path,
    val_data_path: str | Path,
    model_filepath: str | Path,
    model_card_path: str | Path,
    report_path: str | Path,
) -> None:
    """Train, evaluate, and export the churn prediction XGBoost model.

    Orchestrates the complete model training stage by loading processed
    datasets, constructing XGBoost matrices, calculating class-imbalance
    weighting, training the booster, evaluating validation performance,
    and exporting the model, model card, and training report.

    Parameters
    ----------
    train_data_path : str or pathlib.Path
        Path to the processed training Parquet dataset.
    val_data_path : str or pathlib.Path
        Path to the processed validation Parquet dataset.
    model_filepath : str or pathlib.Path
        Destination path for the trained XGBoost model.
    model_card_path : str or pathlib.Path
        Destination path for the machine-readable model card.
    report_path : str or pathlib.Path
        Destination path for the comprehensive training report.

    Raises
    ------
    FileNotFoundError
        If either processed dataset cannot be found.
    RuntimeError
        If model training, model serialization, model-card generation, or
        training-report generation fails.
    ValueError
        If the processed datasets are incompatible with the configured
        training schema or model.
    """
    logger = get_logger()
    settings = get_settings()

    # Resolve paths to absolute paths
    train_data_path = Path(train_data_path).expanduser().resolve()
    val_data_path = Path(val_data_path).expanduser().resolve()
    model_filepath = Path(model_filepath).expanduser().resolve()
    model_card_path = Path(model_card_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()

    logger.debug(
        "Loading configuration from settings. Target: %s",
        settings.PARAMS.schema_config.target_column,
    )

    # Extract parameters from the validated pipeline configuration
    target_col = settings.PARAMS.schema_config.target_column
    feature_cols = settings.PARAMS.schema_config.feature_columns
    verbose_eval = settings.PARAMS.train.verbose_eval
    num_boost_round = settings.PARAMS.train.num_boost_round
    early_stopping_rounds = settings.PARAMS.train.early_stopping_rounds
    classification_threshold = settings.PARAMS.evaluate.classification_threshold
    xgb_params = dict(settings.PARAMS.train.xgb_params)

    # 0. Load the datasets
    logger.debug("Reading Parquet files: train=%s, val=%s", train_data_path, val_data_path)
    train_df = pd.read_parquet(train_data_path)
    val_df = pd.read_parquet(val_data_path)
    train_rows, train_cols = train_df.shape
    val_rows, val_cols = val_df.shape

    # 1. Build DMatrix objects
    dtrain, dval, scale_pos_weight = _build_dmatrices(
        train_df,
        val_df,
        target_col,
        feature_cols,
    )

    # 2. Train the model
    booster, evals_result, params = _train_model(
        dtrain,
        dval,
        xgb_params=xgb_params,
        scale_pos_weight=scale_pos_weight,
        n_estimators=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    # 3. Evaluate the model
    metrics, clf_report, cm, feature_importances = _evaluate_model(
        booster,
        dval,
        scale_pos_weight=scale_pos_weight,
        classification_threshold=classification_threshold,
    )

    # Unpack confusion matrix for clarity
    (tn, fp), (fn, tp) = cm

    sorted_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # 4. Save the trained model
    try:
        logger.debug("Saving booster model to %s", model_filepath)
        booster.save_model(str(model_filepath))
        logger.info("Trained model saved successfully to: %s", model_filepath)
    except Exception as err:
        logger.error("Failed to save the trained model: %s", err)
        raise RuntimeError(
            f"Failed to save the trained model to '{model_filepath}': {err}"
        ) from err

    if not model_filepath.exists() or model_filepath.stat().st_size == 0:
        logger.error("Model file is empty or does not exist: %s", model_filepath)
        raise RuntimeError(f"Model file is empty or does not exist: '{model_filepath}'")

    # 5. Generate and save model card
    _export_model_card(
        output_path=model_card_path,
        params=params,
        metrics=metrics,
        clf_report=clf_report,
        feature_importances=sorted_importances,
        feature_columns=feature_cols,
        target_column=target_col,
        train_shape=train_df.shape,
        val_shape=val_df.shape,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )

    # 6. Export comprehensive training report
    try:
        logger.debug("Exporting training report to %s", report_path)

        best_iter = getattr(booster, "best_iteration", num_boost_round - 1)
        best_score_raw = getattr(booster, "best_score", float("nan"))
        best_score = float(best_score_raw) if not np.isnan(best_score_raw) else None

        report_data: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "target_column": target_col,
                "feature_count": len(feature_cols),
                "feature_names": feature_cols,
                "dataset_dimensions": {
                    "train_rows": train_rows,
                    "train_cols": train_cols,
                    "val_rows": val_rows,
                    "val_cols": val_cols,
                },
                "class_imbalance": {
                    "scale_pos_weight": scale_pos_weight,
                },
            },
            "training_summary": {
                "num_boost_round": num_boost_round,
                "early_stopping_rounds": early_stopping_rounds,
                "best_iteration": best_iter,
                "best_score": best_score,
                "hyperparameters": params,
                "evals_result": evals_result,
            },
            "performance_evaluation": {
                **metrics,
                "confusion_matrix_optimal_2x2": cm,
                "confusion_matrix_unpacked": {
                    "true_negatives": tn,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_positives": tp,
                },
                "classification_report": clf_report,
                "feature_importance_gain": sorted_importances,
            },
        }

        report_json_bytes = orjson.dumps(
            report_data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        )
        report_path.write_bytes(report_json_bytes)
        logger.info("Training report exported successfully to: %s", report_path)
    except Exception as err:
        logger.error("Failed to export training report: %s", err)
        raise RuntimeError(f"Failed to export training report to '{report_path}': {err}") from err
