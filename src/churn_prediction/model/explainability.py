"""Module for exporting SHAP explainability artifacts for dynamic dashboards.

This module computes SHAP values for XGBoost models and exports structured,
high-performance data artifacts (Parquet, JSON) designed for direct consumption
by interactive frontend dashboards (e.g., Dash, React, FastAPI backends).
"""

from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import shap
import xgboost as xgb

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings


class SHAPArtifactExporter:
    """Compute and export SHAP artifacts for churn model interpretation.

    This class transforms model predictions and SHAP explanations into
    dashboard-ready Parquet and JSON artifacts. It supports deterministic
    sampling for large datasets and produces both global feature importance
    metrics and customer-level churn/retention drivers.
    """

    def __init__(
        self,
        batch_id: str,
        df: pd.DataFrame,
        booster: xgb.Booster,
        output_dir: Path,
        *,
        max_samples: int = 10000,
        random_state: int = 42,
    ) -> None:
        """Initialize the SHAP artifact exporter.

        Parameters
        ----------
        batch_id : str
            Identifier for the current batch of explanations.
        df : pandas.DataFrame
            Feature matrix containing the customer records to explain. The
            DataFrame index is treated as the customer identifier.
        booster : xgboost.Booster
            Trained XGBoost booster for which SHAP explanations will be generated.
        output_dir : pathlib.Path
            Directory in which explainability artifacts will be written.
        max_samples : int, default=10000
            Maximum number of customer records to evaluate with SHAP. Larger
            datasets are sampled deterministically.
        random_state : int, default=42
            Random seed used when sampling the evaluation dataset.

        Raises
        ------
        Exception
            If the SHAP tree explainer cannot be initialized for the supplied
            XGBoost model.
        """
        # Initialize logging and settings
        self.logger = get_logger()
        self.settings = get_settings()

        self.logger.debug(
            "Initializing SHAPArtifactExporter: rows=%d, features=%d, output_dir=%s",
            len(df),
            df.shape[1],
            output_dir,
        )
        try:
            self.batch_id = batch_id
            self.output_dir = output_dir
            self.max_samples = max_samples
            self.random_state = random_state
            self.features_df = df.copy()
            self.customer_ids = df.index.values
            self.booster = booster
            self.explainer = shap.TreeExplainer(self.booster)
        except Exception:
            self.logger.exception("Failed to initialize SHAPArtifactExporter")
            raise

        self.logger.info(
            "SHAPArtifactExporter initialized: batch_id=%s",
            self.batch_id,
        )

    def _sample_dataset(self) -> pd.DataFrame:
        """Create a deterministic SHAP evaluation sample.

        Returns the complete feature dataset when it contains no more than
        ``max_samples`` records. Otherwise, returns a deterministic random
        sample and stores the corresponding customer identifiers.

        Returns
        -------
        pandas.DataFrame
            Feature matrix used for SHAP evaluation.
        """
        if len(self.features_df) > self.max_samples:
            self.logger.info(
                "Sampling evaluation set from %d to %d rows",
                len(self.features_df),
                self.max_samples,
            )

            sampled_df = self.features_df.sample(
                n=self.max_samples,
                random_state=self.random_state,
            )
            self.sampled_customer_ids = sampled_df.index.to_numpy()
            return sampled_df

        self.sampled_customer_ids = self.features_df.index.to_numpy()
        return self.features_df

    def _compute_shap_explanation(self, eval_df: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values for the supplied evaluation dataset.

        Converts the feature matrix into an XGBoost ``DMatrix``, evaluates the
        configured tree explainer, and wraps the resulting values together with
        feature data and the model baseline into a ``shap.Explanation``.

        Parameters
        ----------
        eval_df : pandas.DataFrame
            Feature matrix containing the records to explain.

        Returns
        -------
        shap.Explanation
            SHAP explanation containing per-feature contributions, baseline
            value, feature values, and feature names.

        Raises
        ------
        Exception
            If the feature matrix cannot be converted to an XGBoost ``DMatrix``
            or SHAP value computation fails.
        """
        self.logger.info(
            "Computing SHAP values for %d samples across %d features...",
            eval_df.shape[0],
            eval_df.shape[1],
        )
        try:
            feature_names = eval_df.columns.tolist()
            dmatrix = xgb.DMatrix(eval_df, feature_names=feature_names)
            shap_values = self.explainer.shap_values(dmatrix)

            # Convert base value to float for compatibility with dashboards
            base_val: float = self.explainer.expected_value
            base_val = float(base_val[0]) if isinstance(base_val, np.ndarray) else float(base_val)

            return shap.Explanation(
                values=shap_values,
                base_values=base_val,
                data=eval_df.values,
                feature_names=feature_names,
            )
        except Exception:
            self.logger.exception("Failed to compute SHAP values.")
            raise

    def export_artifacts(self, top_k_drivers: int = 5) -> Path:
        """Generate and persist dashboard-ready SHAP artifacts.

        The export contains the raw SHAP matrix, the aligned input feature
        matrix, customer-level risk profiles, and global feature-importance
        metadata.

        Parameters
        ----------
        top_k_drivers : int, default=5
            Maximum number of positive churn drivers and negative retention
            factors retained for each customer.

        Returns
        -------
        pathlib.Path
            Directory containing all generated explainability artifacts.

        Raises
        ------
        OSError
            If an artifact cannot be written to disk.
        Exception
            If SHAP computation or artifact generation fails.
        """
        batch_output_dir = self.output_dir
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting SHAP artifact generation run: destination=%s", batch_output_dir)

        try:
            eval_df = self._sample_dataset()
            explanation = self._compute_shap_explanation(eval_df)

            # 1. Export SHAP Value Matrix
            shap_cols: list[str] = [f"shap_{col}" for col in explanation.feature_names]
            shap_values = np.asarray(explanation.values)
            shap_df = pd.DataFrame(
                data=shap_values,
                columns=pd.Index(shap_cols),
                index=pd.Index(self.sampled_customer_ids, name="customer_id"),
            )
            shap_parquet_path = batch_output_dir / "raw_shap_values.parquet"
            shap_df.to_parquet(
                shap_parquet_path,
                engine="pyarrow",
                compression="snappy",
            )
            self.logger.debug("Exported SHAP matrix: %s", shap_parquet_path)

            # 2. Export Feature Matrix (Aligned with SHAP Values)
            eval_df.index = self.sampled_customer_ids
            eval_df.index.name = "customer_id"
            features_parquet_path = batch_output_dir / "input_features.parquet"
            eval_df.to_parquet(features_parquet_path, engine="pyarrow", compression="snappy")
            self.logger.debug("Exported feature matrix: %s", features_parquet_path)

            # 3. Export Customer Risk Profiles (Top Drivers)
            risk_profiles_df = self._build_customer_risk_profiles(
                explanation=explanation,
                customer_ids=self.sampled_customer_ids,
                top_k=top_k_drivers,
            )
            risk_parquet_path = batch_output_dir / "customer_risk_profiles.parquet"
            risk_profiles_df.to_parquet(risk_parquet_path, engine="pyarrow", compression="snappy")
            self.logger.debug("Exported customer risk profiles: %s", risk_parquet_path)

            # 4. Export Metadata (Global Metrics)
            self._export_metadata(
                explanation=explanation,
                output_path=batch_output_dir / "metadata.json",
                total_samples=len(eval_df),
            )

        except Exception:
            self.logger.exception("Export run failed for output directory: %s", batch_output_dir)
            raise

        self.logger.info("Successfully exported all SHAP artifacts to: %s", batch_output_dir)
        return batch_output_dir

    def _build_customer_risk_profiles(
        self,
        explanation: shap.Explanation,
        customer_ids: np.ndarray,
        top_k: int,
    ) -> pd.DataFrame:
        """Build customer-level churn and retention driver profiles.

        For each customer, ranks SHAP contributions and extracts the strongest
        positive contributions as churn drivers and strongest negative
        contributions as retention factors.

        Parameters
        ----------
        explanation : shap.Explanation
            SHAP explanation containing feature contributions and feature values.
        customer_ids : numpy.ndarray
            Customer identifiers aligned with the rows of ``explanation``.
        top_k : int
            Maximum number of positive and negative contributors retained per
            customer.

        Returns
        -------
        pandas.DataFrame
            Customer-level risk profile indexed by customer ID. The DataFrame
            contains the SHAP baseline, predicted model margin, serialized
            churn drivers, and serialized retention factors.
        """
        self.logger.debug("Extracting top %d risk drivers per customer...", top_k)
        shap_values = explanation.values
        feature_names = np.array(explanation.feature_names)
        base_val = explanation.base_values

        total_risk_scores = base_val + shap_values.sum(axis=1)

        # Vectorized sort indices (ascending order)
        sorted_indices = np.argsort(shap_values, axis=1)

        records: list[dict[str, Any]] = []

        for idx, cid in enumerate(customer_ids):
            row_shap = shap_values[idx]
            row_indices = sorted_indices[idx]

            # Top positive drivers (increase churn risk)
            top_pos_idx = row_indices[-top_k:][::-1]
            pos_drivers = [
                {
                    "feature": str(feature_names[i]),
                    "shap_value": float(row_shap[i]),
                    "feature_value": float(explanation.data[idx, i]),
                }
                for i in top_pos_idx
                if row_shap[i] > 0
            ]

            # Top negative drivers (decrease churn risk / protect retention)
            top_neg_idx = row_indices[:top_k]
            neg_drivers = [
                {
                    "feature": str(feature_names[i]),
                    "shap_value": float(row_shap[i]),
                    "feature_value": float(explanation.data[idx, i]),
                }
                for i in top_neg_idx
                if row_shap[i] < 0
            ]

            records.append(
                {
                    "customer_id": cid,
                    "base_value": base_val,
                    "predicted_margin": float(total_risk_scores[idx]),
                    "top_churn_drivers": orjson.dumps(pos_drivers).decode("utf-8"),
                    "top_retention_factors": orjson.dumps(neg_drivers).decode("utf-8"),
                }
            )

        profiles_df = pd.DataFrame(records)
        profiles_df.set_index("customer_id", inplace=True)
        return profiles_df

    def _export_metadata(
        self,
        explanation: shap.Explanation,
        output_path: Path,
        total_samples: int,
    ) -> None:
        """Compute global SHAP importance and export explainability metadata.

        Calculates mean absolute SHAP values for every feature, ranks features
        by their global contribution magnitude, and persists the resulting
        metadata as formatted JSON.

        Parameters
        ----------
        explanation : shap.Explanation
            SHAP explanation used to calculate global feature importance.
        output_path : pathlib.Path
            Destination path for the metadata JSON file.
        total_samples : int
            Number of customer records included in the SHAP evaluation.

        Raises
        ------
        OSError
            If the metadata file cannot be written to disk.
        """
        self.logger.debug("Computing global feature importance for metadata export...")
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)

        importance_ranking = sorted(
            zip(explanation.feature_names, mean_abs_shap, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        metadata = {
            "batch_id": self.batch_id,
            "base_value": float(explanation.base_values),
            "sample_count": total_samples,
            "feature_count": len(explanation.feature_names),
            "global_feature_importance": [
                {
                    "rank": rank + 1,
                    "feature": feat,
                    "mean_abs_shap": float(val),
                }
                for rank, (feat, val) in enumerate(importance_ranking)
            ],
        }

        with output_path.open("wb") as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        self.logger.debug("Exported metadata JSON: %s", output_path)


def generate_shap_artifacts(
    batch_id: str,
    df: pd.DataFrame,
    booster: xgb.Booster,
    shap_report_dir: Path,
) -> Path:
    """Generate SHAP explainability artifacts for a trained churn model.

    This is the public entry point for the explainability workflow. It
    initializes :class:`SHAPArtifactExporter`, computes the requested SHAP
    outputs, and persists the resulting artifacts to the configured report
    directory.

    Parameters
    ----------
    batch_id : str
        Identifier for the current batch of explanations.
    df : pandas.DataFrame
        Model feature matrix to explain.
    booster : xgboost.Booster
        Trained XGBoost model.
    shap_report_dir : pathlib.Path
        Directory in which SHAP artifacts will be generated.

    Returns
    -------
    pathlib.Path
        Directory containing the generated SHAP artifacts.

    Raises
    ------
    Exception
        If SHAP initialization, computation, or artifact export fails.
    """
    logger = get_logger()
    try:
        exporter = SHAPArtifactExporter(
            batch_id=batch_id,
            df=df,
            booster=booster,
            output_dir=shap_report_dir,
        )
        return exporter.export_artifacts()
    except Exception:
        logger.exception("Failed to complete SHAP artifact generation workflow.")
        raise
