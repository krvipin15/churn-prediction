"""
Churn Prediction Interpretation Module.

This module provides utilities for interpreting machine learning model outputs
related to customer churn. It includes functionality to load model artifacts,
decode ordinal features back to their original categories, map feature
importance to human-readable recommendations, and extract SHAP-based risk
breakdowns for individual customers.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.config.settings import get_settings

# Feature-specific recommendations for churn risk factors.
RECOMMENDATIONS = {
    "tenure": (
        "High early-tenure risk detected. Initiate a proactive onboarding review "
        "and offer a 60-day milestone loyalty check-in."
    ),
    "monthly_charges": (
        "Price sensitivity identified. Perform a plan right-sizing review and "
        "offer a targeted loyalty tier discount."
    ),
    "total_charges": (
        "High cumulative spend vs. perceived value gap. Schedule an executive account "
        "review to demonstrate ROI and discuss long-term value alignment."
    ),
    "contract": (
        "High-risk short-term/month-to-month contract. Offer a discounted annual or "
        "multi-year contract upgrade with a locked-in rate."
    ),
    "payment_method": (
        "Manual payment friction detected. Incentive auto-pay enrollment (e.g., ACH/Credit Card) "
        "with a one-time billing credit."
    ),
    "paperless_billing": (
        "Paper billing friction or low invoice visibility. Transition customer to paperless "
        "billing with automated digital payment reminders."
    ),
    "internet_service": (
        "Service tier mismatch or connectivity friction. Audit local network quality, "
        "test bandwidth metrics, and evaluate a fiber/high-speed upgrade path."
    ),
    "multiple_lines": (
        "Single-line account with expansion potential. Pitch multi-line family or business "
        "bundle savings."
    ),
    "tech_support": (
        "Unresolved technical friction or lack of dedicated assistance. Route to a Senior "
        "Technical Account Manager for issue resolution and priority queue access."
    ),
    "online_security": (
        "Missing essential security feature. Offer a 90-day complimentary trial of the "
        "Online Security package to increase account stickiness."
    ),
    "online_backup": (
        "Unprotected customer data footprint. Highlight disaster recovery risks and propose a "
        "discounted Cloud Backup add-on."
    ),
    "device_protection": (
        "Unprotected hardware footprint. Offer a bundled Device Protection and hardware "
        "refresh plan."
    ),
    "streaming_tv": (
        "Low content engagement. Provide promotional access to premium entertainment bundles "
        "to drive daily active usage."
    ),
    "streaming_movies": (
        "Unutilized media features. Target customer with personalized content recommendations "
        "or a media add-on discount."
    ),
    "partner": (
        "Unattached single-user profile. Offer household sharing incentives or multi-account "
        "linking benefits."
    ),
    "dependents": (
        "Family household indicators present. Propose parental controls and multi-user bundle "
        "discounts."
    ),
    "senior_citizen": (
        "Specialized customer segment. Provide dedicated human-assisted support access and "
        "simplified billing formats."
    ),
}


def recommend_for_feature(feature_name: str) -> str:
    """Return a retention recommendation for a churn-risk feature.

    Normalizes the supplied feature name and maps it to the corresponding
    business recommendation. Unknown features receive a generic customer
    health recommendation.

    Parameters
    ----------
    feature_name : str
        Feature name associated with the identified churn risk factor.

    Returns
    -------
    str
        Business recommendation corresponding to the feature, or a generic
        account-health recommendation when no feature-specific recommendation
        exists.
    """
    key = feature_name.strip().lower().replace("_", "").replace(" ", "")
    normalized_map = {k.replace("_", ""): k for k in RECOMMENDATIONS}
    matched_key = normalized_map.get(key)
    if matched_key:
        return RECOMMENDATIONS[matched_key]
    return "Review overall account engagement and conduct a proactive customer health check-in."


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Transform model margins into probabilities.

    Applies the logistic sigmoid function element-wise to convert model
    decision margins into values in the ``[0, 1]`` probability range.

    Parameters
    ----------
    x : numpy.ndarray
        Array of model decision margins.

    Returns
    -------
    numpy.ndarray
        Element-wise sigmoid-transformed probabilities.
    """
    return 1.0 / (1.0 + np.exp(-x))


@lru_cache(maxsize=1)
def _load_encoder() -> tuple[OrdinalEncoder, list[str]]:
    """Load and cache the fitted ordinal encoder.

    Loads the preprocessing artifact associated with the configured model
    version and retrieves the ordinal feature names from the application
    settings.

    Returns
    -------
    encoder : sklearn.preprocessing.OrdinalEncoder
        Fitted ordinal encoder used during model preprocessing.
    ordinal_columns : list of str
        Feature names transformed by the ordinal encoder.

    Raises
    ------
    FileNotFoundError
        If the configured preprocessing artifact does not exist.
    """
    settings = get_settings()

    preprocessor_path = Path(settings.MODEL_DIR) / "preprocessor.joblib"
    encoder: OrdinalEncoder = joblib.load(preprocessor_path)
    ordinal_columns = settings.PARAMS.schema_config.ordinal_columns

    return encoder, ordinal_columns


def decode_features(feature: str, value: float) -> str:
    """Decode an ordinal feature value into its original category.

    Uses the fitted training encoder to translate a numeric ordinal value
    back into its original categorical representation. Features that are not
    ordinally encoded are returned unchanged.

    Parameters
    ----------
    feature : str
        Name of the feature containing the encoded value.
    value : float
        Numeric ordinal value produced by the preprocessing pipeline.

    Returns
    -------
    str
        Original categorical value when the feature is ordinal and the value
        is valid. Returns the string representation of ``value`` for
        non-ordinal features and ``"Unknown"`` for invalid ordinal values.
    """
    encoder, ordinal_columns = _load_encoder()
    if feature not in ordinal_columns:
        return str(value)

    categories = encoder.categories_[ordinal_columns.index(feature)]
    if 0 <= value < len(categories):
        return str(categories[int(value)])
    return "Unknown"


def load_artifacts(artifacts_dir: str | Path) -> dict[str, Any]:
    """Load dashboard artifacts generated by the explainability pipeline.

    Reads global feature-importance metadata and customer-level SHAP risk
    profiles from the supplied artifact directory and converts them into
    dashboard-ready Python and pandas objects.

    Parameters
    ----------
    artifacts_dir : str or pathlib.Path
        Directory containing ``metadata.json`` and
        ``customer_risk_profiles.parquet``.

    Returns
    -------
    dict
        Dictionary containing:

        - ``metadata`` : dict
            Global explainability metadata.
        - ``feature_importance`` : pandas.DataFrame
            Ranked global feature importance values.
        - ``risk_profiles`` : pandas.DataFrame
            Customer-level churn probabilities and SHAP drivers.
        - ``id_col`` : str
            Name of the customer identifier column.
        - ``prob_col`` : str
            Name of the calculated churn probability column.

    Raises
    ------
    FileNotFoundError
        If the artifact directory or required artifact files do not exist.
    NotADirectoryError
        If ``artifacts_dir`` is not a directory.
    """
    base = Path(artifacts_dir)
    if not base.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Artifacts directory is not a directory: {base}")

    # Load metadata from the JSON file
    metadata: dict[str, Any] = json.loads((base / "metadata.json").read_text())
    feature_importance: pd.DataFrame = pd.DataFrame(metadata["global_feature_importance"]).head(10)[
        ["feature", "mean_abs_shap"]
    ]

    # Load risk profiles from the Parquet file
    risk_profiles = pd.read_parquet(base / "customer_risk_profiles.parquet")
    risk_profiles.index.name = "customer_id"
    risk_profiles = risk_profiles.reset_index()
    risk_profiles["churn_probability"] = _sigmoid(risk_profiles["predicted_margin"].to_numpy())
    risk_profiles["top_churn_drivers"] = risk_profiles["top_churn_drivers"].apply(json.loads)
    risk_profiles["top_retention_factors"] = risk_profiles["top_retention_factors"].apply(
        json.loads
    )

    return {
        "metadata": metadata,
        "feature_importance": feature_importance,
        "risk_profiles": risk_profiles,
        "id_col": "customer_id",
        "prob_col": "churn_probability",
    }


def customer_shap_breakdown(
    risk_profiles: pd.DataFrame,
    customer_id: int,
) -> pd.DataFrame:
    """Extract the SHAP contribution breakdown for one customer.

    Combines the customer's strongest churn drivers and retention factors,
    decodes categorical feature values, and returns the contributions sorted
    by SHAP value.

    Parameters
    ----------
    risk_profiles : pandas.DataFrame
        Customer risk profile table produced by the explainability pipeline.
    customer_id : int
        Customer identifier for which the SHAP breakdown should be retrieved.

    Returns
    -------
    pandas.DataFrame
        SHAP contribution table containing ``feature``, ``shap_value``,
        ``feature_value``, and ``display_value`` columns. An empty DataFrame
        with the expected columns is returned when the customer does not
        exist or has no recorded drivers.
    """
    row: pd.DataFrame = risk_profiles[risk_profiles["customer_id"] == customer_id]
    if row.empty:
        return pd.DataFrame(
            {
                "feature": [],
                "shap_value": [],
                "feature_value": [],
                "display_value": [],
            }
        )

    drivers: list[dict[str, Any]] = (
        row.iloc[0]["top_churn_drivers"] + row.iloc[0]["top_retention_factors"]
    )
    if not drivers:
        return pd.DataFrame(
            {
                "feature": [],
                "shap_value": [],
                "feature_value": [],
                "display_value": [],
            }
        )

    breakdown = pd.DataFrame(drivers)
    breakdown["display_value"] = breakdown.apply(
        lambda r: decode_features(r["feature"], r["feature_value"]), axis=1
    )
    return breakdown.sort_values("shap_value").reset_index(drop=True)
