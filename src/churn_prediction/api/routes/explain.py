"""Explainability API Module.

This module provides endpoints to generate model interpretability reports using
SHAP (SHapley Additive exPlanations) values for the churn prediction model.
It integrates with the XGBoost booster and processed datasets to visualize
feature contributions.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, status
from pydantic import BaseModel, Field

from churn_prediction.api.dependencies import get_model
from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings
from churn_prediction.model.explainability import generate_shap_artifacts

# Define a regex pattern for validating batch IDs
BATCH_ID_PATTERN = r"^[a-zA-Z0-9_\-]+$"

# Initialize the API router for explainability endpoints
router = APIRouter(tags=["Explainability"])


class ExplainabilityResponse(BaseModel):
    """Response model for the explainability endpoint."""

    batch_id: str = Field(..., description="Unique execution batch identifier")
    status: str = Field(..., description="Status of the explainability request")
    artifacts_path: str = Field(..., description="Path to the generated SHAP explainability plots")


@router.post(
    "/explain/{batch_id}",
    response_model=ExplainabilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate SHAP Explainability Plots",
    description=(
        "Generates SHAP explainability plots for the churn prediction model using "
        "the processed dataset corresponding to the current session. The plots are "
        "saved to disk, and the endpoint returns the path to the generated plots."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully generated SHAP explainability plots.",
            "model": ExplainabilityResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Processed data file for the current session not found.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "An unexpected error occurred during plot generation.",
        },
    },
)
async def generate_explanation(
    batch_id: Annotated[
        str,
        PathParam(
            ...,
            description="Unique identifier for the target processing batch",
            pattern=BATCH_ID_PATTERN,
            min_length=1,
            max_length=128,
        ),
    ],
    xgb_booster: Annotated[xgb.Booster, Depends(get_model)],
) -> ExplainabilityResponse:
    """Generate SHAP explainability artifacts for a prediction batch.

    Loads the processed input features associated with the specified batch,
    computes SHAP explanations using the supplied XGBoost booster, and
    persists dashboard-ready explainability artifacts.

    Parameters
    ----------
    batch_id : str
        Unique identifier of the prediction batch to explain.
    xgb_booster : xgboost.Booster
        Trained XGBoost model used to calculate SHAP contributions.

    Returns
    -------
    ExplainabilityResponse
        Response containing the batch identifier and location of the
        generated explainability artifacts.

    Raises
    ------
    fastapi.HTTPException
        HTTP 404 if the batch input artifact does not exist.
        HTTP 500 if data loading or SHAP artifact generation fails.
    """
    # Initialize logger and settings
    logger = get_logger()
    settings = get_settings()

    base_processed_dir = Path(settings.PROCESSED_DATA_DIR).expanduser().resolve()
    base_artifacts_dir = Path(settings.SHAP_REPORT_DIR).expanduser().resolve()

    processed_filepath = (base_processed_dir / f"processed_{batch_id}.parquet").resolve()
    logger.debug("Resolved processed_filepath: %s", processed_filepath)

    artifacts_dir = (base_artifacts_dir / batch_id).resolve()
    logger.debug("Resolved artifacts_dir: %s", artifacts_dir)

    try:
        # Load the processed data
        if not processed_filepath.exists():
            logger.error("Processed data file not found: %s", processed_filepath)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processed data for batch {batch_id} not found at {processed_filepath}",
            )

        # Load the processed data into a DataFrame
        logger.debug("Loading processed data from %s", processed_filepath)
        df = pd.read_parquet(processed_filepath)
        logger.info("Loaded dataframe with shape: %s", df.shape)

        # Generate SHAP explainability plots
        logger.info("Generating SHAP plots for batch: %s", batch_id)
        generate_shap_artifacts(batch_id, df, xgb_booster, artifacts_dir)

        logger.info("Successfully generated SHAP plots at: %s", artifacts_dir)

        return ExplainabilityResponse(
            batch_id=batch_id,
            status="success",
            artifacts_path=str(artifacts_dir),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error occurred while generating SHAP plots for batch %s: %s",
            batch_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating SHAP plots for batch {batch_id}: {e!s}",
        ) from e
