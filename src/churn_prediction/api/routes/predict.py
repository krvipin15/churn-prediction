"""Inference API Module.

This module defines the FastAPI router and endpoints for generating batch
predictions using a pre-trained XGBoost model and a scikit-learn OrdinalEncoder.
It handles CSV file uploads, converts the incoming CSV payload into an optimized
Parquet file on disk asynchronously, triggers the prediction pipeline, and provides
a mechanism to download prediction artifacts.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO
from uuid import uuid4

import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Depends, File, HTTPException, Path as PathParam, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.api.dependencies import get_encoder, get_model
from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings
from churn_prediction.pipelines.prediction_pipeline import run_prediction_pipeline

# Regular expression pattern to validate batch IDs
BATCH_ID_PATTERN = r"^\d{8}T\d{6}_[a-f0-9]{4}$"

# Initialize the API router for inference endpoints
router = APIRouter(prefix="/api/v1", tags=["Inference"])


class PredictionResponse(BaseModel):
    """Response model for the prediction endpoint."""

    batch_id: str = Field(..., description="Unique execution batch identifier")
    filename: str = Field(..., description="Name of uploaded input file")
    status: str = Field(..., description="Processing status of the prediction job")
    download_url: str = Field(..., description="URL endpoint to fetch generated predictions")


def _to_parquet(file_obj: BinaryIO, output_path: Path) -> None:
    """Convert an uploaded CSV stream into a compressed Parquet dataset.

    Reads the uploaded CSV into a pandas DataFrame and persists it as a
    Snappy-compressed Parquet file.

    Parameters
    ----------
    file_obj : BinaryIO
        Binary file-like object containing the uploaded CSV data.
    output_path : pathlib.Path
        Destination path for the generated Parquet file.

    Raises
    ------
    ValueError
        If the CSV is empty, malformed, or cannot be converted into a
        DataFrame.
    OSError
        If the output file cannot be written.
    """
    df = pd.read_csv(file_obj, index_col=0)
    if df.empty:
        raise ValueError("Uploaded CSV payload contains no data records.")
    df.to_parquet(output_path, index=False, compression="snappy")


@router.post(
    "/predict/batch",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Batch Prediction Job",
    description=(
        "Upload a CSV file containing feature records to generate churn predictions. "
        "The system validates the file format, persists it to storage, and executes "
        "the prediction pipeline using a pre-loaded XGBoost model and OrdinalEncoder. "
        "Returns a `batch_id` which can be used to download the results."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Predictions successfully generated"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid file format provided"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Pipeline failed during feature transformation or inference"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error while persisting file"
        },
    },
)
async def predict_batch(
    file: Annotated[UploadFile, File(description="CSV File containing tabular feature records")],
    encoder: Annotated[OrdinalEncoder, Depends(get_encoder)],
    model: Annotated[xgb.Booster, Depends(get_model)],
) -> PredictionResponse:
    """Process an uploaded dataset and initiate batch churn prediction.

    Accepts a customer CSV upload, persists the input data, executes the
    prediction pipeline using the pre-loaded encoder and XGBoost model, and
    returns metadata identifying the generated prediction batch.

    Parameters
    ----------
    file : fastapi.UploadFile
        Uploaded CSV file containing customer records to score.
    encoder : sklearn.preprocessing.OrdinalEncoder
        Pre-fitted encoder used to transform categorical input features.
    model : xgboost.Booster
        Trained XGBoost booster used to generate churn predictions.

    Returns
    -------
    PredictionResponse
        Response containing the batch identifier, original filename,
        processing status, number of records, and prediction download URL.

    Raises
    ------
    fastapi.HTTPException
        If the uploaded file is invalid, cannot be persisted, or the
        prediction pipeline fails.
    ValueError
        If CSV-to-Parquet conversion or dataset preprocessing fails.
    pandas.errors.ParserError
        If the uploaded CSV cannot be parsed.
    """
    logger = get_logger()
    settings = get_settings()

    # Generate a unique batch identifier for this prediction job
    batch_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:4]}"
    original_filename = Path(file.filename).name if file.filename else "upload.csv"
    logger.info(
        "Received batch prediction request. Batch ID: %s, Filename: %s",
        batch_id,
        original_filename,
    )

    # Validate the uploaded file format
    if not original_filename.lower().endswith(".csv"):
        logger.warning("Invalid file upload attempt: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid CSV file (.csv).",
        )

    # Persist the uploaded CSV file to the designated raw data directory
    raw_file_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    clean_stem = Path(original_filename).stem
    raw_file_path = (raw_file_dir / f"{clean_stem}_{batch_id}.parquet").resolve()

    logger.debug("Persisting uploaded file to: %s", raw_file_path)
    try:
        await run_in_threadpool(
            _to_parquet,
            file_obj=file.file,
            output_path=raw_file_path,
        )
        logger.debug("Successfully persisted file to disk.")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as parse_err:
        logger.warning("Corrupted or malformed CSV upload: %s", parse_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV payload: {parse_err}",
        ) from parse_err
    except Exception as err:
        logger.error("Failed to write uploaded file to disk: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded payload: {err}",
        ) from err
    finally:
        await file.close()

    # Run the prediction pipeline in a separate thread to avoid blocking the event loop
    try:
        logger.info("Triggering prediction pipeline for batch %s", batch_id)
        await run_in_threadpool(
            run_prediction_pipeline,
            batch_id=batch_id,
            raw_file_path=raw_file_path,
            loaded_encoder=encoder,
            loaded_booster=model,
        )
        logger.info("Prediction pipeline completed successfully for batch %s", batch_id)
    except Exception as pipeline_err:
        logger.exception("Batch prediction execution failed: %s", pipeline_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Batch execution pipeline failed: {pipeline_err}",
        ) from pipeline_err

    return PredictionResponse(
        batch_id=batch_id,
        filename=original_filename,
        status="SUCCESS",
        download_url=f"/api/v1/predictions/download/{batch_id}",
    )


@router.get(
    "/predictions/download/{batch_id}",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download Prediction Results",
    description=(
        "Fetch the generated predictions CSV file using the filename provided as a query parameter. "
        "Ensures that the requested file exists within the secure predictions directory "
        "and prevents directory traversal attacks."
    ),
    responses={
        status.HTTP_200_OK: {"description": "CSV file successfully retrieved"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid file path or security violation"},
        status.HTTP_404_NOT_FOUND: {"description": "The requested prediction file was not found"},
    },
)
async def download_predictions(
    batch_id: Annotated[
        str,
        PathParam(
            ...,
            description="Unique execution batch identifier",
            pattern=BATCH_ID_PATTERN,
            min_length=1,
            max_length=128,
        ),
    ],
) -> FileResponse:
    """Return the generated prediction CSV for a batch.

    Locates the prediction artifact associated with ``batch_id`` and returns
    it as a downloadable HTTP file response.

    Parameters
    ----------
    batch_id : str
        Unique prediction batch identifier used to locate the output artifact.

    Returns
    -------
    fastapi.responses.FileResponse
        HTTP response containing the generated prediction CSV.

    Raises
    ------
    fastapi.HTTPException
        HTTP 400 if ``batch_id`` violates the expected identifier format or
        represents an unsafe path.
        HTTP 404 if the corresponding prediction artifact does not exist.
    """
    logger = get_logger()
    settings = get_settings()

    logger.debug("Download artifact requested for batch_id: %s", batch_id)

    prediction_dir = Path(settings.PREDICTION_DATA_DIR).expanduser().resolve()
    target_filename = f"prediction_{batch_id}.csv"
    file_path = (prediction_dir / target_filename).resolve()

    if not file_path.is_relative_to(prediction_dir):
        logger.warning("Attempted directory traversal attack: %s", file_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path")

    if not file_path.is_file():
        logger.error("Requested prediction download artifact not found: %s", file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction artifact for batch '{batch_id}' not found.",
        )

    logger.info("Serving prediction file: %s", file_path)
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=target_filename,
    )
