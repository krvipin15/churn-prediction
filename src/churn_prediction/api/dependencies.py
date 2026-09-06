"""
Model and Encoder Dependency Providers.

This module contains utility functions used as FastAPI dependencies to retrieve
the preprocessor (OrdinalEncoder) and the predictive model (XGBoost Booster)
from the application state. It ensures that these artifacts are properly loaded
before being used in request handlers, raising HTTP 503 errors if they are missing.
"""

import xgboost as xgb
from fastapi import HTTPException, Request, status
from sklearn.preprocessing import OrdinalEncoder


def get_encoder(request: Request) -> OrdinalEncoder:
    """Retrieve the fitted encoder from FastAPI application state.

    Parameters
    ----------
    request : fastapi.Request
        Incoming FastAPI request providing access to the application state.

    Returns
    -------
    sklearn.preprocessing.OrdinalEncoder
        Fitted ordinal encoder loaded during application startup.

    Raises
    ------
    fastapi.HTTPException
        HTTP 503 error if the encoder has not been loaded into application
        state.
    """
    encoder: OrdinalEncoder | None = getattr(request.app.state, "encoder", None)
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder artifact is not loaded in memory.",
        )
    return encoder


def get_model(request: Request) -> xgb.Booster:
    """Retrieve the trained XGBoost model from application state.

    Parameters
    ----------
    request : fastapi.Request
        Incoming FastAPI request providing access to the application state.

    Returns
    -------
    xgboost.Booster
        Trained XGBoost booster loaded during application startup.

    Raises
    ------
    fastapi.HTTPException
        HTTP 503 error if the model has not been loaded into application
        state.
    """
    model: xgb.Booster | None = getattr(request.app.state, "booster", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost model artifact is not loaded in memory.",
        )
    return model
