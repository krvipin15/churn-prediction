"""Churn Prediction Package.

A machine learning application for predicting customer churn using XGBoost,
featuring automated training, inference, and model explainability.

Modules
--------
api
    Provides the REST API layer for exposing prediction and explainability
    endpoints, including route definitions and dependency injection.
client
    Contains tools for interacting with the API, including a dedicated
    API client and a dashboard for visualizing results.
config
    Handles application-wide configurations, environment settings, and
    logging initialization.
data
    Manages the data lifecycle, including ingestion, schema definition,
    and data validation.
features
    Contains preprocessing logic for transforming raw data into model-ready
    features for both training and inference phases.
model
    Core machine learning logic including functions for training the XGBoost
    model, making predictions, and generating SHAP artifacts for explainability.
pipelines
    Orchestrates the end-to-end workflow by combining data, features, and
    model modules into cohesive training and prediction pipelines.
"""


def main() -> None:
    """Entry point for the churn prediction package."""
    pass
