"""Data Management Package.

This package provides a unified interface for the data lifecycle of the
churn prediction pipeline, including dataset acquisition from Kaggle,
and structural validation using Pandera schemas.

Modules
-------
ingestion
    Handles the download and local preparation of raw competition datasets.
schemas
    Defines the Pandera DataFrame schemas for raw and processed data stages.
validation
    Implements the validation pipeline and diagnostic report generation.
"""
