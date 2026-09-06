# Churn Prediction

Production-grade MLOps pipeline, serving infrastructure, and containerized deployment for predicting customer churn. The project covers the full end-to-end lifecycle — automated data ingestion, Pandera schema validation, feature engineering, XGBoost model training, SHAP explainability, FastAPI serving, interactive Streamlit UI, and multi-container orchestration with Podman.

## Features

- **Reproducible Pipeline**: DVC-orchestrated stages (ingest → validate → preprocess → validate → train).
- **Schema-Validated Data**: Pandera schemas enforce data integrity, structural constraints, and types on raw and processed datasets with automated diagnostic reporting.
- **XGBoost & Explainability**: Imbalance-aware training with probability calibration, optimal F1 threshold selection, auto-generated model cards, and SHAP-based feature attribution.
- **Inference API**: High-performance FastAPI application supporting batch CSV prediction, status checks, and downloadable SHAP explainability reports.
- **Interactive UI**: Reactive Streamlit dashboard powered by Plotly for exploring customer churn risks, key KPIs, and localized risk drivers.
- **Containerized Orchestration**: Podman and Podman-Compose architecture supporting multi-stage container builds, GHCR publishing, and isolated Pod networking.
- **Documentation**: MkDocs Material site with API reference auto-generated from docstrings, published to GitHub Pages.

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | XGBoost, Scikit-Learn, SHAP |
| Data Validation | Pandera |
| Configuration | Pydantic Settings |
| API & Backend | FastAPI, Uvicorn, Structlog, Sentry |
| Dashboard | Streamlit, Plotly |
| Pipeline & Data Versioning | DVC, KaggleHub |
| Containerization | Podman, Podman-Compose, GHCR |
| Quality | Ruff, Ty, Pre-commit, Hadolint Just |
| Docs & Package management | MkDocs Material, UV |

## Where to go next

- [Getting Started](getting-started.md) — install dependencies and configure the environment.
- [Usage](usage.md) — run the training pipeline, serve the API, launch the dashboard.
- [Deployment](deployment.md) — build and run with Podman, publish images to GHCR.
- **API Reference** — auto-generated documentation for every module, grouped by layer (config, data, features, model, pipelines, api, client).
