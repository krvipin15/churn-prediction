# Churn Prediction

Production-grade MLOps pipeline and serving infrastructure for predicting customer churn. The project covers the full lifecycle — automated data ingestion, Pandera schema validation, feature engineering, XGBoost model training, SHAP explainability, FastAPI serving, interactive Streamlit UI, and containerized deployment.

## Features

- **Reproducible pipeline** — DVC-orchestrated stages (ingest → validate → preprocess → validate → train) with versioned data and model artifacts pushed to DagsHub.
- **Schema-validated data** — Pandera schemas enforce structure and value constraints on both raw and processed datasets, with JSON diagnostic reports on failure.
- **XGBoost training** — Class-imbalance-aware training (`scale_pos_weight`) with probability calibration, F1-optimal threshold selection, and an auto-generated model card.
- **Explainability** — SHAP-based per-customer risk drivers and global feature importance, exported as dashboard-ready Parquet/JSON artifacts.
- **Batch inference API** — FastAPI service for CSV upload, prediction, download, and SHAP explanation endpoints.
- **Interactive dashboard** — Streamlit app for uploading data, viewing KPIs, risk distributions, and per-customer explanations.
- **Containerized deployment** — Multi-stage, non-root Podman builds with required model/data artifacts baked in at build time; images built, security-scanned, and published to GHCR automatically via CI/CD.
- **Structured logging & monitoring** — `structlog`-based logging with rotating file handlers and optional Sentry integration.

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | XGBoost, scikit-learn, SHAP |
| Data validation | Pandera |
| Configuration | Pydantic Settings |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Pipeline & versioning | DVC, DagsHub |
| Logging & monitoring | structlog, Sentry |
| Package management | uv, Just |
| Containerization & deployment | Podman, Podman Compose, GHCR, Render |
| Quality | Ruff, Ty, Pytest, Pre-commit, Hadolint, Trivy |

## Where to go next

- [Getting Started](getting-started.md) — install dependencies and configure the environment.
- [Usage](usage.md) — run the training pipeline, serve the API, launch the dashboard.
- [Deployment](deployment.md) — run with Podman, automated CI/CD image publishing, and deploying to Render.
- **API Reference** — auto-generated documentation for every module, grouped by layer (config, data, features, model, pipelines, api, client).
