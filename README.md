# Churn Prediction

[![CI](https://github.com/krvipin15/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/krvipin15/churn-prediction/actions/workflows/ci.yml)
[![Security](https://github.com/krvipin15/churn-prediction/actions/workflows/security.yml/badge.svg)](https://github.com/krvipin15/churn-prediction/actions/workflows/security.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://krvipin15.github.io/churn-prediction/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/github/license/krvipin15/churn-prediction)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Production-grade MLOps pipeline, serving infrastructure, and containerized deployment for predicting customer churn. The project covers the full end-to-end lifecycle — automated data ingestion, Pandera schema validation, feature engineering, XGBoost model training, SHAP explainability, FastAPI serving, interactive Streamlit UI, and multi-container orchestration with Podman.

**Full documentation:** [krvipin15.github.io/churn-prediction](https://krvipin15.github.io/churn-prediction/)

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

## Project Structure

```
└── 📁churn-prediction
    └── 📁.dvc
    └── 📁.github
        └── 📁workflows
            ├── ci.yml
            └── security.yml
    └── 📁.streamlit
        └── config.toml
    └── 📁data
        └── 📁predictions
        └── 📁processed
        └── 📁raw
    └── 📁docs
    └── 📁logs
    └── 📁models
    └── 📁reports
        └── 📁shap
        └── 📁training
        └── 📁validation
    └── 📁src
        └── 📁churn_prediction
            └── 📁api
                └── 📁routes
                    ├── __init__.py
                    ├── explain.py
                    ├── health.py
                    └── predict.py
                ├── __init__.py
                ├── app.py
                └── dependencies.py
            └── 📁client
                ├── __init__.py
                ├── api_client.py
                ├── dashboard.py
                └── main.py
            └── 📁config
                ├── __init__.py
                ├── logger.py
                └── settings.py
            └── 📁data
                ├── __init__.py
                ├── ingestion.py
                ├── schemas.py
                └── validation.py
            └── 📁features
                ├── __init__.py
                ├── inference_preprocessing.py
                └── train_preprocessing.py
            └── 📁model
                ├── __init__.py
                ├── explainability.py
                ├── inference.py
                └── training.py
            └── 📁pipelines
                ├── __init__.py
                ├── prediction_pipeline.py
                └── training_pipeline.py
            ├── __init__.py
            └── cli.py
    └── 📁tests
        └── 📁e2e
            └── test_cli.py
        └── 📁integration
            ├── test_api_client.py
            ├── test_app.py
            ├── test_dashboard.py
            ├── test_dependencies.py
            ├── test_explain_route.py
            ├── test_health_route.py
            ├── test_main.py
            ├── test_predict_route.py
            ├── test_prediction_pipeline.py
            └── test_training_pipeline.py
        └── 📁unit
            ├── test_explainability.py
            ├── test_inference_preprocessing.py
            ├── test_inference.py
            ├── test_ingestion.py
            ├── test_train_preprocessing.py
            ├── test_training.py
            └── test_validation.py
    ├── .containerignore
    ├── .dvcignore
    ├── .env
    ├── .env.example
    ├── .gitignore
    ├── .pre-commit-config.yaml
    ├── .python-version
    ├── .secrets.baseline
    ├── compose.yaml
    ├── Containerfile.api
    ├── Containerfile.dashboard
    ├── dvc.lock
    ├── dvc.yaml
    ├── Justfile
    ├── LICENSE
    ├── mkdocs.yml
    ├── params.yaml
    ├── pyproject.toml
    ├── README.md
    ├── SECURITY.md
    └── uv.lock
```
## Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/) package manager
- [Just](https://github.com/casey/just) command runner
- [Podman](https://podman.io/) (for containerization)
- [Podman Compose](https://github.com/containers/podman-compose) (for multi-container orchestration)
- [Hadolint](https://github.com/hadolint/hadolint) (for validating Containerfiles)
- A [Kaggle](https://www.kaggle.com/settings) account and API token (for dataset ingestion)
- A [DagsHub](https://dagshub.com/) repository (for DVC remote storage), optional
- A [Sentry](https://sentry.io/) DSN (for error monitoring in staging/production), optional

## Installation

```bash
# Clone the repository
git clone https://github.com/krvipin15/churn-prediction.git
cd churn-prediction

# Install dependencies and set up pre-commit hooks
just env-setup
```

Alternatively, without `just`:

```bash
uv self update
uv sync --all-groups --all-extras
uvx detect-secrets scan > .secrets.baseline
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg -t pre-push
uv run pre-commit autoupdate
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `ENVIRONMENT` | `development`, `staging`, `production`, or `test` |
| `LOGGER_NAME` | Name used for the application logger |
| `SENTRY_DSN` | Sentry DSN (required in staging/production) |
| `GHCR_USER` | GitHub username used to tag and log in to GHCR |
| `GHCR_PAT` | GitHub personal access token (`write:packages` scope) used to log in to GHCR |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API credentials (required in staging/production) |
| `FASTAPI_HOST` / `FASTAPI_PORT` | Host and port for the FastAPI server |

Model, training, and schema hyperparameters live in [`params.yaml`](params.yaml) rather than environment variables.

To configure the DVC locally:

```bash
dvc init
```

## Usage Guide

### Run the training pipeline

Run all stages end-to-end via DVC (respects the dependency graph and caches unchanged stages):

```bash
dvc repro
```

Or run an individual stage directly:

```bash
python -m churn_prediction.pipelines.training_pipeline <stage>
```

Available stages: `ingest-data`, `validate-raw`, `preprocess-raw`, `validate-train`, `validate-val`, `train-model`, `all`.

### Serve the prediction API

```bash
just serve
```

Server will run at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Reports encoder/model load status and memory usage |
| `POST` | `/api/v1/predict/batch` | Upload a CSV file to generate batch predictions |
| `GET` | `/api/v1/predictions/download/{batch_id}` | Download predictions for a batch |
| `POST` | `/explain/{batch_id}` | Generate SHAP explainability artifacts for a batch |

### Launch the dashboard

```bash
churn-prediction
```

This starts the Streamlit application (talks to the FastAPI service configured via `API_BASE_URL`) for uploading data, reviewing predictions, and exploring SHAP-based risk drivers.

### Maintenance & Dependency Management

```bash
just lock           # Upgrade all project lockfile dependencies via uv
just hooks-update   # Autoupdate pre-commit hook repositories
just hooks-run      # Run pre-commit checks across all staged/unstaged files
```

### Running with Podman Compose

```bash
just container-up     # Build and start application containers in detached mode
just container-down   # Stop and remove application containers

# Without just command
podman-compose up --build -d --remove-orphans --force-recreate
podman-compose down
```

Managing the stack manually:

```bash
podman-compose ps            # View running container status
podman-compose logs -f       # Stream container logs
```

### Building & Publishing Container Images

```bash
just ghcr-login              # Log in to GHCR using GHCR_USER / GHCR_PAT
just ghcr-build              # Build API & Dashboard images with version and latest tags
just ghcr-push               # Push version and latest images to GHCR

# Without just command
echo "${GHCR_PAT}" | podman login ghcr.io -u $GHCR_USER --password-stdin

podman build -f Containerfile.api \
  -t ghcr.io/${GHCR_USER}/churn-prediction-api:v0.1.0 \
  -t ghcr.io/${GHCR_USER}/churn-prediction-api:latest .
podman build -f Containerfile.dashboard \
  -t ghcr.io/${GHCR_USER}/churn-prediction-dashboard:v0.1.0 \
  -t ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest .

podman push ghcr.io/${GHCR_USER}/churn-prediction-api:v0.1.0
podman push ghcr.io/${GHCR_USER}/churn-prediction-api:latest
podman push ghcr.io/${GHCR_USER}/churn-prediction-dashboard:v0.1.0
podman push ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest
```

### Isolated Pod Deployment (Podman Pods)

```bash
just pod-up                  # Create pod, launch API and Dashboard containers
just pod-down                # Stop running pod containers
just clean-pod               # Force-remove the pod

# Target specific steps or custom pod names
just pod-create pod=churn-pod
just pod-run-api pod=churn-pod owner=$GHCR_USER tag=latest
just pod-run-dashboard pod=churn-pod owner=$GHCR_USER tag=latest

# Without just command
podman pod create --name churn-pod --userns=keep-id -p 8000:8000 -p 8501:8501

podman run -d \
  --pod churn-pod \
  --name churn-api \
  --restart unless-stopped \
  --env-file .env \
  -e FASTAPI_HOST=0.0.0.0 \
  -e FASTAPI_PORT=8000 \
  -v ./data:/app/data:z \
  -v ./models:/app/models:ro,z \
  -v ./reports:/app/reports:z \
  -v ./logs:/app/logs:z \
  ghcr.io/${GHCR_USER}/churn-prediction-api:latest

podman run -d \
  --pod churn-pod \
  --name churn-dashboard \
  --restart unless-stopped \
  --env-file .env \
  -e FASTAPI_HOST=127.0.0.1 \
  -e FASTAPI_PORT=8000 \
  -e API_BASE_URL=http://127.0.0.1:8000 \
  -v ./data:/app/data:ro,z \
  -v ./models:/app/models:ro,z \
  -v ./reports:/app/reports:ro,z \
  ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest
```

## Documentation

Full project documentation — getting started, usage, deployment, and an API reference auto-generated from docstrings — is published at **[krvipin15.github.io/churn-prediction](https://krvipin15.github.io/churn-prediction/)**.

To work on the docs locally:

```bash
just docs-serve    # Serve locally with live reload at http://127.0.0.1:5050
just docs-deploy   # Publish the documentation to GitHub Pages

# Without just command
uv run mkdocs serve -a 127.0.0.1:5050 --strict
uv run mkdocs gh-deploy
```

## Testing & Quality Assurance

```bash
just test          # Run the test suite with coverage reporting

# Or use command without just
uv run pytest -v
```

Linting and formatting are managed by `ruff`, type checking by `ty`, secret scanning via `detect-secrets`, and Containerfile validation via `hadolint`. A pytest suite with high coverage checks is configured in `pyproject.toml`.

## Cleanup

```bash
just clean-cache      # Remove Python, pytest, ruff, mypy, pyright, and build caches
just clean-docs       # Remove generated documentation site directory
just clean-logs       # Remove log files in logs directory
just clean-generated  # Remove generated models, datasets, and SHAP/training reports
just clean-all        # Execute clean-cache, clean-generated, clean-logs, and clean-docs

just clean-container  # Prune Podman containers, volumes, and images
just clean-pod        # Force-remove the active Podman pod
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Author

**Vipin Kumar** — [krvipin15@tutamail.com](mailto:krvipin15@tutamail.com)

- Source: https://github.com/krvipin15/churn-prediction
- Documentation: https://krvipin15.github.io/churn-prediction/
- Issues: https://github.com/krvipin15/churn-prediction/issues
