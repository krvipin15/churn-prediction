# Getting Started

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [Just](https://github.com/casey/just) command runner
- [Podman](https://podman.io/) (for containerization)
- [Podman Compose](https://github.com/containers/podman-compose) (for multi-container orchestration)
- [Hadolint](https://github.com/hadolint/hadolint) (for validating Containerfiles)
- A [Kaggle](https://www.kaggle.com/settings) account and API token (for dataset ingestion)
- A [DagsHub](https://dagshub.com/) repository (for DVC remote storage)
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
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API credentials (required in staging/production) |
| `DAGSHUB_ACCESS_ID` | DagsHub access token used to authenticate the DVC remote (required in staging/production) |
| `SENTRY_DSN` | Sentry DSN (required in staging/production) |

`FASTAPI_HOST` / `FASTAPI_PORT` can also be overridden but default to `0.0.0.0:8000` and aren't required.

Model, training, and schema hyperparameters live in `params.yaml` rather than environment variables.

To initialize DVC and configure the DagsHub remote:

```bash
just dvc-setup
```
