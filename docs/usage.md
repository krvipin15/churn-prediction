# Usage

## Run the training pipeline

Run all stages end-to-end via DVC (respects the dependency graph and caches unchanged stages):

```bash
dvc repro
```

Or run an individual stage directly:

```bash
python -m churn_prediction.pipelines.training_pipeline <stage>
```

Available stages: `ingest-data`, `validate-raw`, `preprocess-raw`, `validate-train`, `validate-val`, `train-model`, `all`.

## Serve the prediction API

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

See the [API Reference](reference/api.md) for full route and dependency documentation.

## Launch the dashboard

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

## Testing & Quality

```bash
just test          # Run the pytest suite with coverage reporting

# Or use command without just
uv run pytest -v
```

Linting is handled by `ruff`, type checking by `ty`, and Containerfiles are validated with `hadolint`. A pytest suite with an 85% coverage gate is configured in `pyproject.toml`.

## Documentation

Full project documentation — getting started, usage, deployment, and an API reference auto-generated from docstrings — is published at **[krvipin15.github.io/churn-prediction](https://krvipin15.github.io/churn-prediction/)**.

To work on the docs locally:

```bash
just docs-serve     # Serve locally with live reload at http://127.0.0.1:5050
just docs-deploy    # Publish the documentation to GitHub Pages

# Without just command
uv run mkdocs serve -a 127.0.0.1:5050 --strict
uv run mkdocs gh-deploy
```

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
