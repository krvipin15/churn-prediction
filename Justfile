# Automatically load .env file if it exists
set dotenv-load := true

# Default configuration variables
PYPROJECT_VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "([^"]+)"/\1/'`
IMAGE_VERSION := env_var_or_default("IMAGE_VERSION", "v" + PYPROJECT_VERSION)
MKDOCS_PORT := env_var_or_default("MKDOCS_PORT", "5050")
GHCR_USER := env_var_or_default("GHCR_USER", "")
POD_NAME := "churn-pod"

# Display available commands with descriptions
default:
    @just --list --unsorted

## --- Setup & Configuration ---

# Install dependencies with uv and setup pre-commit hooks
env-setup:
    uv self update
    uv sync --all-groups --all-extras
    uvx detect-secrets scan > .secrets.baseline
    uv run pre-commit install --install-hooks -t pre-commit -t commit-msg -t pre-push
    @echo "success: Environment setup complete. Pre-commit hooks installed."

## --- Development & Maintenance ---

# Upgrade project lockfile dependencies
lock:
    uv lock --upgrade

# Update git pre-commit hook versions to latest
hooks-update:
    uv run pre-commit autoupdate
    sed -i 's|rev: v1$|rev: v1.50.1|' .pre-commit-config.yaml
    @echo "warning: Check https://github.com/crate-ci/typos/releases for the latest typos version — v1.50.1 was hardcoded above and may be stale"

# Run pre-commit checks across all staged/unstaged files
hooks-run:
    uv run pre-commit run --all-files

## --- Testing & Quality ---

# Run pytest test suite with coverage report
test:
    uv run pytest tests/ -v

## --- Documentation ---

# Serve the documentation site locally with live reload
docs-serve:
    uv run mkdocs serve -a 127.0.0.1:{{MKDOCS_PORT}} --strict


# Publish the documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy -csm "Deploy the latest documentation" -b gh-pages --shell --force

## --- Application Execution ---

# Run the FastAPI application with uvicorn for development
serve:
    uv run uvicorn churn_prediction.api.app:app --host ${FASTAPI_HOST} --port ${FASTAPI_PORT} --reload

# Build and start the application containers in detached mode
container-up:
    podman-compose up --build -d --remove-orphans --force-recreate

# Stop and remove the application containers
container-down:
    podman-compose down

## --- Podman & GHCR Registry ---

# Log in to GitHub Container Registry using GHCR_PAT environment variable
ghcr-login user=GHCR_USER:
    @if [ -z "${GHCR_PAT}" ]; then echo "Error: GHCR_PAT environment variable is missing."; exit 1; fi
    echo "${GHCR_PAT}" | podman login ghcr.io -u {{user}} --password-stdin
    @echo "success: Logged into GHCR as {{user}}"

# Build both API and Dashboard images with version and latest tags
ghcr-build owner=GHCR_USER tag=IMAGE_VERSION:
    podman build -f Containerfile.api \
        -t ghcr.io/{{owner}}/churn-prediction-api:{{tag}} \
        -t ghcr.io/{{owner}}/churn-prediction-api:latest .
    podman build -f Containerfile.dashboard \
        -t ghcr.io/{{owner}}/churn-prediction-dashboard:{{tag}} \
        -t ghcr.io/{{owner}}/churn-prediction-dashboard:latest .
    @echo "success: Built images for API and Dashboard (tag: {{tag}})"

# Push API and Dashboard images to GitHub Container Registry
ghcr-push owner=GHCR_USER tag=IMAGE_VERSION:
    podman push ghcr.io/{{owner}}/churn-prediction-api:{{tag}}
    podman push ghcr.io/{{owner}}/churn-prediction-api:latest
    podman push ghcr.io/{{owner}}/churn-prediction-dashboard:{{tag}}
    podman push ghcr.io/{{owner}}/churn-prediction-dashboard:latest
    @echo "success: Pushed images to ghcr.io/{{owner}}"

## --- Podman Pod Operations ---

# Create an isolated Pod sharing port namespace (8000 & 8501) and keep-id user namespace
pod-create pod=POD_NAME:
    podman pod create --name {{pod}} --userns=keep-id -p 8000:8000 -p 8501:8501
    @echo "success: Pod {{pod}} created."

# Run FastAPI API service inside the pod
pod-run-api pod=POD_NAME owner=GHCR_USER tag="latest":
    podman run -d \
        --pod {{pod}} \
        --name churn-api \
        --restart unless-stopped \
        --env-file .env \
        -e FASTAPI_HOST=0.0.0.0 \
        -e FASTAPI_PORT=8000 \
        -v ./data:/app/data:z \
        -v ./models:/app/models:ro,z \
        -v ./reports:/app/reports:z \
        -v ./logs:/app/logs:z \
        ghcr.io/{{owner}}/churn-prediction-api:{{tag}}

# Run Streamlit Dashboard inside the pod (talks to API via localhost)
pod-run-dashboard pod=POD_NAME owner=GHCR_USER tag="latest":
    podman run -d \
        --pod {{pod}} \
        --name churn-dashboard \
        --restart unless-stopped \
        --env-file .env \
        -e FASTAPI_HOST=127.0.0.1 \
        -e FASTAPI_PORT=8000 \
        -e API_BASE_URL=http://127.0.0.1:8000 \
        -v ./data:/app/data:ro,z \
        -v ./models:/app/models:ro,z \
        -v ./reports:/app/reports:ro,z \
        ghcr.io/{{owner}}/churn-prediction-dashboard:{{tag}}

# Spin up the entire pod architecture
pod-up pod=POD_NAME owner=GHCR_USER tag="latest": (pod-create pod) (pod-run-api pod owner tag) (pod-run-dashboard pod owner tag)
    @echo "success: Pod {{pod}} is running with API and Dashboard services."

# Stop the pod containers
pod-down pod=POD_NAME:
    podman pod stop {{pod}}
    @echo "success: Pod {{pod}} stopped."

## --- Cleanup ---

# Clean everything; cache, generated files and logs
clean-all: clean-cache clean-generated clean-logs clean-docs

# Clean temporary Python, pytest, and build cache directories
clean-cache:
    find . -type d -regex ".*\(__pycache__\|\.pytest_cache\|\.ruff_cache\|\.mypy_cache\|\.pyright_cache\)" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf .coverage htmlcov/ site/ dist/ build/ *.egg-info
    rm -f coverage.xml
    @echo "success: Cleaned all the cache"

# Clean the generated documentation site
clean-docs:
    rm -rf site/

# Clean the generated log files
clean-logs:
    sudo rm -f logs/*.log
    @echo "success: Cleaned log files in the logs directory."

# Clean the generated files such as models, data, and reports
clean-generated:
    sudo rm -f models/*.{joblib,ubj,json}
    sudo rm -f data/raw/*.{parquet,csv} data/processed/*.parquet data/predictions/*.csv
    sudo rm -f reports/{validation,training}/*.json
    sudo rm -rf reports/shap/*/
    @echo "success: Cleaned generated files in models, data, and reports directories."

# Clean podman containers, images, and volumes associated with the project
clean-container:
    podman-compose down --volumes --rmi all
    podman system prune -f
    @echo "success: Containers, volumes, and images cleaned."

# Clean the podman pod
clean-pod pod=POD_NAME:
    podman pod rm -f {{pod}}
    @echo "success: Pod {{pod}} removed."
