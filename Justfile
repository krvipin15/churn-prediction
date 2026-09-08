# Automatically load .env file if it exists
set dotenv-load := true

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

# Initialize DVC and configure remote storage with optional force flag (-f)
dvc-setup force_flag="":
    uv run dvc init {{force_flag}}
    uv run dvc remote add origin s3://dvc
    uv run dvc remote modify origin endpointurl https://dagshub.com/vipinkr/churn-prediction.s3
    uv run dvc remote modify origin --local access_key_id ${DAGSHUB_ACCESS_ID}
    uv run dvc remote modify origin --local secret_access_key ${DAGSHUB_ACCESS_ID}
    @echo "success: DVC initialized and remote storage configured."

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
docs-serve mkdocs_port="5050":
    uv run mkdocs serve -a 127.0.0.1:{{mkdocs_port}} --strict

# Publish the documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy -csm "Deploy the latest documentation" -b gh-pages --shell --force

## --- Application Execution ---

# Run the FastAPI application with uvicorn for development
serve:
    uv run uvicorn churn_prediction.api.app:app --reload

# Build and start the application containers in detached mode
container-up:
    podman-compose up --build -d --remove-orphans --force-recreate

# Stop and remove the application containers
container-down:
    podman-compose down

## --- Cleanup ---

# Clean everything; cache, generated files and logs
clean-all: clean-cache clean-generated clean-logs

# Clean temporary Python, pytest, and build cache directories
clean-cache:
    find . -type d -regex ".*\(__pycache__\|\.pytest_cache\|\.ruff_cache\|\.mypy_cache\|\.pyright_cache\)" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf .coverage htmlcov/ site/ dist/ build/ *.egg-info
    rm -f coverage.xml
    @echo "success: Cleaned all the cache"

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
