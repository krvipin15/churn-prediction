"""
FastAPI application entry point for the Churn Prediction Batch Inference API.

This module initializes the FastAPI application, configures global middleware,
and manages the application lifecycle—specifically the loading and unloading
of ML artifacts (OrdinalEncoder and XGBoost Booster) into the application state.
"""

import shutil
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import metadata

import joblib
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import OrdinalEncoder

from churn_prediction.api.routes import explain, health, predict
from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings


def _pull_artifacts(artifacts: list[str]) -> None:
    """Pull required artifacts from DagsHub using DVC.

    Parameters
    ----------
    artifacts : list[str]
        List of artifact paths to pull from DagsHub.

    Raises
    ------
    RuntimeError
        If the 'uv' or 'dvc' executables are not found in the system path.
    """
    logger = get_logger()

    # Check if 'uv' is available in the system path
    uv_path = shutil.which("uv")
    if uv_path is None:
        logger.critical("The 'uv' executable was not found in the system path.")
        raise RuntimeError("uv is not installed or not in PATH")

    try:
        result = subprocess.run(  # noqa: S603
            [uv_path, "run", "dvc", "pull", *artifacts],
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
        logger.info("DVC pull successful: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.critical("Failed to pull artifacts from DagsHub. Error: %s", e.stderr)
        raise RuntimeError(
            "Required artifacts are missing and 'dvc pull' failed. "
            "Ensure you have the correct DAGSHUB_ACCESS_ID configured."
        ) from e
    except FileNotFoundError as err:
        logger.critical("The 'uv' or 'dvc' executable was not found in the system path.")
        raise RuntimeError("DVC environment not set up correctly. Cannot pull artifacts.") from err


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of the FastAPI application.

    During startup, loads the fitted preprocessing encoder and trained
    XGBoost model from the configured artifact directory and stores them in
    ``app.state`` for dependency injection.

    During shutdown, clears the references from application state so that
    model artifacts can be released from memory.

    Parameters
    ----------
    app : fastapi.FastAPI
        FastAPI application instance whose lifecycle is being managed.

    Yields
    ------
    None
        Control is yielded to the running FastAPI application after all
        required ML artifacts have been loaded.

    Raises
    ------
    FileNotFoundError
        If the required model, preprocessing artifact, or demonstration
        dataset cannot be found.
    Exception
        If an ML artifact cannot be loaded successfully.
    """
    logger = get_logger()
    settings = get_settings()

    # Resolve paths to required artifacts
    demo_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()
    required_artifacts = [
        demo_dir / "test.csv",
        model_dir / "preprocessor.joblib",
        model_dir / "model.ubj",
        model_dir / "card.json",
    ]
    logger.debug("Resolved required artifact paths: %s", required_artifacts)

    # Check for the existence of required artifacts; if missing, attempt to pull from DagsHub via DVC
    if not all(path.exists() for path in required_artifacts):
        logger.warning("One or more artifacts missing. Attempting to pull from DagsHub via DVC...")
        _pull_artifacts([str(path) for path in required_artifacts])

    # After attempting to pull, verify that all required artifacts are now present
    if not all(path.exists() for path in required_artifacts):
        logger.critical("Artifacts still missing after dvc pull attempt.")
        raise FileNotFoundError("Required ML artifacts could not be retrieved from remote.")

    # Load the pre-fitted encoder and XGBoost model
    try:
        logger.info("Loading preprocessor artifact (joblib)...")
        encoder: OrdinalEncoder = joblib.load(required_artifacts[1])

        logger.info("Loading XGBoost booster model...")
        booster = xgb.Booster()
        booster.load_model(str(required_artifacts[2]))

        app.state.encoder = encoder
        app.state.booster = booster
        logger.info("Successfully loaded all ML artifacts into memory.")
    except Exception as err:
        logger.error("Critical error loading ML artifacts: %s", err)
        raise

    yield

    logger.info("Shutting down application. Clearing app state...")
    app.state.encoder = None
    app.state.booster = None
    logger.debug("Application state cleared.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Initializes the application metadata, lifecycle management, CORS
    configuration, and API routers.

    Returns
    -------
    fastapi.FastAPI
        Fully configured FastAPI application instance ready to serve
        prediction, explainability, download, and health endpoints.
    """
    logger = get_logger()
    logger.info("Creating FastAPI application instance...")

    # Fetch version from pyproject.toml via installed package metadata
    try:
        app_version = metadata.version("churn-prediction")
    except metadata.PackageNotFoundError:
        app_version = "0.1.0-dev"
        logger.warning("Package version not found; defaulting to %s", app_version)

    app = FastAPI(
        title="Churn Prediction Batch Inference API",
        description=(
            "This API provides high-performance batch inference for customer churn modeling. "
            "It utilizes a pre-trained XGBoost model and Scikit-Learn preprocessors to "
            "predict the likelihood of customer attrition based on behavioral data."
        ),
        version=app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS middleware to allow requests from any origin
    logger.debug("Configuring CORSMiddleware with allow_origins=['*']")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers for health checks and prediction endpoints
    logger.info("Registering API routers: health, predict, explain")
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(explain.router)

    logger.info("FastAPI application created successfully.")
    return app


# Initialize the FastAPI application
app = create_app()
