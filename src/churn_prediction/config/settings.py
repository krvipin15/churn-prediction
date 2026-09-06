"""Application Settings and Configuration Management.

This module defines the type-safe application configuration schema using
Pydantic Settings and Pydantic v2. It manages environment variables, `.env` file
overrides, runtime directories, and hyperparameter configuration (`params.yaml`).
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(current_path: Path) -> Path:
    """Locate the project root by searching for known anchor files.

    Starting from ``current_path``, recursively walks up the directory tree
    until a directory containing the project's anchor files is found.

    Parameters
    ----------
    current_path : pathlib.Path
        Starting path from which the parent directories are searched.

    Returns
    -------
    pathlib.Path
        Path to the detected project root directory.
    """
    for parent in [current_path, *list(current_path.parents)]:
        if (
            (parent / "pyproject.toml").exists()
            or (parent / "params.yaml").exists()
            or (parent / ".git").exists()
        ):
            return parent

    return current_path.resolve().parents[2]


# Get the base directory for the project
BASE_DIR = _find_project_root(Path(__file__).resolve())
DEFAULT_PARAMS_FILE = BASE_DIR / "params.yaml"


class Environment(StrEnum):
    """Enumeration of supported application logging levels."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    """Enumeration of logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ImmutableBaseModel(BaseModel):
    """Base Pydantic model enforcing immutable and strict configuration.

    Subclasses inherit validation behavior intended to prevent accidental
    mutation of application configuration after initialization.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )


class SchemaParams(ImmutableBaseModel):
    """Define dataset schema and feature configuration.

    Stores the column definitions and feature groups used to validate,
    preprocess, and model the churn prediction datasets.
    """

    target_column: str = Field(default="Churn", min_length=1)
    binary_columns: list[str] = Field(
        default_factory=lambda: [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
        ]
    )
    ordinal_columns: list[str] = Field(
        default_factory=lambda: [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
    )
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "tenure",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
    )


class PreprocessingParams(ImmutableBaseModel):
    """Define preprocessing configuration for model-ready datasets.

    Contains the feature transformation settings used consistently during
    training and inference.
    """

    test_size: float = Field(default=0.2, gt=0.0, le=1.0)
    random_state: int = Field(default=42, ge=0)


class XGBoostParams(ImmutableBaseModel):
    """Define hyperparameters used to configure the XGBoost model."""

    objective: str = Field(default="binary:logistic")
    learning_rate: float = Field(default=0.03, gt=0.0, le=1.0)
    max_depth: int = Field(default=6, gt=0)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    min_child_weight: int = Field(default=3, ge=0)
    tree_method: str = Field(default="hist")
    scale_pos_weight: float = Field(default=1.0, gt=0.0)
    random_state: int = Field(default=42, ge=0)
    nthread: int = Field(default=-1)
    eval_metric: list[str] = Field(default_factory=lambda: ["logloss", "auc", "aucpr"])


class TrainParams(ImmutableBaseModel):
    """Define model training configuration.

    Stores XGBoost parameters and training controls such as boosting rounds,
    early stopping, and evaluation verbosity.
    """

    verbose_eval: int = Field(default=100, ge=0)
    num_boost_round: int = Field(default=3000, gt=0)
    early_stopping_rounds: int = Field(default=100, gt=0)
    xgb_params: XGBoostParams = Field(default_factory=XGBoostParams)


class EvaluateParams(ImmutableBaseModel):
    """Define model evaluation and classification configuration.

    Contains the decision threshold and other settings used to evaluate
    model performance on the validation dataset.
    """

    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MetadataParams(ImmutableBaseModel):
    """Define metadata used for model tracking and registry records."""

    registered_model_name: str = Field(default="churn_model", min_length=1)
    model_type: str = Field(default="XGBoost Classifier", min_length=1)
    developer: str = Field(default="ML Engineering Team", min_length=1)


class PipelineParams(ImmutableBaseModel):
    """Define the complete validated pipeline configuration.

    Aggregates dataset schema, preprocessing, training, evaluation, and
    metadata configuration into a single type-safe parameter object.
    """

    schema_config: SchemaParams = Field(default_factory=SchemaParams, alias="schema")
    preprocessing: PreprocessingParams = Field(default_factory=PreprocessingParams)
    train: TrainParams = Field(default_factory=TrainParams)
    evaluate: EvaluateParams = Field(default_factory=EvaluateParams)
    metadata: MetadataParams = Field(default_factory=MetadataParams)

    @classmethod
    def load_from_yaml(cls, path: Path = DEFAULT_PARAMS_FILE) -> "PipelineParams":
        """Load and validate pipeline parameters from a YAML configuration file.

        The YAML contents are parsed into the validated parameter model. If the
        configuration file does not exist, the model's default configuration is
        returned instead.

        Parameters
        ----------
        path : pathlib.Path, default=DEFAULT_PARAMS_FILE
            Path to the YAML parameter configuration file.

        Returns
        -------
        PipelineParams
            Validated pipeline parameter configuration.

        Raises
        ------
        ValueError
            If the YAML file exists but contains invalid parameter values or
            cannot be parsed into the expected configuration schema.
        """
        if not path.is_file():
            return cls()

        try:
            with path.open(encoding="utf-8") as f:
                raw_data: dict[str, Any] = yaml.safe_load(f) or {}
            return cls.model_validate(raw_data)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Malformed YAML configuration in '{path}': {exc}") from exc
        except ValidationError as exc:
            raise RuntimeError(f"Invalid pipeline parameter schema in '{path}': {exc}") from exc


class Settings(BaseSettings):
    """Define validated application-wide runtime configuration.

    Configuration values are loaded from environment variables and the
    project-level ``.env`` file, with environment variables taking
    precedence.

    The settings object contains runtime paths, logging configuration,
    external service credentials, API configuration, model settings, and
    computed artifact locations.

    Notes
    -----
    Unknown configuration values are ignored, and environment variable
    names are matched case-insensitively.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime configuration
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOG_BACKUP_COUNT: int = Field(default=14, ge=1)
    LOG_ROTATION_WHEN: str = Field(default="midnight", pattern=r"^(midnight|[Hh]|[Dd]|[Ww])$")
    LOGGER_NAME: str = Field(default="chun_forecast", min_length=1)
    FASTAPI_HOST: str = Field(default="0.0.0.0")
    FASTAPI_PORT: int = Field(default=8000, ge=1, le=65535)

    # API URLs
    API_BASE_URL: str = ""
    PREDICT_API_URL: str = ""
    DOWNLOAD_API_URL: str = ""
    EXPLAIN_API_URL: str = ""
    HEALTH_API_URL: str = ""

    @model_validator(mode="after")
    def assemble_api_urls(self) -> "Settings":
        """Construct API endpoint URLs from the configured host and port.

        Derives the URLs used by the prediction, download, explainability, and
        health-check clients from the application's API host and port settings.
        """
        self.API_BASE_URL = f"http://{self.FASTAPI_HOST}:{self.FASTAPI_PORT}"
        self.PREDICT_API_URL = f"{self.API_BASE_URL}/api/v1/predict/batch"
        self.DOWNLOAD_API_URL = f"{self.API_BASE_URL}/api/v1/predictions/download"
        self.EXPLAIN_API_URL = f"{self.API_BASE_URL}/explain"
        self.HEALTH_API_URL = f"{self.API_BASE_URL}/health"

        return self

    # Workspace Paths
    LOGS_DIR: Path = Field(default_factory=lambda: BASE_DIR / "logs")
    RAW_DATA_DIR: Path = Field(default_factory=lambda: BASE_DIR / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: BASE_DIR / "data" / "processed")
    PREDICTION_DATA_DIR: Path = Field(default_factory=lambda: BASE_DIR / "data" / "predictions")
    MODEL_DIR: Path = Field(default_factory=lambda: BASE_DIR / "models")
    TRAINING_REPORT_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "training")
    VALIDATION_REPORT_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "validation")
    SHAP_REPORT_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "shap")

    # External Services & Credentials
    SENTRY_DSN: HttpUrl | None = Field(default=None)
    KAGGLE_USERNAME: SecretStr | None = Field(default=None)
    KAGGLE_KEY: SecretStr | None = Field(default=None)
    GHCR_USER: str | None = Field(default=None)
    GHCR_PAT: SecretStr | None = Field(default=None)

    # Internal Parameter Cache
    _params_cache: PipelineParams | None = None

    @property
    def PARAMS(self) -> PipelineParams:
        """Return the lazily loaded pipeline configuration.

        Loads and validates ``params.yaml`` on first access and caches the
        resulting configuration for subsequent accesses.

        Returns
        -------
        PipelineParams
            Validated pipeline hyperparameters and processing configuration.
        """
        if self._params_cache is None:
            self._params_cache = PipelineParams.load_from_yaml(DEFAULT_PARAMS_FILE)
        return self._params_cache

    def ensure_directories(self) -> None:
        """Create required application workspace directories.

        Ensures that directories required for data ingestion, preprocessing,
        models, reports, predictions, logs, and other runtime artifacts exist.
        Existing directories are left unchanged.
        """
        directories = [
            self.LOGS_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.PREDICTION_DATA_DIR,
            self.MODEL_DIR,
            self.VALIDATION_REPORT_DIR,
            self.TRAINING_REPORT_DIR,
            self.SHAP_REPORT_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def _validate_environment_credentials(self) -> Self:
        """Validate credentials required by the configured runtime environment.

        Checks that external service credentials required by the current
        environment are available and appropriately configured.

        Raises
        ------
        ValueError
            If a required credential is missing or invalid for the selected
            environment.
        """
        if self.ENVIRONMENT in {Environment.PRODUCTION, Environment.STAGING}:
            missing: list[str] = []

            if not self.SENTRY_DSN:
                missing.append("SENTRY_DSN")
            if not self.KAGGLE_USERNAME or not self.KAGGLE_USERNAME.get_secret_value():
                missing.append("KAGGLE_USERNAME")
            if not self.KAGGLE_KEY or not self.KAGGLE_KEY.get_secret_value():
                missing.append("KAGGLE_KEY")
            if not self.GHCR_USER:
                missing.append("GHCR_USER")
            if not self.GHCR_PAT:
                missing.append("GHCR_PAT")

            if missing:
                raise ValueError(
                    f"Missing required secrets for {self.ENVIRONMENT.value} environment: "
                    f"{', '.join(missing)}"
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance.

    Creates the application settings on first invocation, validates the
    configuration, initializes required workspace directories, and caches
    the resulting instance for subsequent calls.

    Returns
    -------
    Settings
        Fully initialized and validated application settings.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
