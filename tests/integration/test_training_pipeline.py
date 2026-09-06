"""Test the churn prediction training pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import churn_prediction.pipelines.training_pipeline as module


@pytest.fixture
def settings(tmp_path: Path) -> MagicMock:
    """Create pipeline settings backed by temporary directories."""
    settings = MagicMock()

    settings.RAW_DATA_DIR = tmp_path / "raw"
    settings.PROCESSED_DATA_DIR = tmp_path / "processed"
    settings.MODEL_DIR = tmp_path / "models"
    settings.VALIDATION_REPORT_DIR = tmp_path / "validation_reports"
    settings.TRAINING_REPORT_DIR = tmp_path / "training_reports"

    return settings


def test_run_ingestion(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ingestion resolves the raw directory and passes it to get_dataset."""
    get_dataset = MagicMock()
    monkeypatch.setattr(module, "get_dataset", get_dataset)

    module.run_ingestion(settings)

    get_dataset.assert_called_once_with(settings.RAW_DATA_DIR.expanduser().resolve())


def test_run_raw_validation(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Raw validation builds the expected dataset and report paths."""
    validate_dataset = MagicMock()
    monkeypatch.setattr(module, "validate_dataset", validate_dataset)

    module.run_raw_validation(settings)

    raw_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    validate_dataset.assert_called_once_with(
        dataset_path=raw_dir / "train.parquet",
        report_path=report_dir / "raw.json",
        schema=module.RAW_SCHEMA,
    )


def test_run_preprocessing(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Preprocessing builds all expected input and output artifact paths."""
    preprocess = MagicMock()
    monkeypatch.setattr(
        module,
        "preprocess_train_dataset",
        preprocess,
    )

    module.run_preprocessing(settings)

    raw_dir = settings.RAW_DATA_DIR.expanduser().resolve()
    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()

    preprocess.assert_called_once_with(
        raw_input_path=raw_dir / "train.parquet",
        train_output_path=(processed_dir / "processed_train.parquet"),
        val_output_path=(processed_dir / "processed_val.parquet"),
        preprocessor_path=(model_dir / "preprocessor.joblib"),
    )


def test_run_train_validation(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Training validation targets the processed training dataset."""
    validate_dataset = MagicMock()
    monkeypatch.setattr(module, "validate_dataset", validate_dataset)

    module.run_train_validation(settings)

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    validate_dataset.assert_called_once_with(
        dataset_path=(processed_dir / "processed_train.parquet"),
        report_path=(report_dir / "train.json"),
        schema=module.PROCESSED_SCHEMA,
    )


def test_run_val_validation(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Validation targets the processed validation dataset."""
    validate_dataset = MagicMock()
    monkeypatch.setattr(module, "validate_dataset", validate_dataset)

    module.run_val_validation(settings)

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    report_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    validate_dataset.assert_called_once_with(
        dataset_path=(processed_dir / "processed_val.parquet"),
        report_path=(report_dir / "val.json"),
        schema=module.PROCESSED_SCHEMA,
    )


def test_run_training(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Training builds all model, card, and report artifact paths."""
    train_model = MagicMock()
    monkeypatch.setattr(
        module,
        "train_xgb_model",
        train_model,
    )

    module.run_training(settings)

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()
    report_dir = settings.TRAINING_REPORT_DIR.expanduser().resolve()

    train_model.assert_called_once_with(
        train_data_path=(processed_dir / "processed_train.parquet"),
        val_data_path=(processed_dir / "processed_val.parquet"),
        model_filepath=(model_dir / "model.ubj"),
        model_card_path=(model_dir / "card.json"),
        report_path=(report_dir / "training.json"),
    )


def test_run_all_stages_executes_stages_in_order(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """The complete pipeline executes every stage in the correct order."""
    calls: list[str] = []

    monkeypatch.setattr(
        module,
        "run_ingestion",
        lambda _: calls.append("ingestion"),
    )
    monkeypatch.setattr(
        module,
        "run_raw_validation",
        lambda _: calls.append("raw_validation"),
    )
    monkeypatch.setattr(
        module,
        "run_preprocessing",
        lambda _: calls.append("preprocessing"),
    )
    monkeypatch.setattr(
        module,
        "run_train_validation",
        lambda _: calls.append("train_validation"),
    )
    monkeypatch.setattr(
        module,
        "run_val_validation",
        lambda _: calls.append("val_validation"),
    )
    monkeypatch.setattr(
        module,
        "run_training",
        lambda _: calls.append("training"),
    )

    module.run_all_stages(settings)

    assert calls == [
        "ingestion",
        "raw_validation",
        "preprocessing",
        "train_validation",
        "val_validation",
        "training",
    ]


def test_stages_contains_all_pipeline_stages():
    """The stage registry exposes every supported pipeline stage."""
    assert set(module.STAGES) == {
        "ingest-data",
        "validate-raw",
        "preprocess-raw",
        "validate-train",
        "validate-val",
        "train-model",
        "all",
    }

    assert module.STAGES["ingest-data"] is module.run_ingestion
    assert module.STAGES["validate-raw"] is module.run_raw_validation
    assert module.STAGES["preprocess-raw"] is module.run_preprocessing
    assert module.STAGES["validate-train"] is module.run_train_validation
    assert module.STAGES["validate-val"] is module.run_val_validation
    assert module.STAGES["train-model"] is module.run_training
    assert module.STAGES["all"] is module.run_all_stages


def test_build_parser_requires_stage():
    """The CLI parser requires a pipeline stage."""
    parser = module._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize(
    "stage",
    [
        "ingest-data",
        "validate-raw",
        "preprocess-raw",
        "validate-train",
        "validate-val",
        "train-model",
        "all",
    ],
)
def test_build_parser_registers_stage(stage):
    """Every registered stage can be parsed by the CLI."""
    parser = module._build_parser()

    args = parser.parse_args([stage])

    assert args.stage == stage
    assert args.func is module.STAGES[stage]
    assert args.verbose is False


def test_build_parser_verbose_flag():
    """The verbose flag is parsed correctly."""
    parser = module._build_parser()

    args = parser.parse_args(["--verbose", "train-model"])

    assert args.verbose is True
    assert args.stage == "train-model"
    assert args.func is module.run_training


def test_build_parser_verbose_short_flag():
    """The short verbose option is parsed correctly."""
    parser = module._build_parser()

    args = parser.parse_args(["-v", "train-model"])

    assert args.verbose is True
    assert args.stage == "train-model"


def test_build_parser_stage_help_uses_function_docstring():
    """Stage help text is populated from the stage function docstring."""
    parser = module._build_parser()

    args = parser.parse_args(["ingest-data"])

    assert args.func is module.run_ingestion


def test_stage_function_propagates_dependency_error(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Pipeline stage errors from dependencies are not swallowed."""
    error = RuntimeError("ingestion failed")

    monkeypatch.setattr(
        module,
        "get_dataset",
        MagicMock(side_effect=error),
    )

    with pytest.raises(RuntimeError, match="ingestion failed"):
        module.run_ingestion(settings)


def test_run_all_stages_stops_when_stage_fails(
    settings: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """The complete pipeline stops when an earlier stage fails."""
    calls: list[str] = []

    def ingestion(_):
        calls.append("ingestion")
        raise RuntimeError("pipeline stopped")

    monkeypatch.setattr(
        module,
        "run_ingestion",
        ingestion,
    )
    monkeypatch.setattr(
        module,
        "run_raw_validation",
        lambda _: calls.append("raw_validation"),
    )

    with pytest.raises(RuntimeError, match="pipeline stopped"):
        module.run_all_stages(settings)

    assert calls == ["ingestion"]
