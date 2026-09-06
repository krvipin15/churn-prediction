"""Test the churn prediction pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from sklearn.preprocessing import OrdinalEncoder
from xgboost import Booster

import churn_prediction.pipelines.prediction_pipeline as module


@pytest.fixture
def settings(tmp_path: Path) -> MagicMock:
    """Create temporary pipeline settings."""
    settings = MagicMock()

    settings.PROCESSED_DATA_DIR = tmp_path / "processed"
    settings.PREDICTION_DATA_DIR = tmp_path / "predictions"
    settings.MODEL_DIR = tmp_path / "models"
    settings.VALIDATION_REPORT_DIR = tmp_path / "validation"

    return settings


@pytest.fixture
def encoder() -> MagicMock:
    """Create a mock fitted encoder."""
    return MagicMock(spec=OrdinalEncoder)


@pytest.fixture
def booster() -> MagicMock:
    """Create a mock XGBoost booster."""
    return MagicMock(spec=Booster)


def test_run_prediction_pipeline_executes_all_stages(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The complete prediction workflow calls every stage correctly."""
    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    raw_file = tmp_path / "input" / "customers.parquet"

    module.run_prediction_pipeline(
        batch_id="batch-001",
        raw_file_path=raw_file,
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    prediction_dir = settings.PREDICTION_DATA_DIR.expanduser().resolve()
    model_dir = settings.MODEL_DIR.expanduser().resolve()
    validation_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    resolved_raw_file = raw_file.expanduser().resolve()
    processed_file = processed_dir / "processed_batch-001.parquet"
    prediction_file = prediction_dir / "prediction_batch-001.csv"
    model_card = model_dir / "card.json"
    raw_report = validation_dir / "raw_batch-001.json"
    processed_report = validation_dir / "processed_batch-001.json"

    assert validate_dataset.call_args_list == [
        call(
            dataset_path=resolved_raw_file,
            report_path=raw_report,
            schema=module.RAW_BASE_SCHEMA,
        ),
        call(
            dataset_path=processed_file,
            report_path=processed_report,
            schema=module.PROCESSED_BASE_SCHEMA,
        ),
    ]

    preprocess.assert_called_once_with(
        raw_input_path=resolved_raw_file,
        processed_output_path=processed_file,
        fitted_encoder=encoder,
    )

    inference.assert_called_once_with(
        booster=booster,
        raw_filepath=resolved_raw_file,
        input_filepath=processed_file,
        output_filepath=prediction_file,
        model_card_path=model_card,
    )


def test_run_prediction_pipeline_strips_batch_id(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whitespace around the batch ID is removed before paths are built."""
    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    raw_file = tmp_path / "customers.parquet"

    module.run_prediction_pipeline(
        batch_id="  batch-123  ",
        raw_file_path=raw_file,
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    prediction_dir = settings.PREDICTION_DATA_DIR.expanduser().resolve()
    validation_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    processed_file = processed_dir / "processed_batch-123.parquet"
    prediction_file = prediction_dir / "prediction_batch-123.csv"

    assert validate_dataset.call_args_list[0] == call(
        dataset_path=raw_file.resolve(),
        report_path=validation_dir / "raw_batch-123.json",
        schema=module.RAW_BASE_SCHEMA,
    )

    preprocess.assert_called_once_with(
        raw_input_path=raw_file.resolve(),
        processed_output_path=processed_file,
        fitted_encoder=encoder,
    )

    assert validate_dataset.call_args_list[1] == call(
        dataset_path=processed_file,
        report_path=validation_dir / "processed_batch-123.json",
        schema=module.PROCESSED_BASE_SCHEMA,
    )

    inference.assert_called_once_with(
        booster=booster,
        raw_filepath=raw_file.resolve(),
        input_filepath=processed_file,
        output_filepath=prediction_file,
        model_card_path=(settings.MODEL_DIR.expanduser().resolve() / "card.json"),
    )


@pytest.mark.parametrize(
    "batch_id",
    [
        "batch-001",
        "20260905-120000",
        "customer_batch",
        "batch.with.dots",
    ],
)
def test_batch_id_is_used_consistently(
    batch_id: str,
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
) -> None:
    """The normalized batch ID is used in every batch-specific artifact."""
    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    raw_file = tmp_path / "raw.parquet"

    module.run_prediction_pipeline(
        batch_id=batch_id,
        raw_file_path=raw_file,
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    processed_dir = settings.PROCESSED_DATA_DIR.expanduser().resolve()
    prediction_dir = settings.PREDICTION_DATA_DIR.expanduser().resolve()
    validation_dir = settings.VALIDATION_REPORT_DIR.expanduser().resolve()

    processed_file = processed_dir / f"processed_{batch_id}.parquet"
    prediction_file = prediction_dir / f"prediction_{batch_id}.csv"

    assert validate_dataset.call_args_list[0].kwargs["report_path"] == (
        validation_dir / f"raw_{batch_id}.json"
    )

    assert preprocess.call_args.kwargs["processed_output_path"] == (processed_file)

    assert validate_dataset.call_args_list[1].kwargs["report_path"] == (
        validation_dir / f"processed_{batch_id}.json"
    )

    assert inference.call_args.kwargs["output_filepath"] == (prediction_file)


def test_raw_file_path_accepts_string(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A string input path is converted to a resolved Path."""
    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    raw_file = tmp_path / "input.parquet"

    module.run_prediction_pipeline(
        batch_id="batch-001",
        raw_file_path=str(raw_file),
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    resolved_path = raw_file.expanduser().resolve()

    assert validate_dataset.call_args_list[0].kwargs["dataset_path"] == (resolved_path)

    assert preprocess.call_args.kwargs["raw_input_path"] == (resolved_path)

    assert inference.call_args.kwargs["raw_filepath"] == (resolved_path)


def test_pipeline_preserves_loaded_encoder_and_booster(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The supplied encoder and booster are passed through unchanged."""
    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    module.run_prediction_pipeline(
        batch_id="batch-001",
        raw_file_path=tmp_path / "raw.parquet",
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    assert preprocess.call_args.kwargs["fitted_encoder"] is encoder
    assert inference.call_args.kwargs["booster"] is booster


def test_pipeline_stops_when_raw_validation_fails(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preprocessing and inference are not executed after raw validation fails."""
    error = ValueError("raw validation failed")

    validate_dataset = MagicMock(side_effect=error)
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )
    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    with pytest.raises(ValueError, match="raw validation failed"):
        module.run_prediction_pipeline(
            batch_id="batch-001",
            raw_file_path=tmp_path / "raw.parquet",
            loaded_encoder=encoder,
            loaded_booster=booster,
        )

    preprocess.assert_not_called()
    inference.assert_not_called()


def test_pipeline_stops_when_preprocessing_fails(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inference is not executed when preprocessing fails."""
    validate_dataset = MagicMock()
    preprocess = MagicMock(side_effect=ValueError("preprocessing failed"))
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )
    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    with pytest.raises(ValueError, match="preprocessing failed"):
        module.run_prediction_pipeline(
            batch_id="batch-001",
            raw_file_path=tmp_path / "raw.parquet",
            loaded_encoder=encoder,
            loaded_booster=booster,
        )

    assert validate_dataset.call_count == 1
    inference.assert_not_called()


def test_pipeline_stops_when_processed_validation_fails(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inference is not executed when processed validation fails."""
    validate_dataset = MagicMock(
        side_effect=[
            None,
            ValueError("processed validation failed"),
        ]
    )
    preprocess = MagicMock()
    inference = MagicMock()

    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )
    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    with pytest.raises(
        ValueError,
        match="processed validation failed",
    ):
        module.run_prediction_pipeline(
            batch_id="batch-001",
            raw_file_path=tmp_path / "raw.parquet",
            loaded_encoder=encoder,
            loaded_booster=booster,
        )

    assert validate_dataset.call_count == 2
    preprocess.assert_called_once()
    inference.assert_not_called()


def test_pipeline_propagates_inference_error(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inference errors propagate to the pipeline caller."""
    validate_dataset = MagicMock()
    preprocess = MagicMock()
    inference = MagicMock(side_effect=RuntimeError("inference failed"))

    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )
    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate_dataset,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    with pytest.raises(
        RuntimeError,
        match="inference failed",
    ):
        module.run_prediction_pipeline(
            batch_id="batch-001",
            raw_file_path=tmp_path / "raw.parquet",
            loaded_encoder=encoder,
            loaded_booster=booster,
        )

    assert validate_dataset.call_count == 2
    preprocess.assert_called_once()
    inference.assert_called_once()


def test_pipeline_stage_order(
    settings: MagicMock,
    encoder: MagicMock,
    booster: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pipeline stages execute in the documented order."""
    execution_order: list[str] = []

    def validate(*args, **kwargs):
        """Record validation execution."""
        execution_order.append("validate")

    def preprocess(*args, **kwargs):
        """Record preprocessing execution."""
        execution_order.append("preprocess")

    def inference(*args, **kwargs):
        """Record inference execution."""
        execution_order.append("inference")

    monkeypatch.setattr(
        module,
        "get_settings",
        MagicMock(return_value=settings),
    )
    monkeypatch.setattr(
        module,
        "validate_dataset",
        validate,
    )
    monkeypatch.setattr(
        module,
        "preprocess_inference_dataset",
        preprocess,
    )
    monkeypatch.setattr(
        module,
        "run_inference",
        inference,
    )

    module.run_prediction_pipeline(
        batch_id="batch-001",
        raw_file_path=tmp_path / "raw.parquet",
        loaded_encoder=encoder,
        loaded_booster=booster,
    )

    assert execution_order == [
        "validate",
        "preprocess",
        "validate",
        "inference",
    ]
