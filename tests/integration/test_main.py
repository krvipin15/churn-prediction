"""Integration tests for the Streamlit churn prediction dashboard."""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import churn_prediction.client.main as module


class MockSessionState(dict):
    """Dictionary-backed replacement for Streamlit session state.

    Streamlit's SessionState supports both mapping-style and attribute-style
    access. The dashboard uses attribute assignment, for example:

    ``st.session_state.batch_id = ...``

    while initialization uses:

    ``st.session_state[key] = ...``

    This test double supports both forms.
    """

    def __getattr__(self, name: str):
        """Return a state value using attribute-style access."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        """Store a state value using attribute-style assignment."""
        self[name] = value


class NullContext:
    """Minimal context manager used for Streamlit containers and spinners."""

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context without suppressing exceptions."""
        return False


class MockUploadedFile:
    """Minimal Streamlit UploadedFile-compatible object."""

    def __init__(self, content: bytes, name: str = "customers.csv") -> None:
        """Initialize the uploaded file."""
        self._content = content
        self._buffer = BytesIO(content)
        self.name = name

    def getbuffer(self):
        """Return the uploaded file contents as a memory view."""
        return memoryview(self._content)

    def seek(self, position: int) -> int:
        """Move the internal buffer position."""
        return self._buffer.seek(position)

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the internal buffer."""
        return self._buffer.read(size)


class MockColumn(NullContext):
    """Fake Streamlit column supporting context-manager usage."""


@pytest.fixture
def dashboard():
    """Return the dashboard module under test."""
    return module


@pytest.fixture
def session_state() -> MockSessionState:
    """Return a clean Streamlit-compatible session state."""
    return MockSessionState(
        done=False,
        batch_id=None,
        predictions_df=None,
        predictions_path=None,
        artifacts=None,
    )


@pytest.fixture
def risk_profiles() -> pd.DataFrame:
    """Return representative customer prediction profiles."""
    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "churn_probability": [0.91, 0.72, 0.31, 0.05],
        }
    )


@pytest.fixture
def feature_importance() -> pd.DataFrame:
    """Return representative global SHAP feature importance."""
    return pd.DataFrame(
        {
            "feature": [
                "tenure",
                "monthly_charges",
                "contract",
            ],
            "mean_abs_shap": [
                0.12,
                0.35,
                0.61,
            ],
        }
    )


@pytest.fixture
def artifacts(
    risk_profiles: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict:
    """Return representative explainability artifacts."""
    return {
        "risk_profiles": risk_profiles,
        "feature_importance": feature_importance,
        "id_col": "customer_id",
        "prob_col": "churn_probability",
    }


@pytest.fixture
def patch_spinner(monkeypatch, dashboard):
    """Patch Streamlit spinners with lightweight context managers."""
    monkeypatch.setattr(
        dashboard.st,
        "spinner",
        Mock(side_effect=lambda _message: NullContext()),
    )


@pytest.fixture
def patch_basic_streamlit(monkeypatch, dashboard):
    """Patch basic Streamlit rendering functions."""
    monkeypatch.setattr(dashboard.st, "markdown", Mock())
    monkeypatch.setattr(dashboard.st, "caption", Mock())
    monkeypatch.setattr(dashboard.st, "subheader", Mock())
    monkeypatch.setattr(dashboard.st, "warning", Mock())
    monkeypatch.setattr(dashboard.st, "success", Mock())
    monkeypatch.setattr(dashboard.st, "write", Mock())
    monkeypatch.setattr(dashboard.st, "toast", Mock())
    monkeypatch.setattr(dashboard.st, "error", Mock())
    monkeypatch.setattr(dashboard.st, "exception", Mock())


@pytest.fixture
def patch_columns(monkeypatch, dashboard):
    """Patch Streamlit columns to support integer and ratio specifications."""

    def fake_columns(spec):
        count = spec if isinstance(spec, int) else len(spec)

        return [MockColumn() for _ in range(count)]

    monkeypatch.setattr(
        dashboard.st,
        "columns",
        fake_columns,
    )


@pytest.fixture
def patch_streamlit_ui(monkeypatch, dashboard):
    """Patch Streamlit UI primitives used by ``main``."""
    monkeypatch.setattr(dashboard.st, "markdown", Mock())
    monkeypatch.setattr(dashboard.st, "caption", Mock())
    monkeypatch.setattr(dashboard.st, "subheader", Mock())
    monkeypatch.setattr(dashboard.st, "dataframe", Mock())
    monkeypatch.setattr(dashboard.st, "download_button", Mock())
    monkeypatch.setattr(dashboard.st, "button", Mock(return_value=False))

    def fake_columns(spec):
        """Return fake columns for integer or ratio specifications."""
        count = spec if isinstance(spec, int) else len(spec)
        return [MockColumn() for _ in range(count)]

    monkeypatch.setattr(
        dashboard.st,
        "columns",
        fake_columns,
    )

    monkeypatch.setattr(
        dashboard.st,
        "container",
        lambda *_args, **_kwargs: NullContext(),
    )

    monkeypatch.setattr(
        dashboard.st,
        "expander",
        lambda *_args, **_kwargs: NullContext(),
    )


@pytest.fixture
def patch_column_config(monkeypatch, dashboard):
    """Provide lightweight Streamlit column configuration objects."""
    monkeypatch.setattr(
        dashboard.st,
        "column_config",
        SimpleNamespace(
            TextColumn=Mock(),
            ProgressColumn=Mock(),
        ),
    )


def test_init_session_state_initializes_all_defaults(
    dashboard,
    monkeypatch,
):
    """All required session-state keys should be initialized."""
    state = MockSessionState()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        state,
    )

    dashboard.init_session_state()

    assert state == {
        "done": False,
        "batch_id": None,
        "predictions_df": None,
        "predictions_path": None,
        "artifacts": None,
    }


def test_init_session_state_preserves_existing_values(
    dashboard,
    monkeypatch,
):
    """Existing session-state values should not be overwritten."""
    state = MockSessionState(
        done=True,
        batch_id="existing-batch",
        predictions_df="existing-data",
        predictions_path=Path(tempfile.gettempdir()) / "existing.csv",
        artifacts={"existing": True},
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        state,
    )

    dashboard.init_session_state()

    assert state.done is True
    assert state.batch_id == "existing-batch"
    assert state.predictions_df == "existing-data"
    assert state.predictions_path == Path(tempfile.gettempdir()) / "existing.csv"
    assert state.artifacts == {"existing": True}


def test_init_session_state_initializes_only_missing_keys(
    dashboard,
    monkeypatch,
):
    """Only missing state keys should receive default values."""
    state = MockSessionState(
        done=True,
        batch_id="existing-batch",
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        state,
    )

    dashboard.init_session_state()

    assert state.done is True
    assert state.batch_id == "existing-batch"
    assert state.predictions_df is None
    assert state.predictions_path is None
    assert state.artifacts is None


def test_run_pipeline_local_file_success(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
):
    """A local dataset should execute the complete pipeline successfully."""
    input_file = tmp_path / "customers.csv"
    input_file.write_text("customer_id,value\n101,1\n")

    predictions_content = b"customer_id,churn_probability\n101,0.91\n"

    predict_batch = Mock(
        return_value={
            "batch_id": "batch-001",
        }
    )

    download_predictions = Mock(
        side_effect=lambda _batch_id, destination: Path(destination).write_bytes(
            predictions_content
        )
    )

    explain_batch = Mock(
        return_value={
            "artifacts_path": Path(tempfile.gettempdir()) / "artifacts",
        }
    )

    load_artifacts = Mock(
        return_value=artifacts,
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        predict_batch,
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        download_predictions,
    )
    monkeypatch.setattr(
        dashboard,
        "explain_batch",
        explain_batch,
    )
    monkeypatch.setattr(
        dashboard,
        "load_artifacts",
        load_artifacts,
    )
    monkeypatch.setattr(
        dashboard.st,
        "toast",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        Mock(),
    )

    dashboard.run_pipeline(input_file)

    predict_batch.assert_called_once_with(str(input_file))

    download_predictions.assert_called_once()

    assert download_predictions.call_args.args[0] == "batch-001"

    explain_batch.assert_called_once_with("batch-001")

    load_artifacts.assert_called_once_with(Path(tempfile.gettempdir()) / "artifacts")

    assert session_state.done is True
    assert session_state.batch_id == "batch-001"
    assert session_state.predictions_path is not None
    assert session_state.artifacts == artifacts

    pd.testing.assert_frame_equal(
        session_state.predictions_df,
        pd.DataFrame(
            {
                "customer_id": [101],
                "churn_probability": [0.91],
            }
        ),
    )


def test_run_pipeline_string_path_success(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
):
    """A dataset supplied as a string path should work."""
    input_file = tmp_path / "customers.csv"
    input_file.write_text("customer_id,value\n101,1\n")

    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text("customer_id,churn_probability\n101,0.81\n")

    predict_batch = Mock(
        return_value={
            "batch_id": "batch-string",
        }
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        predict_batch,
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        lambda _batch_id, destination: Path(destination).write_bytes(predictions_file.read_bytes()),
    )
    monkeypatch.setattr(
        dashboard,
        "explain_batch",
        Mock(
            return_value={
                "artifacts_path": Path(tempfile.gettempdir()) / "artifacts",
            }
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_artifacts",
        Mock(return_value=artifacts),
    )
    monkeypatch.setattr(
        dashboard.st,
        "toast",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        Mock(),
    )

    dashboard.run_pipeline(str(input_file))

    predict_batch.assert_called_once_with(str(input_file))

    assert session_state.done is True
    assert session_state.batch_id == "batch-string"


def test_run_pipeline_uploaded_file_success(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
):
    """An uploaded CSV should be persisted and processed successfully."""
    uploaded_file = MockUploadedFile(b"customer_id,value\n101,42\n")

    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text("customer_id,churn_probability\n101,0.77\n")

    predict_batch = Mock(
        return_value={
            "batch_id": "uploaded-batch",
        }
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        predict_batch,
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        lambda _batch_id, destination: Path(destination).write_bytes(predictions_file.read_bytes()),
    )
    monkeypatch.setattr(
        dashboard,
        "explain_batch",
        Mock(
            return_value={
                "artifacts_path": Path(tempfile.gettempdir()) / "artifacts",
            }
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_artifacts",
        Mock(return_value=artifacts),
    )
    monkeypatch.setattr(
        dashboard.st,
        "toast",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        Mock(),
    )

    dashboard.run_pipeline(uploaded_file)

    predict_batch.assert_called_once()

    input_path = Path(predict_batch.call_args.args[0])

    assert input_path.exists() is False

    assert session_state.done is True
    assert session_state.batch_id == "uploaded-batch"

    pd.testing.assert_frame_equal(
        session_state.predictions_df,
        pd.DataFrame(
            {
                "customer_id": [101],
                "churn_probability": [0.77],
            }
        ),
    )


def test_run_pipeline_uploaded_file_cleanup(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
):
    """Uploaded temporary input files must be deleted after execution."""
    uploaded_file = MockUploadedFile(b"customer_id,value\n101,42\n")

    captured_input_path = {}

    def fake_predict_batch(path):
        """Capture the temporary input path."""
        captured_input_path["path"] = Path(path)

        return {
            "batch_id": "cleanup-batch",
        }

    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text("customer_id,churn_probability\n101,0.40\n")

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        fake_predict_batch,
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        lambda _batch_id, destination: Path(destination).write_bytes(predictions_file.read_bytes()),
    )
    monkeypatch.setattr(
        dashboard,
        "explain_batch",
        Mock(
            return_value={
                "artifacts_path": Path(tempfile.gettempdir()) / "artifacts",
            }
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_artifacts",
        Mock(return_value=artifacts),
    )
    monkeypatch.setattr(
        dashboard.st,
        "toast",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        Mock(),
    )

    dashboard.run_pipeline(uploaded_file)

    assert "path" in captured_input_path
    assert captured_input_path["path"].exists() is False


def test_run_pipeline_missing_local_file(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
):
    """Missing local datasets should stop before API execution."""
    input_file = tmp_path / "missing.csv"

    predict_batch = Mock()
    error = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        predict_batch,
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        error,
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        Mock(),
    )

    dashboard.run_pipeline(input_file)

    error.assert_called_once_with(f"Dataset file not found at: `{input_file}`")

    predict_batch.assert_not_called()

    assert session_state.done is False


def test_run_pipeline_prediction_api_error(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_spinner,
):
    """Prediction API failures should be surfaced through Streamlit."""
    input_file = tmp_path / "customers.csv"
    input_file.write_text("customer_id,value\n101,1\n")

    request_error = dashboard.requests.RequestException("prediction API unavailable")

    error = Mock()
    exception = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        Mock(side_effect=request_error),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        error,
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        exception,
    )

    dashboard.run_pipeline(input_file)

    error.assert_called_once_with("Inference Pipeline Error: prediction API unavailable")

    exception.assert_called_once_with(request_error)

    assert session_state.done is False


def test_run_pipeline_download_api_error(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_spinner,
):
    """Download API failures should be handled as request exceptions."""
    input_file = tmp_path / "customers.csv"
    input_file.write_text("customer_id,value\n101,1\n")

    request_error = dashboard.requests.RequestException("download failed")

    error = Mock()
    exception = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        Mock(
            return_value={
                "batch_id": "batch-download-error",
            }
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        Mock(side_effect=request_error),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        error,
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        exception,
    )

    dashboard.run_pipeline(input_file)

    error.assert_called_once_with("Inference Pipeline Error: download failed")

    exception.assert_called_once_with(request_error)

    assert session_state.done is False


def test_run_pipeline_explain_api_error(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_spinner,
):
    """Explainability API failures should be reported."""
    input_file = tmp_path / "customers.csv"
    input_file.write_text("customer_id,value\n101,1\n")

    request_error = dashboard.requests.RequestException("explainability service unavailable")

    error = Mock()
    exception = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        dashboard,
        "predict_batch",
        Mock(
            return_value={
                "batch_id": "batch-explain-error",
            }
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "download_predictions",
        lambda _batch_id, destination: Path(destination).write_text(
            "customer_id,churn_probability\n101,0.55\n"
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "explain_batch",
        Mock(side_effect=request_error),
    )
    monkeypatch.setattr(
        dashboard.st,
        "error",
        error,
    )
    monkeypatch.setattr(
        dashboard.st,
        "exception",
        exception,
    )

    dashboard.run_pipeline(input_file)

    error.assert_called_once_with("Inference Pipeline Error: explainability service unavailable")

    exception.assert_called_once_with(request_error)

    assert session_state.done is False


def test_render_kpis(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """KPI rendering should calculate all executive metrics."""
    metric = Mock()
    caption = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "metric",
        metric,
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        caption,
    )
    monkeypatch.setattr(
        dashboard.st,
        "container",
        lambda **_kwargs: NullContext(),
    )

    dashboard.render_kpis(
        risk_profiles,
        "churn_probability",
    )

    assert metric.call_count == 4

    assert metric.call_args_list[0].args == (
        "Total Cohort",
        "4",
    )

    assert metric.call_args_list[1].args == (
        "Avg. Churn Rate",
        "49.8%",
    )

    assert metric.call_args_list[2].args == (
        "High-Risk Count",
        "2",
    )

    assert metric.call_args_list[3].args == (
        "At Risk %",
        "50.0%",
    )


def test_render_feature_importance(
    dashboard,
    monkeypatch,
    feature_importance,
):
    """Global feature importance should render a horizontal bar chart."""
    plotly_chart = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "plotly_chart",
        plotly_chart,
    )

    dashboard.render_feature_importance(feature_importance)

    plotly_chart.assert_called_once()

    figure = plotly_chart.call_args.args[0]

    assert figure.layout.height == 360
    assert len(figure.data) == 1
    assert figure.data[0].orientation == "h"

    assert list(figure.data[0].y) == [
        "tenure",
        "monthly_charges",
        "contract",
    ]


def test_render_risk_distribution(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """Risk distribution should include the 0.5 classification threshold."""
    plotly_chart = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "plotly_chart",
        plotly_chart,
    )

    dashboard.render_risk_distribution(
        risk_profiles,
        "churn_probability",
    )

    plotly_chart.assert_called_once()

    figure = plotly_chart.call_args.args[0]

    assert len(figure.data) == 1
    assert figure.data[0].type == "histogram"

    assert len(figure.layout.shapes) == 1

    threshold_line = figure.layout.shapes[0]

    assert threshold_line.x0 == 0.5
    assert threshold_line.x1 == 0.5
    assert threshold_line.line.dash == "dash"


def test_render_customer_table_sorts_by_risk(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """Customer records should be displayed from highest to lowest risk."""
    dataframe = Mock()
    selectbox = Mock(return_value=101)

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        dataframe,
    )
    monkeypatch.setattr(
        dashboard.st,
        "selectbox",
        selectbox,
    )

    selected_customer = dashboard.render_customer_table(
        risk_profiles,
        "customer_id",
        "churn_probability",
    )

    assert selected_customer == 101

    dataframe.assert_called_once()

    rendered = dataframe.call_args.args[0]

    assert rendered["customer_id"].tolist() == [
        101,
        102,
        103,
        104,
    ]

    assert rendered["churn_probability"].tolist() == [
        0.91,
        0.72,
        0.31,
        0.05,
    ]


def test_render_customer_table_selects_customer(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """The selected customer should be returned from the selectbox."""
    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "selectbox",
        Mock(return_value=103),
    )

    result = dashboard.render_customer_table(
        risk_profiles,
        "customer_id",
        "churn_probability",
    )

    assert result == 103


def test_render_customer_detail_empty_breakdown(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """An empty SHAP breakdown should display a warning."""
    warning = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "warning",
        warning,
    )
    monkeypatch.setattr(
        dashboard,
        "customer_shap_breakdown",
        Mock(return_value=pd.DataFrame()),
    )

    dashboard.render_customer_detail(
        risk_profiles,
        101,
    )

    warning.assert_called_once_with(
        "⚠️ No detailed feature attribution drivers found for this customer."
    )


def test_render_customer_detail_positive_and_negative_drivers(
    dashboard,
    monkeypatch,
    risk_profiles,
    patch_columns,
):
    """Positive and negative SHAP drivers should be rendered."""
    breakdown = pd.DataFrame(
        {
            "feature": [
                "contract",
                "tenure",
                "monthly_charges",
            ],
            "shap_value": [
                0.50,
                -0.20,
                0.30,
            ],
            "display_value": [
                "Month-to-month",
                "60 months",
                "$100",
            ],
        }
    )

    plotly_chart = Mock()
    recommendation = Mock(side_effect=lambda feature: f"Recommendation for {feature}")

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "plotly_chart",
        plotly_chart,
    )
    monkeypatch.setattr(
        dashboard.st,
        "write",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "container",
        lambda **_kwargs: NullContext(),
    )
    monkeypatch.setattr(
        dashboard,
        "customer_shap_breakdown",
        Mock(return_value=breakdown),
    )
    monkeypatch.setattr(
        dashboard,
        "recommend_for_feature",
        recommendation,
    )

    dashboard.render_customer_detail(
        risk_profiles,
        101,
    )

    plotly_chart.assert_called_once()

    figure = plotly_chart.call_args.args[0]

    assert len(figure.data) == 1
    assert figure.data[0].orientation == "h"

    assert list(figure.data[0].x) == [
        0.50,
        -0.20,
        0.30,
    ]

    assert list(figure.data[0].y) == [
        "contract",
        "tenure",
        "monthly_charges",
    ]

    recommendation.assert_any_call("contract")

    recommendation.assert_any_call("monthly_charges")


def test_render_customer_detail_only_negative_drivers(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """Customers with no positive SHAP drivers should receive a low-risk message."""
    breakdown = pd.DataFrame(
        {
            "feature": [
                "tenure",
                "support_calls",
            ],
            "shap_value": [
                -0.30,
                -0.10,
            ],
            "display_value": [
                "60",
                "0",
            ],
        }
    )

    success = Mock()
    recommendation = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "plotly_chart",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "success",
        success,
    )
    monkeypatch.setattr(
        dashboard.st,
        "container",
        lambda **_kwargs: NullContext(),
    )
    monkeypatch.setattr(
        dashboard,
        "customer_shap_breakdown",
        Mock(return_value=breakdown),
    )
    monkeypatch.setattr(
        dashboard,
        "recommend_for_feature",
        recommendation,
    )

    dashboard.render_customer_detail(
        risk_profiles,
        101,
    )

    success.assert_called_once_with(
        "✅ This customer exhibits low risk across all primary operational drivers."
    )

    recommendation.assert_not_called()


def test_render_customer_detail_limits_recommendations_to_three(
    dashboard,
    monkeypatch,
    risk_profiles,
):
    """Only the three strongest positive SHAP drivers should be recommended."""
    breakdown = pd.DataFrame(
        {
            "feature": [
                "feature_a",
                "feature_b",
                "feature_c",
                "feature_d",
                "feature_e",
            ],
            "shap_value": [
                0.10,
                0.50,
                0.40,
                0.30,
                0.20,
            ],
            "display_value": [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
        }
    )

    recommendation = Mock(return_value="Retention recommendation")

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "caption",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "plotly_chart",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "write",
        Mock(),
    )
    monkeypatch.setattr(
        dashboard.st,
        "container",
        lambda **_kwargs: NullContext(),
    )

    monkeypatch.setattr(
        dashboard,
        "customer_shap_breakdown",
        Mock(return_value=breakdown),
    )

    monkeypatch.setattr(
        dashboard,
        "recommend_for_feature",
        recommendation,
    )

    dashboard.render_customer_detail(
        risk_profiles,
        101,
    )

    assert recommendation.call_count == 3

    recommended_features = [call.args[0] for call in recommendation.call_args_list]

    assert recommended_features == [
        "feature_b",
        "feature_c",
        "feature_d",
    ]


def test_inject_custom_css(
    dashboard,
    monkeypatch,
):
    """Dashboard CSS should be injected as HTML."""
    markdown = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "markdown",
        markdown,
    )

    dashboard.inject_custom_css()

    markdown.assert_called_once()

    css = markdown.call_args.args[0]

    assert "<style>" in css
    assert "</style>" in css
    assert ".hero-banner" in css
    assert ".hero-title" in css
    assert ".hero-subtitle" in css
    assert ".custom-card" in css
    assert "stMetricValue" in css

    assert dashboard.THEME_PALETTE["primary"] in css

    assert dashboard.THEME_PALETTE["neutral_dark"] in css

    assert markdown.call_args.kwargs["unsafe_allow_html"] is True


def test_main_without_completed_pipeline(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_streamlit_ui,
):
    """Main should render the ingestion UI when no pipeline is complete."""
    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=None),
    )

    dashboard.main()

    assert session_state.done is False

    dashboard.st.subheader.assert_any_call("1. Ingest Batch Data")


def test_main_demo_button_runs_pipeline(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_streamlit_ui,
):
    """The Demo Dataset button should execute the local test dataset."""
    demo_file = tmp_path / "test.csv"

    demo_file.write_text("customer_id,value\n101,1\n")

    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    run_pipeline = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard,
        "run_pipeline",
        run_pipeline,
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=None),
    )

    # Only the Demo Dataset button exists when no file is uploaded.
    monkeypatch.setattr(
        dashboard.st,
        "button",
        Mock(
            side_effect=[
                True,
            ]
        ),
    )

    dashboard.main()

    run_pipeline.assert_called_once_with(demo_file)


def test_main_uploaded_file_preview(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_streamlit_ui,
):
    """Uploaded data should be previewed before execution."""
    uploaded_file = MockUploadedFile(b"customer_id,value\n101,10\n102,20\n")

    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    dataframe = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=uploaded_file),
    )

    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        dataframe,
    )

    # Demo button = False.
    # Execute button = False.
    monkeypatch.setattr(
        dashboard.st,
        "button",
        Mock(
            side_effect=[
                False,
                False,
            ]
        ),
    )

    dashboard.main()

    dataframe.assert_called_once()

    preview = dataframe.call_args.args[0]

    assert isinstance(
        preview,
        pd.DataFrame,
    )

    assert list(preview["customer_id"]) == [
        101,
        102,
    ]

    assert list(preview["value"]) == [
        10,
        20,
    ]

    uploaded_file.seek(0)

    assert uploaded_file.read() == (b"customer_id,value\n101,10\n102,20\n")


def test_main_uploaded_file_execute_pipeline(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    patch_streamlit_ui,
):
    """The execution button should pass the uploaded file to run_pipeline."""
    uploaded_file = MockUploadedFile(b"customer_id,value\n101,10\n")

    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    run_pipeline = Mock()

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=uploaded_file),
    )

    monkeypatch.setattr(
        dashboard,
        "run_pipeline",
        run_pipeline,
    )

    monkeypatch.setattr(
        dashboard.st,
        "button",
        Mock(
            side_effect=[
                False,
                True,
            ]
        ),
    )

    dashboard.main()

    run_pipeline.assert_called_once_with(uploaded_file)


def test_main_completed_pipeline(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
    *,
    patch_streamlit_ui,
    patch_column_config,
):
    """Completed pipeline state should render the complete dashboard."""
    predictions_file = tmp_path / "predictions.csv"

    predictions_file.write_text("customer_id,churn_probability\n101,0.91\n")

    session_state.done = True
    session_state.batch_id = "batch-999"
    session_state.predictions_df = pd.DataFrame(
        {
            "customer_id": [101],
            "churn_probability": [0.91],
        }
    )
    session_state.predictions_path = str(predictions_file)
    session_state.artifacts = artifacts

    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=None),
    )

    render_kpis = Mock()
    render_feature_importance = Mock()
    render_risk_distribution = Mock()
    render_customer_table = Mock(return_value=101)
    render_customer_detail = Mock()

    monkeypatch.setattr(
        dashboard,
        "render_kpis",
        render_kpis,
    )

    monkeypatch.setattr(
        dashboard,
        "render_feature_importance",
        render_feature_importance,
    )

    monkeypatch.setattr(
        dashboard,
        "render_risk_distribution",
        render_risk_distribution,
    )

    monkeypatch.setattr(
        dashboard,
        "render_customer_table",
        render_customer_table,
    )

    monkeypatch.setattr(
        dashboard,
        "render_customer_detail",
        render_customer_detail,
    )

    dashboard.main()

    render_kpis.assert_called_once_with(
        artifacts["risk_profiles"],
        "churn_probability",
    )

    render_feature_importance.assert_called_once_with(
        artifacts["feature_importance"],
    )

    render_risk_distribution.assert_called_once_with(
        artifacts["risk_profiles"],
        "churn_probability",
    )

    render_customer_table.assert_called_once_with(
        artifacts["risk_profiles"],
        "customer_id",
        "churn_probability",
    )

    render_customer_detail.assert_called_once_with(
        artifacts["risk_profiles"],
        101,
    )

    dashboard.st.download_button.assert_called_once()

    download_kwargs = dashboard.st.download_button.call_args.kwargs

    assert download_kwargs["file_name"] == "churn_predictions_batch_batch-999.csv"

    assert download_kwargs["mime"] == "text/csv"


def test_main_completed_pipeline_uses_customer_selection(
    dashboard,
    monkeypatch,
    tmp_path: Path,
    session_state,
    artifacts,
    *,
    patch_streamlit_ui,
    patch_column_config,
):
    """Customer detail should receive the selected customer ID."""
    predictions_file = tmp_path / "predictions.csv"

    predictions_file.write_text("customer_id,churn_probability\n101,0.91\n")

    session_state.done = True
    session_state.batch_id = "batch-selection"
    session_state.predictions_path = str(predictions_file)
    session_state.artifacts = artifacts

    settings = SimpleNamespace(
        RAW_DATA_DIR=tmp_path,
    )

    monkeypatch.setattr(
        dashboard.st,
        "session_state",
        session_state,
    )

    monkeypatch.setattr(
        dashboard,
        "get_settings",
        Mock(return_value=settings),
    )

    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        Mock(return_value=None),
    )

    monkeypatch.setattr(
        dashboard,
        "render_kpis",
        Mock(),
    )

    monkeypatch.setattr(
        dashboard,
        "render_feature_importance",
        Mock(),
    )

    monkeypatch.setattr(
        dashboard,
        "render_risk_distribution",
        Mock(),
    )

    monkeypatch.setattr(
        dashboard,
        "render_customer_table",
        Mock(return_value=103),
    )

    render_customer_detail = Mock()

    monkeypatch.setattr(
        dashboard,
        "render_customer_detail",
        render_customer_detail,
    )

    dashboard.main()

    render_customer_detail.assert_called_once_with(
        artifacts["risk_profiles"],
        103,
    )
