"""
Customer Churn Analytics & Intelligence Dashboard.

This module serves as the primary entrypoint for the Streamlit-based enterprise
churn prediction and explainability web application. It orchestrates batch data
ingestion, asynchronous inference execution, global feature importance visualization,
and individual customer SHAP breakdown diagnostic analyses.
"""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from churn_prediction.client.api_client import (
    download_predictions,
    explain_batch,
    predict_batch,
)
from churn_prediction.client.dashboard import (
    customer_shap_breakdown,
    load_artifacts,
    recommend_for_feature,
)
from churn_prediction.config.settings import get_settings

# Theme color palette
THEME_PALETTE = {
    "primary": "#4F46E5",  # Modern Indigo
    "primary_hover": "#4338CA",
    "risk_high": "#E11D48",  # Rose Red (Increased Risk)
    "risk_low": "#059669",  # Emerald Green (Decreased Risk)
    "neutral_dark": "#0F172A",  # Slate 900
    "neutral_light": "#F8FAFC",  # Slate 50
    "grid_color": "#E2E8F0",  # Slate 200
}

# Plotly chart template
PLOTLY_TEMPLATE = "plotly_white"

# Streamlit page configuration
st.set_page_config(
    page_title="Enterprise Churn Analytics & Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/10419/10419629.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "**Customer Churn Analytics & Intelligence**\n\n"
            "An end-to-end batch inference dashboard for customer churn prediction. "
            "Upload a customer cohort, run the ML pipeline, and explore SHAP-based "
            "explainability to identify retention risk drivers at both the population "
            "and individual customer level.\n\n"
            "**Version:** 0.1.0\n\n"
            "Built with Streamlit, Plotly, and SHAP."
        ),
        "Report a bug": "https://github.com/krvipin15/churn-prediction/issues",
    },
)


def inject_custom_css() -> None:
    """Inject the dashboard's custom CSS styling.

    Adds the visual styling used by the dashboard for headers, metric cards,
    containers, charts, and other presentation components.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* White Hero Header Styling */
        .hero-banner {{
            background: #FFFFFF;
            padding: 2rem 2.25rem;
            border-radius: 16px;
            color: {THEME_PALETTE["neutral_dark"]};
            border: 1px solid #E2E8F0;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }}
        .hero-badge {{
            background: #EEF2FF;
            color: {THEME_PALETTE["primary"]};
            border: 1px solid #C7D2FE;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            display: inline-block;
            margin-bottom: 0.75rem;
        }}
        .hero-title {{
            font-size: 2.2rem;
            font-weight: 700;
            color: {THEME_PALETTE["neutral_dark"]};
            margin: 0 0 0.5rem 0;
            line-height: 1.2;
        }}
        .hero-subtitle {{
            font-size: 1rem;
            color: #64748B;
            margin: 0;
            max-width: 700px;
        }}

        /* Metric Cards */
        div[data-testid="stMetricValue"] {{
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: {THEME_PALETTE["neutral_dark"]};
        }}

        /* Custom Card Container */
        .custom-card {{
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }}

        /* Hide default Streamlit headers padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        /* Primary Buttons */
        button[kind="primary"] {{
            background-color: {THEME_PALETTE["primary"]} !important;
            border-color: {THEME_PALETTE["primary"]} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: background-color 0.15s ease, box-shadow 0.15s ease;
        }}
        button[kind="primary"]:hover {{
            background-color: {THEME_PALETTE["primary_hover"]} !important;
            border-color: {THEME_PALETTE["primary_hover"]} !important;
            box-shadow: 0 4px 12px -2px rgba(79, 70, 229, 0.35);
        }}

        /* Secondary Buttons (e.g. Download) */
        button[kind="secondary"] {{
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: border-color 0.15s ease, color 0.15s ease;
        }}
        button[kind="secondary"]:hover {{
            border-color: {THEME_PALETTE["primary"]} !important;
            color: {THEME_PALETTE["primary"]} !important;
        }}

        /* File Uploader Dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 10px !important;
            border: 1.5px dashed {THEME_PALETTE["grid_color"]} !important;
            background-color: {THEME_PALETTE["neutral_light"]} !important;
            transition: border-color 0.15s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {THEME_PALETTE["primary"]} !important;
        }}

        /* Spinner */
        div[data-testid="stSpinner"] {{
            color: {THEME_PALETTE["primary"]} !important;
            font-weight: 500;
        }}

        /* Container/Expander Spacing */
        div[data-testid="stExpander"] {{
            border-radius: 10px !important;
            border-color: {THEME_PALETTE["grid_color"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


def init_session_state() -> None:
    """Initialize default values in Streamlit session state.

    Creates the dashboard state variables required for uploaded data,
    prediction results, explainability artifacts, and user selections when
    they are not already present.
    """
    defaults: dict[str, Any] = {
        "done": False,
        "batch_id": None,
        "predictions_df": None,
        "predictions_path": None,
        "artifacts": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def run_pipeline(data_source: Path | str | Any) -> None:
    """Execute batch prediction and explainability workflows.

    Accepts either a local dataset path or an uploaded Streamlit file,
    submits the data for batch inference, retrieves the resulting
    predictions, and requests SHAP explainability artifacts.

    Parameters
    ----------
    data_source : pathlib.Path, str, or streamlit.runtime.uploaded_file_manager.UploadedFile
        Source customer dataset. May be a local filesystem path or a
        Streamlit uploaded file object.

    Raises
    ------
    requests.RequestException
        If communication with the prediction API fails.
    OSError
        If temporary files or downloaded prediction artifacts cannot be
        created.
    """
    tmp_path: Path | None = None
    local_path: Path | None = None

    try:
        with st.spinner("Executing model inference..."):
            # Step 1: Resolve input dataset path (local file vs uploaded buffer)
            if isinstance(data_source, (str, Path)):
                input_path = Path(data_source)
                if not input_path.exists():
                    st.error(f"Dataset file not found at: `{input_path}`")
                    return
            else:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp.write(data_source.getbuffer())
                    tmp_path = Path(tmp.name)
                input_path = tmp_path

            result = predict_batch(str(input_path))

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as prediction_tmp:
                local_path = Path(prediction_tmp.name)

            download_predictions(result["batch_id"], str(local_path))

            st.session_state.batch_id = result["batch_id"]
            st.session_state.predictions_df = pd.read_csv(local_path)
            st.session_state.predictions_path = str(local_path)

        # Step 2: Request SHAP explainability artifacts
        with st.spinner("Extracting feature attribution vectors (SHAP)..."):
            explain_response: dict = explain_batch(result["batch_id"])
            st.session_state.artifacts = load_artifacts(explain_response["artifacts_path"])

        st.session_state.done = True
        st.toast("Pipeline execution completed successfully!", icon="🚀")

    except requests.RequestException as e:
        st.error(f"Inference Pipeline Error: {e!s}")
        st.exception(e)
    finally:
        # Guarantee cleanup only for uploaded stream temporary files
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def render_kpis(risk_profiles: pd.DataFrame, prob_col: str) -> None:
    """Render executive churn-risk key performance indicators.

    Calculates and displays summary metrics derived from the customer risk
    profile dataset.

    Parameters
    ----------
    risk_profiles : pandas.DataFrame
        Customer-level prediction and risk profile data.
    prob_col : str
        Name of the column containing churn probabilities.
    """
    min_prob = 0.5
    total_customers = len(risk_profiles)
    avg_prob = float(risk_profiles[prob_col].mean()) * 100
    high_risk_count = int((risk_profiles[prob_col] >= min_prob).sum())
    high_risk_pct = float((risk_profiles[prob_col] >= min_prob).mean()) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.caption("TOTAL CUSTOMERS SCORED")
        st.metric("Total Cohort", f"{total_customers:,}")
    with c2, st.container(border=True):
        st.caption("AVERAGE RISK SCORE")
        st.metric("Avg. Churn Rate", f"{avg_prob:.1f}%")
    with c3, st.container(border=True):
        st.caption("HIGH RISK SEGMENT")
        st.metric("High-Risk Count", f"{high_risk_count:,}")
    with c4, st.container(border=True):
        st.caption("CRITICAL RISK PCT")
        st.metric("At Risk %", f"{high_risk_pct:.1f}%")


def render_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Render global feature importance as a horizontal bar chart.

    Parameters
    ----------
    feature_importance : pandas.DataFrame
        Feature importance table containing feature names and their global
        SHAP importance values.
    """
    st.markdown("### Global Churn Drivers")
    st.caption("Mean absolute SHAP impact across all customer cohorts")

    sorted_df = feature_importance.sort_values("mean_abs_shap", ascending=True)

    fig = px.bar(
        sorted_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        text_auto=".3f",
        labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": ""},
    )
    fig.update_traces(
        marker_color=THEME_PALETTE["primary"],
        marker_line_color=THEME_PALETTE["primary_hover"],
        marker_line_width=1,
        opacity=0.85,
        textposition="outside",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        margin={"l": 10, "r": 30, "t": 10, "b": 30},
        xaxis={"showgrid": True, "gridcolor": THEME_PALETTE["grid_color"], "zeroline": False},
        yaxis={"showgrid": False},
    )
    st.plotly_chart(fig, width="stretch")


def render_risk_distribution(risk_profiles: pd.DataFrame, prob_col: str) -> None:
    """Render the distribution of predicted churn probabilities.

    Displays a probability histogram and highlights the configured
    classification threshold used to distinguish higher-risk customers.

    Parameters
    ----------
    risk_profiles : pandas.DataFrame
        Customer-level risk profile dataset.
    prob_col : str
        Name of the churn probability column.
    """
    st.markdown("### Churn Risk Distribution")
    st.caption("Density distribution of predicted probability values")

    fig = px.histogram(
        risk_profiles,
        x=prob_col,
        nbins=30,
        labels={prob_col: "Predicted Churn Probability"},
    )
    fig.update_traces(
        marker_color="#818CF8",
        marker_line_color=THEME_PALETTE["primary"],
        marker_line_width=1,
        opacity=0.75,
    )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color=THEME_PALETTE["risk_high"],
        line_width=2,
        annotation_text="Critical Threshold (0.50)",
        annotation_position="top right",
        annotation_font_color=THEME_PALETTE["risk_high"],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        xaxis={"showgrid": True, "gridcolor": THEME_PALETTE["grid_color"], "tickformat": ".0%"},
        yaxis={
            "showgrid": True,
            "gridcolor": THEME_PALETTE["grid_color"],
            "title": "Customer Count",
        },
    )
    st.plotly_chart(fig, width="stretch")


def render_customer_table(
    risk_profiles: pd.DataFrame,
    id_col: str,
    prob_col: str,
) -> int:
    """Render an interactive customer risk table.

    Displays customer identifiers, churn probabilities, and risk information
    using the dashboard's interactive data-grid presentation.

    Parameters
    ----------
    risk_profiles : pandas.DataFrame
        Customer-level prediction and explainability data.
    id_col : str
        Name of the customer identifier column.
    prob_col : str
        Name of the churn probability column.
    """
    st.markdown("### Customer Cohort Risk Table")

    sorted_profiles = (
        risk_profiles[[id_col, prob_col]]
        .sort_values(prob_col, ascending=False)
        .reset_index(drop=True)
    )

    # Render formatted interactive table
    st.dataframe(
        sorted_profiles,
        width="stretch",
        height=320,
        column_config={
            id_col: st.column_config.TextColumn("Customer ID", help="Unique Identifier"),
            prob_col: st.column_config.ProgressColumn(
                "Churn Probability",
                help="Model confidence output (0 to 1)",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        hide_index=True,
    )

    # Dropdown selector for customer detailed inspection
    customer_list = sorted_profiles[id_col].tolist()
    return st.selectbox("Select Customer for Deep-Dive Analysis:", options=customer_list)


def render_customer_detail(risk_profiles: pd.DataFrame, customer_id: int) -> None:
    """Render SHAP-based risk details for a selected customer.

    Displays the customer's strongest churn drivers and retention factors
    using a SHAP contribution visualization and human-readable feature values.

    Parameters
    ----------
    risk_profiles : pandas.DataFrame
        Customer-level SHAP risk profiles.
    customer_id : int
        Identifier of the customer whose explanation should be displayed.
    """
    st.markdown(f"### Diagnostic Analysis: Customer `{customer_id}`")

    breakdown = customer_shap_breakdown(risk_profiles, customer_id)
    if breakdown.empty:
        st.warning("⚠️ No detailed feature attribution drivers found for this customer.")
        return

    # Assign directional risk colors
    colors = [
        THEME_PALETTE["risk_high"] if v > 0 else THEME_PALETTE["risk_low"]
        for v in breakdown["shap_value"]
    ]

    fig = go.Figure(
        go.Bar(
            x=breakdown["shap_value"],
            y=breakdown["feature"],
            orientation="h",
            marker_color=colors,
            customdata=breakdown["display_value"],
            hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>SHAP Impact: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        margin={"l": 10, "r": 20, "t": 10, "b": 30},
        xaxis={
            "title": "SHAP Value (Impact on Churn Probability)",
            "showgrid": True,
            "gridcolor": THEME_PALETTE["grid_color"],
            "zeroline": True,
            "zerolinecolor": "#94A3B8",
            "zerolinewidth": 1.5,
        },
        yaxis={"showgrid": False, "title": ""},
    )

    c_chart, c_recs = st.columns([1.6, 1.0])

    with c_chart:
        st.plotly_chart(fig, width="stretch")
        st.caption("🔴 Red: Increases Churn Risk | 🟢 Green: Decreases Churn Risk")

    with c_recs:
        st.markdown("#### Automated Retention Prescriptions")
        positive_drivers = (
            breakdown[breakdown["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(3)
        )

        if positive_drivers.empty:
            st.success("✅ This customer exhibits low risk across all primary operational drivers.")
        else:
            for row in positive_drivers.itertuples():
                with st.container(border=True):
                    st.markdown(f"**Trigger:** `{row.feature}`")
                    st.write(f"💡 {recommend_for_feature(row.feature)}")


def main() -> None:
    """Run the Churn Prediction dashboard.

    Initializes the Streamlit session, loads available model artifacts,
    presents the batch-upload interface, and renders prediction,
    explainability, and customer-level risk analysis views.
    """
    settings = get_settings()

    # Top Hero Section (White Card Styling)
    st.markdown(
        """
        <div class="hero-banner">
            <span class="hero-badge">Production v0.1.0 • Active Inference</span>
            <h1 class="hero-title">Customer Churn Analytics & Intelligence</h1>
            <p class="hero-subtitle">
                Upload batch customer cohorts to execute machine learning inference, identify key retention risk vectors, and receive targeted engagement prescriptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File Upload & Data Ingestion Block
    with st.container(border=True):
        st.subheader("1. Ingest Batch Data")

        col_upload, col_divider, col_demo = st.columns([1, 0.1, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Customer Dataset (CSV Format)",
                type=["csv"],
                help="Maximum recommended batch size: 50,000 records.",
            )

        with col_divider:
            st.markdown(
                "<div style='text-align: center; margin-top: 40px; color: #94A3B8;'><b>OR</b></div>",
                unsafe_allow_html=True,
            )

        with col_demo:
            st.markdown("##### Quick Demo")
            st.caption("Run pipeline using locally saved test dataset.")
            demo_path = settings.RAW_DATA_DIR / "test.csv"

            if st.button("Demo Dataset"):
                run_pipeline(demo_path)

        if uploaded_file is not None:
            # Efficient head preview without exhausting stream
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)

            with st.expander("Raw Dataset Schema Preview", expanded=True):
                st.dataframe(df_preview, width="stretch")

            if st.button("Execute Prediction & Explainability Pipeline", type="primary"):
                run_pipeline(uploaded_file)

    # Dashboard Rendering Section
    if st.session_state.get("done"):
        st.markdown("---")
        st.subheader("2. Prediction & Insights Dashboard")

        # Download & Batch Action Bar
        col_dl, col_space = st.columns([1, 3])
        _ = col_space
        with col_dl:
            st.download_button(
                label="Download Full Predictions CSV",
                data=Path(st.session_state.predictions_path).read_bytes(),
                file_name=f"churn_predictions_batch_{st.session_state.batch_id}.csv",
                mime="text/csv",
                width="stretch",
            )

        artifacts = st.session_state.artifacts
        risk_profiles = artifacts["risk_profiles"]
        id_col = artifacts["id_col"]
        prob_col = artifacts["prob_col"]

        # Render Core KPIs
        render_kpis(risk_profiles, prob_col)

        # Global Charts Layout
        g_col1, g_col2 = st.columns(2)
        with g_col1, st.container(border=True):
            render_feature_importance(artifacts["feature_importance"])
        with g_col2, st.container(border=True):
            render_risk_distribution(risk_profiles, prob_col)

        # Individual Customer Drilldown Layout
        with st.container(border=True):
            selected_customer = render_customer_table(risk_profiles, id_col, prob_col)
            st.markdown("---")
            render_customer_detail(risk_profiles, selected_customer)


if __name__ == "__main__":
    main()
