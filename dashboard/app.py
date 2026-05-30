"""Streamlit Monitoring Dashboard — calls FastAPI backend for data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.config import get_config

# ── Config ──────────────────────────────────────────────────────────
cfg = get_config()
API_URL = cfg["dashboard"]["api_url"]
REFRESH = cfg["dashboard"]["refresh_interval"]

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)


# ── Helper functions ────────────────────────────────────────────────
def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, json_data: dict | None = None) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=json_data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview", "Predict", "Live Monitor", "Model Performance"])

# Health check
health = api_get("/health")
if health:
    status_color = "🟢" if health["status"] == "healthy" else "🔴"
    st.sidebar.markdown(f"{status_color} API: **{health['status']}**")
    st.sidebar.markdown(f"Model: **{'Loaded' if health['model_loaded'] else 'Not loaded'}**")
    st.sidebar.markdown(f"Dask: **{health['dask_status']}**")
else:
    st.sidebar.markdown("🔴 API: **Unreachable**")

st.sidebar.markdown("---")
st.sidebar.markdown(f"API: `{API_URL}`")


# ── Page: Overview ──────────────────────────────────────────────────
if page == "Overview":
    st.title("🏭 Predictive Maintenance Dashboard")
    st.markdown("Real-time equipment failure prediction and monitoring system.")

    col1, col2, col3, col4 = st.columns(4)

    status = api_get("/monitor/status")
    if status:
        col1.metric("Total Predictions", f"{status['total_predictions']:,}")
        col2.metric("Failure Rate", f"{status['failure_rate']:.2%}")
        col3.metric("Avg Probability", f"{status['avg_probability']:.4f}")
        col4.metric("Total Alerts", status["total_alerts"])

        # Recent alerts table
        if status.get("recent_alerts"):
            st.subheader("🚨 Recent Alerts")
            alerts_df = pd.DataFrame(status["recent_alerts"])
            st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("Start the streaming pipeline to see metrics.")


# ── Page: Predict ───────────────────────────────────────────────────
elif page == "Predict":
    st.title("🔮 Equipment Failure Prediction")
    st.markdown("Enter sensor readings to predict failure probability.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.number_input("Air Temperature [K]", value=300.0, min_value=250.0, max_value=350.0)
            process_temp = st.number_input("Process Temperature [K]", value=310.0, min_value=280.0, max_value=370.0)
            rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500, min_value=500, max_value=3000)
        with col2:
            torque = st.number_input("Torque [Nm]", value=40.0, min_value=0.0, max_value=100.0)
            tool_wear = st.number_input("Tool Wear [min]", value=100, min_value=0, max_value=300)
            product_type = st.selectbox("Product Type", ["L", "M", "H"])

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        payload = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": product_type,
        }
        result = api_post("/predict", payload)
        if result:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Failure Probability", f"{result['failure_probability']:.2%}")
            c2.metric("Risk Level", result["risk_level"].upper())
            c3.metric("Predicted Failure", "⚠️ YES" if result["predicted_failure"] else "✅ NO")

            if result.get("contributing_factors"):
                st.subheader("Top Contributing Factors")
                for f in result["contributing_factors"]:
                    st.markdown(f"- `{f}`")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["failure_probability"] * 100,
                title={"text": "Failure Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 50], "color": "#d4edda"},
                        {"range": [50, 70], "color": "#fff3cd"},
                        {"range": [70, 90], "color": "#f8d7da"},
                        {"range": [90, 100], "color": "#721c24"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Live Monitor ──────────────────────────────────────────────
elif page == "Live Monitor":
    st.title("📡 Real-Time Monitoring")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start Stream", use_container_width=True):
            api_post("/monitor/start")
            st.success("Streaming started!")
    with col2:
        if st.button("⏹️ Stop Stream", use_container_width=True):
            api_post("/monitor/stop")
            st.info("Streaming stopped.")
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    st.markdown("---")

    # Live metrics
    placeholder = st.empty()

    status = api_get("/monitor/status")
    if status:
        with placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predictions", f"{status['total_predictions']:,}")
            m2.metric("Failure Rate", f"{status['failure_rate']:.2%}")
            m3.metric("Avg Risk", f"{status['avg_probability']:.4f}")
            m4.metric("Alerts", status["total_alerts"])

            if status.get("recent_alerts"):
                st.subheader("Recent Alerts")
                for alert in status["recent_alerts"][-5:]:
                    severity = alert.get("severity", "unknown")
                    icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(severity, "⚪")
                    st.markdown(f"{icon} **{alert.get('message', '')}**")

    if auto_refresh:
        time.sleep(REFRESH)
        st.rerun()


# ── Page: Model Performance ─────────────────────────────────────────
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Visualizations of trained model metrics.")

    figures_dir = Path(cfg["paths"]["reports"]) / "figures"

    if figures_dir.exists():
        images = sorted(figures_dir.glob("*.png"))
        if images:
            for img in images:
                st.image(str(img), caption=img.stem.replace("_", " ").title(),
                         use_container_width=True)
        else:
            st.info("No performance figures found. Run the training pipeline first.")
    else:
        st.info("Reports directory not found. Run the training pipeline first.")
