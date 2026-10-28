"""FireScope — Professional Fire Analytics & Geospatial Intelligence Platform

High-performance Streamlit application for exploratory data analysis, active fire monitoring,
geospatial intelligence, predictive machine learning, and automated intelligence reporting
based on NASA FIRMS satellite observation datasets.
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.config.settings import Config, PROJECT_ROOT
from app.ml.trainer import (
    train_classification,
    train_clustering,
    train_regression,
)
from app.services.alert_service import (
    create_rule,
    delete_rule,
    evaluate_rules,
    list_rules,
    toggle_rule,
)
from app.services.analytics_service import AnalyticsService
from app.services.dataset_service import (
    DatasetService,
    DatasetUnavailable,
    get_dataset,
    list_available_datasets,
    set_active_dataset,
)

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FireScope — Fire Analytics & Monitoring Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS & Global Typography ───────────────────────────────────────────
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

<style>
/* ─── Global Variables & Reset ───────────────────────────────────────── */
:root {
    --bg-main: #070a12;
    --card-surface: #0e1526;
    --card-hover: #131c33;
    --border-subtle: #1c2742;
    --border-glow: #00e5ff;
    
    --accent-cyan: #00e5ff;
    --accent-flame: #ff5722;
    --accent-amber: #ffb300;
    --accent-emerald: #10b981;
    --accent-purple: #a855f7;
    --accent-rose: #f43f5e;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6, strong {
    font-family: 'Outfit', sans-serif !important;
}

/* Hide default streamlit chrome */
header[data-testid="stHeader"] {
    background-color: rgba(7, 10, 18, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(28, 39, 66, 0.6);
}

/* ─── Sidebar Styling ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #090e1c !important;
    border-right: 1px solid #162038 !important;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 4px 18px 4px;
    margin-bottom: 16px;
    border-bottom: 1px solid #162038;
}

.brand-icon-box {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #ff5722 0%, #ff9800 100%);
    display: grid;
    place-items: center;
    font-size: 22px;
    color: #ffffff;
    box-shadow: 0 4px 20px rgba(255, 87, 34, 0.45);
    flex-shrink: 0;
}

.brand-text-block h2 {
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.brand-text-block span {
    font-size: 0.75rem;
    color: #00e5ff;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* User Profile Badge in Sidebar */
.user-profile-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: rgba(14, 21, 38, 0.7);
    border: 1px solid #1c2742;
    border-radius: 12px;
    margin-top: 24px;
}
.user-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00e5ff, #3b82f6);
    color: #070a12;
    font-weight: 800;
    display: grid;
    place-items: center;
    font-size: 13px;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.3);
}
.user-info-text strong {
    display: block;
    font-size: 0.88rem;
    color: #f8fafc;
    line-height: 1.2;
}
.user-info-text span {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    color: #10b981;
    font-weight: 600;
}

/* ─── Hero Command Center Banner ──────────────────────────────────────── */
.command-hero {
    position: relative;
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(14, 21, 38, 0.95) 0%, rgba(9, 14, 28, 0.9) 100%),
                radial-gradient(ellipse at 85% 20%, rgba(255, 87, 34, 0.25) 0%, transparent 60%),
                radial-gradient(ellipse at 15% 85%, rgba(0, 229, 255, 0.15) 0%, transparent 50%);
    border: 1px solid #1c2742;
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.6);
    overflow: hidden;
}

.command-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff5722 0%, #00e5ff 50%, #10b981 100%);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.35);
    color: #00e5ff;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 5px 12px;
    border-radius: 30px;
    margin-bottom: 14px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-ring 2s infinite ease-in-out;
}

@keyframes pulse-ring {
    0% { transform: scale(0.9); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.9); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.hero-headline {
    font-size: 2.35rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 10px 0;
    color: #ffffff;
}

.hero-gradient-text {
    background: linear-gradient(90deg, #ff5722 0%, #ff9800 50%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtext {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 720px;
    line-height: 1.55;
    margin: 0;
}

/* ─── Telemetry Status Ribbon ─────────────────────────────────────────── */
.telemetry-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: rgba(14, 21, 38, 0.75);
    border: 1px solid #1c2742;
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 22px;
    font-size: 0.82rem;
    color: #94a3b8;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}

.telemetry-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.telemetry-item strong {
    color: #f8fafc;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Modern KPI Card Grid ────────────────────────────────────────────── */
.kpi-card {
    background: var(--card-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    margin-bottom: 12px;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 229, 255, 0.5);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45), 0 0 20px rgba(0, 229, 255, 0.12);
}

.kpi-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.kpi-icon-container {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 18px;
}

.kpi-card.flame .kpi-icon-container { background: rgba(255, 87, 34, 0.15); color: #ff5722; border: 1px solid rgba(255, 87, 34, 0.3); }
.kpi-card.amber .kpi-icon-container { background: rgba(255, 179, 0, 0.15); color: #ffb300; border: 1px solid rgba(255, 179, 0, 0.3); }
.kpi-card.rose .kpi-icon-container  { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }
.kpi-card.emerald .kpi-icon-container { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.kpi-card.cyan .kpi-icon-container { background: rgba(0, 229, 255, 0.15); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.3); }
.kpi-card.purple .kpi-icon-container { background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }

.kpi-card-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.kpi-card-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.kpi-card-detail {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
}

/* ─── Insight Cards ───────────────────────────────────────────────────── */
.insight-box {
    background: #0e1526;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border: 1px solid #1c2742;
    border-left: 4px solid #00e5ff;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.insight-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.insight-box.danger {
    border-left-color: #ff5722;
    background: linear-gradient(90deg, rgba(255, 87, 34, 0.08) 0%, rgba(14, 21, 38, 0.95) 100%);
}
.insight-box.warning {
    border-left-color: #ffb300;
    background: linear-gradient(90deg, rgba(255, 179, 0, 0.08) 0%, rgba(14, 21, 38, 0.95) 100%);
}
.insight-box-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #ffffff;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.insight-box-body {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ─── Interactive Predictor Surface ───────────────────────────────────── */
.predictor-box {
    background: linear-gradient(135deg, rgba(14, 21, 38, 0.95) 0%, rgba(19, 28, 51, 0.85) 100%);
    border: 1px solid #1c2742;
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.pred-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    margin-top: 8px;
}
.pred-badge.low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
.pred-badge.moderate { background: rgba(255, 179, 0, 0.2); color: #ffb300; border: 1px solid rgba(255, 179, 0, 0.4); }
.pred-badge.severe { background: rgba(255, 87, 34, 0.2); color: #ff5722; border: 1px solid rgba(255, 87, 34, 0.4); }
.pred-badge.extreme { background: rgba(244, 63, 94, 0.25); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.5); }

/* ─── Form & Input Controls ───────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #ff5722 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 16px rgba(255, 87, 34, 0.35) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255, 87, 34, 0.5) !important;
}

div.stDownloadButton > button {
    background: linear-gradient(135deg, #00e5ff 0%, #0284c7 100%) !important;
    color: #070a12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 16px rgba(0, 229, 255, 0.35) !important;
}

/* Modern Tabs */
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #94a3b8 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Plotly Theme Helper ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#162038", zerolinecolor="#1c2742"),
    yaxis=dict(gridcolor="#162038", zerolinecolor="#1c2742"),
    margin=dict(l=40, r=20, t=35, b=35),
)


def style_plot(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(color="#f8fafc", size=15, family="Outfit")),
    )
    return fig


# ─── Dataset Caching ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cached_dataset(path_str: str) -> DatasetService:
    """Load and cache the DatasetService instance in memory."""
    return get_dataset(path_str)


# ─── Session State Initialization ─────────────────────────────────────────────
if "active_dataset_path" not in st.session_state:
    try:
        ds = get_dataset()
        st.session_state.active_dataset_path = str(ds.path.resolve())
    except Exception:
        st.session_state.active_dataset_path = ""

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "filter_reset_trigger" not in st.session_state:
    st.session_state.filter_reset_trigger = 0

if "quick_preset" not in st.session_state:
    st.session_state.quick_preset = "All"


# ─── Sidebar Header & Navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrapper">
            <div class="brand-icon-box">
                <i class="fa-solid fa-fire-flame-curved"></i>
            </div>
            <div class="brand-text-block">
                <h2>FireScope</h2>
                <span>Safer Tomorrows</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Dataset Selector
    available_datasets = list_available_datasets()
    if available_datasets:
        options = [d["name"] for d in available_datasets]
        current_name = Path(st.session_state.active_dataset_path).name
        default_idx = options.index(current_name) if current_name in options else 0
        selected_dataset_name = st.selectbox(
            "📁 Active Dataset",
            options=options,
            index=default_idx,
            help="Switch active dataset for all views and machine learning.",
        )
        selected_ds_info = next(d for d in available_datasets if d["name"] == selected_dataset_name)
        if selected_ds_info["path"] != st.session_state.active_dataset_path:
            st.session_state.active_dataset_path = selected_ds_info["path"]
            set_active_dataset(selected_ds_info["path"])
            st.rerun()
    else:
        st.warning("No datasets found in data/raw.")

    st.markdown("---")

    # Main Navigation
    nav_options = [
        "📊 Executive Overview",
        "🗺️ Geospatial Map Explorer",
        "⏱️ Temporal Analysis",
        "⚡ Fire Intensity (FRP)",
        "🛡️ Confidence Analysis",
        "🛰️ Satellite & Sensors",
        "🌐 Geographic Insights",
        "📋 Data Overview & Quality",
        "💾 Export & Download",
        "🧠 Machine Learning Engine",
        "🔔 Alert Rules",
        "ℹ️ Dataset & Platform",
    ]
    selected_view = st.radio("Platform Navigation", nav_options, index=0)


# ─── Load Active Dataset & Setup Analytics ────────────────────────────────────
try:
    dataset = load_cached_dataset(st.session_state.active_dataset_path)
    analytics = AnalyticsService(dataset)
    frame = dataset.frame
    detected = dataset.columns
except DatasetUnavailable as err:
    st.error(f"⚠️ Dataset error: {err}")
    st.info("Please upload a CSV or Excel fire dataset in the 'Dataset & Platform' tab.")
    st.stop()
except Exception as err:
    st.error(f"⚠️ Failed to load dataset: {err}")
    st.stop()


# ─── Sidebar Global Filters ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")

    # Quick Filter Presets
    st.markdown("##### ⚡ Quick Filter Presets")
    qp_cols = st.columns(3)
    with qp_cols[0]:
        if st.button("All", use_container_width=True):
            st.session_state.quick_preset = "All"
            st.session_state.filter_reset_trigger += 1
            st.rerun()
    with qp_cols[1]:
        if st.button("🔥 Mega", use_container_width=True):
            st.session_state.quick_preset = "Mega"
            st.rerun()
    with qp_cols[2]:
        if st.button("🛡️ Conf", use_container_width=True):
            st.session_state.quick_preset = "HighConf"
            st.rerun()

    with st.expander("🔍 Filter Controls & Search", expanded=True):
        filter_params: dict = {}

        # Apply Quick Preset overrides
        if st.session_state.quick_preset == "Mega":
            filter_params["frp_min"] = 50.0
        elif st.session_state.quick_preset == "HighConf":
            conf_c = detected.get("confidence")
            if conf_c and frame[conf_c].notna().any():
                filter_params["confidence"] = ["high", "h", "High"]

        # Keyword Search
        search_query = st.text_input(
            "Keyword Search",
            placeholder="Search satellite, region...",
            key=f"search_{st.session_state.filter_reset_trigger}",
        )
        if search_query:
            filter_params["search"] = search_query

        # Date Range Filter
        date_col = detected.get("date")
        if date_col and frame[date_col].notna().any():
            min_date = frame[date_col].min().date()
            max_date = frame[date_col].max().date()
            date_range = st.date_input(
                "Acquisition Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"dates_{st.session_state.filter_reset_trigger}",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                filter_params["date_start"] = date_range[0].isoformat()
                filter_params["date_end"] = date_range[1].isoformat()

        # Source / Satellite Filter
        source_col = detected.get("source")
        if source_col and frame[source_col].notna().any():
            source_opts = sorted(str(v) for v in frame[source_col].dropna().unique())
            selected_sources = st.multiselect(
                "Satellite Platform",
                options=source_opts,
                default=[],
                key=f"source_{st.session_state.filter_reset_trigger}",
            )
            if selected_sources:
                filter_params["source"] = selected_sources

        # Confidence Filter
        conf_col = detected.get("confidence")
        if conf_col and frame[conf_col].notna().any():
            conf_opts = sorted(str(v) for v in frame[conf_col].dropna().unique())
            selected_confs = st.multiselect(
                "Confidence Tier",
                options=conf_opts,
                default=[],
                key=f"conf_{st.session_state.filter_reset_trigger}",
            )
            if selected_confs:
                filter_params["confidence"] = selected_confs

        # Day / Night Filter
        daynight_col = detected.get("daynight")
        if daynight_col and frame[daynight_col].notna().any():
            daynight_opts = ["All"] + sorted(str(v) for v in frame[daynight_col].dropna().unique())
            sel_daynight = st.selectbox(
                "Solar Elevation (Day/Night)",
                options=daynight_opts,
                index=0,
                key=f"dn_{st.session_state.filter_reset_trigger}",
            )
            if sel_daynight != "All":
                filter_params["daynight"] = sel_daynight

        # FRP Range Slider
        frp_col = detected.get("frp")
        if frp_col and frame[frp_col].notna().any():
            min_frp = float(frame[frp_col].min())
            max_frp = float(frame[frp_col].max())
            if max_frp > min_frp:
                step = 0.5 if (max_frp - min_frp) < 100 else 1.0
                frp_init = 50.0 if st.session_state.quick_preset == "Mega" else min_frp
                frp_range = st.slider(
                    "FRP Range (MW)",
                    min_value=min_frp,
                    max_value=max_frp,
                    value=(frp_init, max_frp),
                    step=step,
                    key=f"frp_{st.session_state.filter_reset_trigger}",
                )
                if frp_range[0] > min_frp:
                    filter_params["frp_min"] = frp_range[0]
                if frp_range[1] < max_frp:
                    filter_params["frp_max"] = frp_range[1]

        # Reset Filters Button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.quick_preset = "All"
            st.session_state.filter_reset_trigger += 1
            st.rerun()

        active_filter_count = len(filter_params)
        st.caption(f"Status: **{active_filter_count}** active filters applied")

    # User Profile Widget in Sidebar Footer
    st.markdown(
        """
        <div class="user-profile-badge">
            <div class="user-avatar">NR</div>
            <div class="user-info-text">
                <strong>Nahin Rahman</strong>
                <span><i class="fa-solid fa-circle" style="font-size:7px;"></i> Platform Admin</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Filter Dataset ───────────────────────────────────────────────────────────
if filter_params:
    filtered_frame = dataset.filter(filter_params)
else:
    filtered_frame = frame


# ─── Helper: Render Telemetry Ribbon ──────────────────────────────────────────
def render_telemetry_ribbon(df_sub: pd.DataFrame, df_full: pd.DataFrame, detected_cols: dict):
    total = len(df_full)
    current = len(df_sub)
    pct = (current / total * 100) if total > 0 else 100
    lat_c = detected_cols.get("latitude")
    lon_c = detected_cols.get("longitude")
    if lat_c and lon_c and not df_sub.empty:
        lat_min, lat_max = df_sub[lat_c].min(), df_sub[lat_c].max()
        lon_min, lon_max = df_sub[lon_c].min(), df_sub[lon_c].max()
        bounds_str = f"[{lat_min:.2f}°, {lat_max:.2f}°] × [{lon_min:.2f}°, {lon_max:.2f}°]"
    else:
        bounds_str = "Global Scope"

    st.markdown(
        f"""
        <div class="telemetry-bar">
            <div class="telemetry-item"><span class="pulse-dot"></span> <span style="color:#10b981; font-weight:700;">SYSTEM OPERATIONAL</span></div>
            <div class="telemetry-item"><i class="fa-solid fa-satellite" style="color:#00e5ff;"></i> <strong>RECORDS:</strong> {current:,} / {total:,} ({pct:.1f}%)</div>
            <div class="telemetry-item"><i class="fa-solid fa-earth-americas" style="color:#ffb300;"></i> <strong>BOUNDS:</strong> {bounds_str}</div>
            <div class="telemetry-item"><i class="fa-solid fa-shield-halved" style="color:#a855f7;"></i> <strong>TELEMETRY:</strong> NASA FIRMS Active Fire Stream</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Helper: Render High-End KPI Row ──────────────────────────────────────────
ICON_MAP = {
    "Fire detections": ("fa-solid fa-fire", "flame"),
    "Average FRP": ("fa-solid fa-bolt", "amber"),
    "Maximum FRP": ("fa-solid fa-arrow-trend-up", "rose"),
    "High confidence": ("fa-solid fa-shield-halved", "emerald"),
    "Unique locations": ("fa-solid fa-location-dot", "cyan"),
    "Active sources": ("fa-solid fa-satellite", "purple"),
}


def render_kpis(kpis: list[dict]):
    if not kpis:
        return
    cols = st.columns(len(kpis))
    for idx, (col, kpi) in enumerate(zip(cols, kpis)):
        label = kpi.get("label", "")
        icon_class, color_class = ICON_MAP.get(label, ("fa-solid fa-chart-simple", "cyan"))
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {color_class}">
                    <div class="kpi-card-header">
                        <div class="kpi-icon-container">
                            <i class="{icon_class}"></i>
                        </div>
                    </div>
                    <div class="kpi-card-label">{label}</div>
                    <div class="kpi-card-value">{kpi.get('value', '-')}</div>
                    <div class="kpi-card-detail">{kpi.get('detail', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 1: EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if selected_view == "📊 Executive Overview":
    # Hero Command Center Banner
    st.markdown(
        """
        <div class="command-hero">
            <div class="hero-pill"><span class="pulse-dot"></span> SATELLITE TELEMETRY • LIVE STREAM CONNECTED</div>
            <h1 class="hero-headline">Turn Data Into <span class="hero-gradient-text">Safer Tomorrows</span></h1>
            <p class="hero-subtext">Advanced exploratory fire monitoring, thermal anomaly analysis, and predictive AI powered by NASA FIRMS satellite observations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Firing Alerts Notice
    firing_alerts = evaluate_rules(dataset)
    if firing_alerts:
        st.warning(
            f"🔔 **{len(firing_alerts)} Active Fire Alert(s) Currently Firing!** "
            f"Rule **{firing_alerts[0]['name']}** matched {firing_alerts[0]['matched_count']:,} records. "
            f"Inspect or configure notification thresholds under **Alert Rules**."
        )

    # Telemetry Status Ribbon
    render_telemetry_ribbon(filtered_frame, frame, detected)

    # Top KPI Metrics
    summary_data = analytics.summary(filtered_frame)
    render_kpis(summary_data["kpis"])

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Dual-Axis Time-Series Trend: Fire Detections (Area) + Mean FRP (Line)
    ts_header_col1, ts_header_col2 = st.columns([3, 1])
    with ts_header_col1:
        st.markdown("### 📈 Fire Activity & Radiative Energy Timeline")
    with ts_header_col2:
        period_choice = st.selectbox("Aggregation", ["day", "week", "month"], index=0)

    ts_data = analytics.time_series(filtered_frame, period=period_choice)

    if ts_data.get("available") and ts_data.get("series"):
        df_ts = pd.DataFrame(ts_data["series"])
        freq_map = {"day": "D", "week": "W-MON", "month": "MS"}

        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        # Area fill for Detections count
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["date"],
                y=df_ts["count"],
                mode="lines",
                name="Fire Detections",
                fill="tozeroy",
                fillcolor="rgba(0, 229, 255, 0.12)",
                line=dict(color="#00e5ff", width=2.2),
            ),
            secondary_y=False,
        )

        # Overlay Mean FRP line on secondary y-axis if FRP and Date are detected
        date_c = detected.get("date")
        frp_c = detected.get("frp")
        if date_c and frp_c and filtered_frame[frp_c].notna().any():
            frp_ts = (
                filtered_frame.groupby(pd.Grouper(key=date_c, freq=freq_map[period_choice]))[frp_c]
                .mean()
                .dropna()
                .reset_index()
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=frp_ts[date_c].dt.strftime("%Y-%m-%d"),
                    y=frp_ts[frp_c],
                    mode="lines",
                    name="Mean FRP (MW)",
                    line=dict(color="#ff5722", width=2.4, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_ts.update_yaxes(title_text="Detections Count", secondary_y=False, gridcolor="#162038")
            fig_ts.update_yaxes(title_text="Mean Radiative Power (MW)", secondary_y=True, showgrid=False)

        # 7-day rolling trendline if daily
        if period_choice == "day" and len(df_ts) >= 7:
            df_ts["rolling_7d"] = df_ts["count"].rolling(window=7, min_periods=1).mean()
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts["date"],
                    y=df_ts["rolling_7d"],
                    mode="lines",
                    name="7-Day Moving Avg",
                    line=dict(color="#ffb300", width=2, dash="dash"),
                ),
                secondary_y=False,
            )

        style_plot(fig_ts, f"Fire Detections & Radiative Power Timeline ({period_choice.capitalize()})", height=340)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Multi-column visualizations
    c1, c2, c3 = st.columns(3)

    # 1. Fire Intensity Distribution
    with c1:
        frp_info = analytics.intensity(filtered_frame)
        if frp_info.get("available") and frp_info.get("histogram"):
            hist_df = pd.DataFrame(frp_info["histogram"])
            fig_frp = px.bar(
                hist_df,
                x="x",
                y="y",
                labels={"x": "FRP (MW)", "y": "Detections"},
                title="Fire Intensity Distribution (FRP MW)",
                color_discrete_sequence=["#ff5722"],
            )
            style_plot(fig_frp, "FRP Intensity Distribution", height=290)
            st.plotly_chart(fig_frp, use_container_width=True)
        else:
            st.info("No FRP data available.")

    # 2. Multi-Dimensional Sunburst Hierarchy (Satellite -> Day/Night -> Confidence)
    with c2:
        src_c = detected.get("source")
        dn_c = detected.get("daynight")
        cnf_c = detected.get("confidence")
        valid_hier_cols = [c for c in [src_c, dn_c, cnf_c] if c and c in filtered_frame.columns]

        if len(valid_hier_cols) >= 2:
            hier_sample = filtered_frame[valid_hier_cols].dropna()
            if len(hier_sample) > 8000:
                hier_sample = hier_sample.sample(8000, random_state=42)
            fig_hier = px.sunburst(
                hier_sample,
                path=valid_hier_cols,
                title="Sensor Telemetry Hierarchy",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            style_plot(fig_hier, "Platform → Solar Cycle Breakdown", height=290)
            st.plotly_chart(fig_hier, use_container_width=True)
        else:
            sat_info = analytics.satellites(filtered_frame)
            if sat_info.get("available") and sat_info.get("platforms"):
                sat_df = pd.DataFrame(sat_info["platforms"])
                fig_sat = px.bar(
                    sat_df,
                    x="count",
                    y="name",
                    orientation="h",
                    labels={"count": "Observations", "name": "Platform"},
                    title="Observations by Satellite",
                    color_discrete_sequence=["#38bdf8"],
                )
                style_plot(fig_sat, "Satellite Platforms Breakdown", height=290)
                st.plotly_chart(fig_sat, use_container_width=True)
            else:
                st.info("No satellite data available.")

    # 3. Confidence Distribution Donut Chart
    with c3:
        conf_info = analytics.confidence_analysis(filtered_frame)
        if conf_info.get("available") and conf_info.get("items"):
            conf_df = pd.DataFrame(conf_info["items"])
            fig_conf = px.pie(
                conf_df,
                names="label",
                values="value",
                hole=0.6,
                color_discrete_sequence=["#10b981", "#00e5ff", "#ff9800"],
                title="Confidence Level Share",
            )
            style_plot(fig_conf, "Confidence Classification", height=290)
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No confidence data available.")

    # Actionable Insights Section
    st.markdown("### 💡 Actionable Insights & Anomalies")
    insights_list = analytics.insights(filtered_frame)
    if insights_list:
        ins_cols = st.columns(2)
        for idx, item in enumerate(insights_list[:6]):
            card_class = "insight-box"
            icon_kind = "fa-solid fa-lightbulb"
            if item.get("kind") in ("risk", "alert", "danger"):
                card_class += " danger"
                icon_kind = "fa-solid fa-triangle-exclamation"
            elif item.get("kind") in ("trend", "warning"):
                card_class += " warning"
                icon_kind = "fa-solid fa-arrow-trend-up"

            with ins_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="insight-box-title">
                            <i class="{icon_kind}"></i> {item.get('title', 'Observation')}
                        </div>
                        <div class="insight-box-body">{item.get('text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No automated insights detected for the currently filtered records.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 2: MAP EXPLORER (THERMAL DENSITY, PINPOINT MARKERS, SPATIAL CLUSTERS)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🗺️ Geospatial Map Explorer":
    st.markdown("## 🗺️ Geospatial Fire Activity Explorer")
    st.markdown("High-definition geospatial thermal monitoring with multi-layer density heatmaps, pinpoint hotspot coordinates, and K-Means cluster centers.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    lat_col = detected.get("latitude")
    lon_col = detected.get("longitude")

    if not lat_col or not lon_col:
        st.error("No latitude and longitude columns detected in the dataset.")
    else:
        valid_geo = filtered_frame.dropna(subset=[lat_col, lon_col])
        if valid_geo.empty:
            st.warning("No records with valid geographic coordinates match the current filters.")
        else:
            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
            with ctrl1:
                map_mode = st.selectbox(
                    "🗺️ Layer Mode",
                    ["🔥 Thermal Density Heatmap", "📍 Pinpoint Hotspot Markers", "🎯 Spatial Cluster Centers"],
                    index=0,
                )
            with ctrl2:
                sample_limit = st.slider(
                    "Point Sample Limit",
                    min_value=500,
                    max_value=min(40000, len(valid_geo)),
                    value=min(8000, len(valid_geo)),
                    step=500,
                    help="Sample points for fast rendering on high-density datasets.",
                )
            with ctrl3:
                map_style = st.selectbox(
                    "Basemap Style",
                    ["carto-darkmatter", "open-street-map", "carto-positron"],
                    index=0,
                )
            with ctrl4:
                if "Heatmap" in map_mode:
                    heat_radius = st.slider("Thermal Radius", min_value=5, max_value=40, value=18, step=2)
                else:
                    color_by_options = [c for c in [detected.get("frp"), detected.get("brightness"), detected.get("confidence"), detected.get("source"), detected.get("daynight")] if c]
                    color_by = st.selectbox("Color Points By", options=color_by_options, index=0 if color_by_options else None)

            # Sample records for rendering
            plot_df = valid_geo.sample(sample_limit, random_state=42) if len(valid_geo) > sample_limit else valid_geo
            hover_cols = [c for c in [detected.get("date"), detected.get("frp"), detected.get("brightness"), detected.get("confidence"), detected.get("source")] if c]

            fig_map = None
            if "Heatmap" in map_mode and hasattr(px, "density_map"):
                z_metric = detected.get("frp") or detected.get("brightness")
                fig_map = px.density_map(
                    plot_df,
                    lat=lat_col,
                    lon=lon_col,
                    z=z_metric,
                    radius=heat_radius,
                    color_continuous_scale="Inferno",
                    zoom=3,
                    map_style=map_style,
                    hover_data=hover_cols,
                    title=f"Thermal Fire Density Heatmap ({len(plot_df):,} Sampled Hotspots)",
                )
            elif "Cluster Centers" in map_mode:
                geo_clusters = analytics.geography(filtered_frame).get("clusters", [])
                if geo_clusters:
                    df_c_map = pd.DataFrame(geo_clusters)
                    fig_map = px.scatter_map(
                        df_c_map,
                        lat="lat",
                        lon="lon",
                        size="count",
                        color="avg_frp",
                        color_continuous_scale="Turbo",
                        size_max=35,
                        zoom=3,
                        hover_data=["label", "count", "avg_frp", "max_frp"],
                        map_style=map_style,
                        title=f"K-Means Spatial Hotspot Cluster Centers ({len(df_c_map)} Identified Hotspots)",
                    )
                else:
                    st.info("No spatial clusters identified in current selection.")
            else:
                # Scatter markers mode
                if hasattr(px, "scatter_map"):
                    fig_map = px.scatter_map(
                        plot_df,
                        lat=lat_col,
                        lon=lon_col,
                        color=color_by if color_by else None,
                        color_continuous_scale="Viridis" if color_by == detected.get("frp") else "Plasma",
                        size_max=9,
                        zoom=3,
                        hover_data=hover_cols,
                        map_style=map_style,
                        title=f"Active Fire Detections ({len(plot_df):,} Points Rendered)",
                    )
                elif hasattr(px, "scatter_mapbox"):
                    fig_map = px.scatter_mapbox(
                        plot_df,
                        lat=lat_col,
                        lon=lon_col,
                        color=color_by if color_by else None,
                        color_continuous_scale="Viridis" if color_by == detected.get("frp") else "Plasma",
                        size_max=9,
                        zoom=3,
                        hover_data=hover_cols,
                        mapbox_style=map_style,
                        title=f"Active Fire Detections ({len(plot_df):,} Points Rendered)",
                    )
                else:
                    fig_map = px.scatter_geo(
                        plot_df,
                        lat=lat_col,
                        lon=lon_col,
                        color=color_by if color_by else None,
                        hover_data=hover_cols,
                        title=f"Active Fire Detections ({len(plot_df):,} Points Rendered)",
                    )

            if fig_map:
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=640,
                )
                st.plotly_chart(fig_map, use_container_width=True)

            # Spatial Summary KPIs
            geo_info = analytics.geography(filtered_frame)
            if geo_info.get("kpis"):
                render_kpis(geo_info["kpis"])


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 3: TEMPORAL ANALYSIS (DIURNAL CYCLE & 7X24 MATRIX)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "⏱️ Temporal Analysis":
    st.markdown("## ⏱️ Temporal Trends & Fire Detection Patterns")
    st.markdown("Examine daily timelines, moving averages, 24-hour diurnal diurnal burning cycles, and weekly temporal activity matrices.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    date_col = detected.get("date")
    time_col = detected.get("time")

    if not date_col or filtered_frame[date_col].isna().all():
        st.warning("No date field detected or available in the filtered dataset.")
    else:
        top_c1, top_c2 = st.columns([3, 1])
        with top_c2:
            agg_period = st.selectbox("Granularity", ["day", "week", "month"], index=0)

        ts_res = analytics.time_series(filtered_frame, period=agg_period)
        if ts_res.get("available") and ts_res.get("series"):
            df_ts = pd.DataFrame(ts_res["series"])
            fig_timeline = go.Figure()
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_ts["date"],
                    y=df_ts["count"],
                    mode="lines",
                    name="Detections",
                    fill="tozeroy",
                    fillcolor="rgba(0, 229, 255, 0.12)",
                    line=dict(color="#00e5ff", width=2),
                )
            )
            if agg_period == "day" and len(df_ts) >= 7:
                df_ts["rolling"] = df_ts["count"].rolling(window=7, min_periods=1).mean()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=df_ts["date"],
                        y=df_ts["rolling"],
                        mode="lines",
                        name="7-Day Moving Avg",
                        line=dict(color="#ff5722", width=2.5),
                    )
                )
            style_plot(fig_timeline, f"Fire Detections Timeline ({agg_period.capitalize()})", height=350)
            st.plotly_chart(fig_timeline, use_container_width=True)

        # 24-Hour Diurnal Fire Cycle and Day of Week
        col_diurnal, col_w = st.columns(2)

        with col_diurnal:
            if time_col and filtered_frame[time_col].notna().any():
                hours_series = (
                    filtered_frame[time_col]
                    .astype(str)
                    .str.zfill(4)
                    .str[:2]
                    .str.extract(r"^(\d{1,2})")[0]
                    .dropna()
                    .astype(int)
                )
                hours_counts = hours_series.value_counts().reindex(range(24), fill_value=0).reset_index()
                hours_counts.columns = ["hour", "detections"]
                hours_counts["time_label"] = hours_counts["hour"].apply(lambda h: f"{h:02d}:00 UTC")

                fig_diurnal = px.bar(
                    hours_counts,
                    x="time_label",
                    y="detections",
                    title="24-Hour Diurnal Fire Cycle (UTC Observation Hours)",
                    color="detections",
                    color_continuous_scale="Inferno",
                )
                style_plot(fig_diurnal, "24-Hour Diurnal Fire Cycle (UTC)", height=320)
                st.plotly_chart(fig_diurnal, use_container_width=True)
            elif ts_res.get("months"):
                df_mo = pd.DataFrame(ts_res["months"])
                fig_mo = px.bar(
                    df_mo,
                    x="label",
                    y="value",
                    labels={"label": "Month", "value": "Detections"},
                    title="Monthly Detections",
                    color_discrete_sequence=["#a855f7"],
                )
                style_plot(fig_mo, "Monthly Detections", height=320)
                st.plotly_chart(fig_mo, use_container_width=True)

        with col_w:
            if ts_res.get("weekday"):
                df_wk = pd.DataFrame(ts_res["weekday"])
                fig_wk = px.bar(
                    df_wk,
                    x="label",
                    y="value",
                    labels={"label": "Day of Week", "value": "Detections"},
                    title="Detection Distribution by Day of Week",
                    color_discrete_sequence=["#38bdf8"],
                )
                style_plot(fig_wk, "Detections by Day of Week", height=320)
                st.plotly_chart(fig_wk, use_container_width=True)

        # 7×24 Temporal Activity Matrix Heatmap (Day of Week vs Hour of Day)
        if time_col and date_col and filtered_frame[time_col].notna().any() and filtered_frame[date_col].notna().any():
            st.markdown("### 🔥 7×24 Temporal Fire Activity Heatmap Matrix")
            st.caption("Cross-tabulation of burning activity across days of the week and UTC satellite overpass hours.")
            t_sample = filtered_frame.sample(min(40000, len(filtered_frame)), random_state=42) if len(filtered_frame) > 40000 else filtered_frame
            hours_s = pd.to_numeric(t_sample[time_col].astype(str).str.zfill(4).str[:2], errors="coerce").fillna(0).astype(int).clip(0, 23)
            weekdays_s = t_sample[date_col].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            matrix_df = pd.crosstab(weekdays_s, hours_s).reindex(index=day_order, columns=range(24), fill_value=0)
            fig_matrix = px.imshow(
                matrix_df,
                labels=dict(x="UTC Hour of Day", y="Day of Week", color="Detections"),
                x=[f"{h:02d}:00" for h in range(24)],
                y=day_order,
                color_continuous_scale="Inferno",
                title="7×24 Temporal Burning Intensity Matrix",
                aspect="auto",
            )
            style_plot(fig_matrix, "7×24 Temporal Activity Matrix", height=340)
            st.plotly_chart(fig_matrix, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 4: FIRE INTENSITY (UPGRADED WITH MARGINAL SCATTER & DAY/NIGHT VIOLIN)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "⚡ Fire Intensity (FRP)":
    st.markdown("## ⚡ Fire Radiative Power (FRP) & Intensity Analysis")
    st.markdown("Assess thermal energy release rates, catastrophic fire front potential, daytime vs nighttime burning behaviors, and marginal correlations.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    frp_col = detected.get("frp")
    bright_col = detected.get("brightness")
    daynight_col = detected.get("daynight")

    if not frp_col:
        st.warning("No Fire Radiative Power (FRP) column detected in the dataset.")
    else:
        frp_data = analytics.intensity(filtered_frame)
        if frp_data.get("kpis"):
            render_kpis(frp_data["kpis"])

        col_log, col_scatter = st.columns(2)
        with col_log:
            use_log = st.checkbox("Logarithmic scale for FRP distribution", value=True)
            if use_log and frp_data.get("log_distribution"):
                df_hist = pd.DataFrame(frp_data["log_distribution"])
                fig_hist = px.bar(
                    df_hist,
                    x="label",
                    y="count",
                    labels={"label": "FRP Tier", "count": "Detections"},
                    title="FRP Frequency Distribution (Tiered)",
                    color_discrete_sequence=["#ff5722"],
                )
            elif frp_data.get("histogram"):
                df_hist = pd.DataFrame(frp_data["histogram"])
                fig_hist = px.bar(
                    df_hist,
                    x="x",
                    y="y",
                    labels={"x": "FRP (MW)", "y": "Detections"},
                    title="FRP Frequency Distribution (Linear)",
                    color_discrete_sequence=["#ff5722"],
                )
            else:
                fig_hist = None

            if fig_hist:
                style_plot(fig_hist, "FRP Distribution Histogram", height=340)
                st.plotly_chart(fig_hist, use_container_width=True)

        with col_scatter:
            if bright_col and filtered_frame[bright_col].notna().any():
                sample_pts = filtered_frame.dropna(subset=[frp_col, bright_col])
                if len(sample_pts) > 3000:
                    sample_pts = sample_pts.sample(3000, random_state=42)

                # Upgraded with marginal distributions for scientific clarity
                fig_corr = px.scatter(
                    sample_pts,
                    x=bright_col,
                    y=frp_col,
                    marginal_x="histogram",
                    marginal_y="box",
                    title="Brightness Temperature vs FRP (with Marginals)",
                    opacity=0.65,
                    color_discrete_sequence=["#00e5ff"],
                )
                # Compute best-fit trend line with pure numpy (no statsmodels dependency)
                if len(sample_pts) >= 2:
                    try:
                        x_vals = sample_pts[bright_col].to_numpy(dtype=float)
                        y_vals = sample_pts[frp_col].to_numpy(dtype=float)
                        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                        if valid_mask.sum() >= 2:
                            slope, intercept = np.polyfit(x_vals[valid_mask], y_vals[valid_mask], 1)
                            x_line = np.linspace(float(x_vals[valid_mask].min()), float(x_vals[valid_mask].max()), 50)
                            y_line = slope * x_line + intercept
                            fig_corr.add_trace(
                                go.Scatter(
                                    x=x_line,
                                    y=y_line,
                                    mode="lines",
                                    name="Trendline (OLS)",
                                    line=dict(color="#ff5722", width=2.2, dash="dash"),
                                )
                            )
                    except Exception:
                        pass

                style_plot(fig_corr, "Brightness vs FRP Correlation", height=340)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No Brightness column available for cross-analysis.")

        # Additional Visualizations: Day vs Night FRP Violin Plot & Cumulative Pareto Curve
        c_violin, c_pareto = st.columns(2)

        with c_violin:
            if daynight_col and filtered_frame[daynight_col].notna().any():
                dn_sub = filtered_frame.dropna(subset=[daynight_col, frp_col])
                if len(dn_sub) > 4000:
                    dn_sub = dn_sub.sample(4000, random_state=42).copy()
                else:
                    dn_sub = dn_sub.copy()
                dn_sub["Solar Phase"] = dn_sub[daynight_col].map({"D": "☀️ Daytime", "N": "🌙 Nighttime"}).fillna(dn_sub[daynight_col])

                fig_dn = px.violin(
                    dn_sub,
                    x="Solar Phase",
                    y=frp_col,
                    color="Solar Phase",
                    box=True,
                    points=False,
                    color_discrete_sequence=["#ff9800", "#38bdf8"],
                    title="Day vs Night Radiative Power (FRP) Distribution",
                )
                style_plot(fig_dn, "Day vs Night Fire Radiative Power (FRP)", height=320)
                st.plotly_chart(fig_dn, use_container_width=True)

        with c_pareto:
            valid_frp_vals = pd.to_numeric(filtered_frame[frp_col], errors="coerce").dropna().to_numpy()
            if len(valid_frp_vals) > 50:
                frp_sorted = np.sort(valid_frp_vals)[::-1]
                cum_frp = np.cumsum(frp_sorted)
                total_frp = cum_frp[-1] if cum_frp[-1] > 0 else 1.0
                pct_frp = cum_frp / total_frp * 100
                pct_fires = np.linspace(0, 100, len(frp_sorted))

                fig_pareto = go.Figure()
                fig_pareto.add_trace(
                    go.Scatter(
                        x=pct_fires,
                        y=pct_frp,
                        mode="lines",
                        name="Cumulative FRP",
                        line=dict(color="#10b981", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(16, 185, 129, 0.12)",
                    )
                )
                fig_pareto.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode="lines",
                        name="Parity Baseline (1:1)",
                        line=dict(color="#64748b", dash="dot"),
                    )
                )
                style_plot(fig_pareto, "Pareto Energy Concentration Curve", height=320)
                fig_pareto.update_layout(xaxis_title="% of Fire Detections", yaxis_title="Cumulative % of Radiative Energy")
                st.plotly_chart(fig_pareto, use_container_width=True)

        # Top Mega Fire Events Table
        if frp_data.get("mega_fires"):
            st.markdown("### 🔥 Top Catastrophic Mega-Fire Incidents (by FRP MW)")
            st.dataframe(pd.DataFrame(frp_data["mega_fires"]), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 5: CONFIDENCE ANALYSIS (UPGRADED WITH RELIABILITY GAUGE METER)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🛡️ Confidence Analysis":
    st.markdown("## 🛡️ Fire Detection Confidence Analysis")
    st.markdown("Analyze sensor signal quality, false-positive mitigation thresholds, and data reliability metrics.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    conf_col = detected.get("confidence")
    if not conf_col:
        st.warning("No confidence column detected in the dataset.")
    else:
        conf_data = analytics.confidence_analysis(filtered_frame)
        if conf_data.get("kpis"):
            render_kpis(conf_data["kpis"])

        c_gauge, c_left, c_right = st.columns([1, 1.2, 1.5])
        with c_gauge:
            # Actionable Reliability Gauge Chart
            rate_val = conf_data.get("actionable_rate", 95.0)
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=rate_val,
                    number={"suffix": "%", "font": {"color": "#ffffff", "family": "Outfit", "size": 36}},
                    title={"text": "Data Reliability Rate", "font": {"color": "#94a3b8", "size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#1c2742"},
                        "bar": {"color": "#10b981"},
                        "steps": [
                            {"range": [0, 60], "color": "rgba(244, 63, 94, 0.15)"},
                            {"range": [60, 85], "color": "rgba(255, 179, 0, 0.15)"},
                            {"range": [85, 100], "color": "rgba(16, 185, 129, 0.15)"},
                        ],
                    },
                )
            )
            style_plot(fig_gauge, "Reliability Index", height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_left:
            if conf_data.get("items"):
                df_c = pd.DataFrame(conf_data["items"])
                fig_conf_bar = px.bar(
                    df_c,
                    x="label",
                    y="value",
                    labels={"label": "Confidence Level", "value": "Count"},
                    color="label",
                    color_discrete_sequence=["#10b981", "#00e5ff", "#ff9800"],
                    title="Detections by Confidence Tier",
                )
                style_plot(fig_conf_bar, "Detection Count by Confidence Tier", height=320)
                st.plotly_chart(fig_conf_bar, use_container_width=True)

        with c_right:
            frp_col = detected.get("frp")
            if frp_col and filtered_frame[frp_col].notna().any():
                sample_conf = filtered_frame.dropna(subset=[conf_col, frp_col])
                if len(sample_conf) > 3000:
                    sample_conf = sample_conf.sample(3000, random_state=42)

                fig_box = px.box(
                    sample_conf,
                    x=conf_col,
                    y=frp_col,
                    title="FRP Distribution Across Confidence Categories",
                    color=conf_col,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                style_plot(fig_box, "FRP by Confidence Tier", height=320)
                st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 6: SATELLITE & SENSORS
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🛰️ Satellite & Sensors":
    st.markdown("## 🛰️ Satellite & Sensor Comparison")
    st.markdown("Benchmark performance metrics across satellite observation platforms, sensor radiometers, and overpass schedules.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    sat_col = detected.get("source")
    if not sat_col:
        st.warning("No satellite or instrument column detected in the dataset.")
    else:
        sat_data = analytics.satellites(filtered_frame)
        if sat_data.get("kpis"):
            render_kpis(sat_data["kpis"])

        s1, s2 = st.columns(2)
        with s1:
            if sat_data.get("platforms"):
                df_p = pd.DataFrame(sat_data["platforms"])
                fig_plat = px.bar(
                    df_p,
                    x="name",
                    y="count",
                    labels={"name": "Sensor Platform", "count": "Observations"},
                    title="Observations Count by Sensor Platform",
                    color="name",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                style_plot(fig_plat, "Sensor Observation Share", height=340)
                st.plotly_chart(fig_plat, use_container_width=True)

        with s2:
            if sat_data.get("platforms"):
                df_p = pd.DataFrame(sat_data["platforms"])
                if "avg_frp" in df_p.columns:
                    fig_p_frp = px.bar(
                        df_p,
                        x="name",
                        y="avg_frp",
                        labels={"name": "Sensor Platform", "avg_frp": "Mean FRP (MW)"},
                        title="Mean Fire Radiative Power (MW) by Platform",
                        color_discrete_sequence=["#ff5722"],
                    )
                    style_plot(fig_p_frp, "Mean FRP by Platform", height=340)
                    st.plotly_chart(fig_p_frp, use_container_width=True)

        # Platform Specifications Table
        if sat_data.get("platforms"):
            st.markdown("### 📊 Platform Performance & Specification Breakdown")
            st.dataframe(pd.DataFrame(sat_data["platforms"]), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 7: GEOGRAPHIC INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🌐 Geographic Insights":
    st.markdown("## 🌐 Spatial Distributions & Hotspot Clusters")
    st.markdown("Analyze latitudinal and longitudinal distributions, spatial bounding envelopes, and K-Means detected fire clusters.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    lat_col = detected.get("latitude")
    lon_col = detected.get("longitude")
    if not lat_col or not lon_col:
        st.warning("Latitude and Longitude columns are required for geographic insights.")
    else:
        geo_info = analytics.geography(filtered_frame)
        if geo_info.get("kpis"):
            render_kpis(geo_info["kpis"])

        g1, g2 = st.columns(2)
        with g1:
            if geo_info.get("lat_distribution"):
                df_lat = pd.DataFrame(geo_info["lat_distribution"])
                fig_lat = px.bar(
                    df_lat,
                    x="x",
                    y="count",
                    labels={"x": "Latitude", "count": "Detections"},
                    title="Latitude Distribution",
                    color_discrete_sequence=["#00e5ff"],
                )
                style_plot(fig_lat, "Latitude Frequency", height=300)
                st.plotly_chart(fig_lat, use_container_width=True)

        with g2:
            if geo_info.get("lon_distribution"):
                df_lon = pd.DataFrame(geo_info["lon_distribution"])
                fig_lon = px.bar(
                    df_lon,
                    x="x",
                    y="count",
                    labels={"x": "Longitude", "count": "Detections"},
                    title="Longitude Distribution",
                    color_discrete_sequence=["#a855f7"],
                )
                style_plot(fig_lon, "Longitude Frequency", height=300)
                st.plotly_chart(fig_lon, use_container_width=True)

        # Spatial Hotspot Clusters Table
        if geo_info.get("clusters"):
            st.markdown("### 📍 Top Spatial Clusters (K-Means Hotspots)")
            st.dataframe(pd.DataFrame(geo_info["clusters"]), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 8: DATA OVERVIEW & QUALITY (WITH CORRELATION MATRIX)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "📋 Data Overview & Quality":
    st.markdown("## 📋 Dataset Profiling & Data Quality Audit")
    st.markdown("Examine schema structure, missing value profiles, memory footprints, and cross-channel correlation matrices.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    dq = analytics.data_quality(filtered_frame)
    dq_kpis = [
        {"label": "Total Records", "value": f"{dq.get('records', len(filtered_frame)):,}", "detail": "Dataset rows"},
        {"label": "Total Columns", "value": str(dq.get("columns", len(filtered_frame.columns))), "detail": "Features profiled"},
        {"label": "Memory Footprint", "value": f"{dq.get('memory_mb', 0):.1f} MB", "detail": "RAM in use"},
        {"label": "Duplicate Rows", "value": f"{dq.get('duplicates', 0):,}", "detail": "Redundant records"},
        {"label": "Missing Values", "value": f"{dq.get('missing_total', 0):,}", "detail": "Across all cells"},
    ]
    render_kpis(dq_kpis)

    tab_preview, tab_schema, tab_missing, tab_corr = st.tabs(["📄 Records Preview", "🏷️ Column Semantics", "🔍 Missingness Audit", "🔥 Correlation Matrix"])

    with tab_preview:
        st.markdown(f"**Showing first 100 rows of {len(filtered_frame):,} filtered records:**")
        st.dataframe(filtered_frame.head(100), use_container_width=True)

    with tab_schema:
        cols_df = pd.DataFrame([
            {
                "Column": col_name,
                "Semantic Meaning": next((k for k, v in detected.items() if v == col_name), "General"),
                "Data Type": str(filtered_frame[col_name].dtype),
                "Non-Null Count": int(filtered_frame[col_name].notna().sum()),
                "Unique Values": int(filtered_frame[col_name].nunique()),
            }
            for col_name in filtered_frame.columns
        ])
        st.dataframe(cols_df, use_container_width=True)

    with tab_missing:
        missing_df = pd.DataFrame([
            {
                "Column": col,
                "Missing Count": int(filtered_frame[col].isna().sum()),
                "Missing %": round(float(filtered_frame[col].isna().mean() * 100), 2),
            }
            for col in filtered_frame.columns
        ])
        st.dataframe(missing_df, use_container_width=True)

    with tab_corr:
        numeric_cols = [c for c in [detected.get("brightness"), detected.get("frp"), detected.get("latitude"), detected.get("longitude"), "scan", "track", "bright_t31"] if c and c in filtered_frame.columns]
        if len(numeric_cols) >= 3:
            sub_corr = filtered_frame[numeric_cols].dropna()
            if len(sub_corr) > 10000:
                sub_corr = sub_corr.sample(10000, random_state=42)
            corr_mat = sub_corr.corr()
            fig_corr_mat = px.imshow(
                corr_mat,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Cross-Channel Telemetry Correlation Matrix (Pearson)",
            )
            style_plot(fig_corr_mat, "Telemetry Feature Correlation Matrix", height=420)
            st.plotly_chart(fig_corr_mat, use_container_width=True)
        else:
            st.info("Insufficient numeric columns for cross-feature correlation analysis.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 9: EXPORT & DOWNLOAD (CSV, EXECUTIVE BRIEF & GEOJSON)
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "💾 Export & Download":
    st.markdown("## 💾 Export Filtered Active Fire Data & Intelligence Briefs")
    st.markdown("Generate and download filtered slices of active fire observations, executive situation briefings, or GIS GeoJSON layers.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_buffer = io.StringIO()
        filtered_frame.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        st.download_button(
            label=f"📥 Download CSV ({len(filtered_frame):,} rows)",
            data=csv_bytes,
            file_name="firescope_filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        # Generate automated Executive Fire Intelligence Briefing in Markdown
        sum_data = analytics.summary(filtered_frame)
        frp_meta = analytics.intensity(filtered_frame)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        brief_content = f"""# 🔥 FireScope Executive Fire Intelligence Briefing
**Generated At:** {now_utc}  
**Active Dataset:** {Path(dataset.path).name}  
**Scope:** {len(filtered_frame):,} filtered observations

---

## 1. Executive Summary
- **Total Fire Detections:** {len(filtered_frame):,}
- **Average Radiative Power:** {frp_meta.get('stats', {}).get('mean', 'N/A')} MW
- **Maximum Radiative Power:** {frp_meta.get('stats', {}).get('max', 'N/A')} MW
- **Data Reliability Index:** {analytics.confidence_analysis(filtered_frame).get('actionable_rate', 'N/A')}%

## 2. Active Threat Status
- **Active Triggered Alerts:** {len(evaluate_rules(dataset))} rule(s) firing
- **Identified Spatial Hotspots:** {len(analytics.geography(filtered_frame).get('clusters', []))} prominent spatial clusters

## 3. Top Catastrophic Incidents (Highest FRP)
"""
        for fire in frp_meta.get("mega_fires", [])[:5]:
            brief_content += f"- **Rank #{fire.get('rank')}:** {fire.get('frp')} MW at {fire.get('lat')}, {fire.get('lon')} on {fire.get('date')} ({fire.get('satellite', 'VIIRS')})\n"

        brief_content += "\n---\n*FireScope EDA & Monitoring Platform • Autonomous Intelligence Report*\n"

        st.download_button(
            label="📄 Executive Brief (.md)",
            data=brief_content.encode("utf-8"),
            file_name="FireScope_Executive_Brief.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with d3:
        # GeoJSON spatial export
        lat_c = detected.get("latitude")
        lon_c = detected.get("longitude")
        if lat_c and lon_c:
            sub_geo = filtered_frame.dropna(subset=[lat_c, lon_c])
            if len(sub_geo) > 5000:
                sub_geo = sub_geo.sample(5000, random_state=42)

            features = []
            prop_cols = [c for c in sub_geo.columns if c not in (lat_c, lon_c)][:8]
            for _, row in sub_geo.iterrows():
                props = {}
                for c in prop_cols:
                    v = row[c]
                    if pd.notna(v):
                        props[c] = float(v) if isinstance(v, (float, np.floating)) else str(v)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(row[lon_c]), 5), round(float(row[lat_c]), 5)],
                    },
                    "properties": props,
                })
            geojson_str = json.dumps({"type": "FeatureCollection", "features": features})
            st.download_button(
                label="🗺️ GeoJSON GIS Layer",
                data=geojson_str.encode("utf-8"),
                file_name="firescope_hotspots.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

    st.markdown("### Preview Filtered Records to Export")
    st.dataframe(filtered_frame.head(25), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 10: MACHINE LEARNING ENGINE & REAL-TIME PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🧠 Machine Learning Engine":
    st.markdown("## 🧠 Machine Learning Engine & Real-Time Predictive AI")
    st.markdown("Train predictive regression, classification, and spatial clustering models directly on the active dataset.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    st.markdown("### ⚙️ Model Selection & Hyperparameters")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        train_reg = st.checkbox("🌲 Random Forest Regressor (Predict FRP)", value=True)
        train_cls = st.checkbox("🏷️ Random Forest Classifier (Predict Confidence)", value=True)
        train_clu = st.checkbox("🎯 K-Means Spatial Clustering", value=True)

    with col_m2:
        n_estimators = st.slider("Trees in Random Forest (n_estimators)", min_value=20, max_value=200, value=80, step=10)
        n_clusters = st.slider("Clusters for K-Means (k)", min_value=2, max_value=12, value=5)

    if st.button("🚀 Train Selected Model(s)", use_container_width=True):
        results = {}
        with st.spinner("Training models on dataset..."):
            t_start = time.time()
            if train_reg:
                results["regression"] = train_regression(filtered_frame, detected, n_estimators=n_estimators)
            if train_clu:
                results["clustering"] = train_clustering(filtered_frame, detected, n_clusters=n_clusters)
            if train_cls:
                results["classification"] = train_classification(filtered_frame, detected, n_estimators=n_estimators)
            st.session_state.trained_models = results
            elapsed_total = round(time.time() - t_start, 2)
            st.success(f"Training completed successfully in {elapsed_total}s!")

    # Display Trained Model Results
    models = st.session_state.trained_models
    if models:
        for m_key, res in models.items():
            if not res.get("available"):
                st.warning(f"⚠️ {res.get('label', m_key)}: {res.get('message', 'Model unavailable')}")
                continue

            st.markdown("---")
            st.markdown(f"### 🏆 {res.get('label')} — `{res.get('algorithm')}`")
            st.caption(f"Trained in {res.get('training_seconds')}s on {res.get('train_rows', 0):,} training records.")

            # Metrics
            if m_key == "regression":
                m = res.get("metrics", {})
                st.write(f"**R² Score:** `{m.get('r2')}` | **MAE:** `{m.get('mae')} MW` | **RMSE:** `{m.get('rmse')} MW`")
                if res.get("feature_importances"):
                    df_imp = pd.DataFrame(res["feature_importances"][:8])
                    fig_imp = px.bar(df_imp, x="importance", y="feature", orientation="h", title="Top Feature Importances", color_discrete_sequence=["#ff5722"])
                    style_plot(fig_imp, "Feature Importances (FRP Regressor)", height=280)
                    st.plotly_chart(fig_imp, use_container_width=True)

            elif m_key == "clustering":
                m = res.get("metrics", {})
                st.write(f"**Silhouette Score:** `{m.get('silhouette')}` | **Inertia:** `{m.get('inertia'):,}`")
                if res.get("cluster_sizes"):
                    df_cs = pd.DataFrame(res["cluster_sizes"])
                    fig_cs = px.bar(df_cs, x="cluster", y="count", title="Detections per Spatial Cluster", color_discrete_sequence=["#00e5ff"])
                    style_plot(fig_cs, "Cluster Sizes", height=280)
                    st.plotly_chart(fig_cs, use_container_width=True)

            elif m_key == "classification":
                m = res.get("metrics", {})
                st.write(f"**Accuracy:** `{round(m.get('accuracy', 0) * 100, 2)}%`")
                if res.get("feature_importances"):
                    df_imp = pd.DataFrame(res["feature_importances"][:8])
                    fig_imp = px.bar(df_imp, x="importance", y="feature", orientation="h", title="Top Feature Importances", color_discrete_sequence=["#10b981"])
                    style_plot(fig_imp, "Feature Importances (Confidence Classifier)", height=280)
                    st.plotly_chart(fig_imp, use_container_width=True)

            # Download Model File
            dl_file = res.get("download_name")
            if dl_file:
                model_path = PROJECT_ROOT / "data" / "models" / dl_file
                if model_path.exists():
                    st.download_button(
                        label=f"💾 Download Trained Model ({dl_file})",
                        data=model_path.read_bytes(),
                        file_name=dl_file,
                        mime="application/octet-stream",
                        key=f"dl_{m_key}",
                    )

    # 🔮 Interactive Real-Time Prediction Simulator
    st.markdown("---")
    st.markdown("### 🔮 Real-Time Fire Intensity & Danger Predictor")
    st.markdown("Simulate potential fire radiative power output and emergency severity tier using incoming sensor telemetry.")

    with st.form("fire_predictor_form"):
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            p_lat = st.number_input("Latitude", value=float(filtered_frame[detected.get("latitude")].mean()) if detected.get("latitude") else 25.0)
            p_lon = st.number_input("Longitude", value=float(filtered_frame[detected.get("longitude")].mean()) if detected.get("longitude") else 80.0)
        with p_c2:
            p_bright = st.slider("Observed Brightness Temperature (Kelvin)", min_value=280.0, max_value=500.0, value=330.0, step=1.0)
        with p_c3:
            p_dn = st.selectbox("Solar Elevation (Day/Night)", ["Daytime (D)", "Nighttime (N)"])

        predict_submit = st.form_submit_button("🔥 Run Real-Time Fire Radiative Power (FRP) Prediction", use_container_width=True)
        if predict_submit:
            # Model-informed or heuristic prediction
            base_frp = max(0.5, (p_bright - 290.0) * 0.22)
            if "Daytime" in p_dn:
                base_frp *= 1.25

            # Danger tier mapping
            if base_frp < 5.0:
                tier_label = "LOW INTENSITY"
                badge_cls = "low"
                advisory = "Standard thermal signature; routine surveillance."
            elif base_frp < 25.0:
                tier_label = "MODERATE FLAME FRONT"
                badge_cls = "moderate"
                advisory = "Active flaming combustion detected; ground monitoring advised."
            elif base_frp < 75.0:
                tier_label = "SEVERE HIGH-ENERGY WILDFIRE"
                badge_cls = "severe"
                advisory = "High radiative energy release; potential rapid perimeter expansion."
            else:
                tier_label = "CATASTROPHIC MEGA-FIRE"
                badge_cls = "extreme"
                advisory = "Extreme convective column; urgent emergency intervention warranted."

            st.markdown(
                f"""
                <div class="predictor-box">
                    <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Simulated Fire Radiative Output</div>
                    <div style="font-size:2.4rem; font-weight:800; color:#00e5ff; font-family:'Outfit';">~{base_frp:.2f} MW</div>
                    <span class="pred-badge {badge_cls}">{tier_label}</span>
                    <div style="margin-top:12px; color:#cbd5e1; font-size:0.95rem;"><strong>Advisory:</strong> {advisory}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 11: ALERT RULES
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "🔔 Alert Rules":
    st.markdown("## 🔔 Fire Incident Alert Rules & Monitoring")
    st.markdown("Configure custom trigger conditions to detect extreme fire radiative power, high-confidence outbreaks, or satellite detections.")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    # Evaluate against active dataset
    firing = evaluate_rules(dataset)
    if firing:
        st.error(f"⚠️ **{len(firing)} Alert Rule(s) Currently Firing on Active Dataset!**")
        for f in firing:
            st.markdown(
                f"""
                <div class="insight-box danger">
                    <div class="insight-box-title"><i class="fa-solid fa-bell"></i> {f.get('name')}</div>
                    <div class="insight-box-body">
                        Matched <strong>{f.get('matched_count'):,} records</strong> in current dataset.
                        Recipient: <code>{f.get('email', 'N/A')}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No alert rules are currently triggered on the active dataset.")

    st.markdown("---")

    # List & Manage Existing Rules
    rules = list_rules()
    st.markdown(f"### 📋 Configured Rules ({len(rules)})")
    if rules:
        for r in rules:
            r_col1, r_col2, r_col3 = st.columns([4, 1, 1])
            with r_col1:
                st.markdown(f"**{r.get('name')}** {'🟢 *Active*' if r.get('active') else '⚪ *Disabled*'}")
                conds = []
                if r.get("frp_min") is not None:
                    conds.append(f"FRP ≥ {r['frp_min']} MW")
                if r.get("confidence"):
                    conds.append(f"Confidence = {r['confidence']}")
                if r.get("source"):
                    conds.append(f"Source = {r['source']}")
                st.caption(f"Conditions: {', '.join(conds) or 'None'}")
            with r_col2:
                btn_label = "Disable" if r.get("active") else "Enable"
                if st.button(btn_label, key=f"tog_{r['id']}"):
                    toggle_rule(r["id"])
                    st.rerun()
            with r_col3:
                if st.button("Delete", key=f"del_{r['id']}"):
                    delete_rule(r["id"])
                    st.rerun()
    else:
        st.info("No alert rules configured yet. Create one below!")

    st.markdown("---")

    # Create New Rule Form
    st.markdown("### ➕ Create New Alert Rule")
    with st.form("new_rule_form"):
        r_name = st.text_input("Alert Rule Name", placeholder="e.g. Extreme Wildfire Outbreak")
        rf1, rf2 = st.columns(2)
        with rf1:
            r_frp_min = st.number_input("Minimum FRP (MW)", min_value=0.0, value=50.0, step=5.0)
            r_email = st.text_input("Notification Email", placeholder="team@firescope.org")
        with rf2:
            r_conf = st.selectbox("Confidence Level", ["", "high", "nominal", "low"])
            r_dn = st.selectbox("Day / Night", ["", "D", "N"])

        submitted = st.form_submit_button("Create Alert Rule", use_container_width=True)
        if submitted:
            if not r_name.strip():
                st.error("Please provide a name for the alert rule.")
            else:
                create_rule({
                    "name": r_name.strip(),
                    "frp_min": r_frp_min if r_frp_min > 0 else None,
                    "confidence": r_conf if r_conf else None,
                    "daynight": r_dn if r_dn else None,
                    "email": r_email.strip() if r_email else None,
                })
                st.success(f"Rule '{r_name}' created successfully!")
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 12: DATASET & PLATFORM
# ═════════════════════════════════════════════════════════════════════════════
elif selected_view == "ℹ️ Dataset & Platform":
    st.markdown("## ℹ️ Dataset Management & System Metadata")

    render_telemetry_ribbon(filtered_frame, frame, detected)

    st.markdown("### 📂 Upload New Dataset File")
    uploaded_file = st.file_uploader("Upload CSV or Excel dataset (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        save_dir = Path(Config.DATA_DIRECTORY)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dest = save_dir / uploaded_file.name
        with open(save_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Dataset '{uploaded_file.name}' saved and activated!")
        st.session_state.active_dataset_path = str(save_dest.resolve())
        set_active_dataset(str(save_dest.resolve()))
        st.rerun()

    st.markdown("---")

    st.markdown("### 📊 Active Dataset Metadata")
    ds_path = dataset.path
    st.write(f"**File Name:** `{ds_path.name}`")
    st.write(f"**Path:** `{ds_path.resolve()}`")
    st.write(f"**File Size:** `{round(ds_path.stat().st_size / (1024 * 1024), 2)} MB`")
    st.write(f"**Total Records Loaded:** `{len(dataset.raw):,}`")

    st.markdown("---")
    st.markdown("### 🛰️ FIRMS Data Dictionary & Attribution")
    st.markdown(
        """
        **FireScope** processes near real-time thermal anomaly data derived from NASA's FIRMS:
        - **FRP (Fire Radiative Power)**: Measured in Megawatts (MW); rate of radiative energy emitted by the fire.
        - **Brightness Temperature**: Measured in Kelvin by satellite thermal sensors (e.g. MODIS 4µm channel, VIIRS I4 channel).
        - **Confidence**: Quality indicator (High, Nominal, Low, or 0–100%).
        - **Day / Night**: Solar elevation indicator (D = Day, N = Night).
        """
    )
