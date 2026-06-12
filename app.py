#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smart Product Discovery & Recommendation Engine
================================================
Portfolio-quality ML dashboard  |  python -m streamlit run app.py
"""

import os, sys, pickle, random, re, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first) ────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Product Discovery Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models",  "final_model_svd.pkl")
DATA_PATH  = os.path.join(ROOT, "data", "processed", "processed_data.pkl")
sys.path.insert(0, ROOT)

# ── Brand palette ──────────────────────────────────────────────────────────
C_DARK   = "#080A10"
C_NAV    = "#111625"
C_ORANGE = "#FF9900"
C_GOLD   = "#FFB03A"
C_BURN   = "#FF5500"
C_TEAL   = "#00E5FF"
C_PAGE   = "#0A0D16"
C_CARD   = "rgba(17, 22, 37, 0.75)"
C_TEXT   = "#E2E8F0"
C_SUB    = "#94A3B8"

# ══════════════════════════════════════════════════════════════════════════
#  CSS  — single source of truth, applied once
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Reset ─────────────────────────────────────────── */
*, *::before, *::after {{
    font-family: 'Outfit', -apple-system, Arial, sans-serif !important;
    box-sizing: border-box;
}}

/* ── Page background ────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {{
    background-color: {C_PAGE} !important;
}}
.block-container {{
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* ── Force LIGHT/SILVER text everywhere in main area ─────────── */
.stApp p, .stApp li,
.stApp label, .stApp small, .stApp strong, .stApp b,
.element-container p, .element-container li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"],
.stMarkdown p, .stMarkdown li {{
    color: {C_TEXT} !important;
}}
.stApp strong, .stApp b {{ font-weight: 700 !important; color: #FFFFFF !important; }}

/* Caption */
[data-testid="stCaptionContainer"] p,
.stCaption {{
    color: {C_SUB} !important;
    font-size: 0.78rem !important;
}}

/* ── Headings ───────────────────────────────────────── */
h1 {{ color: #FFFFFF !important; font-weight: 800 !important; font-size: 1.65rem !important; text-shadow: 0 0 15px rgba(255, 153, 0, 0.2); }}
h2 {{ color: #FFFFFF !important; font-weight: 700 !important; font-size: 1.25rem !important; }}
h3 {{ color: #FFFFFF !important; font-weight: 600 !important; font-size: 1.0rem  !important; }}
h4 {{ color: #FFFFFF !important; font-weight: 600 !important; }}
hr {{ border-color: rgba(255, 255, 255, 0.1) !important; margin: 0.75rem 0 !important; }}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {C_NAV} !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: #E2E8F0 !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {C_ORANGE} !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.08) !important; }}
section[data-testid="stSidebar"] [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetricValue"] * {{
    color: #FFFFFF !important; font-size: 1.5rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] * {{
    color: {C_GOLD} !important;
}}
section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {{
    color: {C_GOLD} !important;
}}

/* ── Metric cards (main area) ───────────────────────── */
div[data-testid="metric-container"] {{
    background: {C_CARD} !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease;
}}
div[data-testid="metric-container"]:hover {{
    border-color: rgba(255, 153, 0, 0.3) !important;
    box-shadow: 0 8px 32px 0 rgba(255, 153, 0, 0.1) !important;
    transform: translateY(-2px);
}}
/* Metric LABEL — small uppercase text above number */
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] div,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] span {{
    color: {C_SUB} !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
}}
/* Metric VALUE — the big number */
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] div,
div[data-testid="metric-container"] [data-testid="stMetricValue"] span {{
    color: #FFFFFF !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
}}
/* Metric DELTA */
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 0.73rem !important; }}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] span {{ font-size: 0.73rem !important; }}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(17, 22, 37, 0.6) !important;
    border-radius: 8px 8px 0 0;
    border-bottom: 2px solid rgba(255, 255, 255, 0.08) !important;
    padding: 0 8px; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C_SUB} !important;
    font-weight: 500;
    font-size: 0.86rem;
    padding: 10px 18px;
    border-bottom: 3px solid transparent;
    border-radius: 0; margin-bottom: -2px;
}}
.stTabs [aria-selected="true"] {{
    color: {C_ORANGE} !important;
    border-bottom: 3px solid {C_ORANGE} !important;
    font-weight: 700 !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: {C_CARD} !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 20px;
}}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, #FF9900 0%, #C45500 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 9px 20px !important;
    box-shadow: 0 4px 14px rgba(255, 153, 0, 0.15) !important;
    transition: all .2s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #FFB03A 0%, #E86A00 100%) !important;
    box-shadow: 0 6px 20px rgba(255, 153, 0, 0.25) !important;
    transform: translateY(-1px) !important;
}}

/* ── Inputs ─────────────────────────────────────────── */
.stTextInput > div > div > input {{
    background: rgba(25, 32, 51, 0.6) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    transition: all 0.2s;
}}
.stTextInput > div > div > input:focus {{
    border-color: {C_ORANGE} !important;
    box-shadow: 0 0 0 2px rgba(255,153,0,0.2) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #64748B !important; }}

.stSelectbox > div > div {{
    background: rgba(25, 32, 51, 0.6) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}}

/* ── DataFrames ─────────────────────────────────────── */
.stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }}
[data-testid="stDataFrameResizable"] * {{ color: #FFFFFF !important; }}

/* ── Radio (sidebar nav) ────────────────────────────── */
section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.88rem !important; }}

/* ── Custom layout components ───────────────────────── */

.amz-topbar {{
    background: {C_NAV};
    padding: 12px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 0 -2rem 0 -2rem;
    border-bottom: 3px solid {C_ORANGE};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}}
.amz-topbar, .amz-topbar * {{ color: #FFFFFF !important; }}
.amz-logo {{
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: {C_ORANGE} !important;
    letter-spacing: -1px;
    line-height: 1;
}}
.amz-logo em {{ color: {C_GOLD} !important; font-style: normal; }}
.amz-tagline {{ color: #94A3B8 !important; font-size: 0.77rem !important; }}
.amz-pill {{
    margin-left: auto;
    background: linear-gradient(135deg, {C_ORANGE}, {C_BURN});
    color: #FFFFFF !important;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(255, 153, 0, 0.2);
}}

.page-header {{
    background: linear-gradient(90deg, {C_NAV} 0%, rgba(17, 22, 37, 0.4) 100%);
    border-radius: 10px;
    padding: 16px 22px;
    margin: 18px 0 20px 0;
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 5px solid {C_ORANGE};
}}
.page-header h1 {{ color: #FFFFFF !important; margin: 0 !important; font-size: 1.35rem !important; font-weight: 700 !important; }}
.page-header .sub {{ color: #94A3B8 !important; font-size: 0.78rem !important; margin-top: 4px !important; }}

.card {{
    background: {C_CARD};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease;
}}
.card:hover {{
    border-color: rgba(255, 153, 0, 0.25) !important;
    box-shadow: 0 8px 32px 0 rgba(255, 153, 0, 0.08) !important;
}}
.card-title {{
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    padding-bottom: 10px !important;
    margin-bottom: 14px !important;
    border-bottom: 2px solid {C_ORANGE} !important;
    display: block !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.step-box {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid {C_ORANGE};
    border-radius: 0 8px 8px 0;
    padding: 11px 15px;
    margin: 6px 0;
    font-size: 0.86rem;
    color: {C_TEXT};
    line-height: 1.5;
}}
.step-box .step-title {{ font-weight: 700; color: #FFFFFF; display: block; margin-bottom: 2px; }}

.insight-box {{
    background: rgba(255, 153, 0, 0.05);
    border: 1px solid rgba(255, 176, 58, 0.2);
    border-left: 4px solid {C_ORANGE};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: {C_TEXT};
    margin-top: 10px;
    line-height: 1.55;
}}

.rec-card {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 5px solid {C_ORANGE};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: all .2s ease;
}}
.rec-card:hover {{
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    border-left-color: {C_BURN};
}}
.rec-rank  {{ font-size: 1.3rem; font-weight: 800; color: {C_ORANGE}; min-width: 32px; display: inline-block; }}
.rec-prod  {{ font-size: 0.9rem; font-weight: 700; color: {C_TEAL}; font-family: 'Courier New', monospace; }}
.rec-meta  {{ font-size: 0.78rem; color: {C_SUB}; margin-top: 3px; }}
.stars     {{ color: {C_ORANGE}; }}
.xai-label {{ font-size: 0.72rem; color: {C_SUB}; font-weight: 600; margin-bottom: 2px; }}
.xai-track {{ background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 7px; overflow: hidden; margin: 2px 0; }}
.xai-fill  {{ background: linear-gradient(90deg, {C_GOLD}, {C_ORANGE}); height: 100%; border-radius: 4px; }}
.xai-val   {{ font-size: 0.72rem; font-weight: 700; color: {C_BURN}; }}

.prod-row {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 5px 0;
    display: flex;
    gap: 14px;
    align-items: center;
    transition: all 0.2s ease;
}}
.prod-row:hover {{ background: rgba(255, 255, 255, 0.04); box-shadow: 0 3px 10px rgba(0,0,0,0.2); }}
.prod-rank-badge {{
    background: {C_NAV};
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.82rem;
    border-radius: 50%;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.prod-id   {{ font-family: monospace; font-weight: 700; color: {C_TEAL}; font-size: 0.87rem; }}
.prod-meta {{ font-size: 0.76rem; color: {C_SUB}; margin-top: 2px; }}

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer, [data-testid="stDeployButton"] {{ visibility: hidden !important; }}
header[data-testid="stHeader"] {{
    background-color: transparent !important;
    pointer-events: none;
}}
[data-testid="collapsedControl"] {{
    pointer-events: auto !important;
    background: transparent !important;
    border: none !important;
    margin-left: 12px !important;
}}
[data-testid="collapsedControl"] button {{
    color: #FFFFFF !important;
    background: transparent !important;
    border: none !important;
    padding: 4px !important;
}}
[data-testid="collapsedControl"] button:hover {{
    color: {C_ORANGE} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Top header bar ─────────────────────────────────────────────────────────
st.markdown("""
<div class="amz-topbar">
    <div class="amz-logo">amazon<em>.ai</em></div>
    <span class="amz-tagline">Smart Product Discovery &amp; Recommendation Engine</span>
    <span class="amz-pill">ML Portfolio</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading SVD model…")
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["train_df"]

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Computing product statistics…")
def product_stats(_df):
    ps = _df.groupby("prod_id")["rating"].agg(count="count", mean="mean").reset_index()
    g  = _df["rating"].mean()
    k  = ps["count"].mean()
    ps["bayesian"]      = (g * k + ps["count"] * ps["mean"]) / (k + ps["count"])
    ps["pop_pct"]       = ((ps["count"] - ps["count"].min()) /
                           (ps["count"].max() - ps["count"].min()) * 100).round(1)
    return ps.sort_values("bayesian", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner="Computing user statistics…")
def user_stats(_df):
    us = _df.groupby("user_id")["rating"].agg(
        count="count", mean="mean", std="std"
    ).reset_index()
    us["segment"] = pd.cut(
        us["count"],
        bins=[0, 5, 15, 30, 50, 9999],
        labels=["Casual", "Light", "Regular", "Active", "Power"],
    )
    return us.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner="Generating recommendations…")
def get_recs(user_id, top_n, _df, _ps):
    algo, _ = load_model()
    rated      = set(_df[_df["user_id"] == user_id]["prod_id"])
    candidates = list(set(_df["prod_id"]) - rated)
    preds = [(pid, algo.predict(user_id, pid).est) for pid in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    pop_max = _ps["count"].max()
    results = []
    for rank, (pid, svd) in enumerate(preds[:top_n], 1):
        row = _ps[_ps["prod_id"] == pid]
        cnt  = int(row["count"].values[0])   if len(row) else 1
        bay  = float(row["bayesian"].values[0]) if len(row) else svd
        cf_s = svd / 5.0
        pk_s = cnt / pop_max
        rk_s = bay / 5.0
        fin  = 0.5 * cf_s + 0.3 * rk_s + 0.2 * pk_s
        results.append({
            "rank": rank, "prod_id": pid,
            "predicted": round(svd, 3),
            "stars": "★" * int(round(svd)) + "☆" * (5 - int(round(svd))),
            "cf_pct": round(cf_s * 100, 1),
            "pop_pct": round(pk_s * 100, 1),
            "rank_pct": round(rk_s * 100, 1),
            "confidence": min(int(fin * 100), 99),
            "score": round(fin * 100, 1),
            "reviews": cnt,
        })
    return results

def nl_search(query, ps, top_n=20):
    q = query.lower().strip()
    lo, hi = 1.0, 5.0
    m = re.search(r'under\s+([\d.]+)', q)
    if m: hi = float(m.group(1))
    m = re.search(r'(above|over|at ?least)\s+([\d.]+)', q)
    if m: lo = float(m.group(2))
    sort = "bayesian"
    if any(x in q for x in ["popular","trending","most reviewed"]): sort = "count"
    elif any(x in q for x in ["highest","best","top rated"]):       sort = "mean"
    filtered = ps[(ps["mean"] >= lo) & (ps["mean"] <= hi)].copy()
    return filtered.nlargest(top_n, sort).reset_index(drop=True), sort

# ── Static model results ───────────────────────────────────────────────────
RESULTS = pd.DataFrame({
    "Model":       ["Rank-Based","User-User CF","Item-Item CF","SVD","Hybrid"],
    "RMSE":        [0.940, 1.006, 1.018, 0.898, 0.958],
    "MAE":         [0.714, 0.783, 0.798, 0.680, 0.741],
    "Precision@K": [0.841, 0.851, 0.831, 0.842, 0.849],
    "Recall@K":    [0.930, 0.884, 0.868, 0.908, 0.902],
    "F1@K":        [0.883, 0.867, 0.849, 0.874, 0.875],
    "MRR":         [0.952, 0.937, 0.911, 0.952, 0.948],
    "MAP":         [0.924, 0.913, 0.884, 0.922, 0.919],
    "Hit Rate@K":  [0.994, 0.994, 0.994, 0.994, 0.994],
})
MODEL_COLORS = [C_GOLD, C_TEAL, "#146EB4", C_ORANGE, C_BURN]

# ── Chart helper ──────────────────────────────────────────────────────────
def base_layout(fig, height=340, title="", **kw):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=13, family="Outfit")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(25, 32, 51, 0.3)",
        font=dict(color="#E2E8F0", family="Outfit"),
        height=height, margin=dict(t=44 if title else 20, b=16, l=12, r=12),
        **kw,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.15)", color="#E2E8F0")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.15)", color="#E2E8F0")
    return fig

def bar_chart(models, values, title, color_list=None, y_range=None, key="chart"):
    fig = go.Figure()
    colors = color_list or MODEL_COLORS
    for i, (m, v) in enumerate(zip(models, values)):
        fig.add_trace(go.Bar(
            x=[m], y=[v],
            marker_color=colors[i % len(colors)],
            marker_line_color=C_BURN if v == min(values) or v == max(values) else "rgba(255,255,255,0.2)",
            marker_line_width=2,
            text=[f"{v:.3f}"], textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
            name=m,
        ))
    base_layout(fig, title=title, height=340, showlegend=False, bargap=0.35)
    if y_range:
        fig.update_yaxes(range=y_range)
    st.plotly_chart(fig, key=key)

# ── Page header helper ─────────────────────────────────────────────────────
def page_hdr(icon, title, sub=""):
    st.markdown(
        f'<div class="page-header"><h1>{icon} {title}</h1>'
        f'{"<div class=sub>" + sub + "</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
PAGES = {
    "📊  Executive Summary":     "exec",
    "🎯  Smart Recommendations": "recs",
    "🔍  Product Discovery":      "search",
    "📈  ML Analytics":           "analytics",
    "🛍️  Product Intelligence":   "products",
    "👥  User Insights":          "users",
    "🤖  Model Comparison":       "models",
    "⚡  Performance Charts":     "perf",
}
with st.sidebar:
    st.markdown("### 🛒 RecSys Platform")
    st.markdown("---")
    page = PAGES[st.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")]
    st.markdown("---")
    st.markdown("#### Dataset Stats")
    try:
        _sdf = load_data()
        st.metric("Interactions", f"{len(_sdf):,}")
        st.metric("Users",        f"{_sdf['user_id'].nunique():,}")
        st.metric("Products",     f"{_sdf['prod_id'].nunique():,}")
        st.metric("Avg Rating",   f"{_sdf['rating'].mean():.2f} ★")
    except Exception:
        st.info("Data not loaded.")
    st.markdown("---")
    st.markdown("#### Best Model: SVD")
    st.markdown("RMSE **0.898** · MAP **0.922**")
    st.markdown("Hit Rate@K **99.4 %**")
    st.markdown("---")
    st.caption("Python · scikit-surprise · Streamlit · Plotly")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
if page == "exec":
    page_hdr("📊", "Executive Summary",
             "Business overview of the Amazon Product Recommendation Engine")

    df = load_data()
    ps = product_stats(df)

    # KPI strip — use short labels to prevent truncation
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Ratings",       f"{len(df):,}")
    c2.metric("Users",         f"{df['user_id'].nunique():,}")
    c3.metric("Products",      f"{df['prod_id'].nunique():,}")
    c4.metric("Best RMSE",     "0.898",  delta="SVD",       delta_color="inverse")
    c5.metric("Hit Rate@10",   "99.4%",  delta="All models")
    c6.metric("Sparsity",      "99.27%")
    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            st.markdown('<span class="card-title">🏗️ System Architecture</span>', unsafe_allow_html=True)
            for t, d in [
                ("Data Pipeline",       "13 Amazon CSV files → 7.8M raw ratings → 78K filtered interactions"),
                ("Feature Engineering", "Users filtered to ≥5 ratings; Bayesian scoring for rank-based model"),
                ("Model Training",      "5 algorithms: Rank-Based, User-User CF, Item-Item CF, SVD, Hybrid"),
                ("Evaluation Suite",    "7 metrics: RMSE, MAE, Precision@K, Recall@K, F1@K, MRR, MAP, Hit Rate@K"),
                ("Deployment",          "Pre-trained SVD model — sub-second inference via Streamlit dashboard"),
            ]:
                st.markdown(
                    f'<div class="step-box"><span class="step-title">{t}</span>{d}</div>',
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown('<span class="card-title">📈 Model Leaderboard</span>', unsafe_allow_html=True)

            def hl(col):
                if col.name == "Model": return [""]*len(col)
                best = col.min() if col.name in ("RMSE","MAE") else col.max()
                return [f"background:#FFF3CD;font-weight:700;color:{C_BURN}" if v==best else "" for v in col]

            lb = RESULTS[["Model","RMSE","MAE","MAP","Hit Rate@K"]].copy()
            st.dataframe(
                lb.style.apply(hl).format({c:"{:.3f}" for c in lb.columns if c!="Model"}),
                hide_index=True,
            )
            st.markdown(
                f'<div class="insight-box">🏆 <b>SVD wins</b> on RMSE (0.898) and MAE (0.680). '
                f'Rank-Based leads on Recall@K (0.930). Hit Rate@K is 99.4% across <b>all</b> models.</div>',
                unsafe_allow_html=True,
            )

    # Business insights
    with st.container(border=True):
        st.markdown('<span class="card-title">💡 Key Business Insights</span>', unsafe_allow_html=True)
        i1,i2,i3,i4 = st.columns(4)
        five = (df["rating"]==5).mean()*100
        power= int((df.groupby("user_id")["rating"].count()>=30).sum())
        cold = (df.groupby("user_id")["rating"].count()<=6).mean()*100
        top10_pct = ps.head(10)["count"].sum()/len(df)*100
        i1.markdown(f'<div class="insight-box"><b>⭐ Positivity Bias</b><br>{five:.1f}% of ratings are 5-stars. Users rate products they actively chose to buy.</div>', unsafe_allow_html=True)
        i2.markdown(f'<div class="insight-box"><b>🔥 Power Users</b><br>{power:,} users have 30+ ratings. They provide the strongest collaborative filtering signal.</div>', unsafe_allow_html=True)
        i3.markdown(f'<div class="insight-box"><b>🧊 Cold Start</b><br>~{cold:.0f}% of users have ≤6 ratings. The Hybrid model addresses this with a rank-based fallback.</div>', unsafe_allow_html=True)
        i4.markdown(f'<div class="insight-box"><b>📦 Long-Tail</b><br>Top 10 products = {top10_pct:.1f}% of all ratings — classic power-law distribution in e-commerce.</div>', unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns(2)
    with ch1:
        with st.container(border=True):
            st.markdown('<span class="card-title">📊 Rating Distribution</span>', unsafe_allow_html=True)
            rc = df["rating"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=rc.index.astype(str), y=rc.values,
                marker_color=[C_GOLD,"#FFB347",C_ORANGE,"#E88000",C_BURN],
                marker_line_color="#A37000", marker_line_width=1,
                text=[f"{v/len(df)*100:.1f}%" for v in rc.values],
                textposition="outside", textfont=dict(color=C_DARK, size=11),
            ))
            base_layout(fig, height=260, showlegend=False)
            fig.update_xaxes(title_text="Stars")
            fig.update_yaxes(title_text="Reviews")
            st.plotly_chart(fig, key="exec_rating_dist")

    with ch2:
        with st.container(border=True):
            st.markdown('<span class="card-title">🏅 Top 10 Products — Bayesian Score</span>', unsafe_allow_html=True)
            top10 = ps.head(10)[["prod_id","count","mean","bayesian"]].copy()
            top10.index = range(1, 11)
            top10.columns = ["Product ID","Reviews","Avg ★","Bayesian"]
            top10["Avg ★"]   = top10["Avg ★"].round(2)
            top10["Bayesian"] = top10["Bayesian"].round(3)
            st.dataframe(top10, hide_index=False)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SMART RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == "recs":
    page_hdr("🎯", "Smart Recommendations",
             "Personalised picks with Explainable AI — powered by SVD Matrix Factorisation")

    algo, _ = load_model()
    if algo is None:
        st.error("Model not found. Run `python run_project.py` first.")
        st.stop()

    df = load_data()
    ps = product_stats(df)
    uc = df.groupby("user_id")["rating"].count().sort_values(ascending=False)

    if "selected_user" not in st.session_state:
        st.session_state["selected_user"] = uc.index[0]
    
    if st.session_state["selected_user"] not in uc.index:
        st.session_state["selected_user"] = uc.index[0]

    user_list = uc.index.tolist()
    default_idx = user_list.index(st.session_state["selected_user"])

    ctrl1, ctrl2, ctrl3 = st.columns([2.5, 1, 1])
    with ctrl1:
        sel = st.selectbox("👤 Customer ID", user_list,
                           index=default_idx,
                           format_func=lambda u: f"{u}   ({uc[u]} ratings)")
        st.session_state["selected_user"] = sel
    with ctrl2:
        top_n = st.slider("Top-N", 3, 20, 10)
    with ctrl3:
        st.write("")
        if st.button("🎲 Random Customer"):
            st.session_state["selected_user"] = random.choice(uc[uc >= 10].index.tolist())
            st.rerun()

    udf = df[df["user_id"] == sel]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Ratings Given",   len(udf))
    k2.metric("Avg Rating",      f"{udf['rating'].mean():.2f} ★")
    k3.metric("5-Star Reviews",  int((udf["rating"]==5).sum()))
    k4.metric("1-Star Reviews",  int((udf["rating"]==1).sum()))
    k5.metric("Products Unseen", df["prod_id"].nunique() - len(udf))

    with st.spinner("Running SVD…"):
        t0   = time.time()
        recs = get_recs(sel, top_n, df, ps)
        elapsed = time.time() - t0
    st.caption(f"⚡ {top_n} recommendations generated in {elapsed:.3f}s")

    col_recs, col_hist = st.columns([1.4, 1])

    with col_recs:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">🏆 Recommendations + XAI Breakdown</span>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"""
<div class="rec-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">
    <span class="rec-rank">#{r['rank']}</span>
    <div style="flex:1">
      <div class="rec-prod">{r['prod_id']}</div>
      <div class="rec-meta">
        <span class="stars">{r['stars']}</span>
        &nbsp;Predicted <b>{r['predicted']}/5.0</b>
        &nbsp;·&nbsp; Confidence <b>{r['confidence']}%</b>
        &nbsp;·&nbsp; {r['reviews']:,} reviews
      </div>
    </div>
    <div style="background:#FFF3CD;padding:4px 12px;border-radius:20px;
                font-size:0.75rem;font-weight:700;color:{C_BURN};flex-shrink:0">
      Score: {r['score']}
    </div>
  </div>
  <div style="font-size:0.7rem;font-weight:700;color:{C_SUB};
              text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
    Why Recommended
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
    <div>
      <div class="xai-label">🤝 Collaborative Filtering</div>
      <div class="xai-track"><div class="xai-fill" style="width:{r['cf_pct']}%"></div></div>
      <div class="xai-val">{r['cf_pct']:.0f}%</div>
    </div>
    <div>
      <div class="xai-label">📊 Product Popularity</div>
      <div class="xai-track"><div class="xai-fill" style="width:{r['pop_pct']}%"></div></div>
      <div class="xai-val">{r['pop_pct']:.0f}%</div>
    </div>
    <div>
      <div class="xai-label">⭐ Bayesian Rank</div>
      <div class="xai-track"><div class="xai-fill" style="width:{r['rank_pct']}%"></div></div>
      <div class="xai-val">{r['rank_pct']:.0f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_hist:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">📋 Customer Rating History</span>', unsafe_allow_html=True)
        hist = udf.sort_values("rating", ascending=False).head(12).copy()
        hist["Stars"] = hist["rating"].apply(lambda r: "★"*int(r)+"☆"*(5-int(r)))
        hist = hist[["prod_id","rating","Stars"]].rename(
            columns={"prod_id":"Product ID","rating":"Rating"})
        hist.index = range(1, len(hist)+1)
        st.dataframe(hist, hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">📊 Rating Profile</span>', unsafe_allow_html=True)
        ud = udf["rating"].value_counts().sort_index()
        fig_u = go.Figure(go.Bar(
            x=ud.index.astype(str), y=ud.values,
            marker_color=[C_GOLD,"#FFB347",C_ORANGE,"#E88000",C_BURN],
            text=ud.values, textposition="outside",
            textfont=dict(color=C_DARK, size=10),
        ))
        base_layout(fig_u, height=200, showlegend=False)
        fig_u.update_xaxes(title_text="Stars")
        st.plotly_chart(fig_u, key="recs_profile")
        avg_conf = int(sum(r["confidence"] for r in recs)/len(recs))
        st.markdown(
            f'<div class="insight-box">🎯 <b>Avg Confidence:</b> '
            f'<span style="color:{C_ORANGE};font-size:1.1rem;font-weight:800">{avg_conf}%</span><br>'
            f'<small>Weighted SVD score + popularity + Bayesian rank.</small></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PRODUCT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════
elif page == "search":
    page_hdr("🔍", "Product Discovery",
             "Natural language search, filtering, and catalog exploration")

    df = load_data()
    ps = product_stats(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="card-title">🔍 Natural Language Search</span>', unsafe_allow_html=True)

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    query = st.text_input(
        "query",
        value=st.session_state["search_query"],
        placeholder='e.g. "best rated"  |  "most popular"  |  "under 4 stars"  |  "trending"',
        label_visibility="collapsed",
    )
    st.session_state["search_query"] = query

    st.markdown("**Quick Filters:**")
    qf = st.columns(5)
    if qf[0].button("🔥 Trending"):
        st.session_state["search_query"] = "trending"
        st.rerun()
    if qf[1].button("⭐ Highest Rated"):
        st.session_state["search_query"] = "highest rated"
        st.rerun()
    if qf[2].button("📦 Most Reviewed"):
        st.session_state["search_query"] = "most reviewed"
        st.rerun()
    if qf[3].button("🎯 Under 4 Stars"):
        st.session_state["search_query"] = "under 4"
        st.rerun()
    if qf[4].button("💎 Best Overall"):
        st.session_state["search_query"] = "best"
        st.rerun()

    top_n_s = st.slider("Results to show", 5, 50, 20, key="search_n")
    st.markdown('</div>', unsafe_allow_html=True)

    if query:
        results, sort_used = nl_search(query, ps, top_n_s)
        st.markdown(
            f'<div class="insight-box">🔍 Query: <b>"{query}"</b> &nbsp;·&nbsp; '
            f'Sorted by: <b>{sort_used}</b> &nbsp;·&nbsp; Found: <b>{len(results)}</b> products</div>',
            unsafe_allow_html=True,
        )
        for i, row in results.iterrows():
            pop  = min(int(row["pop_pct"]), 100)
            avg  = row["mean"]
            stars = "★"*int(round(avg))+"☆"*(5-int(round(avg)))
            st.markdown(f"""
<div class="prod-row">
  <div class="prod-rank-badge">{i+1}</div>
  <div style="flex:1">
    <div class="prod-id">{row['prod_id']}</div>
    <div class="prod-meta">
      <span class="stars">{stars}</span> {avg:.2f} avg
      &nbsp;·&nbsp; {int(row['count']):,} reviews
      &nbsp;·&nbsp; Bayesian: {row['bayesian']:.3f}
    </div>
  </div>
  <div style="text-align:right;min-width:110px">
    <div style="font-size:0.72rem;color:{C_SUB};margin-bottom:3px">Popularity</div>
    <div class="xai-track" style="width:100px">
      <div class="xai-fill" style="width:{pop}%"></div>
    </div>
    <div class="xai-val">{pop}%</div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">📊 Full Catalog Map</span>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=ps["count"], y=ps["mean"], mode="markers",
            marker=dict(size=5, opacity=0.4,
                        color=ps["bayesian"],
                        colorscale=[[0,C_GOLD],[0.5,C_ORANGE],[1,C_BURN]],
                        showscale=True,
                        colorbar=dict(title="Bayesian Score",
                                      tickfont=dict(color=C_DARK))),
            hovertemplate="<b>%{customdata}</b><br>Reviews: %{x}<br>Avg: %{y:.2f}<extra></extra>",
            customdata=ps["prod_id"],
        ))
        base_layout(fig, height=420, showlegend=False)
        fig.update_xaxes(title_text="Rating Count")
        fig.update_yaxes(title_text="Average Rating")
        st.plotly_chart(fig, key="search_catalog")
        st.caption("Each dot = one product. Colour intensity = Bayesian score.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — ML ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "analytics":
    page_hdr("📈", "ML Analytics Dashboard",
             "Model evaluation metrics, accuracy benchmarks, and performance KPIs")

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Best RMSE",    "0.898",  delta="SVD",         delta_color="inverse")
    m2.metric("Best MAE",     "0.680",  delta="SVD",         delta_color="inverse")
    m3.metric("Precision@10", "85.1 %", delta="User-User CF")
    m4.metric("Recall@10",    "93.0 %", delta="Rank-Based")
    m5.metric("Hit Rate@10",  "99.4 %", delta="All models")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="card-title">📋 Complete Evaluation Results — Test Set (K=10, Threshold=3.5)</span>', unsafe_allow_html=True)

    def hl(col):
        if col.name == "Model": return [""]*len(col)
        best = col.min() if col.name in ("RMSE","MAE") else col.max()
        return [f"background:#FFF3CD;font-weight:700;color:{C_BURN}" if v==best else "" for v in col]

    st.dataframe(
        RESULTS.style.apply(hl).format({c:"{:.3f}" for c in RESULTS.columns if c!="Model"}),
        hide_index=True,
    )
    st.caption("🟡 Gold = best per metric. RMSE/MAE: lower is better. Ranking metrics: higher is better.")
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Accuracy (RMSE & MAE)", "📊 Ranking Metrics", "🕸️ Radar"])

    with tab1:
        ch1, ch2 = st.columns(2)
        with ch1:
            bar_chart(RESULTS["Model"], RESULTS["RMSE"],
                      "RMSE — Lower is Better", y_range=[0.82,1.08], key="an_rmse")
        with ch2:
            bar_chart(RESULTS["Model"], RESULTS["MAE"],
                      "MAE — Lower is Better", y_range=[0.60,0.85], key="an_mae")

    with tab2:
        met = st.selectbox("Metric", ["Precision@K","Recall@K","F1@K","MRR","MAP","Hit Rate@K"], key="an_m")
        bar_chart(RESULTS["Model"], RESULTS[met],
                  f"{met} — Higher is Better", y_range=[0.82,1.02], key="an_rank")

    with tab3:
        radar_m = ["Precision@K","Recall@K","F1@K","MRR","MAP"]
        fig = go.Figure()
        for i, row in RESULTS.iterrows():
            vals = [row[m] for m in radar_m] + [row[radar_m[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=radar_m+[radar_m[0]],
                fill="toself", name=row["Model"],
                line_color=MODEL_COLORS[i], opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(bgcolor="#FAFAFA",
                radialaxis=dict(visible=True, range=[0.85,1.0],
                    tickfont=dict(color=C_SUB), gridcolor="#DDD"),
                angularaxis=dict(tickfont=dict(color=C_DARK, size=11))),
            paper_bgcolor=C_CARD, font=dict(color=C_DARK),
            title=dict(text="Radar — All Ranking Metrics", font=dict(color=C_DARK,size=13)),
            height=460,
            legend=dict(bgcolor="#F9F9F9", bordercolor="#DDD", borderwidth=1),
        )
        st.plotly_chart(fig, key="an_radar")
        st.markdown('<div class="insight-box">📌 Larger shaded area = better overall performance. SVD and Rank-Based dominate most axes.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PRODUCT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "products":
    page_hdr("🛍️", "Product Intelligence",
             "Most popular, highest rated, trending and distribution analytics")

    df = load_data()
    ps = product_stats(df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Most Popular","⭐ Highest Rated","🔥 Trending","📊 Distributions"]
    )

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_p = ps.nlargest(20,"count")
        fig = go.Figure(go.Bar(
            x=top_p["count"], y=top_p["prod_id"], orientation="h",
            marker_color=C_ORANGE, marker_line_color=C_BURN, marker_line_width=1,
            text=top_p["count"].astype(str), textposition="outside",
            textfont=dict(color=C_DARK, size=9),
        ))
        base_layout(fig, height=520, title="Top 20 Most Reviewed Products", showlegend=False)
        fig.update_xaxes(title_text="Rating Count")
        fig.update_yaxes(tickfont=dict(size=8))
        st.plotly_chart(fig, key="prod_pop")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_r = ps[ps["count"]>=10].nlargest(20,"mean")
        fig2 = go.Figure(go.Bar(
            x=top_r["mean"], y=top_r["prod_id"], orientation="h",
            marker_color=[C_BURN if v>=4.9 else C_ORANGE if v>=4.5 else C_GOLD for v in top_r["mean"]],
            text=top_r["mean"].round(3).astype(str), textposition="outside",
            textfont=dict(color=C_DARK, size=9),
        ))
        base_layout(fig2, height=520, title="Top 20 Highest Rated (min 10 reviews)", showlegend=False)
        fig2.update_xaxes(title_text="Average Rating", range=[4.0,5.15])
        fig2.update_yaxes(tickfont=dict(size=8))
        st.plotly_chart(fig2, key="prod_rated")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_t = ps.nlargest(50,"bayesian")
        fig3 = px.scatter(
            top_t, x="count", y="mean", size="bayesian",
            color="bayesian",
            color_continuous_scale=[[0,C_GOLD],[0.5,C_ORANGE],[1,C_BURN]],
            hover_data={"prod_id":True,"count":True,"mean":":.2f","bayesian":":.3f"},
            title="Trending: Bayesian Score vs Popularity (top 50)",
        )
        fig3.update_layout(paper_bgcolor=C_CARD, plot_bgcolor="#FAFAFA",
                           font=dict(color=C_DARK), height=420,
                           margin=dict(t=44,b=20,l=40,r=20),
                           title_font=dict(size=13, color=C_DARK))
        fig3.update_xaxes(title_text="Rating Count", gridcolor="#EEE", color=C_DARK)
        fig3.update_yaxes(title_text="Average Rating", gridcolor="#EEE", color=C_DARK)
        st.plotly_chart(fig3, key="prod_trend")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig4 = px.histogram(ps, x="count", nbins=60,
                                color_discrete_sequence=[C_ORANGE], template="plotly_white")
            base_layout(fig4, height=280, title="Rating Count Distribution")
            fig4.update_xaxes(title_text="Rating Count")
            st.plotly_chart(fig4, key="prod_d1")
            st.markdown('</div>', unsafe_allow_html=True)
        with d2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig5 = px.histogram(ps, x="mean", nbins=40,
                                color_discrete_sequence=[C_TEAL], template="plotly_white")
            base_layout(fig5, height=280, title="Average Rating Distribution")
            fig5.update_xaxes(title_text="Average Rating")
            st.plotly_chart(fig5, key="prod_d2")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 6 — USER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "users":
    page_hdr("👥", "User Insights",
             "Activity analysis, rating behaviour, and engagement segmentation")

    df = load_data()
    us = user_stats(df)

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Users",   f"{len(us):,}")
    m2.metric("Avg Ratings",   f"{us['count'].mean():.1f}")
    m3.metric("Max Ratings",   f"{us['count'].max():,}")
    m4.metric("Avg Rating",    f"{us['mean'].mean():.2f} ★")
    m5.metric("Power Users",   str(int((us["count"]>=30).sum())), delta="≥30 ratings")

    tab1, tab2, tab3 = st.tabs(["📊 Activity","🔥 Top Users","🗺️ Behaviour Heatmap"])

    with tab1:
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.histogram(us, x="count", nbins=60,
                               color_discrete_sequence=[C_ORANGE], template="plotly_white")
            base_layout(fig, height=280, title="Ratings per User")
            fig.update_xaxes(title_text="Ratings Given")
            st.plotly_chart(fig, key="usr_a1")
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig2 = px.histogram(us, x="mean", nbins=40,
                                color_discrete_sequence=[C_TEAL], template="plotly_white")
            base_layout(fig2, height=280, title="Avg Rating per User")
            fig2.update_xaxes(title_text="Average Rating")
            st.plotly_chart(fig2, key="usr_a2")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">👥 Engagement Segments</span>', unsafe_allow_html=True)
        seg = us["segment"].value_counts().reset_index()
        seg.columns = ["Segment","Count"]
        fig3 = go.Figure(go.Pie(
            labels=seg["Segment"], values=seg["Count"],
            marker_colors=[C_GOLD,C_ORANGE,C_BURN,C_TEAL,C_NAV],
            textinfo="label+percent", hole=0.4,
        ))
        fig3.update_layout(paper_bgcolor=C_CARD, font=dict(color=C_DARK),
                            height=300, margin=dict(t=20,b=10),
                            legend=dict(bgcolor="#F9F9F9"))
        st.plotly_chart(fig3, key="usr_seg")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top20u = us.head(20)[["user_id","count","mean"]].copy()
        top20u.index = range(1, 21)
        top20u.columns = ["User ID","Ratings","Avg Rating"]
        top20u["Avg Rating"] = top20u["Avg Rating"].round(2)
        fig4 = px.bar(top20u, x="User ID", y="Ratings",
                      color="Avg Rating",
                      color_continuous_scale=[[0,C_GOLD],[0.5,C_ORANGE],[1,C_BURN]],
                      title="Top 20 Most Active Users")
        fig4.update_layout(paper_bgcolor=C_CARD, plot_bgcolor="#FAFAFA",
                           font=dict(color=C_DARK), height=360,
                           margin=dict(t=44,b=60,l=8,r=8),
                           title_font=dict(size=13,color=C_DARK))
        fig4.update_xaxes(tickangle=45, tickfont_size=7, gridcolor="#EEE", color=C_DARK)
        fig4.update_yaxes(gridcolor="#EEE", color=C_DARK)
        st.plotly_chart(fig4, key="usr_top20")
        st.dataframe(top20u, hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        heat_data = df.merge(us[["user_id","segment"]], on="user_id")
        heat = heat_data.groupby(["segment","rating"]).size().reset_index(name="count")
        heat_piv = heat.pivot(index="segment", columns="rating", values="count").fillna(0)
        fig5 = px.imshow(heat_piv,
                         color_continuous_scale=[[0,"#FFF8E7"],[0.5,C_ORANGE],[1,C_BURN]],
                         title="Rating Heatmap by User Segment",
                         labels=dict(x="Star Rating", y="Segment", color="Count"))
        fig5.update_layout(paper_bgcolor=C_CARD, font=dict(color=C_DARK), height=340,
                           margin=dict(t=44,b=30,l=80,r=20),
                           title_font=dict(size=13,color=C_DARK))
        fig5.update_xaxes(color=C_DARK)
        fig5.update_yaxes(color=C_DARK)
        st.plotly_chart(fig5, key="usr_heat")
        st.markdown('<div class="insight-box">Power users lean heavily toward 5-star ratings, amplifying the positivity bias in the training data. The Hybrid model down-weights this using a Bayesian prior.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 7 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
elif page == "models":
    page_hdr("🤖", "Model Comparison",
             "Side-by-side comparison of all 5 recommendation algorithms")

    m1,m2,m3,m4,m5 = st.columns(5)
    for col, (_, row) in zip([m1,m2,m3,m4,m5], RESULTS.iterrows()):
        col.metric(row["Model"], f"RMSE {row['RMSE']:.3f}",
                   delta="Best" if row["RMSE"]==RESULTS["RMSE"].min() else None,
                   delta_color="inverse")

    # Algorithm cards
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="card-title">🔬 Algorithm Explainer</span>', unsafe_allow_html=True)
    a1,a2,a3,a4,a5 = st.columns(5)
    for col, (name, algo, desc, rmse) in zip([a1,a2,a3,a4,a5], [
        ("Rank-Based",   "Bayesian Avg",      "Surfaces globally popular products. Shrinks sparse ratings toward the global mean — fairer than simple averaging.", 0.940),
        ("User-User CF", "KNNBasic (cosine)", "Finds users with similar rating histories and borrows their preferences. Intuitive but slow at scale.", 1.006),
        ("Item-Item CF", "KNNBasic (cosine)", "Finds products similar to what the user liked. Memory-heavy but stable in dense regions.", 1.018),
        ("SVD ⭐",        "Matrix Factorisation","Decomposes the rating matrix into latent factors. Best RMSE (0.898). Generalises well to sparse data.", 0.898),
        ("Hybrid",       "CF×0.6 + Rank×0.4", "Blends SVD personalisation with Bayesian popularity — handles cold-start and sparse users.", 0.958),
    ]):
        with col:
            top_border = f"border-top:4px solid {C_ORANGE}" if rmse==RESULTS['RMSE'].min() else f"border-top:4px solid {C_GOLD}"
            badge = f'<span style="background:{C_ORANGE};color:{C_DARK};font-size:0.65rem;font-weight:700;padding:2px 7px;border-radius:10px;margin-left:5px">BEST</span>' if rmse==RESULTS['RMSE'].min() else ''
            st.markdown(f"""
<div style="background:#FAFAFA;border:1px solid #E0E0E0;{top_border};border-radius:8px;padding:14px;min-height:190px">
  <div style="font-weight:700;color:{C_DARK};font-size:0.88rem">{name}{badge}</div>
  <div style="font-size:0.73rem;color:{C_TEAL};margin:4px 0;font-style:italic">{algo}</div>
  <div style="font-size:0.77rem;color:{C_TEXT};margin-top:8px;line-height:1.5">{desc}</div>
  <div style="margin-top:10px;font-size:0.75rem;color:{C_SUB}">RMSE: <b style="color:{C_BURN}">{rmse}</b></div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Grouped comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="card-title">📊 Metrics Comparison</span>', unsafe_allow_html=True)
    sel_metrics = st.multiselect(
        "Select metrics",
        list(RESULTS.columns[1:]),
        default=["RMSE","MAE","MAP","Hit Rate@K"],
        key="mc_sel",
    )
    if sel_metrics:
        fig = go.Figure()
        for i, row in RESULTS.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"], x=sel_metrics,
                y=[row[m] for m in sel_metrics],
                marker_color=MODEL_COLORS[i],
            ))
        fig.update_layout(paper_bgcolor=C_CARD, plot_bgcolor="#FAFAFA",
                          font=dict(color=C_DARK), height=360,
                          margin=dict(t=20,b=30), barmode="group",
                          legend=dict(bgcolor="#F9F9F9",bordercolor="#DDD",borderwidth=1))
        fig.update_xaxes(gridcolor="#EEE", color=C_DARK)
        fig.update_yaxes(gridcolor="#EEE", color=C_DARK)
        st.plotly_chart(fig, key="mc_grouped")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 8 — PERFORMANCE CHARTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "perf":
    page_hdr("⚡", "Performance Visualisations",
             "Interactive charts: distributions, heatmaps, trend analysis")

    df = load_data()
    ps = product_stats(df)
    us = user_stats(df)

    tab1, tab2, tab3 = st.tabs(["📊 Score Distributions","🗺️ Heatmaps","📈 Trend Lines"])

    with tab1:
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="card-title">Bayesian Score Distribution</span>', unsafe_allow_html=True)
            fig = px.histogram(ps, x="bayesian", nbins=50,
                               color_discrete_sequence=[C_ORANGE], template="plotly_white")
            base_layout(fig, height=250)
            fig.update_xaxes(title_text="Bayesian Score")
            fig.update_yaxes(title_text="Products")
            st.plotly_chart(fig, key="pf_bay")
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="card-title">Popularity Score Distribution</span>', unsafe_allow_html=True)
            fig2 = px.histogram(ps, x="pop_pct", nbins=50,
                                color_discrete_sequence=[C_TEAL], template="plotly_white")
            base_layout(fig2, height=250)
            fig2.update_xaxes(title_text="Popularity %")
            fig2.update_yaxes(title_text="Products")
            st.plotly_chart(fig2, key="pf_pop")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="card-title">📉 RMSE & MAE Across Models</span>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=RESULTS["Model"], y=RESULTS["RMSE"], mode="lines+markers",
            name="RMSE", line=dict(color=C_ORANGE, width=3),
            marker=dict(size=10, color=C_ORANGE),
        ))
        fig3.add_trace(go.Scatter(
            x=RESULTS["Model"], y=RESULTS["MAE"], mode="lines+markers",
            name="MAE", line=dict(color=C_TEAL, width=3, dash="dot"),
            marker=dict(size=10, color=C_TEAL),
        ))
        base_layout(fig3, height=280, showlegend=True,
                    legend=dict(bgcolor="#F9F9F9",bordercolor="#DDD",borderwidth=1))
        fig3.update_yaxes(title_text="Error (lower = better)")
        st.plotly_chart(fig3, key="pf_line")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ps_b = ps.copy()
        ps_b["count_bin"]  = pd.cut(ps["count"], bins=[0,5,10,20,50,100,9999],
                                     labels=["1-5","6-10","11-20","21-50","51-100","100+"])
        ps_b["rating_bin"] = pd.cut(ps["mean"],  bins=[0,2,3,3.5,4,4.5,5.01],
                                     labels=["<2","2-3","3-3.5","3.5-4","4-4.5","4.5+"])
        heat = ps_b.groupby(["rating_bin","count_bin"]).size().reset_index(name="products")
        heat_piv = heat.pivot(index="rating_bin", columns="count_bin", values="products").fillna(0)
        fig4 = px.imshow(heat_piv,
                         color_continuous_scale=[[0,"#FFF"],[0.3,C_GOLD],[0.7,C_ORANGE],[1,C_BURN]],
                         title="Products by Rating Range × Review Count",
                         labels=dict(x="Review Count Range", y="Avg Rating Range", color="# Products"))
        fig4.update_layout(paper_bgcolor=C_CARD, font=dict(color=C_DARK), height=360,
                           margin=dict(t=44,b=40,l=80,r=20),
                           title_font=dict(size=13,color=C_DARK))
        fig4.update_xaxes(color=C_DARK)
        fig4.update_yaxes(color=C_DARK)
        st.plotly_chart(fig4, key="pf_heat")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        metrics_t = ["Precision@K","Recall@K","F1@K","MAP","MRR"]
        fig5 = go.Figure()
        for i, m in enumerate(metrics_t):
            fig5.add_trace(go.Scatter(
                x=RESULTS["Model"], y=RESULTS[m], mode="lines+markers",
                name=m, line=dict(color=MODEL_COLORS[i], width=2),
                marker=dict(size=8),
            ))
        base_layout(fig5, height=340, showlegend=True,
                    legend=dict(bgcolor="#F9F9F9",bordercolor="#DDD",borderwidth=1))
        fig5.update_yaxes(title_text="Score (higher = better)", range=[0.84,0.96])
        st.plotly_chart(fig5, key="pf_trend")
        st.markdown(
            '<div class="insight-box">📌 All models exceed 0.88 on every ranking metric. '
            'The real differentiator is RMSE: SVD\'s 0.898 vs Item-Item CF\'s 1.018 is a '
            '<b>13.7% improvement</b> in predictive accuracy.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
