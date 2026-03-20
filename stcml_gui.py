# =============================================================================
# S-TCML GUI v2  —  Single-Screen Thermal Comfort Dashboard
# Layout: [Inputs col] | [Results col] | [3D Room col]
# No scrolling — everything fits one viewport
# =============================================================================
# Run:  streamlit run stcml_gui.py
# Deps: streamlit plotly joblib numpy pandas xgboost scikit-learn
# =============================================================================

# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import math
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S-TCML - Thermal Comfort",
    page_icon=":thermometer:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─────────────────────────────────────────────────────────────────────────────
# CSS — injected dynamically based on theme
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        # ── Dark palette ──
        bg       = "#0d1117"
        s1       = "#161b26"
        s2       = "#1c2333"
        s3       = "#222a3a"
        border   = "#2d3650"
        text     = "#ffffff"
        text_dim = "#a0aabf"
        text_fnt = "#5a6480"
        gold_dim = "rgba(240,165,0,0.15)"
        sel_bg   = "#1c2333"
        pop_bg   = "#1c2333"
        metric_bg= "#1c2333"
        gauge_axis_color = "#a0aabf"
    else:
        # ── Light palette ──
        bg       = "#f4f6fa"
        s1       = "#ffffff"
        s2       = "#eef0f6"
        s3       = "#e4e8f0"
        border   = "#c8cedc"
        text     = "#0d1117"
        text_dim = "#4a5270"
        text_fnt = "#7a8299"
        gold_dim = "rgba(200,120,0,0.10)"
        sel_bg   = "#ffffff"
        pop_bg   = "#ffffff"
        metric_bg= "#ffffff"
        gauge_axis_color = "#4a5270"

    gold = "#f0a500"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {{
    --bg:       {bg};
    --s1:       {s1};
    --s2:       {s2};
    --s3:       {s3};
    --border:   {border};
    --gold:     {gold};
    --gold-dim: {gold_dim};
    --text:     {text};
    --text-dim: {text_dim};
    --text-fnt: {text_fnt};
    --ff-head:  'Syne', sans-serif;
    --ff-body:  'IBM Plex Sans', sans-serif;
    --ff-mono:  'IBM Plex Mono', monospace;
}}

/* ── Kill chrome ── */
#MainMenu, footer, header {{ display:none !important; }}
[data-testid="collapsedControl"] {{ display:none !important; }}
[data-testid="stSidebar"]        {{ display:none !important; }}

/* ── Single-screen lock ── */
html, body {{ overflow:hidden !important; height:100vh !important; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container {{
    background-color: {bg} !important;
    color: {text} !important;
    font-family: var(--ff-body) !important;
}}
.block-container {{
    padding: 0.4rem 0.9rem 0.1rem !important;
    max-width: 100% !important;
}}

/* ── All text ── */
p, span, label, div, h1, h2, h3, h4, li,
[data-testid="stMarkdownContainer"] *,
[data-testid="metric-container"] *,
.stSelectbox label, .stSlider label,
.stSelectbox div, .stSlider div,
[data-baseweb="select"] *, [data-baseweb="slider"] *,
[data-testid="stCaptionContainer"] {{
    color: {text} !important;
}}
caption, .caption, [data-testid="stCaptionContainer"] * {{
    color: {text_dim} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {metric_bg};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 5px 9px !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--ff-head) !important;
    font-size: 1rem !important;
    color: {text} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.6rem !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: {text_dim} !important;
}}

/* ── Select box ── */
[data-baseweb="select"] > div {{
    background: {sel_bg} !important;
    border-color: {border} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] div {{ color: {text} !important; }}
[data-baseweb="popover"] * {{
    background: {pop_bg} !important;
    color: {text} !important;
}}

/* ── Slider thumb ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: {gold} !important;
}}

/* ── Button ── */
.stButton > button {{
    background: {gold} !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--ff-head) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    width: 100%;
}}

/* ── Expander ── */
[data-testid="stExpander"] * {{ color: {text_dim} !important; }}

/* ── Custom components ── */
.card-title {{
    font-family: var(--ff-mono);
    font-size: 0.58rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: {text_dim};
    border-bottom: 1px solid {border};
    padding-bottom: 4px;
    margin-bottom: 7px;
}}
.hdr {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 3px 0 5px;
    border-bottom: 1px solid {border};
    margin-bottom: 6px;
}}
.hdr-title {{
    font-family: var(--ff-head);
    font-size: 1.05rem;
    font-weight: 800;
    color: {text};
}}
.hdr-sub {{
    font-family: var(--ff-mono);
    font-size: 0.58rem;
    color: {text_fnt};
    letter-spacing: .06em;
}}
.pill {{
    font-family: var(--ff-mono);
    font-size: 0.56rem;
    background: {gold_dim};
    border: 1px solid rgba(240,165,0,.3);
    color: {gold};
    padding: 1px 7px;
    border-radius: 20px;
    white-space: nowrap;
}}
.big-val {{
    font-family: var(--ff-head);
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1;
    color: {text};
}}
.badge {{
    display: inline-block;
    font-family: var(--ff-mono);
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    border: 1px solid;
    margin-top: 3px;
    color: {text};
}}
.result-card {{
    background: {s1};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 7px 10px;
}}
.interval-row {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    color: {text_dim};
    background: {s3};
    border-radius: 5px;
    padding: 5px 8px;
    margin-top: 5px;
    line-height: 1.7;
}}
.interval-row b {{ color: {text}; }}
.guidance-box {{
    background: {s2};
    border-left: 3px solid {gold};
    border-radius: 0 6px 6px 0;
    padding: 6px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: {text};
    line-height: 1.55;
    margin-top: 5px;
}}
.ref-box {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: {text_dim};
    line-height: 1.7;
    background: {s3};
    border-radius: 6px;
    padding: 6px 9px;
    margin-top: 6px;
}}
.ref-box b {{ color: {text}; }}
</style>
""", unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = str(pathlib.Path(__file__).parent.resolve())
FEATURES   = ["month", "wall_width", "room_depth", "orientation", "WWR"]

ORIENTATION_MAP = {"North up": 0, "East right": 90, "South down": 180, "West left": 270}
MONTH_SHORT = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
MONTH_FULL  = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
SEASON_MAP  = {12:("Winter","[W]"),1:("Winter","[W]"),2:("Winter","[W]"),
               3:("Spring","[Sp]"),4:("Spring","[Sp]"),5:("Spring","[Sp]"),
               6:("Summer","[Su]"),7:("Summer","[Su]"),8:("Summer","[Su]"),
               9:("Autumn","[Au]"),10:("Autumn","[Au]"),11:("Autumn","[Au]")}

def pmv_label(v):
    if   v < -2.5: return "Cold",           "#3d9be9"
    elif v < -1.5: return "Cool",           "#6db8f0"
    elif v < -0.5: return "Slightly Cool",  "#9dd0f5"
    elif v <=  0.5: return "Comfortable",    "#3ecf6e"
    elif v <=  1.5: return "Slightly Warm", "#f0c040"
    elif v <=  2.5: return "Warm",          "#e87040"
    else:           return "Hot",           "#e84040"

def ppd_label(v):
    if   v <= 10: return "Acceptable",              "#3ecf6e"
    elif v <= 20: return "Marginal",               "#f0c040"
    elif v <= 40: return "Uncomfortable",          "#e87040"
    else:         return "Highly Uncomfortable",   "#e84040"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    m, errs = {}, []
    for t in ["PMV", "PPD"]:
        p = os.path.join(MODELS_DIR, f"XGBoost_{t}.pkl")
        if os.path.exists(p): m[f"XGBoost_{t}"] = joblib.load(p)
        else: errs.append(f"XGBoost_{t}.pkl")
        for q in ["q05", "q50", "q95"]:
            p = os.path.join(MODELS_DIR, f"XGBoost_{t}_{q}.pkl")
            if os.path.exists(p): m[f"XGBoost_{t}_{q}"] = joblib.load(p)
            else: errs.append(f"XGBoost_{t}_{q}.pkl")
    return m, errs

def predict_all(models, month, ww, rd, ori_deg, wwr):
    X = pd.DataFrame([[month, ww, rd, ori_deg, wwr]], columns=FEATURES)
    out = {}
    for t in ["PMV", "PPD"]:
        k = f"XGBoost_{t}"
        if k in models:
            out[f"{t}_point"] = float(models[k].predict(X)[0])
        for q in ["q05", "q50", "q95"]:
            k2 = f"XGBoost_{t}_{q}"
            if k2 in models:
                out[f"{t}_{q}"] = float(models[k2].predict(X)[0])
    return out

# ─────────────────────────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────
def gauge(val, target, text_color="#ffffff"):
    if target == "PMV":
        rng   = [-3, 3]
        steps = [
            dict(range=[-3,-2.5], color="#0d2d4d"),
            dict(range=[-2.5,-1.5],color="#1a4870"),
            dict(range=[-1.5,-0.5],color="#2a6a9a"),
            dict(range=[-0.5, 0.5],color="#1a4d30"),
            dict(range=[ 0.5, 1.5],color="#5a4800"),
            dict(range=[ 1.5, 2.5],color="#6a3010"),
            dict(range=[ 2.5,   3],color="#6a1010"),
        ]
        suffix = ""
    else:
        rng   = [0, 100]
        steps = [
            dict(range=[0,  10], color="#1a4d30"),
            dict(range=[10, 20], color="#4a4800"),
            dict(range=[20, 40], color="#5a3800"),
            dict(range=[40, 70], color="#5a2010"),
            dict(range=[70,100], color="#6a1010"),
        ]
        suffix = "%"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = val,
        number= dict(font=dict(size=26, color=text_color, family="Syne"),
                     suffix=suffix),
        gauge = dict(
            axis    = dict(range=rng, tickcolor="#2d3650",
                           tickfont=dict(color=text_color, size=7), nticks=7),
            bar     = dict(color="#f0a500", thickness=0.17),
            bgcolor = "rgba(0,0,0,0)", borderwidth=0,
            steps   = steps,
        ),
    ))
    fig.update_layout(
        height=120,
        margin=dict(t=5, b=0, l=5, r=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=text_color,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# INTERVAL CHART
# ─────────────────────────────────────────────────────────────────────────────
def interval_chart(results, text_color="#ffffff", text_dim="#a0aabf", grid_color="#2d3650"):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
                        subplot_titles=["PMV — 90% Interval", "PPD — 90% Interval"])

    for col, tgt, color in [(1,"PMV","#3d9be9"), (2,"PPD","#e87040")]:
        pt  = results.get(f"{tgt}_point")
        lo  = results.get(f"{tgt}_q05")
        med = results.get(f"{tgt}_q50")
        hi  = results.get(f"{tgt}_q95")
        if None in (pt, lo, med, hi):
            continue
        r, g_c, b_c = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

        # shaded band
        fig.add_trace(go.Scatter(
            x=[lo,lo,hi,hi], y=[0.15,0.85,0.85,0.15],
            fill="toself",
            fillcolor=f"rgba({r},{g_c},{b_c},0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

        # interval whisker
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[0.5, 0.5], mode="lines",
            line=dict(color=color, width=2),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

        # tick marks at q05, q50, q95
        for xv, lbl in [(lo, f"{lo:.2f}"), (med, f"{med:.2f}"), (hi, f"{hi:.2f}")]:
            fig.add_trace(go.Scatter(
                x=[xv], y=[0.5], mode="markers+text",
                marker=dict(symbol="line-ns", size=14,
                            line=dict(color=color, width=2)),
                text=[lbl], textposition="top center",
                textfont=dict(size=8, color=text_color),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

        # gold diamond = point prediction
        fig.add_trace(go.Scatter(
            x=[pt], y=[0.5], mode="markers",
            marker=dict(symbol="diamond", size=11, color="#f0a500",
                        line=dict(color="#0d1117", width=1.5)),
            showlegend=False,
            hovertemplate=f"Point: {pt:.3f}<extra></extra>",
        ), row=1, col=col)

    fig.update_layout(
        height=115,
        margin=dict(t=20, b=4, l=4, r=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=text_color,
        font_family="IBM Plex Mono",
    )
    for c in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=grid_color,
                         tickfont=dict(size=7, color=text_dim),
                         zeroline=False, row=1, col=c)
        fig.update_yaxes(visible=False, row=1, col=c)
    for ann in fig.layout.annotations:
        ann.font.color = text_dim
        ann.font.size  = 8
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 3-D ROOM
# ─────────────────────────────────────────────────────────────────────────────
def make_3d_room(wall_width, room_depth, wwr, orientation_label, pmv_val=None):
    W = float(wall_width)
    D = float(room_depth)
    H = 3.0

    traces = []

    def mesh_face(xs, ys, zs, color, opacity):
        return go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=opacity,
            flatshading=True, showscale=False, hoverinfo="skip",
        )

    # ── Room shell ──────────────────────────────────────────────────────────
    # Floor
    traces.append(mesh_face([0,W,W,0], [0,0,D,D], [0,0,0,0],  "#1a2535", 0.92))
    # Ceiling
    traces.append(mesh_face([0,W,W,0], [0,0,D,D], [H,H,H,H],  "#141c2b", 0.45))
    # Back wall (y=D)
    traces.append(mesh_face([0,W,W,0], [D,D,D,D], [0,0,H,H],  "#1c2a3d", 0.65))
    # Left wall (x=0)
    traces.append(mesh_face([0,0,0,0], [0,D,D,0], [0,0,H,H],  "#162030", 0.60))
    # Right wall (x=W)
    traces.append(mesh_face([W,W,W,W], [0,D,D,0], [0,0,H,H],  "#162030", 0.60))

    # ── Facade (y=0) with glazing cutout ────────────────────────────────────
    gz_bot = H * 0.10
    gz_h   = H * 0.72
    gz_top = gz_bot + gz_h
    gw     = W * wwr
    gx0    = (W - gw) / 2
    gx1    = gx0 + gw

    # solid strips around window
    # bottom sill
    traces.append(mesh_face([0,W,W,0],[0,0,0,0],[0,0,gz_bot,gz_bot], "#1c2a3d", 0.78))
    # top header
    traces.append(mesh_face([0,W,W,0],[0,0,0,0],[gz_top,gz_top,H,H], "#1c2a3d", 0.78))
    # left jamb
    if gx0 > 0.01:
        traces.append(mesh_face([0,gx0,gx0,0],[0,0,0,0],
                                [gz_bot,gz_bot,gz_top,gz_top], "#1c2a3d", 0.78))
    # right jamb
    if gx1 < W - 0.01:
        traces.append(mesh_face([gx1,W,W,gx1],[0,0,0,0],
                                [gz_bot,gz_bot,gz_top,gz_top], "#1c2a3d", 0.78))

    # Window glass
    glass_col = "#1a5a90" if (pmv_val is None or pmv_val <= 0.5) else "#7a2018"
    traces.append(go.Mesh3d(
        x=[gx0,gx1,gx1,gx0], y=[0,0,0,0], z=[gz_bot,gz_bot,gz_top,gz_top],
        i=[0,0], j=[1,2], k=[2,3],
        color=glass_col, opacity=0.50,
        flatshading=True, showscale=False, hoverinfo="skip",
    ))

    # Window frame lines
    fw = dict(color="#7ab8e0", width=1.8)
    for xl, yl, zl in [
        ([gx0,gx1],[0,0],[gz_bot,gz_bot]),
        ([gx0,gx1],[0,0],[gz_top,gz_top]),
        ([gx0,gx0],[0,0],[gz_bot,gz_top]),
        ([gx1,gx1],[0,0],[gz_bot,gz_top]),
        ([(gx0+gx1)/2]*2,[0,0],[gz_bot,gz_top]),
        ([gx0,gx1],[0,0],[(gz_bot+gz_top)/2]*2),
    ]:
        traces.append(go.Scatter3d(x=xl,y=yl,z=zl,mode="lines",
                                   line=fw,hoverinfo="skip",showlegend=False))

    # ── Sun indicator ────────────────────────────────────────────────────────
    ori_deg = ORIENTATION_MAP.get(orientation_label, 180)
    ang = math.radians(ori_deg)
    sx  = W/2 + 1.4 * math.sin(ang)
    sy  = D/2 + 1.4 * math.cos(ang)
    ax  = W/2 + 0.5 * math.sin(ang)
    ay  = D/2 + 0.5 * math.cos(ang)
    traces.append(go.Scatter3d(
        x=[sx, ax], y=[sy, ay], z=[H*0.65, H*0.55],
        mode="lines+markers+text",
        line=dict(color="#f0c040", width=3),
        marker=dict(size=[10,3], color="#f0c040"),
        text=["☀", ""], textfont=dict(color="#f0c040", size=12),
        textposition="top center",
        hoverinfo="skip", showlegend=False,
    ))

    # ── Floor grid ──────────────────────────────────────────────────────────
    grid_lines = []
    for xi in np.linspace(0, W, 5):
        grid_lines += [([xi,xi],[0,D],[0,0])]
    for yi in np.linspace(0, D, 5):
        grid_lines += [([0,W],[yi,yi],[0,0])]
    for xl,yl,zl in grid_lines:
        traces.append(go.Scatter3d(x=xl,y=yl,z=zl,mode="lines",
                                   line=dict(color="#1e2a3a",width=1),
                                   hoverinfo="skip",showlegend=False))

    # ── Dimension annotations ────────────────────────────────────────────────
    for xv,yv,zv,txt in [
        (W/2,  -0.3, 0.05, f"W = {W:.1f} m"),
        (-0.3, D/2,  0.05, f"D = {D:.1f} m"),
        (-0.3, 0,    H/2,  f"H = 3.0 m"),
        ((gx0+gx1)/2, -0.15, gz_top+0.12, f"WWR {wwr:.0%}"),
    ]:
        traces.append(go.Scatter3d(
            x=[xv], y=[yv], z=[zv], mode="text",
            text=[txt], textfont=dict(color="#a0aabf", size=9),
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=370,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor    = "rgba(0,0,0,0)",
            xaxis      = dict(visible=False, range=[-0.6, W+0.6]),
            yaxis      = dict(visible=False, range=[-0.6, D+0.6]),
            zaxis      = dict(visible=False, range=[0,    H+0.4]),
            camera     = dict(eye=dict(x=1.55, y=-1.55, z=1.05)),
            aspectmode = "manual",
            aspectratio= dict(x=W/3.0, y=D/3.0, z=H/3.0),
        ),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
models, load_errors = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLORS — used in both CSS and inline HTML blocks
# ─────────────────────────────────────────────────────────────────────────────
dark = st.session_state.dark_mode
T = {
    "bg":       "#0d1117" if dark else "#f4f6fa",
    "s1":       "#161b26" if dark else "#ffffff",
    "s2":       "#1c2333" if dark else "#eef0f6",
    "s3":       "#222a3a" if dark else "#e4e8f0",
    "border":   "#2d3650" if dark else "#c8cedc",
    "text":     "#ffffff" if dark else "#0d1117",
    "text_dim": "#a0aabf" if dark else "#4a5270",
    "gold":     "#f0a500",
    "toggle_label": "Light Mode" if dark else "Dark Mode",
    "toggle_icon":  "☀" if dark else "☽",
}

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hdr_left, hdr_mid, hdr_right = st.columns([2.2, 0.38, 0.8], gap="small")

with hdr_left:
    st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">S-TCML &nbsp;&middot;&nbsp; Thermal Comfort Predictor</div>
    <div class="hdr-sub">SURROGATE ML &nbsp;&middot;&nbsp; WEST CAIRO OFFICE BUILDINGS &nbsp;&middot;&nbsp; ENERGYPLUS + XGBOOST</div>
  </div>
  <span class="pill">PMV / PPD</span>
  <span class="pill">90% Quantile Intervals</span>
  <span class="pill">ASHRAE 55</span>
  <span class="pill">v2.0</span>
</div>
""", unsafe_allow_html=True)

with hdr_mid:
    st.markdown(f"""
<div style="padding:3px 0 5px;border-bottom:1px solid {T['border']};height:100%;
            display:flex;flex-direction:column;justify-content:center;align-items:center;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
              color:{T['text_dim']};text-align:center;margin-bottom:4px;">
    Display
  </div>
</div>
""", unsafe_allow_html=True)
    if st.button(f"{T['toggle_icon']}  {T['toggle_label']}", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

with hdr_right:
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:3px 0 5px;
            border-bottom:1px solid {T['border']};height:100%;">
  <div>
    <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=R78MZ3FzakWFcN8uuKSnuwjDcFCkSUpOgNR3aIEY0WRUM01LNEEyWDFMOEFZWEJNRUwySzlXWDkwMC4u"
       target="_blank"
       style="display:inline-block;background:#f0a500;color:#000000;
              font-family:'IBM Plex Mono',monospace;font-weight:700;
              font-size:0.75rem;letter-spacing:.05em;text-decoration:none;
              padding:5px 14px;border-radius:6px;white-space:nowrap;">
      &#128172; Share Feedback
    </a>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                color:{T['text_dim']};margin-top:4px;line-height:1.4;">
      Your feedback helps improve this tool.<br>
      It takes less than 2 minutes &mdash; thank you!
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_in, col_res, col_3d = st.columns([1.0, 1.25, 1.75], gap="small")

# ══════════════════════════════════════════
# COL 1 — INPUTS
# ══════════════════════════════════════════
with col_in:
    st.markdown('<div class="card-title">INPUT PARAMETERS</div>', unsafe_allow_html=True)

    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=7,
        format_func=lambda m: MONTH_SHORT[m],
    )
    season_name, season_emoji = SEASON_MAP[month]
    st.caption(f"{season_emoji} {season_name}  -  {MONTH_FULL[month]}")

    orientation_label = st.selectbox(
        "Glazing Orientation",
        ["North up", "East right", "South down", "West left"],
        index=2,
    )
    ori_deg = ORIENTATION_MAP[orientation_label]

    wall_width = st.slider("Wall Width (m)  (exterior glazed wall width)", 2.0, 9.0, 5.0, 0.5)
    room_depth = st.slider("Room Depth (m)", 2.0, 9.0, 5.0, 0.5)
    wwr        = st.slider("WWR (Window-to-Wall Ratio)", 0.10, 0.95, 0.40, 0.05, format="%.2f")
    st.caption(f"Glazing covers {wwr:.0%} of facade")

    st.markdown("""
<div class="ref-box">
  <b>PMV Scale</b><br>
  &minus;3 Cold &nbsp;&middot;&nbsp; &minus;0.5&rarr;+0.5 &#x2705; &nbsp;&middot;&nbsp; +3 Hot<br>
  <b>PPD Target</b> &nbsp; &le; 10% &nbsp;(ASHRAE 55)
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# COL 2 — RESULTS
# ══════════════════════════════════════════
with col_res:
    results = predict_all(models, month, wall_width, room_depth, ori_deg, wwr)
    has_point = ("PMV_point" in results and "PPD_point" in results)
    has_qi    = ("PMV_q05"   in results and "PPD_q05"   in results)

    if not has_point:
        st.error("⛔ Point prediction models not loaded — check MODELS_DIR path.")
        st.stop()

    pmv = results["PMV_point"]
    ppd = results["PPD_point"]
    pmv_lbl, pmv_col = pmv_label(pmv)
    ppd_lbl, ppd_col = ppd_label(ppd)

    st.markdown('<div class="card-title">PREDICTION RESULTS</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="result-card" style="border-top:3px solid {pmv_col};">
          <div style="font-family:'IBM Plex Mono';font-size:0.58rem;
                      color:{T['text_dim']};letter-spacing:.1em;">PMV</div>
          <div class="big-val" style="color:{pmv_col};">{pmv:+.2f}</div>
          <span class="badge" style="color:{pmv_col};border-color:{pmv_col};">{pmv_lbl}</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(gauge(pmv, "PMV", T['text']), use_container_width=True,
                        config={"displayModeBar": False})

    with r2:
        st.markdown(f"""
        <div class="result-card" style="border-top:3px solid {ppd_col};">
          <div style="font-family:'IBM Plex Mono';font-size:0.58rem;
                      color:{T['text_dim']};letter-spacing:.1em;">PPD</div>
          <div class="big-val" style="color:{ppd_col};">{ppd:.1f}%</div>
          <span class="badge" style="color:{ppd_col};border-color:{ppd_col};">{ppd_lbl}</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(gauge(ppd, "PPD", T['text']), use_container_width=True,
                        config={"displayModeBar": False})

    # Intervals
    if has_qi:
        st.markdown('<div class="card-title" style="margin-top:3px;">~ 90% PREDICTION INTERVALS</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(interval_chart(results, T['text'], T['text_dim'], T['border']),
                        use_container_width=True, config={"displayModeBar": False})

        ic1, ic2 = st.columns(2)
        with ic1:
            lo, med, hi = results["PMV_q05"], results["PMV_q50"], results["PMV_q95"]
            st.markdown(f"""
            <div class="interval-row">
              <span style="color:{T['text']};">PMV</span> &nbsp;
              <b style="color:{T['text']};">{lo:+.2f}</b> &rarr;
              <b style="color:#f0a500;">{med:+.2f}</b> &rarr;
              <b style="color:{T['text']};">{hi:+.2f}</b><br>
              <span style="color:{T['text_dim']};">width</span>
              <b style="color:{T['text']};">{hi-lo:.2f}</b>
            </div>""", unsafe_allow_html=True)
        with ic2:
            lo, med, hi = results["PPD_q05"], results["PPD_q50"], results["PPD_q95"]
            st.markdown(f"""
            <div class="interval-row">
              <span style="color:{T['text']};">PPD</span> &nbsp;
              <b style="color:{T['text']};">{lo:.1f}%</b> &rarr;
              <b style="color:#f0a500;">{med:.1f}%</b> &rarr;
              <b style="color:{T['text']};">{hi:.1f}%</b><br>
              <span style="color:{T['text_dim']};">width</span>
              <b style="color:{T['text']};">{hi-lo:.1f}%</b>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:0.62rem;color:{T['text_dim']};
            background:{T['s2']};border-radius:5px;padding:6px 9px;margin-top:5px;
            line-height:1.75;border-left:2px solid {T['border']};">
  <b style="color:{T['text']};">How to read the intervals:</b><br>
  <b style="color:#f0a500;">&#9670; Gold diamond</b> = point prediction (XGBoost best model)<br>
  <b style="color:{T['text']};">q05</b> = lower bound &nbsp;&middot;&nbsp;
  <b style="color:#f0a500;">q50</b> = median &nbsp;&middot;&nbsp;
  <b style="color:{T['text']};">q95</b> = upper bound<br>
  The shaded band is the <b style="color:{T['text']};">90% prediction interval</b> &mdash;
  the true value has a 90% probability of falling within this range.
  A <b style="color:{T['text']};">narrow band</b> indicates higher model confidence.
</div>
""", unsafe_allow_html=True)

    # Design guidance — moved to col_3d, no longer rendered here

# ══════════════════════════════════════════
# COL 3 — 3D ROOM
# ══════════════════════════════════════════
with col_3d:
    st.markdown('<div class="card-title">3D ROOM GEOMETRY - LIVE PREVIEW</div>',
                unsafe_allow_html=True)

    fig_3d = make_3d_room(wall_width, room_depth, wwr, orientation_label,
                          results.get("PMV_point"))
    st.plotly_chart(fig_3d, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False})

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric("Width",   f"{wall_width} m")
    with g2: st.metric("Depth",   f"{room_depth} m")
    with g3: st.metric("WWR",     f"{wwr:.0%}")
    with g4: st.metric("Facing",  orientation_label.split()[0])

    # Design Insights — dynamic tip based on PMV
    if pmv < -0.5:
        insight_icon  = "&#10052;"   # snowflake
        insight_color = "#3d9be9"
        insight_text  = f"Space is <b>too cool</b> (PMV {pmv:+.2f}). Try increasing WWR, South orientation, or reducing room depth."
    elif pmv > 0.5:
        insight_icon  = "&#9728;"    # sun
        insight_color = "#e87040"
        insight_text  = f"Space is <b>too warm</b> (PMV {pmv:+.2f}). Reduce WWR, use North/East facing orientation."
    else:
        insight_icon  = "&#10003;"   # checkmark
        insight_color = "#3ecf6e"
        insight_text  = f"<b>Thermally comfortable</b> &mdash; PMV {pmv:+.2f}, PPD {ppd:.1f}%. Meets ASHRAE 55 criteria &#x2705;"

    st.markdown(f"""
<div style="margin-top:8px;background:{T['s2']};border-left:3px solid {insight_color};
            border-radius:0 8px 8px 0;padding:9px 13px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
              letter-spacing:.12em;text-transform:uppercase;
              color:{insight_color};margin-bottom:5px;">
    &#128161; Design Insights
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
              color:{T['text']};line-height:1.65;">
    {insight_text}
  </div>
</div>
""", unsafe_allow_html=True)

    if load_errors:
        with st.expander(f"WARNING: {len(load_errors)} model file(s) missing"):
            for e in load_errors:
                st.caption(f"• {e}")
            st.caption(f"Expected path: {MODELS_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# FULL-WIDTH ABOUT ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:{T['text_dim']};
            background:rgba(240,165,0,0.06);border-left:3px solid #f0a500;
            border-radius:0 6px 6px 0;padding:8px 16px;margin-top:6px;
            line-height:1.8;white-space:nowrap;">
  <b style="color:#f0a500;">About this Tool</b> &nbsp;&mdash;&nbsp;
  This tool is part of a <b style="color:{T['text']};">PhD research project</b>. &nbsp;
  It predicts PMV &amp; PPD for <b style="color:{T['text']};">air-conditioned</b> office rooms in
  <b style="color:{T['text']};">West Cairo, Egypt</b> using a surrogate ML model trained on
  EnergyPlus parametric simulations.
</div>
""", unsafe_allow_html=True)
