import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Thermal Comfort AI - PhD", layout="wide")

st.markdown("""
    <style>
    /* ── Global layout ── */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] { display: none; }

    /* ── White background ── */
    .stApp, body, [data-testid="stAppViewContainer"] {
        background-color: #f5f7fa !important;
    }
    [data-testid="stHeader"] { background-color: #f5f7fa !important; }

    /* ── Headings ── */
    h1 {
        font-size: clamp(1.4rem, 2.2vw, 2rem) !important;
        margin-bottom: 0.2rem !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        font-size: clamp(1rem, 1.4vw, 1.3rem) !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }

    /* ── Body text ── */
    p, div, span, label {
        font-size: clamp(0.88rem, 1.1vw, 1.05rem) !important;
        color: #333 !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetricValue"] {
        font-size: clamp(1.8rem, 2.5vw, 2.6rem) !important;
        color: #0097a7 !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 2px solid #b2dfdb;
        border-radius: 12px;
        padding: 0.7rem 1rem 0.5rem !important;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(0,150,136,0.08);
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: clamp(0.82rem, 1vw, 0.98rem) !important;
        color: #00796b !important;
        font-weight: 600 !important;
    }

    /* ── Description text ── */
    .description-text {
        font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
        color: #555 !important;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    /* ── Slider track ── */
    div[data-testid="stSlider"] > div > div > div {
        height: 8px !important;
        border-radius: 4px !important;
        background: #cfd8dc !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #00897b, #00bcd4) !important;
        height: 8px !important;
    }
    /* ── Slider THUMB — large & teal ── */
    div[data-testid="stSlider"] span[role="slider"] {
        width: 28px !important;
        height: 28px !important;
        background: #00897b !important;
        border: 4px solid #ffffff !important;
        border-radius: 50% !important;
        box-shadow: 0 0 0 3px #00897b55, 0 3px 10px rgba(0,0,0,0.2) !important;
        top: -10px !important;
        cursor: grab !important;
    }
    div[data-testid="stSlider"] span[role="slider"]:active {
        background: #00695c !important;
        cursor: grabbing !important;
    }
    /* Slider label */
    div[data-testid="stSlider"] label {
        font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
        color: #37474f !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    /* Slider min/max tick labels */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        font-size: clamp(0.78rem, 0.95vw, 0.9rem) !important;
        color: #78909c !important;
    }

    /* ── Select-slider ── */
    div[data-testid="stSelectSlider"] label {
        font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
        color: #37474f !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSelectSlider"] span[role="slider"] {
        width: 28px !important;
        height: 28px !important;
        background: #00897b !important;
        border: 4px solid #ffffff !important;
        border-radius: 50% !important;
        box-shadow: 0 0 0 3px #00897b55, 0 3px 10px rgba(0,0,0,0.2) !important;
        top: -10px !important;
    }

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] label {
        font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
        color: #37474f !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        border: 1.5px solid #b0bec5 !important;
        border-radius: 8px !important;
        font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
    }

    /* ── Control panel card ── */
    div[data-testid="column"]:first-child {
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }

    /* ── Right panel card ── */
    div[data-testid="column"]:last-child {
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }

    /* ── Divider ── */
    hr { margin: 0.6rem 0 !important; border-color: #e0e0e0 !important; }

    /* ── Alert / info box ── */
    div[data-testid="stAlert"] {
        background: #e0f7fa !important;
        border: 1px solid #80deea !important;
        border-radius: 8px !important;
        font-size: clamp(0.85rem, 1.05vw, 0.98rem) !important;
        color: #00600f !important;
    }

    /* ── Link button ── */
    .stLinkButton a {
        font-size: clamp(0.9rem, 1.1vw, 1rem) !important;
        padding: 0.5rem 1.2rem !important;
        background: #00897b !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* ── Caption ── */
    div[data-testid="stCaptionContainer"] p {
        font-size: clamp(0.75rem, 0.9vw, 0.88rem) !important;
        color: #90a4ae !important;
    }

    /* ── Column gaps ── */
    div[data-testid="column"] {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
PMV_PATH = r"XGBoost_PMV_model.pkl"
PPD_PATH = r"XGBoost_PPD_model.pkl"

@st.cache_resource
def load_research_models():
    try:
        pmv_m = joblib.load(PMV_PATH)
        ppd_m = joblib.load(PPD_PATH)
        return pmv_m, ppd_m
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

pmv_model, ppd_model = load_research_models()

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
def get_predictions(wall_w, depth, orient, wwr, month_str):
    orient_map = {"North": 0, "East": 90, "South": 180, "West": 270}
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    features = np.array([[month_map[month_str], wall_w, depth, orient_map[orient], wwr]])
    if pmv_model is not None and ppd_model is not None::
        pred_pmv = pmv_model.predict(features)[0]
        pred_ppd = ppd_model.predict(features)[0]
        return round(float(pred_pmv), 2), round(float(pred_ppd), 1)
    else:
        return 0.0, 5.0

# ==========================================
# 4. ROOM 3D GEOMETRY BUILDER
# ==========================================
def build_room_figure(wall_width, room_depth, wwr, height=3.0):
    fig = go.Figure()

    W, D, H = wall_width, room_depth, height
    frame_t = 0.06  # frame thickness

    # ── Floor ──
    fig.add_trace(go.Mesh3d(
        x=[0, 0, W, W], y=[0, D, D, 0], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#d0d8e4', opacity=0.95, flatshading=True, showscale=False, name="Floor"
    ))

    # ── Ceiling ──
    fig.add_trace(go.Mesh3d(
        x=[0, 0, W, W], y=[0, D, D, 0], z=[H, H, H, H],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#e8edf3', opacity=0.6, flatshading=True, showscale=False, name="Ceiling"
    ))

    # ── Back wall (y=D) ──
    fig.add_trace(go.Mesh3d(
        x=[0, 0, W, W], y=[D, D, D, D], z=[0, H, H, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#c8d0dc', opacity=0.9, flatshading=True, showscale=False, name="Back Wall"
    ))

    # ── Left wall (x=0) ──
    fig.add_trace(go.Mesh3d(
        x=[0, 0, 0, 0], y=[0, D, D, 0], z=[0, 0, H, H],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#bcc6d4', opacity=0.9, flatshading=True, showscale=False, name="Left Wall"
    ))

    # ── Right wall (x=W) ──
    fig.add_trace(go.Mesh3d(
        x=[W, W, W, W], y=[0, D, D, 0], z=[0, 0, H, H],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#bcc6d4', opacity=0.9, flatshading=True, showscale=False, name="Right Wall"
    ))

    # ── Front wall panels around window (y=0) ──
    # Window dimensions
    win_w = wall_width * np.sqrt(wwr)
    win_h = height * np.sqrt(wwr)
    wx0 = (W - win_w) / 2       # window left x
    wx1 = wx0 + win_w            # window right x
    wz0 = (H - win_h) / 2 + 0.1 # sill height
    wz1 = wz0 + win_h            # window top z

    # left panel
    fig.add_trace(go.Mesh3d(
        x=[0, 0, wx0, wx0], y=[0, 0, 0, 0], z=[0, H, H, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#b8c4d0', opacity=0.85, flatshading=True, showscale=False, name="Front Wall"
    ))
    # right panel
    fig.add_trace(go.Mesh3d(
        x=[wx1, wx1, W, W], y=[0, 0, 0, 0], z=[0, H, H, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#b8c4d0', opacity=0.85, flatshading=True, showscale=False, name="Front Wall"
    ))
    # bottom panel
    fig.add_trace(go.Mesh3d(
        x=[wx0, wx0, wx1, wx1], y=[0, 0, 0, 0], z=[0, wz0, wz0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#b8c4d0', opacity=0.85, flatshading=True, showscale=False, name="Front Wall"
    ))
    # top panel
    fig.add_trace(go.Mesh3d(
        x=[wx0, wx0, wx1, wx1], y=[0, 0, 0, 0], z=[wz1, H, H, wz1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#b8c4d0', opacity=0.85, flatshading=True, showscale=False, name="Front Wall"
    ))

    # ── Glazing panel ──
    fig.add_trace(go.Mesh3d(
        x=[wx0, wx0, wx1, wx1], y=[0, 0, 0, 0], z=[wz0, wz1, wz1, wz0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#00bcd4', opacity=0.45, flatshading=True, showscale=False, name="Glazing"
    ))

    # ── Window frame edges (outer rect) using Scatter3d lines ──
    def edge(xs, ys, zs, color='#aaaaaa', width=4):
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=color, width=width),
            showlegend=False
        )

    # outer frame
    fig.add_trace(edge(
        [wx0, wx1, wx1, wx0, wx0],
        [0,   0,   0,   0,   0],
        [wz0, wz0, wz1, wz1, wz0],
        color='#888888', width=5
    ))

    # mullion vertical (center)
    mid_x = (wx0 + wx1) / 2
    fig.add_trace(edge([mid_x, mid_x], [0, 0], [wz0, wz1], color='#666666', width=3))

    # transom horizontal (center)
    mid_z = (wz0 + wz1) / 2
    fig.add_trace(edge([wx0, wx1], [0, 0], [mid_z, mid_z], color='#666666', width=3))

    # frame border lines (room edges)
    room_edges = [
        # bottom
        ([0, W], [0, 0], [0, 0]), ([0, W], [D, D], [0, 0]), ([0, 0], [0, D], [0, 0]), ([W, W], [0, D], [0, 0]),
        # top
        ([0, W], [0, 0], [H, H]), ([0, W], [D, D], [H, H]), ([0, 0], [0, D], [H, H]), ([W, W], [0, D], [H, H]),
        # verticals
        ([0, 0], [0, 0], [0, H]), ([W, W], [0, 0], [0, H]), ([0, 0], [D, D], [0, H]), ([W, W], [D, D], [0, H]),
    ]
    for xs, ys, zs in room_edges:
        fig.add_trace(edge(xs, ys, zs, color='#90a4b8', width=2))

    # ── Floor grid lines ──
    grid_cols = 4
    grid_rows = max(2, int(round(room_depth / wall_width * grid_cols)))
    for i in range(1, grid_cols):
        x_ = W * i / grid_cols
        fig.add_trace(edge([x_, x_], [0, D], [0.002, 0.002], color='#c8d4e0', width=1))
    for j in range(1, grid_rows):
        y_ = D * j / grid_rows
        fig.add_trace(edge([0, W], [y_, y_], [0.002, 0.002], color='#c8d4e0', width=1))

    # ── Sunlight ray effect through glazing ──
    for ray_x in np.linspace(wx0 + win_w * 0.15, wx1 - win_w * 0.15, 3):
        for ray_z in np.linspace(wz0 + win_h * 0.2, wz1 - win_h * 0.2, 2):
            fig.add_trace(go.Scatter3d(
                x=[ray_x, ray_x + (ray_x - W/2) * 0.3],
                y=[0.02, D * 0.55],
                z=[ray_z, ray_z - win_h * 0.15],
                mode='lines',
                line=dict(color='rgba(0,188,212,0.12)', width=2),
                showlegend=False
            ))

    # ── Camera & layout ──
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-W * 0.1, W * 1.1]),
            yaxis=dict(visible=False, range=[-D * 0.05, D * 1.1]),
            zaxis=dict(visible=False, range=[-0.1, H * 1.15]),
            bgcolor='#f0f4f8',
            camera=dict(
                eye=dict(x=-1.5, y=-2.0, z=1.1),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=wall_width, y=room_depth * 0.65, z=height * 0.9)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_white",
        autosize=True,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        showlegend=False
    )
    return fig

# ==========================================
# 5. MAIN TITLE + FEEDBACK BUTTON (TOP ROW)
# ==========================================

# Inject modal CSS + JS
st.markdown("""
<style>
/* ── Feedback button fixed top-right ── */
.feedback-btn-wrap {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin-bottom: 0.3rem;
}
.feedback-trigger {
    background: #00897b;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1.1rem;
    font-size: clamp(0.85rem, 1.05vw, 0.98rem);
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,137,123,0.25);
    transition: background 0.2s;
    text-decoration: none;
}
.feedback-trigger:hover { background: #00695c; }

/* ── Modal overlay ── */
#fb-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 9998;
    align-items: center;
    justify-content: center;
}
#fb-overlay.open { display: flex; }

/* ── Modal box ── */
#fb-modal {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.22);
    width: min(680px, 92vw);
    height: min(620px, 88vh);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    z-index: 9999;
}
#fb-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.1rem 0.6rem;
    border-bottom: 1px solid #e0e0e0;
    background: #f5f7fa;
}
#fb-modal-header span {
    font-size: 1rem;
    font-weight: 700;
    color: #1a1a2e;
}
#fb-close {
    background: none;
    border: none;
    font-size: 1.4rem;
    cursor: pointer;
    color: #607d8b;
    line-height: 1;
    padding: 0 0.2rem;
    transition: color 0.15s;
}
#fb-close:hover { color: #c0392b; }
#fb-modal iframe {
    flex: 1;
    border: none;
    width: 100%;
}
</style>

<!-- Feedback modal markup -->
<div id="fb-overlay" onclick="if(event.target===this) closeModal()">
  <div id="fb-modal">
    <div id="fb-modal-header">
      <span>📋 PhD Research Feedback Form</span>
      <button id="fb-close" onclick="closeModal()" title="Close">✕</button>
    </div>
    <iframe
      id="fb-iframe"
      src=""
      data-src="https://forms.office.com/r/THfuycGkDZ"
      title="Feedback Form"
      allow="clipboard-write">
    </iframe>
  </div>
</div>

<script>
function openModal() {
    var overlay = document.getElementById('fb-overlay');
    var iframe  = document.getElementById('fb-iframe');
    // Lazy-load the form only on first open
    if (!iframe.src || iframe.src === window.location.href) {
        iframe.src = iframe.getAttribute('data-src');
    }
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
}
function closeModal() {
    document.getElementById('fb-overlay').classList.remove('open');
    document.body.style.overflow = '';
}
// Close on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
});
</script>
""", unsafe_allow_html=True)

# Title row: title on left, feedback button on right
title_col, btn_col = st.columns([5, 1])
with title_col:
    st.title("🏛️ Architectural Thermal Comfort Predictor")
    st.markdown("""<div class="description-text">
    AI-driven prediction of <b>PMV</b> and <b>PPD</b> from architectural parameters — 
    PhD research tool for early-stage thermal performance optimization.
    </div>""", unsafe_allow_html=True)
with btn_col:
    st.markdown("""
    <div class="feedback-btn-wrap" style="height:100%; padding-top:0.6rem;">
        <button class="feedback-trigger" onclick="openModal()">📋 Give Feedback</button>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns([1, 2.2, 1])

# ── COLUMN 1: CONTROL PANEL ──
with col1:
    st.subheader("⚙ Control Panel")
    wall_width = st.slider("Exterior Wall Width (m)", 0.5, 5.0, 3.5, step=0.1)
    room_depth = st.slider("Room Depth (m)", 2.0, 10.0, 5.0, step=0.25)
    wwr        = st.slider("Window-to-Wall Ratio", 0.1, 0.9, 0.4, step=0.05)
    orientation = st.selectbox("Glazing Orientation", ["North", "East", "South", "West"])
    month_val  = st.select_slider("Month",
        options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        value="Jul")

# ── COLUMN 2: 3D VISUALIZATION ──
with col2:
    st.subheader("📐 Room Geometry")
    fig = build_room_figure(wall_width, room_depth, wwr)
    st.plotly_chart(fig, width='stretch')

# ── COLUMN 3: PREDICTIONS ──
with col3:
    st.subheader("📊 Predictions")
    current_pmv, current_ppd = get_predictions(wall_width, room_depth, orientation, wwr, month_val)

    st.metric(label="PMV — Predicted Mean Vote", value=current_pmv)
    st.metric(label="PPD — % Dissatisfied", value=f"{current_ppd:.1f}%")

    # Comfort band gauge
    st.write("")
    comfort_ok = -0.5 <= current_pmv <= 0.5
    bar_color  = "#00897b" if comfort_ok else ("#e64a19" if current_pmv > 0 else "#1976d2")
    label_text = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if current_pmv > 0 else "🔵 Too Cool")
    pct = min(max((current_pmv + 3) / 6, 0), 1) * 100

    st.markdown(f"""
    <div style="margin-top:6px;">
      <div style="background:#e0e8f0; border-radius:8px; height:14px; overflow:hidden;">
        <div style="width:{pct:.1f}%; background:{bar_color}; height:100%; border-radius:8px;
                    transition: width 0.4s ease;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#78909c; margin-top:3px;">
        <span>-3 Cold</span><span style="color:{bar_color}; font-weight:600;">{label_text}</span><span>+3 Hot</span>
      </div>
    </div>
    <p style="font-size:0.75rem; color:#90a4ae; margin-top:8px;">ASHRAE 55 target: −0.5 to +0.5</p>
    """, unsafe_allow_html=True)

# ── end of app ──


