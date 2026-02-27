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
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }
.stApp, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f7fa !important;
}
[data-testid="stHeader"] { background-color: #f5f7fa !important; }
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
p, div, span, label {
    font-size: clamp(0.88rem, 1.1vw, 1.05rem) !important;
    color: #333 !important;
}
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
.description-text {
    font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
    color: #555 !important;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
div[data-testid="stSlider"] > div > div > div {
    height: 8px !important;
    border-radius: 4px !important;
    background: #cfd8dc !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #00897b, #00bcd4) !important;
    height: 8px !important;
}
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
div[data-testid="stSlider"] label {
    font-size: clamp(0.88rem, 1.1vw, 1rem) !important;
    color: #37474f !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    font-size: clamp(0.78rem, 0.95vw, 0.9rem) !important;
    color: #78909c !important;
}
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
div[data-testid="column"]:first-child {
    background: #ffffff;
    border-radius: 14px;
    padding: 1rem 1.1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
div[data-testid="column"]:last-child {
    background: #ffffff;
    border-radius: 14px;
    padding: 1rem 1.1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
hr { margin: 0.6rem 0 !important; border-color: #e0e0e0 !important; }
div[data-testid="column"] {
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
}
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
    padding: 0.55rem 1.2rem;
    font-size: clamp(0.85rem, 1.05vw, 0.98rem);
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,137,123,0.25);
    transition: background 0.2s;
}
.feedback-trigger:hover { background: #00695c; }
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
#fb-modal-header span { font-size: 1rem; font-weight: 700; color: #1a1a2e; }
#fb-close {
    background: none; border: none; font-size: 1.4rem;
    cursor: pointer; color: #607d8b; line-height: 1;
    padding: 0 0.2rem; transition: color 0.15s;
}
#fb-close:hover { color: #c0392b; }
#fb-modal iframe { flex: 1; border: none; width: 100%; }
</style>

<div id="fb-overlay" onclick="if(event.target===this) closeModal()">
  <div id="fb-modal">
    <div id="fb-modal-header">
      <span>📋 PhD Research Feedback Form</span>
      <button id="fb-close" onclick="closeModal()" title="Close">✕</button>
    </div>
    <iframe id="fb-iframe" src="" data-src="https://forms.office.com/r/THfuycGkDZ"
      title="Feedback Form" allow="clipboard-write"></iframe>
  </div>
</div>

<script>
function openModal() {
    var overlay = document.getElementById('fb-overlay');
    var iframe  = document.getElementById('fb-iframe');
    if (!iframe.src || iframe.src === window.location.href)
        iframe.src = iframe.getAttribute('data-src');
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
}
function closeModal() {
    document.getElementById('fb-overlay').classList.remove('open');
    document.body.style.overflow = '';
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });
</script>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD MODELS & SCALER
# — Files must be in the same folder as this script in the GitHub repo
# ==========================================
@st.cache_resource
def load_models_and_scaler():
    try:
        pmv_model = joblib.load("XGBoost_PMV_model.pkl")
        ppd_model = joblib.load("XGBoost_PPD_model.pkl")
        scaler    = joblib.load("scaler_X.pkl")
        return pmv_model, ppd_model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

pmv_model, ppd_model, scaler_x = load_models_and_scaler()


# ==========================================
# 3. PREDICTION
# Feature order (must match training): [Month, WallWidth, Depth, Orientation, WWR]
# ==========================================
def get_predictions(month_str, wall_w, depth, orient, wwr):
    month_map  = {"Jan": 1,  "Feb": 2,  "Mar": 3,  "Apr": 4,
                  "May": 5,  "Jun": 6,  "Jul": 7,  "Aug": 8,
                  "Sep": 9,  "Oct": 10, "Nov": 11, "Dec": 12}
    orient_map = {"North": 0, "East": 90, "South": 180, "West": 270}

    if pmv_model is None or ppd_model is None or scaler_x is None:
        return None, None

    # Build raw feature array in exact training column order
    raw = np.array([[
        month_map[month_str],   # col 0 — Month
        wall_w,                 # col 1 — Wall Width
        depth,                  # col 2 — Room Depth
        orient_map[orient],     # col 3 — Orientation (degrees)
        wwr                     # col 4 — WWR
    ]])

    # Scale exactly as done during training
    scaled = scaler_x.transform(raw)

    pmv = round(float(pmv_model.predict(scaled)[0]), 2)
    ppd = round(float(ppd_model.predict(scaled)[0]), 1)
    return pmv, ppd


# ==========================================
# 4. 3D ROOM GEOMETRY
# ==========================================
def build_room_figure(wall_width, room_depth, wwr, height=3.0):
    fig = go.Figure()
    W, D, H = wall_width, room_depth, height

    def mesh(x, y, z, color, opacity=0.9):
        return go.Mesh3d(x=x, y=y, z=z, i=[0,0], j=[1,2], k=[2,3],
                         color=color, opacity=opacity, flatshading=True,
                         showscale=False, showlegend=False)

    def edge(xs, ys, zs, color='#90a4b8', width=2):
        return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                            line=dict(color=color, width=width), showlegend=False)

    # Walls
    fig.add_trace(mesh([0,0,W,W],[0,D,D,0],[0,0,0,0],   '#d0d8e4', 0.95))  # floor
    fig.add_trace(mesh([0,0,W,W],[0,D,D,0],[H,H,H,H],   '#e8edf3', 0.60))  # ceiling
    fig.add_trace(mesh([0,0,W,W],[D,D,D,D],[0,H,H,0],   '#c8d0dc'))         # back
    fig.add_trace(mesh([0,0,0,0],[0,D,D,0],[0,0,H,H],   '#bcc6d4'))         # left
    fig.add_trace(mesh([W,W,W,W],[0,D,D,0],[0,0,H,H],   '#bcc6d4'))         # right

    # Window
    win_w = W * np.sqrt(wwr)
    win_h = H * np.sqrt(wwr)
    wx0 = (W - win_w) / 2;  wx1 = wx0 + win_w
    wz0 = (H - win_h) / 2 + 0.1;  wz1 = wz0 + win_h

    for x_verts, z_verts in [
        ([0,   0,   wx0, wx0], [0,   H,   H,   0  ]),
        ([wx1, wx1, W,   W  ], [0,   H,   H,   0  ]),
        ([wx0, wx0, wx1, wx1], [0,   wz0, wz0, 0  ]),
        ([wx0, wx0, wx1, wx1], [wz1, H,   H,   wz1]),
    ]:
        fig.add_trace(mesh(x_verts, [0,0,0,0], z_verts, '#b8c4d0', 0.85))

    # Glazing
    fig.add_trace(mesh([wx0,wx0,wx1,wx1],[0,0,0,0],[wz0,wz1,wz1,wz0],'#00bcd4',0.45))

    # Window frame + mullion + transom
    fig.add_trace(edge([wx0,wx1,wx1,wx0,wx0],[0,0,0,0,0],[wz0,wz0,wz1,wz1,wz0],'#888888',5))
    mid_x = (wx0+wx1)/2;  mid_z = (wz0+wz1)/2
    fig.add_trace(edge([mid_x,mid_x],[0,0],[wz0,wz1],'#666666',3))
    fig.add_trace(edge([wx0,wx1],[0,0],[mid_z,mid_z],'#666666',3))

    # Room edges
    for xs,ys,zs in [
        ([0,W],[0,0],[0,0]),([0,W],[D,D],[0,0]),([0,0],[0,D],[0,0]),([W,W],[0,D],[0,0]),
        ([0,W],[0,0],[H,H]),([0,W],[D,D],[H,H]),([0,0],[0,D],[H,H]),([W,W],[0,D],[H,H]),
        ([0,0],[0,0],[0,H]),([W,W],[0,0],[0,H]),([0,0],[D,D],[0,H]),([W,W],[D,D],[0,H]),
    ]:
        fig.add_trace(edge(xs,ys,zs,'#90a4b8',2))

    # Floor grid
    for i in range(1,4):
        x_ = W*i/4
        fig.add_trace(edge([x_,x_],[0,D],[0.002,0.002],'#c8d4e0',1))
    for j in range(1, max(2, int(round(D/W*4)))):
        y_ = D*j/max(2,int(round(D/W*4)))
        fig.add_trace(edge([0,W],[y_,y_],[0.002,0.002],'#c8d4e0',1))

    # Light rays
    for rx in np.linspace(wx0+win_w*0.15, wx1-win_w*0.15, 3):
        for rz in np.linspace(wz0+win_h*0.2, wz1-win_h*0.2, 2):
            fig.add_trace(go.Scatter3d(
                x=[rx, rx+(rx-W/2)*0.3], y=[0.02, D*0.55],
                z=[rz, rz-win_h*0.15], mode='lines',
                line=dict(color='rgba(0,188,212,0.12)', width=2), showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-W*0.1, W*1.1]),
            yaxis=dict(visible=False, range=[-D*0.05, D*1.1]),
            zaxis=dict(visible=False, range=[-0.1, H*1.15]),
            bgcolor='#f0f4f8',
            camera=dict(eye=dict(x=-1.5,y=-2.0,z=1.1), up=dict(x=0,y=0,z=1)),
            aspectmode='manual',
            aspectratio=dict(x=W, y=D*0.65, z=H*0.9)
        ),
        margin=dict(l=0,r=0,b=0,t=0),
        template="plotly_white",
        autosize=True,
        paper_bgcolor='#ffffff',
        showlegend=False
    )
    return fig


# ==========================================
# 5. TITLE + FEEDBACK BUTTON
# ==========================================
title_col, btn_col = st.columns([5, 1])
with title_col:
    st.title("🏛️S-TCML V.01 Architectural Thermal Comfort Predictor")
    st.markdown("""<div class="description-text">
    AI-driven prediction of <b>PMV</b> and <b>PPD</b> from architectural parameters —
    PhD research tool for early-stage thermal performance optimization.
    </div>""", unsafe_allow_html=True)
with btn_col:
    st.markdown("""
    <div class="feedback-btn-wrap" style="height:100%; padding-top:0.8rem;">
        <button class="feedback-trigger" onclick="openModal()">📋 Give Feedback</button>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ==========================================
# 6. THREE-COLUMN LAYOUT
# ==========================================
col1, col2, col3 = st.columns([1, 2.2, 1])

with col1:
    st.subheader("⚙ Control Panel")
    month_val   = st.select_slider("Month",
        options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        value="Jul")
    wall_width  = st.slider("Exterior Wall Width (m)", 0.5, 5.0, 3.5, step=0.1)
    room_depth  = st.slider("Room Depth (m)", 2.0, 10.0, 5.0, step=0.25)
    orientation = st.selectbox("Glazing Orientation", ["North", "East", "South", "West"])
    wwr         = st.slider("Window-to-Wall Ratio", 0.1, 0.9, 0.4, step=0.05)

with col2:
    st.subheader("📐 Room Geometry")
    st.plotly_chart(build_room_figure(wall_width, room_depth, wwr), width='stretch')

with col3:
    st.subheader("📊 Predictions")
    pmv, ppd = get_predictions(month_val, wall_width, room_depth, orientation, wwr)

    if pmv is not None:
        st.metric("PMV — Predicted Mean Vote", pmv)
        st.metric("PPD — % Dissatisfied", f"{ppd:.1f}%")

        comfort_ok = -0.5 <= pmv <= 0.5
        bar_color  = "#00897b" if comfort_ok else ("#e64a19" if pmv > 0 else "#1976d2")
        label_text = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv > 0 else "🔵 Too Cool")
        pct = min(max((pmv + 3) / 6, 0), 1) * 100

        st.markdown(f"""
        <div style="margin-top:6px;">
          <div style="background:#e0e8f0;border-radius:8px;height:14px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:8px;transition:width 0.4s;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#78909c;margin-top:3px;">
            <span>-3 Cold</span>
            <span style="color:{bar_color};font-weight:600;">{label_text}</span>
            <span>+3 Hot</span>
          </div>
        </div>
        <p style="font-size:0.75rem;color:#90a4ae;margin-top:8px;">ASHRAE 55 target: −0.5 to +0.5</p>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not load model files. Make sure XGBoost_PMV_model.pkl, XGBoost_PPD_model.pkl, and scaler_X.pkl are in the same folder as this script.")
