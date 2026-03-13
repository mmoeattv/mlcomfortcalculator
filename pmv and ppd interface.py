import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & COMPACT STYLING
# ==========================================
st.set_page_config(page_title="Thermal Comfort AI - PhD", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* 1. COMPACT LAYOUT - Eliminate scrolling */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 98% !important;
}
/* Hide scrollbars for a clean "Dashboard" feel if content fits */
body { overflow: hidden; } 

section[data-testid="stSidebar"] { display: none; }

.stApp { background-color: #f8fafd !important; }

/* Elegant Headers */
h1 {
    font-size: 1.6rem !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #1a1a2e !important;
}
h3 {
    font-size: 1rem !important;
    margin-bottom: 0.4rem !important;
    color: #455a64 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Metric styling */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #00796b !important;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e6ed;
    padding: 0.4rem !important;
    border-radius: 10px;
}

/* FEEDBACK BUTTON FIX - Specific ID and z-index */
.feedback-trigger {
    background: #00897b;
    color: white !important;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}
.feedback-trigger:hover { background: #004d40; }

/* Modal Overlay */
#fb-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 10000;
    align-items: center;
    justify-content: center;
}
#fb-overlay.open { display: flex; }
#fb-modal {
    background: white;
    width: 600px;
    height: 500px;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
}
#fb-modal iframe { flex: 1; border: none; }
#close-btn { 
    position: absolute; right: 10px; top: 10px; 
    cursor: pointer; font-weight: bold; font-size: 20px; 
}

/* Insights Strip - Smaller and more compact */
.insights-strip {
    display: flex;
    gap: 10px;
    margin-top: 5px;
}
.insight-card {
    flex: 1;
    padding: 8px;
    border-radius: 8px;
    font-size: 0.8rem !important;
    border-left: 4px solid #ccc;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>

<div id="fb-overlay">
  <div id="fb-modal">
    <div id="close-btn" onclick="closeModal()">✕</div>
    <iframe src="https://forms.office.com/r/THfuycGkDZ"></iframe>
  </div>
</div>

<script>
function openModal() {
    document.getElementById('fb-overlay').classList.add('open');
}
function closeModal() {
    document.getElementById('fb-overlay').classList.remove('open');
}
</script>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models_and_scaler():
    try:
        pmv_model = joblib.load("XGBoost_PMV_model.pkl")
        ppd_model = joblib.load("XGBoost_PPD_model.pkl")
        scaler    = joblib.load("scaler_X.pkl")
        return pmv_model, ppd_model, scaler
    except:
        return None, None, None

pmv_model, ppd_model, scaler_x = load_models_and_scaler()

# ==========================================
# 3. 3D GEOMETRY - REALISTIC + NORTH ARROW + ZOOM FIX
# ==========================================
def build_room_figure(W, D, wwr, orient_deg, height=3.0):
    fig = go.Figure()
    
    # ── Realistic Textures & Colors ──
    # Floor (Polished Concrete/Wood feel)
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[0,0,0,0], i=[0,0], j=[1,2], k=[2,3], 
                           color='#dcdde1', opacity=1, name="Floor"))
    # Ceiling
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[height,height,height,height], i=[0,0], j=[1,2], k=[2,3], 
                           color='#f5f6fa', opacity=0.5))
    # Walls (Light Grey/Off-white)
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,D,D,0], z=[0,0,height,height], i=[0,0], j=[1,2], k=[2,3], color='#ecf0f1')) # Left
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,D,D,0], z=[0,0,height,height], i=[0,0], j=[1,2], k=[2,3], color='#ecf0f1')) # Right
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[D,D,D,D], z=[0,height,height,0], i=[0,0], j=[1,2], k=[2,3], color='#bdc3c7')) # Back

    # ── Glazing Calculation ──
    win_w = W * np.sqrt(wwr)
    win_h = height * np.sqrt(wwr)
    wx0, wx1 = (W - win_w)/2, (W + win_w)/2
    wz0, wz1 = (height - win_h)/2, (height + win_h)/2

    # Glass (Realistic cyan transparent)
    fig.add_trace(go.Mesh3d(x=[wx0,wx0,wx1,wx1], y=[0,0,0,0], z=[wz0,wz1,wz1,wz0], i=[0,0], j=[1,2], k=[2,3], 
                           color='#00d2ff', opacity=0.3, name="Glazing"))

    # ── NORTH ARROW (Representative of facade orientation) ──
    # Place arrow on the ground in front of the window
    arrow_len = 0.8
    # Pivot logic: The window is ALWAYS at y=0. 
    # If orientation is North, North is Y negative. 
    rad = np.radians(orient_deg)
    nx, ny = arrow_len * np.sin(rad), -arrow_len * np.cos(rad)
    
    # Arrow Stem
    fig.add_trace(go.Scatter3d(x=[W/2, W/2 - nx], y=[-0.2, -0.2 - ny], z=[0.05, 0.05],
                               mode='lines', line=dict(color='red', width=6), name="North"))
    # Arrow Head
    fig.add_trace(go.Scatter3d(x=[W/2 - nx], y=[-0.2 - ny], z=[0.05],
                               mode='markers+text', marker=dict(symbol='diamond', size=5, color='red'),
                               text=["N"], textposition="top center", showlegend=False))

    # ── ZOOM ISSUE FIX (Fixed camera eye and projection) ──
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=1.2), # Standardized Perspective
                projection=dict(type='perspective')
            ),
            aspectmode='data' # Keeps proportions realistic
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=380,
        uirevision='constant' # FIX: Keeps zoom/pan even when sliders move
    )
    return fig

# ==========================================
# 4. PREDICTIONS & LOGIC
# ==========================================
def get_predictions(m, w, d, o, wwr):
    if not pmv_model: return 0.0, 0.0
    m_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    raw = np.array([[m_map[m], w, d, o, wwr]])
    scaled = scaler_x.transform(raw)
    return pmv_model.predict(scaled)[0], ppd_model.predict(scaled)[0]

# ==========================================
# 5. UI LAYOUT (ONE SCREEN)
# ==========================================
t1, t2 = st.columns([4, 1])
with t1:
    st.title("🏛️ S-TCML V.01: PhD Thermal Predictor")
with t2:
    st.markdown('<button class="feedback-trigger" onclick="openModal()">📋 Feedback</button>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.1, 2.2, 1])

with c1:
    st.subheader("⚙️ Inputs")
    month = st.select_slider("Month", options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], value="Jul")
    w_w = st.slider("Wall Width", 2.0, 6.0, 4.0)
    r_d = st.slider("Room Depth", 2.0, 8.0, 5.0)
    orient = st.select_slider("Facade Orientation", options=[0, 90, 180, 270], format_func=lambda x: {0:"North", 90:"East", 180:"South", 270:"West"}[x])
    wwr_val = st.slider("WWR", 0.1, 0.8, 0.3)

pmv, ppd = get_predictions(month, w_w, r_d, orient, wwr_val)

with c2:
    st.subheader("📐 3D Preview")
    st.plotly_chart(build_room_figure(w_w, r_d, wwr_val, orient), use_container_width=True)

with c3:
    st.subheader("📊 Results")
    st.metric("PMV (Thermal Vote)", f"{pmv:.2f}")
    st.metric("PPD (Dissatisfied)", f"{ppd:.1f}%")
    
    # Comfort Gauge
    color = "#27ae60" if -0.5 <= pmv <= 0.5 else "#e67e22"
    st.markdown(f"""
        <div style="background:#eee; height:15px; border-radius:10px;">
            <div style="background:{color}; width:{min(100, (pmv+3)*16.6)}%; height:100%; border-radius:10px;"></div>
        </div>
        <center><small>-3 (Cold) | 0 (Neutral) | +3 (Hot)</small></center>
    """, unsafe_allow_html=True)

# 6. COMPACT INSIGHTS STRIP
st.markdown('<div class="insights-strip">', unsafe_allow_html=True)
if pmv > 0.5:
    st.markdown('<div class="insight-card" style="border-color:#e67e22"><b>🔥 Overheating:</b> Consider reducing WWR or adding shading.</div>', unsafe_allow_html=True)
elif pmv < -0.5:
    st.markdown('<div class="insight-card" style="border-color:#3498db"><b>❄️ Underheating:</b> Increase South-facing glazing for solar gain.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="insight-card" style="border-color:#2ecc71"><b>✅ Optimal:</b> Parameters meet ASHRAE 55 standards.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="insight-card"><b>🧭 Orientation:</b> Facade is facing {orient}°. North arrow updated in 3D.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
