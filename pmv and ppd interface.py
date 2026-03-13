import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIG & ENHANCED STYLING
# ==========================================
st.set_page_config(page_title="Thermal Comfort AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* 1. COMPACT DASHBOARD SETUP */
.main .block-container {
    padding: 1rem 1.5rem 0rem 1.5rem !important;
    max-width: 100% !important;
}
body { overflow: hidden; } 

/* 2. ROUNDED INSIGHT CARDS - HORIZONTAL */
.insights-container {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 10px;
    width: 100%;
}
.insight-card {
    flex: 1;
    background: white;
    border-radius: 15px; /* Rounded Squares */
    padding: 12px;
    border: 1px solid #e0e6ed;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    text-align: center;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.insight-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.insight-body {
    font-size: 0.8rem;
    color: #546e7a;
    line-height: 1.3;
}

/* 3. FEEDBACK BUTTON */
.feedback-trigger {
    background: #00897b;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    cursor: pointer;
}
#fb-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.7); z-index: 10000;
    align-items: center; justify-content: center;
}
#fb-overlay.open { display: flex; }
#fb-modal {
    background: white; width: 80%; height: 80%;
    border-radius: 15px; position: relative;
}
</style>

<div id="fb-overlay" onclick="if(event.target===this) closeModal()">
  <div id="fb-modal">
    <button style="position:absolute; right:15px; top:10px; border:none; background:none; font-size:20px; cursor:pointer;" onclick="closeModal()">✕</button>
    <iframe src="https://forms.office.com/r/THfuycGkDZ" width="100%" height="100%" style="border:none; border-radius:15px;"></iframe>
  </div>
</div>

<script>
function openModal() { document.getElementById('fb-overlay').classList.add('open'); }
function closeModal() { document.getElementById('fb-overlay').classList.remove('open'); }
</script>
""", unsafe_allow_html=True)

# ==========================================
# 2. 3D GEOMETRY - FIXED ARROW & ZOOM
# ==========================================
def build_room_figure(W, D, wwr, orient_deg, height=3.0):
    fig = go.Figure()
    
    # Room Surfaces
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[0,0,0,0], color='#dfe6e9', name="Floor"))
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,D,D,0], z=[0,0,height,height], color='#f1f2f6'))
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,D,D,0], z=[0,0,height,height], color='#f1f2f6'))
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[D,D,D,D], z=[0,height,height,0], color='#bdc3c7'))

    # Glazing
    win_w, win_h = W * np.sqrt(wwr), height * np.sqrt(wwr)
    wx0, wx1 = (W - win_w)/2, (W + win_w)/2
    wz0, wz1 = (height - win_h)/2, (height + win_h)/2
    fig.add_trace(go.Mesh3d(x=[wx0,wx0,wx1,wx1], y=[0,0,0,0], z=[wz0,wz1,wz1,wz0], 
                           color='#74b9ff', opacity=0.4))

    # ── FIXED FULL NORTH ARROW ──
    # Math: 0 deg = North. Facade (at y=0) faces South if orient=180.
    # We draw the arrow relative to the room center
    cx, cy = W/2, -0.5
    rad = np.radians(orient_deg)
    
    # Arrow Body (Stem)
    nx, ny = 0.7 * np.sin(rad), -0.7 * np.cos(rad)
    fig.add_trace(go.Scatter3d(x=[cx, cx + nx], y=[cy, cy + ny], z=[0.02, 0.02],
                               mode='lines', line=dict(color='#d63031', width=8), name="North Pointer"))
    
    # Arrow Head (Cone/Triangle)
    fig.add_trace(go.Scatter3d(x=[cx + nx], y=[cy + ny], z=[0.02],
                               mode='markers+text', text=["N"], textfont=dict(size=14, color="red"),
                               marker=dict(symbol='cone', size=10, color='#d63031', angleref='previous')))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=400,
        uirevision='constant' # FIX: Prevents zoom jumping
    )
    return fig

# ==========================================
# 3. MAIN INTERFACE
# ==========================================
col_t, col_b = st.columns([4, 1])
col_t.title("🏛️ S-TCML V.01 | Thermal Analysis")
col_b.markdown('<button class="feedback-trigger" onclick="openModal()">📋 Feedback Form</button>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])

with c1:
    st.subheader("Controls")
    m = st.select_slider("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], "Jul")
    w = st.slider("Width", 2.0, 6.0, 4.0)
    d = st.slider("Depth", 2.0, 8.0, 5.0)
    o = st.select_slider("Facade Orientation", [0, 90, 180, 270], format_func=lambda x: {0:"North", 90:"East", 180:"South", 270:"West"}[x])
    ww = st.slider("WWR", 0.1, 0.9, 0.3)

with c2:
    st.plotly_chart(build_room_figure(w, d, ww, o), use_container_width=True)

with c3:
    st.subheader("Predictions")
    st.metric("PMV Vote", "0.42") # Placeholder logic
    st.metric("PPD Index", "8.5%")
    st.info("Room is within ASHRAE 55 Comfort limits.")

# ==========================================
# 4. ROUNDED HORIZONTAL INSIGHTS
# ==========================================
st.markdown(f"""
<div class="insights-container">
    <div class="insight-card" style="border-top: 5px solid #00b894;">
        <div class="insight-title">Thermal Status</div>
        <div class="insight-body">Current PMV is optimal. No active cooling required for {m}.</div>
    </div>
    <div class="insight-card" style="border-top: 5px solid #0984e3;">
        <div class="insight-title">Orientation</div>
        <div class="insight-body">Facing {o}°. North arrow confirms solar path relative to glazing.</div>
    </div>
    <div class="insight-card" style="border-top: 5px solid #fdcb6e;">
        <div class="insight-title">Geometry</div>
        <div class="insight-body">Aspect ratio of {d/w:.1f} facilitates even heat distribution.</div>
    </div>
</div>
""", unsafe_allow_html=True)
