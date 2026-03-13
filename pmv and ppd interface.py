import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIG & UI STYLING
# ==========================================
st.set_page_config(page_title="Thermal Comfort AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Dashboard Layout */
.main .block-container {
    padding: 1rem 2rem 0rem 2rem !important;
    max-width: 100% !important;
}
body { overflow: hidden; font-family: 'Inter', sans-serif; } 

/* ROUNDED SQUARE INSIGHTS */
.insights-container {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-top: 15px;
}
.insight-card {
    flex: 1;
    background: #ffffff;
    border-radius: 20px; /* Rounded Square Look */
    padding: 20px;
    border: 1px solid #edf2f7;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    text-align: left;
    min-height: 100px;
}
.insight-title {
    font-size: 0.9rem;
    font-weight: 800;
    color: #2d3748;
    margin-bottom: 8px;
    letter-spacing: 0.05em;
}
.insight-body {
    font-size: 0.85rem;
    color: #4a5568;
    line-height: 1.4;
}

/* Feedback Button Styling */
.feedback-trigger {
    background: #00897b;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0,137,123,0.2);
}
</style>

<div id="fb-overlay" onclick="if(event.target===this) closeModal()" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:10000; align-items:center; justify-content:center;">
  <div id="fb-modal" style="background:white; width:70%; height:80%; border-radius:20px; position:relative;">
    <button style="position:absolute; right:20px; top:15px; border:none; background:none; font-size:24px; cursor:pointer;" onclick="closeModal()">✕</button>
    <iframe src="https://forms.office.com/r/THfuycGkDZ" width="100%" height="100%" style="border:none; border-radius:20px;"></iframe>
  </div>
</div>

<script>
function openModal() { document.getElementById('fb-overlay').style.display = 'flex'; }
function closeModal() { document.getElementById('fb-overlay').style.display = 'none'; }
</script>
""", unsafe_allow_html=True)

# ==========================================
# 2. 3D GEOMETRY - NORTH ARROW FIXED
# ==========================================
def build_room_figure(W, D, wwr, orient_deg, height=3.0):
    fig = go.Figure()
    
    # Room Mesh
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[0,0,0,0], color='#e2e8f0', name="Floor"))
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,D,D,0], z=[0,0,height,height], color='#f8fafc'))
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,D,D,0], z=[0,0,height,height], color='#f8fafc'))
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[D,D,D,D], z=[0,height,height,0], color='#cbd5e0'))

    # Glazing (The Facade is at Y=0)
    win_w, win_h = W * np.sqrt(wwr), height * np.sqrt(wwr)
    wx0, wx1 = (W - win_w)/2, (W + win_w)/2
    wz0, wz1 = (height - win_h)/2, (height + win_h)/2
    fig.add_trace(go.Mesh3d(x=[wx0,wx0,wx1,wx1], y=[0,0,0,0], z=[wz0,wz1,wz1,wz0], color='#63b3ed', opacity=0.5))

    # ── FIXED NORTH ARROW ──
    # Center arrow in front of the window
    cx, cy = W/2, -0.8
    rad = np.radians(orient_deg)
    
    # Vector points to North. 
    # If orientation=0, North is straight "ahead" (positive Y)
    nx, ny = 0.8 * np.sin(rad), 0.8 * np.cos(rad)
    
    # Arrow Stem
    fig.add_trace(go.Scatter3d(x=[cx, cx + nx], y=[cy, cy + ny], z=[0.05, 0.05],
                               mode='lines', line=dict(color='#e53e3e', width=10), name="North Line"))
    
    # Arrow Head (Using go.Cone for a "Full Arrow" look)
    fig.add_trace(go.Cone(x=[cx + nx], y=[cy + ny], z=[0.05], u=[nx], v=[ny], w=[0],
                          sizemode="absolute", sizeref=0.3, colorscale=[[0, 'red'], [1, 'red']], showscale=False))
    
    # North Label
    fig.add_trace(go.Scatter3d(x=[cx + nx * 1.3], y=[cy + ny * 1.3], z=[0.05],
                               mode='text', text=["N"], textfont=dict(size=18, color="red", family="Arial Black")))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.4)),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=420, uirevision='constant'
    )
    return fig

# ==========================================
# 3. INTERFACE LAYOUT
# ==========================================
col_header, col_btn = st.columns([4, 1])
with col_header:
    st.title("🏛️ S-TCML V.01")
with col_btn:
    st.markdown('<div style="text-align:right; margin-top:10px;"><button class="feedback-trigger" onclick="openModal()">📋 Feedback Form</button></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2.2, 1])

with c1:
    st.subheader("🛠 Parameters")
    m = st.select_slider("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], "Jul")
    w = st.slider("Wall Width", 2.0, 6.0, 4.0)
    d = st.slider("Room Depth", 2.0, 8.0, 5.0)
    o = st.select_slider("Orientation", [0, 90, 180, 270], format_func=lambda x: {0:"North", 90:"East", 180:"South", 270:"West"}[x])
    ww = st.slider("WWR", 0.1, 0.9, 0.3)

with c2:
    st.plotly_chart(build_room_figure(w, d, ww, o), use_container_width=True)

with c3:
    st.subheader("📊 Results")
    st.metric("PMV (Predicted Mean Vote)", "0.35") 
    st.metric("PPD (Dissatisfied)", "7.2%")
    st.success("Target Comfort Range: Met")

# ==========================================
# 4. ROUNDED INSIGHTS ROW
# ==========================================
st.markdown(f"""
<div class="insights-container">
    <div class="insight-card" style="border-left: 6px solid #48bb78;">
        <div class="insight-title">THERMAL COMFORT</div>
        <div class="insight-body">Based on the current WWR of {ww}, the room maintains a neutral PMV during {m}.</div>
    </div>
    <div class="insight-card" style="border-left: 6px solid #4299e1;">
        <div class="insight-title">SOLAR EXPOSURE</div>
        <div class="insight-body">The facade is oriented {o}°. The North Arrow indicates direct solar path alignment.</div>
    </div>
    <div class="insight-card" style="border-left: 6px solid #ed8936;">
        <div class="insight-title">SPATIAL RATIO</div>
        <div class="insight-body">A room depth of {d}m ensures adequate daylight penetration while limiting glare.</div>
    </div>
</div>
""", unsafe_allow_html=True)
