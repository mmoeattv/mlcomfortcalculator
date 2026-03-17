import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="S-TCML V.01 - Thermal Comfort AI", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
.stApp { background-color: #f5f7fa !important; }
h1 { font-size: 2rem !important; color: #1a1a2e !important; font-weight: 700 !important; }
div[data-testid="stMetricValue"] { font-size: 2.2rem !important; color: #0097a7 !important; }
.uncertainty-box {
    background-color: #fff3e0; 
    padding: 12px; 
    border-radius: 10px; 
    border-left: 6px solid #ffb74d; 
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.range-text { color: #e65100; font-weight: 700; font-size: 1.1rem; }
.range-label { color: #8d6e63; font-size: 0.85rem; display: block; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS & SCALER
# ==========================================
@st.cache_resource
def load_all_assets():
    try:
        pmv_model = joblib.load("XGBoost_PMV_model.pkl")
        ppd_model = joblib.load("XGBoost_PPD_model.pkl")
        # تحميل موديلات اليقين (Quantiles)
        pmv_q05   = joblib.load("XGBoost_PMV_q05.pkl") 
        pmv_q95   = joblib.load("XGBoost_PMV_q95.pkl")
        scaler    = joblib.load("scaler_X.pkl")
        return pmv_model, ppd_model, pmv_q05, pmv_q95, scaler
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
        return [None]*5

pmv_model, ppd_model, pmv_q05, pmv_q95, scaler_x = load_all_assets()

# ==========================================
# 3. PREDICTION ENGINE (With Range Logic)
# ==========================================
def get_predictions(month_str, wall_w, depth, orient, wwr):
    month_map  = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, 
                  "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    orient_map = {"North":0, "East":90, "South":180, "West":270}

    if any(m is None for m in [pmv_model, ppd_model, pmv_q05, pmv_q95, scaler_x]):
        return None, None, None, None

    # ترتيب الخصائص كما في التدريب
    raw = np.array([[month_map[month_str], wall_w, depth, orient_map[orient], wwr]])
    scaled = scaler_x.transform(raw)

    # التوقعات الثلاثة (الوسيط، الأدنى، الأعلى)
    pmv_mid  = round(float(pmv_model.predict(scaled)[0]), 2)
    pmv_low  = round(float(pmv_q05.predict(scaled)[0]), 2)
    pmv_high = round(float(pmv_q95.predict(scaled)[0]), 2)
    
    ppd = round(float(ppd_model.predict(scaled)[0]), 1)
    
    return pmv_mid, ppd, pmv_low, pmv_high

# ==========================================
# 4. 3D ROOM VISUALIZATION (Plotly)
# ==========================================
def build_room_figure(wall_width, room_depth, wwr, height=3.0):
    fig = go.Figure()
    W, D, H = wall_width, room_depth, height
    
    # Floor & Ceiling
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[0,0,0,0], color='#d0d8e4', opacity=0.9))
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[H,H,H,H], color='#e8edf3', opacity=0.6))
    
    # Back & Side Walls
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[D,D,D,D], z=[0,H,H,0], color='#c8d0dc'))
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,D,D,0], z=[0,0,H,H], color='#bcc6d4'))
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,D,D,0], z=[0,0,H,H], color='#bcc6d4'))

    # Window logic (Simplified for speed)
    win_w, win_h = W * np.sqrt(wwr), H * np.sqrt(wwr)
    wx0, wz0 = (W-win_w)/2, (H-win_h)/2 + 0.1
    fig.add_trace(go.Mesh3d(x=[wx0,wx0,wx0+win_w,wx0+win_w], y=[0,0,0,0], 
                           z=[wz0,wz0+win_h,wz0+win_h,wz0], color='#00bcd4', opacity=0.5))

    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                      aspectratio=dict(x=W, y=D*0.7, z=H)), margin=dict(l=0,r=0,b=0,t=0))
    return fig

# ==========================================
# 5. INTERFACE LAYOUT
# ==========================================
st.title("🏛️ S-TCML Architectural Thermal Comfort Predictor")
st.markdown("Quantitative Analysis for Early-Stage Design Optimization")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2.2, 1])

with col1:
    st.subheader("⚙ Parameters")
    month_val   = st.select_slider("Month", options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], value="Jul")
    wall_width  = st.slider("Exterior Wall Width (m)", 0.5, 5.0, 3.5, step=0.1)
    room_depth  = st.slider("Room Depth (m)", 2.0, 10.0, 5.0, step=0.25)
    orientation = st.selectbox("Glazing Orientation", ["North", "East", "South", "West"])
    wwr         = st.slider("Window-to-Wall Ratio", 0.1, 0.9, 0.4, step=0.05)

with col2:
    st.subheader("📐 Room Geometry")
    st.plotly_chart(build_room_figure(wall_width, room_depth, wwr), use_container_width=True)

with col3:
    st.subheader("📊 Quantitative Results")
    pmv, ppd, low_p, high_p = get_predictions(month_val, wall_width, room_depth, orientation, wwr)

    if pmv is not None:
        # 1. Main PMV Metric
        st.metric("PMV (Predicted Mean Vote)", pmv)
        
        # 2. Uncertainty Range Box (The PhD addition)
        st.markdown(f"""
        <div class="uncertainty-box">
            <span class="range-label">90% Prediction Interval (QR Method):</span>
            <span class="range-text">Range: [{low_p} to {high_p}]</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. PPD Metric
        st.metric("PPD (% Dissatisfied)", f"{ppd}%")

        # 4. Visual Comfort Scale
        comfort_ok = -0.5 <= pmv <= 0.5
        color = "#00897b" if comfort_ok else ("#e64a19" if pmv > 0 else "#1976d2")
        status = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv > 0 else "🔵 Too Cool")
        
        pct = min(max((pmv + 3) / 6, 0), 1) * 100
        st.markdown(f"""
        <div style="background:#e0e8f0; border-radius:10px; height:18px; margin-top:10px;">
            <div style="width:{pct}%; background:{color}; height:100%; border-radius:10px;"></div>
        </div>
        <p style="text-align:center; color:{color}; font-weight:bold; margin-top:5px;">{status}</p>
        <p style="font-size:0.8rem; color:gray;">ASHRAE 55 Target: [-0.5, +0.5]</p>
        """, unsafe_allow_html=True)

    else:
        st.error("Model files missing. Please check your repository.")
