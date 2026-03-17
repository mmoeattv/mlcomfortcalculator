import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="S-TCML V.01 - Thermal Comfort AI", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #f5f7fa !important; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { 
        font-size: clamp(1.8rem, 2.5vw, 2.6rem) !important; 
        color: #0097a7 !important; 
        font-weight: 700 !important; 
    }
    
    /* Uncertainty Range Box */
    .uncertainty-box {
        background-color: #fff3e0; 
        padding: 15px; 
        border-radius: 12px; 
        border-left: 6px solid #ffb74d; 
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .range-label { color: #8d6e63; font-size: 0.9rem; font-weight: 600; display: block; margin-bottom: 5px; }
    .range-text { color: #e65100; font-weight: 800; font-size: 1.2rem; }
    
    .description-text { color: #555; line-height: 1.6; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (Models & Scaler)
# ==========================================
@st.cache_resource
def load_all_assets():
    try:
        # تحميل الموديلات الثلاثة (الأدنى، الوسيط، الأعلى) لضمان الاتساق
        pmv_q05 = joblib.load("XGBoost_PMV_q05.pkl") 
        pmv_q50 = joblib.load("XGBoost_PMV_q50.pkl") # سنستخدم هذا كالقيمة الأساسية
        pmv_q95 = joblib.load("XGBoost_PMV_q95.pkl")
        
        ppd_model = joblib.load("XGBoost_PPD_model.pkl")
        scaler    = joblib.load("scaler_X.pkl")
        
        return pmv_q05, pmv_q50, pmv_q95, ppd_model, scaler
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}. Ensure all .pkl files are in the same directory.")
        return [None]*5

q05, q50, q95, ppd_model, scaler_x = load_all_assets()

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
def get_predictions(month_str, wall_w, depth, orient, wwr):
    month_map  = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, 
                  "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    orient_map = {"North":0, "East":90, "South":180, "West":270}

    if any(m is None for m in [q05, q50, q95, ppd_model, scaler_x]):
        return None, None, None, None

    # بناء مصفوفة الخصائص (نفس ترتيب التدريب بالظبط)
    raw = np.array([[
        month_map[month_str], # col 0
        wall_w,               # col 1
        depth,                # col 2
        orient_map[orient],   # col 3
        wwr                   # col 4
    ]])

    scaled = scaler_x.transform(raw)

    # حساب التوقعات من موديلات الكوانتايل (لضمان التطابق 100%)
    pmv_low  = round(float(q05.predict(scaled)[0]), 2)
    pmv_mid  = round(float(q50.predict(scaled)[0]), 2)
    pmv_high = round(float(q95.predict(scaled)[0]), 2)
    
    # حساب الـ PPD
    ppd_val = round(float(ppd_model.predict(scaled)[0]), 1)
    
    return pmv_mid, ppd_val, pmv_low, pmv_high

# ==========================================
# 4. 3D ROOM VISUALIZATION
# ==========================================
def build_room_figure(W, D, wwr, H=3.0):
    fig = go.Figure()
    # Floor, Ceiling, Walls
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[0,0,0,0], color='#d0d8e4', opacity=0.9))
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[0,D,D,0], z=[H,H,H,H], color='#e8edf3', opacity=0.5))
    fig.add_trace(go.Mesh3d(x=[0,0,W,W], y=[D,D,D,0], z=[0,H,H,0], color='#c8d0dc')) # Back
    
    # Window
    win_w, win_h = W * np.sqrt(wwr), H * np.sqrt(wwr)
    wx, wz = (W-win_w)/2, (H-win_h)/2 + 0.1
    fig.add_trace(go.Mesh3d(x=[wx, wx, wx+win_w, wx+win_w], y=[0,0,0,0], 
                           z=[wz, wz+win_h, wz+win_h, wz], color='#00bcd4', opacity=0.6))

    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                      aspectratio=dict(x=W, y=D*0.7, z=H)), margin=dict(l=0,r=0,b=0,t=0))
    return fig

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("🏛️ S-TCML Architectural Thermal Comfort Predictor")
st.markdown("<div class='description-text'>PhD Research Tool: Quantitative Uncertainty Modeling for Early Design Stages.</div>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2.2, 1])

with col1:
    st.subheader("⚙ Design Controls")
    month_val   = st.select_slider("Select Month", options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], value="Jul")
    wall_width  = st.slider("Exterior Wall Width (m)", 2.0, 6.0, 3.5, step=0.1)
    room_depth  = st.slider("Room Depth (m)", 2.0, 10.0, 5.0, step=0.25)
    orientation = st.selectbox("Glazing Orientation", ["North", "East", "South", "West"])
    wwr         = st.slider("Window-to-Wall Ratio", 0.1, 0.9, 0.4, step=0.05)

with col2:
    st.subheader("📐 3D Geometry Preview")
    st.plotly_chart(build_room_figure(wall_width, room_depth, wwr), use_container_width=True)

with col3:
    st.subheader("📊 Quantitative Results")
    pmv, ppd, low_p, high_p = get_predictions(month_val, wall_width, room_depth, orientation, wwr)

    if pmv is not None:
        # الـ Metric الأساسي (يعتمد على Median q50)
        st.metric("PMV (Predicted Mean Vote)", pmv)
        
        # صندوق عدم اليقين (يعتمد على q05 و q95)
        st.markdown(f"""
        <div class="uncertainty-box">
            <span class="range-label">90% Prediction Interval (Quantile Method):</span>
            <span class="range-text">Range: [{low_p} to {high_p}]</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("PPD (% Dissatisfied)", f"{ppd}%")

        # مؤشر الحالة الحرارية
        comfort_ok = -0.5 <= pmv <= 0.5
        color = "#00897b" if comfort_ok else ("#e64a19" if pmv > 0 else "#1976d2")
        status = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv > 0 else "🔵 Too Cool")
        
        # حساب موقع المؤشر على المسطرة
        pos = min(max((pmv + 3) / 6, 0), 1) * 100
        st.markdown(f"""
        <div style="background:#e0e8f0; border-radius:10px; height:20px; margin-top:15px; position:relative;">
            <div style="width:{pos}%; background:{color}; height:100%; border-radius:10px; transition: 0.5s;"></div>
        </div>
        <p style="text-align:center; color:{color}; font-weight:bold; margin-top:10px; font-size:1.1rem;">{status}</p>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Error in data processing. Please check model files.")
