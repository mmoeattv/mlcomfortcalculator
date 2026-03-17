import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib
import pandas as pd
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="S-TCML V.01 - Thermal Comfort AI", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #f5f7fa !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; color: #0097a7 !important; font-weight: 700 !important; }
    .uncertainty-box {
        background-color: #fff3e0; 
        padding: 12px; 
        border-radius: 10px; 
        border-left: 6px solid #ffb74d; 
        margin-bottom: 20px;
    }
    .range-text { color: #e65100; font-weight: 700; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (Strict Matching with GitHub Names)
# ==========================================
@st.cache_resource
def load_all_assets():
    try:
        # الأسماء مطابقة تماماً للصورة التي أرسلتِها
        q05 = joblib.load("XGBoost_PMV_q05.pkl") 
        q50 = joblib.load("XGBoost_PMV_q50.pkl") 
        q95 = joblib.load("XGBoost_PMV_q95.pkl")
        ppd = joblib.load("XGBoost_PPD.pkl") # لاحظي الاسم في الصورة لا يحتوي على model
        scaler = joblib.load("scaler_X.pkl") # الـ X كبيرة كما في الصورة
        
        return q05, q50, q95, ppd, scaler
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
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

    raw = np.array([[month_map[month_str], wall_w, depth, orient_map[orient], wwr]])
    scaled = scaler_x.transform(raw)

    pmv_mid  = round(float(q50.predict(scaled)[0]), 2)
    pmv_low  = round(float(q05.predict(scaled)[0]), 2)
    pmv_high = round(float(q95.predict(scaled)[0]), 2)
    ppd_val  = round(float(ppd_model.predict(scaled)[0]), 1)
    
    return pmv_mid, ppd_val, pmv_low, pmv_high

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("🏛️ S-TCML Architectural Thermal Comfort Predictor")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2.2, 1])

with col1:
    st.subheader("⚙ Design Controls")
    month_val   = st.select_slider("Month", options=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], value="Jul")
    wall_width  = st.slider("Wall Width (m)", 2.0, 6.0, 3.5)
    room_depth  = st.slider("Room Depth (m)", 2.0, 10.0, 5.0)
    orientation = st.selectbox("Orientation", ["North", "East", "South", "West"])
    wwr         = st.slider("WWR", 0.1, 0.9, 0.4)

with col3:
    st.subheader("📊 Quantitative Results")
    pmv, ppd, low_p, high_p = get_predictions(month_val, wall_width, room_depth, orientation, wwr)

    if pmv is not None:
        st.metric("PMV (Predicted Mean Vote)", pmv)
        st.markdown(f"""
        <div class="uncertainty-box">
            <span style="color:#8d6e63; font-size:0.85rem;">90% Prediction Range (QR):</span><br>
            <span class="range-text">[{low_p} to {high_p}]</span>
        </div>
        """, unsafe_allow_html=True)
        st.metric("PPD (% Dissatisfied)", f"{ppd}%")
        
        # مؤشر الراحة
        comfort_ok = -0.5 <= pmv <= 0.5
        color = "#00897b" if comfort_ok else ("#e64a19" if pmv > 0 else "#1976d2")
        status = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv > 0 else "🔵 Too Cool")
        st.markdown(f"<p style='text-align:center; color:{color}; font-weight:bold;'>{status}</p>", unsafe_allow_html=True)
