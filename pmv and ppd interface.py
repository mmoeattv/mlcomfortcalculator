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
    .main { background-color: #1e1e1e; color: white; }
    div[data-testid="stMetricValue"] { font-size: 40px; color: #00d1b2; }
    .description-text { font-size: 18px; color: #b0b0b0; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (D: Drive Paths)
# ==========================================
PMV_PATH = r"D:\my phd monthly database\my database monthly pmv and ppd\web_deployed_models\model_pmv_raw.pkl"
PPD_PATH = r"D:\my phd monthly database\my database monthly pmv and ppd\web_deployed_models\model_ppd_raw.pkl"

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
    
    # Corrected Feature Order: [Month, Wall Width, Depth, Orientation, WWR]
    features = np.array([[month_map[month_str], wall_w, depth, orient_map[orient], wwr]])
    
    if pmv_model and ppd_model:
        pred_pmv = pmv_model.predict(features)[0]
        pred_ppd = ppd_model.predict(features)[0]
        return round(float(pred_pmv), 2), round(float(pred_ppd), 1)
    else:
        return 0.0, 5.0

# ==========================================
# 4. MAIN TITLE & DESCRIPTION
# ==========================================
st.title("🏛️ Architectural Thermal Comfort Predictor")
st.markdown("""
    <div class="description-text">
    This AI-driven tool predicts <b>Predicted Mean Vote (PMV)</b> and <b>Predicted Percentage Dissatisfied (PPD)</b> 
    based on architectural parameters. Developed as part of a PhD research project to optimize thermal performance 
    in early-stage design.
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

# --- COLUMN 1: CONTROL PANEL ---
with col1:
    st.subheader("Control Panel")
    wall_width = st.slider("Exterior Wall Width (m)", 0.5, 5.0, 3.5)
    room_depth = st.slider("Room Depth (m)", 2.0, 10.0, 5.0)
    wwr = st.slider("Window to Wall Ratio (WWR)", 0.1, 0.9, 0.4)
    orientation = st.selectbox("Glazing Orientation", ["North", "East", "South", "West"])
    month_val = st.select_slider("Month", options=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], value="Jul")

# --- COLUMN 2: GEOMETRY VISUALIZATION ---
with col2:
    st.subheader("Generated Room Geometry")
    height = 3.0
    
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=[0, 0, wall_width, wall_width, 0, 0, wall_width, wall_width],
        y=[0, room_depth, room_depth, 0, 0, room_depth, room_depth, 0],
        z=[0, 0, 0, 0, height, height, height, height],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.2, color='lightgray', name="Room Volume"
    ))

    win_w, win_h = wall_width * np.sqrt(wwr), height * np.sqrt(wwr)
    off_x, off_z = (wall_width - win_w) / 2, (height - win_h) / 2

    fig.add_trace(go.Mesh3d(
        x=[off_x, off_x, off_x+win_w, off_x+win_w],
        y=[0, 0, 0, 0],
        z=[off_z, off_z+win_h, off_z+win_h, off_z],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='cyan', opacity=0.8, name="Glazing Area"
    ))

    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      margin=dict(l=0, r=0, b=0, t=0), template="plotly_dark", height=500)
    st.plotly_chart(fig, width='stretch')

# --- COLUMN 3: ANALYTICS ---
with col3:
    st.subheader("Predictions")
    current_pmv, current_ppd = get_predictions(wall_width, room_depth, orientation, wwr, month_val)
    st.metric(label="PMV (Predicted Mean Vote)", value=current_pmv)
    st.metric(label="PPD (Percentage Dissatisfied)", value=f"{current_ppd:.1f}%")
    
    st.write("Comfort Range")
    comfort_color = "green" if -0.5 <= current_pmv <= 0.5 else "red"
    st.markdown(f"<div style='width:100%; height:10px; background-color:{comfort_color}; border-radius:5px;'></div>", unsafe_allow_html=True)
    st.caption("Target: -0.5 to 0.5 (ASHRAE 55)")

# ==========================================
# 5. INTEGRATED FEEDBACK SECTION
# ==========================================
st.markdown("---")
st.subheader("📊 Research Feedback & Expert Opinion")
st.write("Your feedback helps validate the AI model's accuracy from an architectural perspective.")

# Link to Microsoft Form
form_url = "https://forms.office.com/r/THfuycGkDZ" # Replace with your full link

f_col1, f_col2 = st.columns([1, 1])
with f_col1:
    st.info("Submit your ratings and comments via our official PhD feedback form.")
    st.link_button("Go to Feedback Form", form_url, type="primary")

with f_col2:
    st.caption("Feedback collected will be used for model validation in accordance with PhD research ethics.")