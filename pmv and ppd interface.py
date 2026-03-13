import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib

# ──────────────────────────────────────────
# 1. PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="S-TCML V.01",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── strip ALL streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }

/* ── page background & zero padding ── */
html, body, .stApp { background:#eef1f6 !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── kill gap Streamlit injects above first element ── */
[data-testid="stAppViewContainer"] > section > div:first-child {
    padding-top: 0 !important;
}

/* ── HEADER bar ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #fff;
    border-bottom: 1.5px solid #dde3ea;
    padding: 9px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.app-header h1 {
    font-size: 19px !important;
    font-weight: 800 !important;
    color: #1a1a2e !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.app-header p {
    font-size: 12px !important;
    color: #607d8b !important;
    margin: 2px 0 0 !important;
}
.fb-btn {
    background: linear-gradient(135deg,#00897b,#00acc1);
    color: #fff !important;
    border: none; border-radius: 8px;
    padding: 9px 20px;
    font-size: 14px; font-weight: 700;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,137,123,0.35);
    white-space: nowrap;
    transition: opacity 0.15s;
    text-decoration: none;
}
.fb-btn:hover { opacity: 0.85; }

/* ── COLUMN PANELS ── */
.panel {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    padding: 12px 14px;
    height: 100%;
}
.panel-hdr {
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #546e7a !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px !important;
    padding-bottom: 7px;
    border-bottom: 1px solid #eceff1;
}

/* ── SLIDER labels ── */
div[data-testid="stSlider"] label,
div[data-testid="stSelectSlider"] label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #37474f !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="stSlider"],
div[data-testid="stSelectSlider"] {
    margin-bottom: 4px !important;
}
/* slider track */
div[data-testid="stSlider"] > div > div > div {
    height: 5px !important;
    border-radius: 3px !important;
    background: #cfd8dc !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg,#00897b,#00bcd4) !important;
    height: 5px !important;
}
div[data-testid="stSlider"] span[role="slider"],
div[data-testid="stSelectSlider"] span[role="slider"] {
    width: 18px !important; height: 18px !important;
    background: #00897b !important;
    border: 3px solid #fff !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 2.5px #00897b66 !important;
    top: -7px !important;
}
/* current value text */
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stSelectSlider"] div[data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #00796b !important;
    text-align: right !important;
}

/* ── SELECT BOX ── */
div[data-testid="stSelectbox"] label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #37474f !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="stSelectbox"] > div > div {
    font-size: 14px !important;
    background: #f5f7fa !important;
    border: 1.5px solid #cfd8dc !important;
    border-radius: 8px !important;
}

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#fff,#f0fffe);
    border: 1.5px solid #b2dfdb;
    border-radius: 10px;
    padding: 10px 14px 8px !important;
    margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(0,150,136,0.09);
}
div[data-testid="stMetricValue"] {
    font-size: 32px !important;
    color: #00838f !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 11px !important;
    color: #00796b !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── ALL other text ── */
p, div, span, label {
    font-size: 13px !important;
    line-height: 1.45 !important;
}

hr { margin: 6px 0 !important; border-color: #dde3ea !important; }

/* ── INSIGHTS STRIP ── */
.ins-strip {
    display: flex;
    gap: 9px;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding: 2px 0 4px;
}
.ins-strip::-webkit-scrollbar { height: 3px; }
.ins-strip::-webkit-scrollbar-thumb { background:#b0bec5; border-radius:2px; }
.ins-card {
    flex: 1 1 0;
    min-width: 160px;
    border-radius: 12px;
    padding: 10px 13px 9px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
}
.ins-icon  { font-size: 15px; line-height: 1; flex-shrink: 0; }
.ins-title { font-size: 12.5px !important; font-weight: 800 !important;
             color: #1a2e35 !important; line-height: 1.2; }
.ins-body  { font-size: 11.5px !important; color: #455a64 !important;
             line-height: 1.5; margin-top: 3px; }
.ins-body b { font-weight: 700; color: #263238 !important; }
.c-warm { background:#fff3e0; border-top:3px solid #e64a19; }
.c-cool { background:#e3f2fd; border-top:3px solid #1976d2; }
.c-ok   { background:#e8f5e9; border-top:3px solid #2e7d32; }
.c-tip  { background:#f3e5f5; border-top:3px solid #7b1fa2; }
.c-info { background:#e0f7fa; border-top:3px solid #00838f; }

/* ── FEEDBACK MODAL ── */
#fb-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.52); z-index: 99999;
    align-items: center; justify-content: center;
}
#fb-overlay.open { display: flex !important; }
#fb-modal {
    background: #fff; border-radius: 14px;
    box-shadow: 0 16px 56px rgba(0,0,0,0.3);
    width: min(700px,94vw); height: min(640px,90vh);
    display: flex; flex-direction: column; overflow: hidden;
}
#fb-mhdr {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; border-bottom: 1px solid #e0e0e0; background: #f5f7fa;
}
#fb-mhdr span { font-size: 14px !important; font-weight: 700; color: #1a1a2e; }
#fb-close {
    background: none; border: none; font-size: 20px; cursor: pointer;
    color: #607d8b; line-height: 1; transition: color 0.15s;
}
#fb-close:hover { color: #c0392b; }
#fb-modal iframe { flex: 1; border: none; width: 100%; }

/* ── SUMMARY TABLE ── */
.sum-box {
    background: #f5f7fa; border-radius: 9px;
    padding: 10px 12px; margin-top: 10px;
}
.sum-title {
    font-size: 11px !important; font-weight: 800 !important;
    color: #546e7a !important; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 7px !important;
}
table.sum { width:100%; border-collapse:collapse; }
table.sum td { padding: 3px 0; font-size: 12.5px !important; color: #455a64; }
table.sum td:last-child { text-align:right; font-weight:700; color:#263238 !important; }

/* make plotly chart fill its container properly */
[data-testid="stPlotlyChart"] {
    height: 100% !important;
}
[data-testid="stPlotlyChart"] > div {
    height: 100% !important;
}
</style>

<!-- FEEDBACK MODAL -->
<div id="fb-overlay" onclick="if(event.target===this)closeFB()">
  <div id="fb-modal">
    <div id="fb-mhdr">
      <span>📋 PhD Research Feedback Form</span>
      <button id="fb-close" onclick="closeFB()">✕</button>
    </div>
    <iframe id="fb-frame" src="" data-src="https://forms.office.com/r/THfuycGkDZ"
            title="Feedback" allow="clipboard-write"></iframe>
  </div>
</div>
<script>
function openFB(){
    var f=document.getElementById('fb-frame');
    if(!f.src||f.src===window.location.href) f.src=f.getAttribute('data-src');
    document.getElementById('fb-overlay').classList.add('open');
}
function closeFB(){
    document.getElementById('fb-overlay').classList.remove('open');
}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeFB();});
</script>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# 2. LOAD MODELS
# ──────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        pm = joblib.load("XGBoost_PMV_model.pkl")
        pp = joblib.load("XGBoost_PPD_model.pkl")
        sc = joblib.load("scaler_X.pkl")
        return pm, pp, sc
    except:
        return None, None, None

pmv_model, ppd_model, scaler_x = load_models()


# ──────────────────────────────────────────
# 3. PREDICTION
# ──────────────────────────────────────────
def get_predictions(month_str, wall_w, depth, orient_deg, wwr):
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    if None in (pmv_model, ppd_model, scaler_x):
        return None, None
    raw = np.array([[month_map[month_str], wall_w, depth, orient_deg, wwr]])
    return (round(float(pmv_model.predict(scaler_x.transform(raw))[0]),2),
            round(float(ppd_model.predict(scaler_x.transform(raw))[0]),1))


# ──────────────────────────────────────────
# 4. INSIGHTS
# ──────────────────────────────────────────
def get_insights(pmv, ppd, month_str, orient_deg, wwr, wall_w, room_depth):
    ins=[]
    hot=["May","Jun","Jul","Aug","Sep"]; cold=["Dec","Jan","Feb"]; trans=["Mar","Apr","Oct","Nov"]
    oname={0:"North",90:"East",180:"South",270:"West"}.get(orient_deg,f"{orient_deg}°")

    if -0.5<=pmv<=0.5:
        ins.append(("ok","✅","Comfort Achieved",
            f"PMV={pmv:+.2f} within ASHRAE 55. Only <b>{ppd:.0f}%</b> dissatisfied."))
    elif pmv>0.5:
        sev="Slight" if pmv<=1 else("Moderate" if pmv<=2 else"Severe")
        ins.append(("warm","🌡️",f"Overheating — {sev}",
            f"PMV={pmv:+.2f} → <b>{ppd:.0f}%</b> dissatisfied. Cooling needed."))
    else:
        sev="Slight" if pmv>=-1 else("Moderate" if pmv>=-2 else"Severe")
        ins.append(("cool","❄️",f"Underheating — {sev}",
            f"PMV={pmv:+.2f} → <b>{ppd:.0f}%</b> dissatisfied. Heating/solar gain needed."))

    if pmv>0.5:
        if orient_deg==270:
            ins.append(("tip","🧭","West — Afternoon Heat",
                "West glazing traps afternoon sun. Rotate to <b>North/NE</b> or add fins."))
        elif orient_deg==180 and month_str in hot:
            ins.append(("tip","🧭","South — Summer Overheating",
                "Add horizontal overhangs or reduce WWR to cut summer solar load."))
        else:
            ins.append(("info","🧭",f"{oname} — Check Shading",
                "Review shading for peak cooling months."))
    elif pmv<-0.5:
        if orient_deg==0:
            ins.append(("tip","🧭","North — No Solar Gain",
                "North glazing loses heat in winter. Rotate to <b>South</b>."))
        else:
            ins.append(("tip","🧭",f"{oname} — Low Solar Gain",
                f"Limited passive solar heating in {month_str}."))
    else:
        ins.append(("info","🧭",f"{oname} — Good Balance",
            "Orientation performing well under current parameters."))

    if pmv>0.5 and wwr>0.4:
        ins.append(("tip","🪟","High WWR — Reduce",
            f"WWR={wwr:.0%} amplifies gain. Target <b>25–35%</b> with shading."))
    elif pmv<-0.5 and wwr<0.35:
        ins.append(("tip","🪟","Low WWR — Increase",
            f"WWR={wwr:.0%} limits gain. Try <b>40–55%</b> on south façade."))
    else:
        ins.append(("info","🪟","WWR — Balanced",
            f"WWR={wwr:.0%} suits current thermal conditions."))

    asp=room_depth/wall_w if wall_w>0 else 0
    if pmv>0.5 and asp<1.2:
        ins.append(("tip","📐","Shallow Plan",
            f"D/W={asp:.1f} concentrates solar load. Deepen to ≥<b>{wall_w*1.5:.1f}m</b>."))
    elif pmv<-0.5 and asp>2.5:
        ins.append(("tip","📐","Deep Plan",
            "D/W ratio blocks solar heat. Use light shelf or shorten depth."))
    else:
        ins.append(("info","📐",f"Geometry OK  D/W={asp:.1f}",
            "Room proportions suit the current thermal condition."))

    if month_str in hot and pmv>0.5:
        ins.append(("warm","📅",f"{month_str} — Peak Cooling",
            "Night ventilation, low-e glazing & reflective roof for West Cairo."))
    elif month_str in cold and pmv<-0.5:
        ins.append(("cool","📅",f"{month_str} — Heating Season",
            "Thermal mass, airtight envelope & passive solar design essential."))
    elif month_str in trans:
        ins.append(("ok","📅",f"{month_str} — Transitional",
            "Operable windows & natural ventilation can maintain comfort."))
    return ins


# ──────────────────────────────────────────
# 5. 3D FIGURE
# ──────────────────────────────────────────
def build_room_figure(wall_width, room_depth, wwr, orientation, height=3.0):
    W,D,H=wall_width,room_depth,height
    fig=go.Figure()

    def quad(xs,ys,zs,color,opacity=1.0,lighting=None,lpos=None):
        kw=dict(x=xs,y=ys,z=zs,i=[0,0],j=[1,2],k=[2,3],
                color=color,opacity=opacity,flatshading=True,
                showscale=False,showlegend=False,hoverinfo='skip')
        if lighting: kw['lighting']=lighting
        if lpos:     kw['lightposition']=lpos
        return go.Mesh3d(**kw)

    def ln(xs,ys,zs,col='#8fa3b5',w=1.5):
        return go.Scatter3d(x=xs,y=ys,z=zs,mode='lines',
                            line=dict(color=col,width=w),showlegend=False,hoverinfo='skip')

    LPOS=dict(x=2,y=-3,z=5)
    LT=dict(ambient=0.72,diffuse=0.90,specular=0.15,roughness=0.60,fresnel=0.05)
    GL=dict(ambient=0.30,diffuse=0.50,specular=0.90,roughness=0.04,fresnel=0.90)

    fig.add_trace(quad([0,W,W,0],[0,0,D,D],[0,0,0,0],'#c8ae7e',1.0,LT,LPOS))
    for xv in np.arange(max(0.28,W/10),W,max(0.28,W/10)):
        fig.add_trace(ln([xv,xv],[0,D],[0.003,0.003],'#b09460',0.55))
    fig.add_trace(quad([0,W,W,0],[0,0,D,D],[H,H,H,H],'#f0f3f6',0.50,LT,LPOS))
    fig.add_trace(quad([0,W,W,0],[D,D,D,D],[0,0,H,H],'#dde5ee',1.0,LT,LPOS))
    fig.add_trace(quad([0,0,0,0],[D,0,0,D],[0,0,H,H],'#cfd9e6',1.0,LT,LPOS))
    fig.add_trace(quad([W,W,W,W],[0,D,D,0],[0,0,H,H],'#cfd9e6',1.0,LT,LPOS))

    ww=W*np.sqrt(wwr); wh=H*np.sqrt(wwr)
    wx0=(W-ww)/2; wx1=wx0+ww; wz0=0.80; wz1=min(wz0+wh,H-0.10)

    fig.add_trace(quad([0,wx0,wx0,0],[0,0,0,0],[0,0,H,H],'#c8d4e0',1.0,LT,LPOS))
    fig.add_trace(quad([wx1,W,W,wx1],[0,0,0,0],[0,0,H,H],'#c8d4e0',1.0,LT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0,0],[0,0,wz0,wz0],'#c8d4e0',1.0,LT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0,0],[wz1,wz1,H,H],'#c8d4e0',1.0,LT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0.12,0.12],[wz0,wz0,wz0,wz0],'#9ab0c2',1.0,LT,LPOS))

    mx=(wx0+wx1)/2
    for (gx0,gx1) in [(wx0+0.05,mx-0.03),(mx+0.03,wx1-0.05)]:
        fig.add_trace(quad([gx0,gx1,gx1,gx0],[0,0,0,0],
                           [wz0+0.05,wz0+0.05,wz1-0.04,wz1-0.04],'#9dd8ef',0.38,GL,LPOS))
        rw=(gx1-gx0)*0.22
        fig.add_trace(quad([gx0+rw,gx0+rw*2.2,gx0+rw*2.2,gx0+rw],[0,0,0,0],
                           [wz1-0.28,wz1-0.28,wz1-0.08,wz1-0.08],'#ffffff',0.14))

    FC='#3d4f5e'; mz=(wz0+wz1)/2
    for (xs,zs,lw) in [([wx0,wx1],[wz0,wz0],4),([wx0,wx1],[wz1,wz1],4),
                        ([wx0,wx0],[wz0,wz1],4),([wx1,wx1],[wz0,wz1],4),
                        ([mx,mx],[wz0,wz1],3),([wx0,wx1],[mz,mz],2)]:
        fig.add_trace(ln(xs,[0,0],zs,FC,lw))

    for (xs,ys,zs) in [
        ([0,W],[0,0],[0,0]),([0,W],[D,D],[0,0]),([0,0],[0,D],[0,0]),([W,W],[0,D],[0,0]),
        ([0,W],[0,0],[H,H]),([0,W],[D,D],[H,H]),([0,0],[0,D],[H,H]),([W,W],[0,D],[H,H]),
        ([0,0],[0,0],[0,H]),([W,W],[0,0],[0,H]),([0,0],[D,D],[0,H]),([W,W],[D,D],[0,H])]:
        fig.add_trace(ln(xs,ys,zs,'#6e8ea4',1.4))

    for rx in np.linspace(wx0+ww*0.18,wx1-ww*0.18,4):
        fig.add_trace(go.Scatter3d(
            x=[rx,rx+(rx-W/2)*0.08],y=[0.01,D*0.50],
            z=[(wz0+wz1)/2,max(0.1,(wz0+wz1)/2-0.35)],
            mode='lines',line=dict(color='rgba(255,224,120,0.09)',width=2),
            showlegend=False,hoverinfo='skip'))

    # North arrow
    az_r=np.radians({"North":0,"East":90,"South":180,"West":270}.get(orientation,0))
    nx=np.sin(az_r); ny=-np.cos(az_r)
    alen=min(W,D)*0.22; cx=W*1.20; cy=D*0.12; cz=0.01
    tx=cx+nx*alen; ty=cy+ny*alen; tx2=cx-nx*alen*0.6; ty2=cy-ny*alen*0.6
    fig.add_trace(ln([tx2,tx],[ty2,ty],[cz,cz],'#e53935',3.5))
    px=-ny*0.20*alen; py=nx*0.20*alen
    bx=tx-nx*0.28*alen; by=ty-ny*0.28*alen
    for s in [1,-1]:
        fig.add_trace(ln([tx,bx+s*px],[ty,by+s*py],[cz,cz],'#e53935',3.5))
    fig.add_trace(go.Scatter3d(
        x=[tx+nx*0.09*alen],y=[ty+ny*0.09*alen],z=[cz+0.22],
        mode='text',text=['N'],textfont=dict(color='#e53935',size=16),
        showlegend=False,hoverinfo='skip'))

    # Camera: diagonal-normalised, stable across all room sizes
    diag=np.sqrt((W*1.45)**2+D**2+H**2)
    ed=diag*0.90
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False,range=[-W*0.02,W*1.48]),
            yaxis=dict(visible=False,range=[-D*0.02,D*1.02]),
            zaxis=dict(visible=False,range=[-0.04,H*1.08]),
            bgcolor='rgba(236,241,247,1)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=ed*0.55,y=-ed*0.72,z=ed*0.42),
                center=dict(x=0,y=0,z=-0.04),
                up=dict(x=0,y=0,z=1),
            ),
        ),
        uirevision='locked',
        margin=dict(l=0,r=0,b=0,t=0),
        paper_bgcolor='rgba(236,241,247,1)',
        showlegend=False,
        autosize=True,
        height=480,       # explicit px height so chart fills the column
    )
    return fig


# ──────────────────────────────────────────
# 6. HEADER
# ──────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <h1>🏛️ S-TCML V.01 — Architectural Thermal Comfort Predictor</h1>
    <p>AI-driven PMV &amp; PPD · Architectural parameters · PhD research · West Cairo · ASHRAE 55</p>
  </div>
  <button class="fb-btn" onclick="openFB()">📋 Feedback</button>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# 7. THREE-COLUMN MAIN LAYOUT
# ──────────────────────────────────────────
ORIENT_MAP = {"North":0,"East":90,"South":180,"West":270}

# Use a small gap and tight column ratios
col1, col2, col3 = st.columns([1, 2.4, 1], gap="small")

with col1:
    st.markdown('<div class="panel"><div class="panel-hdr">⚙ Controls</div>', unsafe_allow_html=True)
    orientation = st.selectbox("Glazing Orientation",
                               ["North","East","South","West"], index=2)
    orient_deg  = ORIENT_MAP[orientation]
    month_val   = st.selectbox("Month",
                               ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"], index=6)
    wall_width  = st.slider("Wall Width (m)",  0.5, 5.0,  3.5, step=0.1)
    room_depth  = st.slider("Room Depth (m)",  2.0, 10.0, 5.0, step=0.25)
    wwr         = st.slider("Window-to-Wall Ratio", 0.10, 0.90, 0.40, step=0.05,
                            format="%.0%%")
    st.markdown('</div>', unsafe_allow_html=True)

# Compute prediction
pmv, ppd = get_predictions(month_val, wall_width, room_depth, orient_deg, wwr)

with col2:
    st.markdown('<div class="panel"><div class="panel-hdr">📐 Room Geometry</div>', unsafe_allow_html=True)
    fig3d = build_room_figure(wall_width, room_depth, wwr, orientation)
    st.plotly_chart(fig3d, use_container_width=True,
                    config=dict(displayModeBar=False, scrollZoom=False))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="panel"><div class="panel-hdr">📊 Predictions</div>', unsafe_allow_html=True)

    if pmv is not None:
        st.metric("PMV — Predicted Mean Vote", f"{pmv:+.2f}")
        st.metric("PPD — % Dissatisfied",      f"{ppd:.1f}%")

        comfort_ok = -0.5<=pmv<=0.5
        bc  = "#2e7d32" if comfort_ok else ("#e64a19" if pmv>0 else "#1565c0")
        lbl = "✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv>0 else "🔵 Too Cool")
        pct = min(max((pmv+3)/6,0),1)*100

        st.markdown(f"""
        <div style="margin-top:2px;">
          <div style="background:#dde4ee;border-radius:5px;height:10px;
                      overflow:hidden;position:relative;">
            <div style="width:{pct:.1f}%;background:{bc};height:100%;border-radius:5px;"></div>
            <div style="position:absolute;top:0;left:50%;width:2px;height:100%;
                        background:rgba(255,255,255,0.7);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-size:11px;color:#90a4ae;margin-top:3px;">
            <span>−3</span>
            <span style="font-weight:700;color:{bc};">{lbl}</span>
            <span>+3</span>
          </div>
          <div style="font-size:10.5px;color:#aab8c2;margin-top:4px;">
            ASHRAE 55 target: −0.5 to +0.5
          </div>
        </div>
        <div class="sum-box">
          <div class="sum-title">Input Summary</div>
          <table class="sum">
            <tr><td>🧭 Orientation</td><td>{orientation} ({orient_deg}°)</td></tr>
            <tr><td>📅 Month</td>      <td>{month_val}</td></tr>
            <tr><td>↔ Wall Width</td>  <td>{wall_width:.1f} m</td></tr>
            <tr><td>↕ Room Depth</td>  <td>{room_depth:.1f} m</td></tr>
            <tr><td>🪟 WWR</td>        <td>{int(wwr*100)}%</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model files not found. Place .pkl files alongside this script.")

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# 8. INSIGHTS FOOTER — FULL WIDTH
# ──────────────────────────────────────────
CMAP={"warm":"#fff3e0","cool":"#e3f2fd","ok":"#e8f5e9","tip":"#f3e5f5","info":"#e0f7fa"}
BMAP={"warm":"#e64a19","cool":"#1976d2","ok":"#2e7d32","tip":"#7b1fa2","info":"#00838f"}

st.markdown('<hr>', unsafe_allow_html=True)

if pmv is not None:
    ins_list = get_insights(pmv,ppd,month_val,orient_deg,wwr,wall_width,room_depth)
    cards_html = '<div class="ins-strip">'
    for (ct,ic,title,body) in ins_list:
        cards_html += f"""
        <div class="ins-card c-{ct}">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
            <span class="ins-icon">{ic}</span>
            <span class="ins-title">{title}</span>
          </div>
          <div class="ins-body">{body}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(
        f'<div style="background:#fff;border-radius:12px;'
        f'box-shadow:0 2px 10px rgba(0,0,0,0.07);padding:10px 14px 12px;margin:0 0 8px;">'
        f'<div style="font-size:12px;font-weight:800;color:#546e7a;text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:8px;">💡 Design Insights</div>'
        f'{cards_html}</div>',
        unsafe_allow_html=True)
