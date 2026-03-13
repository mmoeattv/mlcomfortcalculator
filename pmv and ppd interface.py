import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import joblib

# ─────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="S-TCML V.01 — Thermal Comfort",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html,body{margin:0;padding:0;}
.block-container{padding:0!important;margin:0!important;max-width:100%!important;}
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],footer,#MainMenu{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
.stApp{background:#eef1f6!important;}
iframe{display:block;border:none;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 2. LOAD MODELS
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────
def get_predictions(month_str, wall_w, depth, orient_deg, wwr):
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    if None in (pmv_model, ppd_model, scaler_x):
        return None, None
    raw = np.array([[month_map[month_str], wall_w, depth, orient_deg, wwr]])
    sc  = scaler_x.transform(raw)
    return round(float(pmv_model.predict(sc)[0]),2), round(float(ppd_model.predict(sc)[0]),1)


# ─────────────────────────────────────────
# 4. INSIGHTS
# ─────────────────────────────────────────
def get_insights(pmv, ppd, month_str, orient_deg, wwr, wall_w, room_depth):
    ins = []
    hot   = ["May","Jun","Jul","Aug","Sep"]
    cold  = ["Dec","Jan","Feb"]
    trans = ["Mar","Apr","Oct","Nov"]
    oname = {0:"North",90:"East",180:"South",270:"West"}.get(orient_deg, f"{orient_deg}°")

    if -0.5<=pmv<=0.5:
        ins.append(("ok","✅","Comfort Achieved",
            f"PMV={pmv:+.2f} within ASHRAE 55. Only <b>{ppd:.0f}%</b> dissatisfied."))
    elif pmv>0.5:
        sev="Slight" if pmv<=1 else ("Moderate" if pmv<=2 else "Severe")
        ins.append(("warm","🌡️",f"Overheating — {sev}",
            f"PMV={pmv:+.2f} → <b>{ppd:.0f}%</b> dissatisfied. Cooling strategies needed."))
    else:
        sev="Slight" if pmv>=-1 else ("Moderate" if pmv>=-2 else "Severe")
        ins.append(("cool","❄️",f"Underheating — {sev}",
            f"PMV={pmv:+.2f} → <b>{ppd:.0f}%</b> dissatisfied. Heating/solar gain needed."))

    if pmv>0.5:
        if orient_deg==270:
            ins.append(("tip","🧭","West — Afternoon Heat",
                "West glazing traps intense afternoon sun. Rotate to <b>North/NE</b> or add vertical fins."))
        elif orient_deg==180 and month_str in hot:
            ins.append(("tip","🧭","South — Summer Heat",
                "South glazing increases summer load. Add horizontal overhangs or reduce WWR."))
        elif orient_deg==90:
            ins.append(("info","🧭","East — Morning Sun",
                "East glazing is manageable but verify shading in May–Sep."))
        else:
            ins.append(("info","🧭",f"{oname} — Check Shading",
                "Review shading adequacy for peak cooling months."))
    elif pmv<-0.5:
        if orient_deg==0:
            ins.append(("tip","🧭","North — No Solar Gain",
                "North glazing loses heat in winter. Rotate to <b>South</b> for passive heating."))
        elif orient_deg==180 and month_str in cold:
            ins.append(("ok","🧭","South — Ideal Winter",
                "South orientation maximises winter sun. Consider slightly increasing WWR."))
        else:
            ins.append(("tip","🧭",f"{oname} — Low Solar Gain",
                f"{oname} provides limited solar heating in {month_str}."))
    else:
        ins.append(("info","🧭",f"{oname} — Good Balance",
            "Orientation performing well under current parameters."))

    if pmv>0.5 and wwr>0.4:
        ins.append(("tip","🪟","High WWR — Reduce",
            f"WWR={wwr:.0%} amplifies solar gain. Target <b>25–35%</b> with shading."))
    elif pmv<-0.5 and wwr<0.35:
        ins.append(("tip","🪟","Low WWR — Increase",
            f"WWR={wwr:.0%} limits solar gain. Try <b>40–55%</b> on south façade."))
    else:
        ins.append(("info","🪟","WWR — Balanced",
            f"WWR={wwr:.0%} suits current thermal conditions."))

    asp = room_depth/wall_w if wall_w>0 else 0
    if pmv>0.5 and asp<1.2:
        ins.append(("tip","📐","Shallow Plan",
            f"D/W={asp:.1f} concentrates solar load. Deepen to ≥<b>{wall_w*1.5:.1f}m</b>."))
    elif pmv<-0.5 and asp>2.5:
        ins.append(("tip","📐","Deep Plan",
            f"D/W={asp:.1f} blocks solar heat. Use light shelf or shorten depth."))
    else:
        ins.append(("info","📐",f"Geometry OK (D/W={asp:.1f})",
            "Room proportions suit the current thermal condition."))

    if month_str in hot and pmv>0.5:
        ins.append(("warm","📅",f"{month_str} — Peak Cooling",
            "Night ventilation, low-e glazing & reflective roof advised for West Cairo."))
    elif month_str in cold and pmv<-0.5:
        ins.append(("cool","📅",f"{month_str} — Heating Season",
            "Thermal mass, airtight envelope & passive solar design are essential."))
    elif month_str in trans:
        ins.append(("ok","📅",f"{month_str} — Transitional",
            "Mild climate: operable windows & natural ventilation can maintain comfort."))

    return ins


# ─────────────────────────────────────────
# 5. 3D ROOM FIGURE
# ─────────────────────────────────────────
def build_room_figure(wall_width, room_depth, wwr, orientation, height=3.0):
    W, D, H = wall_width, room_depth, height
    fig = go.Figure()

    def quad(xs, ys, zs, color, opacity=1.0, lighting=None, lpos=None):
        kw = dict(x=xs, y=ys, z=zs, i=[0,0], j=[1,2], k=[2,3],
                  color=color, opacity=opacity, flatshading=True,
                  showscale=False, showlegend=False, hoverinfo='skip')
        if lighting: kw['lighting'] = lighting
        if lpos:     kw['lightposition'] = lpos
        return go.Mesh3d(**kw)

    def ln(xs, ys, zs, col='#8fa3b5', w=1.5):
        return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                            line=dict(color=col, width=w),
                            showlegend=False, hoverinfo='skip')

    LPOS  = dict(x=2, y=-3, z=5)
    LIGHT = dict(ambient=0.72, diffuse=0.90, specular=0.15, roughness=0.60, fresnel=0.05)
    GLIGHT= dict(ambient=0.30, diffuse=0.50, specular=0.90, roughness=0.04, fresnel=0.90)

    # Floor
    fig.add_trace(quad([0,W,W,0],[0,0,D,D],[0,0,0,0],'#c8ae7e',1.0,LIGHT,LPOS))
    plank = max(0.28, W/10)
    for xv in np.arange(plank, W, plank):
        fig.add_trace(ln([xv,xv],[0,D],[0.003,0.003],'#b09460',0.55))
    # Ceiling
    fig.add_trace(quad([0,W,W,0],[0,0,D,D],[H,H,H,H],'#f0f3f6',0.50,LIGHT,LPOS))
    # Back wall
    fig.add_trace(quad([0,W,W,0],[D,D,D,D],[0,0,H,H],'#dde5ee',1.0,LIGHT,LPOS))
    # Side walls
    fig.add_trace(quad([0,0,0,0],[D,0,0,D],[0,0,H,H],'#cfd9e6',1.0,LIGHT,LPOS))
    fig.add_trace(quad([W,W,W,W],[0,D,D,0],[0,0,H,H],'#cfd9e6',1.0,LIGHT,LPOS))

    # Glazing facade (y=0)
    win_w = W * np.sqrt(wwr); win_h = H * np.sqrt(wwr)
    wx0=(W-win_w)/2; wx1=wx0+win_w; wz0=0.80; wz1=min(wz0+win_h,H-0.10)

    fig.add_trace(quad([0,wx0,wx0,0],[0,0,0,0],[0,0,H,H],'#c8d4e0',1.0,LIGHT,LPOS))
    fig.add_trace(quad([wx1,W,W,wx1],[0,0,0,0],[0,0,H,H],'#c8d4e0',1.0,LIGHT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0,0],[0,0,wz0,wz0],'#c8d4e0',1.0,LIGHT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0,0],[wz1,wz1,H,H],'#c8d4e0',1.0,LIGHT,LPOS))
    fig.add_trace(quad([wx0,wx1,wx1,wx0],[0,0,0.12,0.12],[wz0,wz0,wz0,wz0],'#9ab0c2',1.0,LIGHT,LPOS))

    mid_x=(wx0+wx1)/2
    for (gx0,gx1) in [(wx0+0.05,mid_x-0.03),(mid_x+0.03,wx1-0.05)]:
        fig.add_trace(quad([gx0,gx1,gx1,gx0],[0,0,0,0],
                           [wz0+0.05,wz0+0.05,wz1-0.04,wz1-0.04],'#9dd8ef',0.38,GLIGHT,LPOS))
        rw=(gx1-gx0)*0.22
        fig.add_trace(quad([gx0+rw,gx0+rw*2.2,gx0+rw*2.2,gx0+rw],[0,0,0,0],
                           [wz1-0.28,wz1-0.28,wz1-0.08,wz1-0.08],'#ffffff',0.14))

    FC='#3d4f5e'; mid_z=(wz0+wz1)/2
    for (xs,zs,lw) in [([wx0,wx1],[wz0,wz0],4),([wx0,wx1],[wz1,wz1],4),
                        ([wx0,wx0],[wz0,wz1],4),([wx1,wx1],[wz0,wz1],4),
                        ([mid_x,mid_x],[wz0,wz1],3),([wx0,wx1],[mid_z,mid_z],2)]:
        fig.add_trace(ln(xs,[0,0],zs,FC,lw))

    EC='#6e8ea4'
    for (xs,ys,zs) in [
        ([0,W],[0,0],[0,0]),([0,W],[D,D],[0,0]),([0,0],[0,D],[0,0]),([W,W],[0,D],[0,0]),
        ([0,W],[0,0],[H,H]),([0,W],[D,D],[H,H]),([0,0],[0,D],[H,H]),([W,W],[0,D],[H,H]),
        ([0,0],[0,0],[0,H]),([W,W],[0,0],[0,H]),([0,0],[D,D],[0,H]),([W,W],[D,D],[0,H]),
    ]:
        fig.add_trace(ln(xs,ys,zs,EC,1.4))

    for rx in np.linspace(wx0+win_w*0.18, wx1-win_w*0.18, 4):
        fig.add_trace(go.Scatter3d(
            x=[rx,rx+(rx-W/2)*0.08],y=[0.01,D*0.50],
            z=[(wz0+wz1)/2,max(0.1,(wz0+wz1)/2-0.35)],
            mode='lines',line=dict(color='rgba(255,224,120,0.09)',width=2),
            showlegend=False,hoverinfo='skip'))

    # North arrow
    odeg_map={"North":0,"East":90,"South":180,"West":270}
    az_r=np.radians(odeg_map.get(orientation,0))
    nx=np.sin(az_r); ny=-np.cos(az_r)
    alen=min(W,D)*0.22; cx=W*1.22; cy=D*0.15; cz=0.01
    tip_x=cx+nx*alen; tip_y=cy+ny*alen
    tail_x=cx-nx*alen*0.6; tail_y=cy-ny*alen*0.6
    fig.add_trace(ln([tail_x,tip_x],[tail_y,tip_y],[cz,cz],'#e53935',3.5))
    px2=-ny*0.20*alen; py2=nx*0.20*alen
    bx=tip_x-nx*0.28*alen; by=tip_y-ny*0.28*alen
    for s in [1,-1]:
        fig.add_trace(ln([tip_x,bx+s*px2],[tip_y,by+s*py2],[cz,cz],'#e53935',3.5))
    fig.add_trace(go.Scatter3d(
        x=[tip_x+nx*0.09*alen],y=[tip_y+ny*0.09*alen],z=[cz+0.20],
        mode='text',text=['N'],textfont=dict(color='#e53935',size=14),
        showlegend=False,hoverinfo='skip'))

    # Camera — diagonal-normalised so zoom is stable across all room sizes
    diag=np.sqrt((W*1.48)**2+D**2+H**2)
    ed=diag*0.92
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False,range=[-W*0.03,W*1.50]),
            yaxis=dict(visible=False,range=[-D*0.03,D*1.03]),
            zaxis=dict(visible=False,range=[-0.05,H*1.10]),
            bgcolor='rgba(236,241,247,1)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=ed*0.55,y=-ed*0.72,z=ed*0.42),
                center=dict(x=0,y=0,z=-0.05),
                up=dict(x=0,y=0,z=1),
            ),
        ),
        uirevision='locked',
        margin=dict(l=0,r=0,b=0,t=0),
        paper_bgcolor='rgba(236,241,247,1)',
        showlegend=False,
        autosize=True,
    )
    return fig


# ─────────────────────────────────────────
# 6. READ STATE FROM QUERY PARAMS
# ─────────────────────────────────────────
ORIENT_OPTS = ["North","East","South","West"]
MONTH_OPTS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ODEG        = {"North":0,"East":90,"South":180,"West":270}

qp = st.query_params
orientation = qp.get("orientation","South")
month_val   = qp.get("month","Jul")
wall_width  = float(qp.get("ww","3.5"))
room_depth  = float(qp.get("rd","5.0"))
wwr         = float(qp.get("wwr","0.4"))

if orientation not in ORIENT_OPTS: orientation="South"
if month_val   not in MONTH_OPTS:  month_val="Jul"
wall_width=max(0.5,min(5.0,wall_width))
room_depth=max(2.0,min(10.0,room_depth))
wwr=max(0.10,min(0.90,wwr))

orient_deg=ODEG[orientation]
pmv,ppd=get_predictions(month_val,wall_width,room_depth,orient_deg,wwr)


# ─────────────────────────────────────────
# 7. BUILD 3D → JSON
# ─────────────────────────────────────────
fig3d     = build_room_figure(wall_width,room_depth,wwr,orientation)
fig3d_json= pio.to_json(fig3d)


# ─────────────────────────────────────────
# 8. INSIGHTS HTML
# ─────────────────────────────────────────
CMAP={"warm":"#fff3e0","cool":"#e3f2fd","ok":"#e8f5e9","tip":"#f3e5f5","info":"#e0f7fa"}
BMAP={"warm":"#e64a19","cool":"#1976d2","ok":"#2e7d32","tip":"#7b1fa2","info":"#00838f"}

if pmv is not None:
    ins_list=get_insights(pmv,ppd,month_val,orient_deg,wwr,wall_width,room_depth)
    ins_cards=""
    for (ct,ic,title,body) in ins_list:
        ins_cards+=f"""<div style="flex:1 1 0;min-width:155px;max-width:280px;border-radius:10px;
            background:{CMAP[ct]};border-top:3px solid {BMAP[ct]};
            padding:9px 11px 8px;box-shadow:0 1px 5px rgba(0,0,0,0.07);">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
            <span style="font-size:14px;line-height:1;">{ic}</span>
            <span style="font-size:12px;font-weight:800;color:#1a2e35;line-height:1.2;">{title}</span>
          </div>
          <div style="font-size:11.5px;color:#455a64;line-height:1.5;">{body}</div>
        </div>"""
else:
    ins_cards="""<div style="border-radius:10px;background:#e0f7fa;border-top:3px solid #00838f;
        padding:9px 12px;font-size:12px;color:#455a64;">
        ℹ️ <b>Model files not loaded.</b> Place .pkl files alongside this script.</div>"""

# PMV display values
if pmv is not None:
    comfort_ok=-0.5<=pmv<=0.5
    bc="#2e7d32" if comfort_ok else ("#e64a19" if pmv>0 else "#1565c0")
    lbl="✅ Comfortable" if comfort_ok else ("🔴 Too Warm" if pmv>0 else "🔵 Too Cool")
    pct=min(max((pmv+3)/6,0),1)*100
    pmv_display=f"{pmv:+.2f}"; ppd_display=f"{ppd:.1f}%"
else:
    bc="gray"; lbl="No model"; pct=50; pmv_display="—"; ppd_display="—"

# Build orientation options HTML
orient_opts_html="".join(
    f'<option value="{o}"{"selected" if o==orientation else ""}>{o}</option>'
    for o in ORIENT_OPTS)
month_opts_html="".join(
    f'<option value="{m}"{"selected" if m==month_val else ""}>{m}</option>'
    for m in MONTH_OPTS)


# ─────────────────────────────────────────
# 9. FULL HTML COMPONENT
# ─────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',system-ui,sans-serif;}}
html,body{{height:100%;overflow:hidden;background:#eef1f6;}}
body{{display:flex;flex-direction:column;height:100vh;}}

/* HEADER */
#hdr{{display:flex;align-items:center;justify-content:space-between;
  background:#fff;border-bottom:1.5px solid #dde3ea;
  padding:9px 18px;flex-shrink:0;box-shadow:0 1px 6px rgba(0,0,0,0.06);}}
#hdr h1{{font-size:18px;font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;}}
#hdr p{{font-size:12px;color:#607d8b;margin-top:2px;}}
.fb-btn{{background:linear-gradient(135deg,#00897b,#00acc1);color:#fff;
  border:none;border-radius:8px;padding:8px 18px;font-size:13px;
  font-weight:700;cursor:pointer;box-shadow:0 2px 8px rgba(0,137,123,0.35);
  white-space:nowrap;transition:opacity 0.15s;}}
.fb-btn:hover{{opacity:0.85;}}

/* MAIN ROW */
#main{{display:flex;flex:1;gap:10px;padding:10px 14px 8px;min-height:0;overflow:hidden;}}

/* PANELS */
.panel{{background:#fff;border-radius:12px;
  box-shadow:0 2px 10px rgba(0,0,0,0.07);
  display:flex;flex-direction:column;overflow:hidden;}}
.ph{{font-size:11.5px;font-weight:800;color:#546e7a;text-transform:uppercase;
  letter-spacing:0.07em;padding:8px 13px 7px;border-bottom:1px solid #eceff1;flex-shrink:0;}}
.pb{{flex:1;padding:11px 13px;overflow-y:auto;min-height:0;}}

/* CONTROLS */
#col-ctrl{{flex:0 0 220px;}}
.cl{{font-size:11.5px;font-weight:700;color:#37474f;text-transform:uppercase;
  letter-spacing:0.04em;margin-bottom:3px;margin-top:10px;}}
.cl:first-of-type{{margin-top:0;}}
select{{width:100%;font-size:13px;color:#263238;background:#f5f7fa;
  border:1.5px solid #cfd8dc;border-radius:7px;padding:5px 8px;
  cursor:pointer;outline:none;}}
select:focus{{border-color:#00897b;}}
input[type=range]{{width:100%;-webkit-appearance:none;height:5px;border-radius:3px;
  background:linear-gradient(90deg,#00897b,#00bcd4);outline:none;
  cursor:pointer;margin-top:4px;}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;
  width:17px;height:17px;border-radius:50%;background:#00897b;
  border:2.5px solid #fff;box-shadow:0 0 0 2px #00897b55;cursor:grab;}}
.cv{{font-size:12.5px;font-weight:700;color:#00796b;text-align:right;margin-top:1px;}}

/* 3D col */
#col-3d{{flex:1 1 0;min-width:0;}}
#pdiv{{width:100%;height:100%;}}

/* PREDICTIONS */
#col-pred{{flex:0 0 215px;}}
.mbox{{background:linear-gradient(135deg,#fff,#f0fffe);
  border:1.5px solid #b2dfdb;border-radius:10px;
  padding:10px 13px 8px;margin-bottom:9px;
  box-shadow:0 2px 6px rgba(0,150,136,0.09);}}
.mlbl{{font-size:10.5px;font-weight:700;color:#00796b;
  text-transform:uppercase;letter-spacing:0.05em;}}
.mval{{font-size:30px;font-weight:800;color:#00838f;line-height:1.1;margin-top:2px;}}
.bwrap{{margin:7px 0 4px;}}
.bbg{{background:#dde4ee;border-radius:5px;height:10px;overflow:hidden;position:relative;}}
.bfg{{height:100%;border-radius:5px;}}
.bcl{{position:absolute;top:0;left:50%;width:2px;height:100%;background:rgba(255,255,255,0.7);}}
.blbl{{display:flex;justify-content:space-between;font-size:11px;color:#90a4ae;margin-top:3px;}}
.sbox{{background:#f5f7fa;border-radius:9px;padding:10px 12px;margin-top:9px;}}
.stit{{font-size:10.5px;font-weight:800;color:#546e7a;text-transform:uppercase;
  letter-spacing:0.05em;margin-bottom:6px;}}
table.st{{width:100%;border-collapse:collapse;font-size:12.5px;color:#455a64;}}
table.st td{{padding:3px 0;}}
table.st td:last-child{{text-align:right;font-weight:700;color:#263238;}}

/* FOOTER INSIGHTS */
#footer{{flex-shrink:0;background:#fff;border-top:1.5px solid #dde3ea;padding:7px 14px 9px;}}
#ilbl{{font-size:11.5px;font-weight:800;color:#37474f;text-transform:uppercase;
  letter-spacing:0.07em;margin-bottom:6px;}}
#istrip{{display:flex;gap:8px;flex-wrap:nowrap;overflow-x:auto;}}
#istrip::-webkit-scrollbar{{height:3px;}}
#istrip::-webkit-scrollbar-thumb{{background:#b0bec5;border-radius:2px;}}

/* MODAL */
#fbo{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.52);
  z-index:9999;align-items:center;justify-content:center;}}
#fbo.open{{display:flex;}}
#fbm{{background:#fff;border-radius:14px;box-shadow:0 16px 56px rgba(0,0,0,0.3);
  width:min(700px,94vw);height:min(640px,90vh);display:flex;flex-direction:column;overflow:hidden;}}
#fbh{{display:flex;justify-content:space-between;align-items:center;
  padding:10px 16px;border-bottom:1px solid #e0e0e0;background:#f5f7fa;}}
#fbh span{{font-size:14px;font-weight:700;color:#1a1a2e;}}
#fbc{{background:none;border:none;font-size:20px;cursor:pointer;color:#607d8b;}}
#fbc:hover{{color:#c0392b;}}
#fbm iframe{{flex:1;border:none;width:100%;}}
</style>
</head>
<body>

<div id="hdr">
  <div>
    <h1>🏛️ S-TCML V.01 — Architectural Thermal Comfort Predictor</h1>
    <p>AI-driven PMV &amp; PPD · Architectural parameters · PhD research · West Cairo · ASHRAE 55</p>
  </div>
  <button class="fb-btn" onclick="openFB()">📋 Feedback</button>
</div>

<div id="main">

  <!-- CONTROLS -->
  <div class="panel" id="col-ctrl">
    <div class="ph">⚙ Controls</div>
    <div class="pb">
      <div class="cl">Glazing Orientation</div>
      <select id="sel-o" onchange="pushState()">{orient_opts_html}</select>

      <div class="cl">Month</div>
      <select id="sel-m" onchange="pushState()">{month_opts_html}</select>

      <div class="cl">Wall Width (m)</div>
      <input type="range" id="sl-ww" min="0.5" max="5.0" step="0.1"
             value="{wall_width:.1f}" oninput="liveUpdate(this,'val-ww','m');pushState()">
      <div class="cv" id="val-ww">{wall_width:.1f} m</div>

      <div class="cl">Room Depth (m)</div>
      <input type="range" id="sl-rd" min="2.0" max="10.0" step="0.25"
             value="{room_depth:.2f}" oninput="liveUpdate(this,'val-rd','m');pushState()">
      <div class="cv" id="val-rd">{room_depth:.1f} m</div>

      <div class="cl">Window-to-Wall Ratio</div>
      <input type="range" id="sl-wwr" min="0.10" max="0.90" step="0.05"
             value="{wwr:.2f}" oninput="liveUpdate(this,'val-wwr','%');pushState()">
      <div class="cv" id="val-wwr">{int(wwr*100)}%</div>
    </div>
  </div>

  <!-- 3D -->
  <div class="panel" id="col-3d">
    <div class="ph">📐 Room Geometry</div>
    <div id="pdiv" style="flex:1;min-height:0;"></div>
  </div>

  <!-- PREDICTIONS -->
  <div class="panel" id="col-pred">
    <div class="ph">📊 Predictions</div>
    <div class="pb">
      <div class="mbox">
        <div class="mlbl">PMV — Predicted Mean Vote</div>
        <div class="mval">{pmv_display}</div>
      </div>
      <div class="mbox">
        <div class="mlbl">PPD — % Dissatisfied</div>
        <div class="mval">{ppd_display}</div>
      </div>
      <div class="bwrap">
        <div class="bbg">
          <div class="bfg" style="width:{pct:.1f}%;background:{bc};"></div>
          <div class="bcl"></div>
        </div>
        <div class="blbl">
          <span>−3</span>
          <span style="font-weight:700;color:{bc};">{lbl}</span>
          <span>+3</span>
        </div>
        <div style="font-size:10.5px;color:#aab8c2;margin-top:4px;">ASHRAE 55 target: −0.5 to +0.5</div>
      </div>
      <div class="sbox">
        <div class="stit">Input Summary</div>
        <table class="st">
          <tr><td>🧭 Orientation</td><td>{orientation} ({orient_deg}°)</td></tr>
          <tr><td>📅 Month</td><td>{month_val}</td></tr>
          <tr><td>↔ Wall Width</td><td>{wall_width:.1f} m</td></tr>
          <tr><td>↕ Room Depth</td><td>{room_depth:.1f} m</td></tr>
          <tr><td>🪟 WWR</td><td>{int(wwr*100)}%</td></tr>
        </table>
      </div>
    </div>
  </div>

</div>

<!-- INSIGHTS -->
<div id="footer">
  <div id="ilbl">💡 Design Insights</div>
  <div id="istrip">{ins_cards}</div>
</div>

<!-- MODAL -->
<div id="fbo" onclick="if(event.target===this)closeFB()">
  <div id="fbm">
    <div id="fbh">
      <span>📋 PhD Research Feedback Form</span>
      <button id="fbc" onclick="closeFB()">✕</button>
    </div>
    <iframe id="fbf" src="" data-src="https://forms.office.com/r/THfuycGkDZ"
            title="Feedback" allow="clipboard-write"></iframe>
  </div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
// ── Render 3D ─────────────────────────────────────────────
var fig = {fig3d_json};
Plotly.newPlot('pdiv', fig.data, fig.layout,
  {{responsive:true,displayModeBar:false,scrollZoom:false}});

// ── Live slider value display ─────────────────────────────
function liveUpdate(el, labelId, unit) {{
  var v = parseFloat(el.value);
  var txt = unit==='%' ? Math.round(v*100)+'%' : v.toFixed(unit==='m'?1:2)+' '+unit;
  document.getElementById(labelId).textContent = txt;
}}

// ── Push state to URL → triggers Streamlit rerun ──────────
var _timer = null;
function pushState() {{
  clearTimeout(_timer);
  _timer = setTimeout(function() {{
    var url = new URL(window.location.href);
    url.searchParams.set('orientation', document.getElementById('sel-o').value);
    url.searchParams.set('month',       document.getElementById('sel-m').value);
    url.searchParams.set('ww',  parseFloat(document.getElementById('sl-ww').value).toFixed(2));
    url.searchParams.set('rd',  parseFloat(document.getElementById('sl-rd').value).toFixed(2));
    url.searchParams.set('wwr', parseFloat(document.getElementById('sl-wwr').value).toFixed(2));
    window.location.href = url.toString();
  }}, 350);
}}

// ── Modal ─────────────────────────────────────────────────
function openFB() {{
  var f=document.getElementById('fbf');
  if(!f.src||f.src===window.location.href) f.src=f.getAttribute('data-src');
  document.getElementById('fbo').classList.add('open');
}}
function closeFB() {{ document.getElementById('fbo').classList.remove('open'); }}
document.addEventListener('keydown',function(e){{if(e.key==='Escape')closeFB();}});
</script>
</body>
</html>
"""

components.html(HTML, height=820, scrolling=False)
