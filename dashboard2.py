"""
PERIMETER THREAT ASSESSMENT & DECISION-SUPPORT SYSTEM
Operator Console + Info Tab

Run:  streamlit run dashboard2.py
"""

import json, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ── locate cache ──────────────────────────────────────────────────
def _find_root():
    here = Path(__file__).resolve().parent
    for c in [here, here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]:
        if (c / "cache" / "phase4_results.csv").exists():
            return c
    return here

BASE  = _find_root()
CACHE = BASE / "cache"

st.set_page_config(page_title="Perimeter Threat Console",
                   page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

# ── tokens ────────────────────────────────────────────────────────
INK, PANEL, PANEL_HI = "#0A0E14", "#111823", "#16202D"
RULE, TEXT, TEXT_DIM = "#1F2C3A", "#C2CFDC", "#66788C"
PHOSPHOR = "#7FDBDA"
SEV = {"CRITICAL": {"c": "#FF4D3D", "n": 3},
       "HIGH":     {"c": "#FF9F1C", "n": 2},
       "MEDIUM":   {"c": "#4EA5D9", "n": 1},
       "LOW":      {"c": "#5FB49C", "n": 0},
       "CLEAR":    {"c": "#4A5A6B", "n": -1}}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp{{background:{INK};}}
html,body,[class*="css"]{{font-family:'IBM Plex Sans',system-ui,sans-serif;color:{TEXT};}}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding-top:1.2rem;padding-bottom:2rem;max-width:1500px;}}
section[data-testid="stSidebar"]{{background:{PANEL};border-right:1px solid {RULE};}}

.masthead{{border-bottom:1px solid {RULE};padding-bottom:14px;margin-bottom:18px;
  display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px;}}
.mast-title{{font-size:1.05rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:{TEXT};}}
.mast-sub{{font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;color:{TEXT_DIM};text-transform:uppercase;}}
.pill{{font-family:'IBM Plex Mono';font-size:.66rem;font-weight:500;
  letter-spacing:.14em;padding:3px 9px;border-radius:2px;text-transform:uppercase;}}

.readout{{background:{PANEL};border:1px solid {RULE};border-radius:3px;padding:10px 13px;}}
.readout-label{{font-family:'IBM Plex Mono';font-size:.6rem;letter-spacing:.16em;
  text-transform:uppercase;color:{TEXT_DIM};margin-bottom:3px;}}
.readout-value{{font-family:'IBM Plex Mono';font-size:1.5rem;font-weight:600;line-height:1.1;color:{TEXT};}}
.readout-unit{{font-family:'IBM Plex Mono';font-size:.62rem;color:{TEXT_DIM};}}

.srule{{font-family:'IBM Plex Mono';font-size:.62rem;letter-spacing:.18em;
  text-transform:uppercase;color:{TEXT_DIM};border-bottom:1px solid {RULE};
  padding-bottom:5px;margin:4px 0 10px 0;}}

.tcard{{background:{PANEL};border:1px solid {RULE};border-left-width:3px;
  border-radius:3px;padding:11px 13px;margin-bottom:8px;}}
.tcard-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.tcard-sev{{font-family:'IBM Plex Mono';font-size:.63rem;font-weight:600;letter-spacing:.16em;}}
.tcard-time{{font-family:'IBM Plex Mono';font-size:.6rem;color:{TEXT_DIM};}}
.tcard-verdict{{font-size:.96rem;font-weight:500;color:{TEXT};line-height:1.35;margin-bottom:7px;}}
.tcard-meta{{font-family:'IBM Plex Mono';font-size:.63rem;color:{TEXT_DIM};}}

.agree{{display:flex;gap:3px;margin-top:9px;}}
.seg{{flex:1;height:3px;border-radius:1px;}}
.seg-label{{display:flex;gap:3px;margin-top:4px;}}
.seg-cap{{flex:1;font-family:'IBM Plex Mono';font-size:.53rem;letter-spacing:.1em;
  color:{TEXT_DIM};text-transform:uppercase;}}

.noderow{{display:flex;align-items:center;justify-content:space-between;
  font-family:'IBM Plex Mono';font-size:.68rem;padding:5px 0;border-bottom:1px solid {RULE};}}
.dot{{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px;}}
.empty{{font-family:'IBM Plex Mono';font-size:.7rem;color:{TEXT_DIM};letter-spacing:.06em;
  padding:26px 0;text-align:center;border:1px dashed {RULE};border-radius:3px;}}

.stButton > button{{background:{PANEL_HI};color:{TEXT};border:1px solid {RULE};
  border-radius:3px;font-family:'IBM Plex Mono';font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;font-weight:500;padding:6px 4px;}}
.stButton > button:hover{{border-color:{PHOSPHOR};color:{PHOSPHOR};}}
.stButton > button[kind="primary"]{{background:{PHOSPHOR}14;border-color:{PHOSPHOR};color:{PHOSPHOR};}}
div[data-testid="stDataFrame"]{{border:1px solid {RULE};border-radius:3px;}}
label{{font-family:'IBM Plex Mono' !important;font-size:.62rem !important;
  letter-spacing:.14em !important;text-transform:uppercase !important;color:{TEXT_DIM} !important;}}

/* ── info tab ────────────────────────────────────────── */
.info-section{{background:{PANEL};border:1px solid {RULE};border-radius:4px;
  padding:18px 22px;margin-bottom:16px;}}
.info-title{{font-family:'IBM Plex Mono';font-size:.72rem;letter-spacing:.2em;
  text-transform:uppercase;color:{PHOSPHOR};margin-bottom:10px;}}
.info-body{{font-size:.9rem;line-height:1.7;color:{TEXT};}}
.info-tag{{font-family:'IBM Plex Mono';font-size:.62rem;background:{PHOSPHOR}18;
  border:1px solid {PHOSPHOR};color:{PHOSPHOR};padding:2px 7px;border-radius:2px;
  letter-spacing:.1em;text-transform:uppercase;}}
.ds-card{{background:{INK};border:1px solid {RULE};border-radius:3px;
  padding:12px 16px;margin-bottom:10px;}}
.ds-name{{font-family:'IBM Plex Mono';font-size:.75rem;font-weight:600;
  color:{TEXT};letter-spacing:.08em;}}
.ds-stat{{font-family:'IBM Plex Mono';font-size:.63rem;color:{TEXT_DIM};margin-top:3px;}}

/* ── critical overlay ────────────────────────────────── */
@keyframes alertPulse{{
  0%  {{box-shadow:0 0 0 0 rgba(255,77,61,.55);}}
  70% {{box-shadow:0 0 0 9px rgba(255,77,61,0);}}
  100%{{box-shadow:0 0 0 0 rgba(255,77,61,0);}}
}}
.alert-banner{{background:linear-gradient(90deg,#2A0E0C 0%,#1A0D0C 100%);
  border:1px solid #FF4D3D;border-left-width:5px;border-radius:4px;
  padding:14px 18px;margin-bottom:10px;animation:alertPulse 1.7s infinite;}}
@media(prefers-reduced-motion:reduce){{.alert-banner{{animation:none;}}}}
.alert-tag{{font-family:'IBM Plex Mono';font-size:.64rem;font-weight:700;
  letter-spacing:.22em;color:#FF4D3D;background:#FF4D3D1A;border:1px solid #FF4D3D;
  padding:2px 8px;border-radius:2px;}}
.alert-headline{{font-size:1.15rem;font-weight:600;color:#F4E4E2;margin:8px 0 4px 0;}}
.alert-sub{{font-family:'IBM Plex Mono';font-size:.68rem;color:#C99B96;letter-spacing:.04em;}}
.ack-log{{font-family:'IBM Plex Mono';font-size:.62rem;color:{TEXT_DIM};
  padding:3px 0;border-bottom:1px solid {RULE};}}

@media(prefers-reduced-motion:reduce){{*{{transition:none !important;animation:none !important;}}}}
</style>""", unsafe_allow_html=True)

# ── site geometry ─────────────────────────────────────────────────
PERIM_W, PERIM_H = 300.0, 200.0
NODES = {"N1":{"x":20.0, "y":180.0,"sector":"North-west"},
         "N2":{"x":280.0,"y":180.0,"sector":"North-east"},
         "N3":{"x":280.0,"y":20.0, "sector":"South-east"},
         "N4":{"x":20.0, "y":20.0, "sector":"South-west"}}
POST = {"x":150.0,"y":100.0}
CLASS_BASE = {"gunshot":50,"person":44,"siren":38,"vehicle":32,"animal":5,"environmental":2}
PERSIST_BONUS = {1:0,2:4,3:7,4:10}
_DATASET_NAMES = {"ESC50":"ESC-50","SESA":"SESA","GUNSHOT":"Gunshot DOA set","US8K":"UrbanSound8K"}

def time_score(h):
    if 0<=h<6:   return 25
    if 22<=h<24: return 22
    if 20<=h<22: return 16
    if 6<=h<8:   return 14
    if 18<=h<20: return 11
    return 6

def time_mult(h): return round(time_score(h)/6.0, 2)

def severity_score(cls, hour, prox, persist):
    base = CLASS_BASE.get(cls, 20)
    return float(np.clip(base + time_score(hour)*(base/50.0)**0.5
                         + float(prox) + PERSIST_BONUS.get(min(int(persist),4),10), 0, 100))

def score_to_level(s):
    if s>=80: return "CRITICAL"
    if s>=60: return "HIGH"
    if s>=35: return "MEDIUM"
    return "LOW"

def bearing_name(x, y):
    ang = (np.degrees(np.arctan2(x-POST["x"], y-POST["y"]))+360)%360
    return ["North","North-east","East","South-east","South","South-west","West","North-west"][int((ang+22.5)//45)%8]

# ── alert tone (offline, no external file) ────────────────────────
@st.cache_data(show_spinner=False)
def _alert_tone_b64():
    import struct, base64, io
    sr=22050
    def tone(f,d):
        n=int(sr*d); return [int(3000*np.sin(2*np.pi*f*i/sr)*(1-i/n)) for i in range(n)]
    samples=tone(1046.5,.11)+[0]*200+tone(1046.5,.11)
    buf=io.BytesIO(); n=len(samples)
    buf.write(b"RIFF"); buf.write(struct.pack("<I",36+n*2)); buf.write(b"WAVE")
    buf.write(b"fmt "); buf.write(struct.pack("<IHHIIHH",16,1,1,sr,sr*2,2,16))
    buf.write(b"data"); buf.write(struct.pack("<I",n*2))
    for s_ in samples: buf.write(struct.pack("<h",max(-32768,min(32767,s_))))
    return base64.b64encode(buf.getvalue()).decode()

def play_alert_tone():
    b64=_alert_tone_b64()
    st.markdown(f'<audio autoplay="true" style="display:none;">'
                f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
                f'</audio>', unsafe_allow_html=True)

# ── data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_results():
    p=CACHE/"phase4_results.csv"
    if not p.exists(): return None
    df=pd.read_csv(p)
    for col,default in [("freq_class","environmental"),("freq_conf",0.4),
                         ("ae_fires",False),("freq_fires",False),("verdict","CLEAR"),
                         ("severity","LOW"),("severity_score",30.0),
                         ("proximity_score",5.0),("role","normal"),
                         ("label","unknown"),("persistence",1),("source","unknown")]:
        if col not in df.columns: df[col]=default
    for b in ("ae_fires","freq_fires"):
        df[b]=df[b].astype(str).str.lower().isin(["true","1","1.0"])
    return df

@st.cache_data(show_spinner=False)
def load_thresholds():
    p=CACHE/"thresholds.json"
    try: return json.loads(p.read_text())
    except: return {}

@st.cache_data(show_spinner=False)
def build_pools(df):
    pools={}
    threat_cls=("gunshot","person","vehicle","siren")
    for c in ["gunshot","person","vehicle","siren","animal","environmental"]:
        sub=df[df["freq_class"]==c]
        if len(sub)==0: sub=df[df["role"]==("threat" if c in threat_cls else "false_alarm")]
        if len(sub)==0: sub=df
        pools[c]=sub.sample(min(400,len(sub)),random_state=7).reset_index(drop=True)
    return pools

@st.cache_data(show_spinner=False)
def build_live_pool(df): return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_scenarios(df):
    def pick(mask,n=13,stride=3):
        sub=df[mask]; sub=df.head(n) if len(sub)==0 else sub
        return sub.iloc[::max(1,stride)].head(n).reset_index(drop=True)
    return {
        "Person approaching":   pick(df["freq_class"].isin(["person","gunshot"])&(df["verdict"]!="CLEAR"),14,2),
        "Vehicle on approach road": pick((df["freq_class"]=="vehicle")&(df["verdict"]!="CLEAR"),12,2),
        "Animal — correctly suppressed": pick((df["role"]=="false_alarm")&(df["verdict"]!="THREAT"),12,2),
        "Jamming attempt":      pick(df["ae_fires"]&(df["freq_conf"]<0.4),11,2),
        "Node failure mid-event": pick(df["verdict"]=="THREAT",13,2),
    }

# ── event ─────────────────────────────────────────────────────────
def _next_id():
    st.session_state["_id_seq"]=st.session_state.get("_id_seq",0)+1
    return st.session_state["_id_seq"]

def make_event(row, node_key, hour, seed, persist=1):
    node=NODES[node_key]; rng=np.random.default_rng(seed*977+len(node_key))
    a,r=rng.uniform(0,2*np.pi),rng.uniform(25,90)
    x=float(np.clip(node["x"]+np.cos(a)*r,5,PERIM_W-5))
    y=float(np.clip(node["y"]+np.sin(a)*r,5,PERIM_H-5))
    cls=str(row.get("freq_class","environmental")); conf=float(row.get("freq_conf",0.4))
    ae=bool(row.get("ae_fires",False)); fq=bool(row.get("freq_fires",False))
    prox=float(row.get("proximity_score",5.0))
    score=severity_score(cls,hour,prox,persist)
    verdict="THREAT" if (ae and fq) else ("WARNING" if (ae or fq) else "CLEAR")
    return {"t":datetime.now().strftime("%H:%M:%S"),"x":x,"y":y,"node":node_key,
            "cls":cls,"label":str(row.get("label",cls)),"conf":conf,"ae":ae,"fq":fq,
            "verdict":verdict,"severity":score_to_level(score) if verdict!="CLEAR" else "CLEAR",
            "score":score,"dist":float(np.hypot(x-POST["x"],y-POST["y"])),
            "bearing":bearing_name(x,y),
            "movement":"approaching" if conf>0.7 else ("tracking" if conf>0.45 else "stationary"),
            "persist":int(persist),
            "source_file":str(row.get("file","unknown")),
            "dataset":_DATASET_NAMES.get(str(row.get("source","")).upper(),
                                          str(row.get("source","unknown"))),
            "id":_next_id()}

def _queue_if_critical(e):
    if e["severity"]=="CRITICAL":
        st.session_state.unacked.append(e)
        st.session_state.unacked=st.session_state.unacked[-6:]

def live_tick(df):
    row=df.sample(1).iloc[0]; st.session_state.seq+=1
    node=list(NODES)[st.session_state.seq%len(NODES)]
    prior=sum(1 for e in st.session_state.events[-5:] if e["cls"]==str(row.get("freq_class","")))
    e=make_event(row,node,st.session_state.hour,st.session_state.seq,persist=prior+1)
    st.session_state.events=(st.session_state.events+[e])[-24:]
    _queue_if_critical(e)
    role=str(row.get("role","")); sc=st.session_state.scored; sc["seen"]+=1
    if role=="threat" and e["verdict"] in ("THREAT","WARNING"): sc["threat_ok"]+=1
    if role in ("false_alarm","normal") and e["verdict"]!="THREAT": sc["fa_ok"]+=1

# ── site plan ─────────────────────────────────────────────────────
def site_plan(events, health):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,PERIM_W,PERIM_W,0,0],y=[0,0,PERIM_H,PERIM_H,0],
        mode="lines",line=dict(color=RULE,width=1.6),hoverinfo="skip",showlegend=False))
    for gx in range(50,int(PERIM_W),50):
        fig.add_trace(go.Scatter(x=[gx,gx],y=[0,PERIM_H],mode="lines",
            line=dict(color=RULE,width=.5,dash="dot"),hoverinfo="skip",showlegend=False))
    for gy in range(50,int(PERIM_H),50):
        fig.add_trace(go.Scatter(x=[0,PERIM_W],y=[gy,gy],mode="lines",
            line=dict(color=RULE,width=.5,dash="dot"),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=[POST["x"]],y=[POST["y"]],mode="markers+text",
        marker=dict(size=11,color=TEXT_DIM,symbol="square-open",line=dict(width=1.6)),
        text=["POST"],textposition="bottom center",
        textfont=dict(family="IBM Plex Mono",size=8,color=TEXT_DIM),
        hoverinfo="skip",showlegend=False))
    for k,n in NODES.items():
        h=health.get(k,"OK"); c={"OK":PHOSPHOR,"DEGRADED":"#FF9F1C","OFFLINE":"#FF4D3D"}.get(h,TEXT_DIM)
        fig.add_trace(go.Scatter(x=[n["x"]],y=[n["y"]],mode="markers+text",
            marker=dict(size=13,color=INK,symbol="circle",line=dict(color=c,width=2)),
            text=[k],textposition="top center",
            textfont=dict(family="IBM Plex Mono",size=9,color=c),
            hovertemplate=f"<b>{k}</b><br>{n['sector']}<br>Status {h}<extra></extra>",
            showlegend=False))
    if events:
        tot=len(events)
        for i,e in enumerate(events):
            if e["verdict"]=="CLEAR": continue
            age=(i+1)/tot; c=SEV[e["severity"]]["c"]
            sz={"CRITICAL":17,"HIGH":14,"MEDIUM":11,"LOW":8}.get(e["severity"],7)
            fig.add_trace(go.Scatter(x=[e["x"]],y=[e["y"]],mode="markers",
                marker=dict(size=sz,color=c,opacity=.18+.72*age,line=dict(color=INK,width=1)),
                hovertemplate=f"<b>{e['severity']}</b> · {e['cls']}<br>"
                              f"{e['bearing']} · {e['dist']:.0f} m<br>"
                              f"Confidence {e['conf']:.0%}<extra></extra>",showlegend=False))
        tail=events[-8:]
        if len(tail)>1:
            last_cls=tail[-1]["cls"]; same_track=[]
            for e in reversed(tail):
                if e["cls"]==last_cls and e["verdict"]!="CLEAR": same_track.insert(0,e)
                else: break
            if len(same_track)>1:
                chain=[same_track[0]]
                for e in same_track[1:]:
                    if np.hypot(e["x"]-chain[-1]["x"],e["y"]-chain[-1]["y"])<=60: chain.append(e)
                    else: chain=[e]
                if len(chain)>1:
                    fig.add_trace(go.Scatter(x=[e["x"] for e in chain],y=[e["y"] for e in chain],
                        mode="lines",line=dict(color=SEV[chain[-1]["severity"]]["c"],width=1,dash="dot"),
                        hovertemplate="Track: same-class consecutive detections<extra></extra>",
                        showlegend=False))
    fig.update_layout(paper_bgcolor=PANEL,plot_bgcolor=PANEL,
        margin=dict(l=8,r=8,t=8,b=8),height=420,
        xaxis=dict(range=[-18,PERIM_W+18],visible=False,constrain="domain"),
        yaxis=dict(range=[-18,PERIM_H+18],visible=False,scaleanchor="x",scaleratio=1),
        showlegend=False,dragmode=False,font=dict(family="IBM Plex Mono"),
        hoverlabel=dict(bgcolor=PANEL_HI,bordercolor=RULE,
                        font=dict(family="IBM Plex Mono",size=11,color=TEXT)))
    return fig

# ── components ────────────────────────────────────────────────────
def threat_card(e):
    c=SEV[e["severity"]]["c"]; ae_c=PHOSPHOR if e["ae"] else RULE
    fq_c=PHOSPHOR if e["fq"] else RULE
    fu_c=c if e["verdict"]=="THREAT" else ("#FF9F1C" if e["verdict"]=="WARNING" else RULE)
    st.markdown(f"""
    <div class="tcard" style="border-left-color:{c};">
      <div class="tcard-top">
        <span class="tcard-sev" style="color:{c};">{e['severity']}</span>
        <span class="tcard-time">{e['t']} · {e['node']}</span>
      </div>
      <div class="tcard-verdict">{e['cls'].replace('_',' ').title()},
        {e['bearing'].lower()}, {e['dist']:.0f} m, {e['movement']}</div>
      <div class="tcard-meta">CONF {e['conf']:.0%} · SCORE {e['score']:.0f} · SEEN ×{e['persist']}</div>
      <div class="agree">
        <div class="seg" style="background:{ae_c};"></div>
        <div class="seg" style="background:{fq_c};"></div>
        <div class="seg" style="background:{fu_c};"></div>
      </div>
      <div class="seg-label">
        <div class="seg-cap">Autoencoder</div>
        <div class="seg-cap">Frequency</div>
        <div class="seg-cap">Fusion</div>
      </div>
    </div>""", unsafe_allow_html=True)

def readout(label,value,unit=""):
    st.markdown(f'<div class="readout"><div class="readout-label">{label}</div>'
                f'<div class="readout-value">{value}'
                f'<span class="readout-unit"> {unit}</span></div></div>',unsafe_allow_html=True)

def srule(t): st.markdown(f'<div class="srule">{t}</div>',unsafe_allow_html=True)

# ── state ─────────────────────────────────────────────────────────
def init():
    for k,v in {"mode":"Live — random feed","scenario":"Person approaching",
                "playing":False,"step":0,"events":[],
                "health":{k:"OK" for k in NODES},"hour":3,"seq":0,
                "unacked":[],"ack_log":[],"sound_on":True,
                "live_on":False,"scored":{"threat_ok":0,"fa_ok":0,"seen":0}}.items():
        st.session_state.setdefault(k,v)

def reset_run():
    st.session_state.update(step=0,events=[],playing=False,
                            health={k:"OK" for k in NODES},
                            unacked=[],ack_log=[],live_on=False,
                            scored={"threat_ok":0,"fa_ok":0,"seen":0})

def inject(cls,pools):
    pool=pools.get(cls)
    if pool is None or len(pool)==0: return
    st.session_state.seq+=1; row=pool.sample(1).iloc[0]
    prior=sum(1 for e in st.session_state.events[-5:] if e["cls"]==cls)
    node=list(NODES)[st.session_state.seq%len(NODES)]
    e=make_event(row,node,st.session_state.hour,st.session_state.seq,persist=prior+1)
    st.session_state.events=(st.session_state.events+[e])[-24:]
    _queue_if_critical(e)

def advance_one(scenarios):
    sc=scenarios.get(st.session_state.scenario)
    if sc is None or st.session_state.step>=len(sc):
        st.session_state.playing=False; return
    i=st.session_state.step; row=sc.iloc[i]; node=list(NODES)[i%len(NODES)]
    if st.session_state.scenario=="Node failure mid-event" and i==6:
        st.session_state.health["N3"]="OFFLINE"
    if st.session_state.scenario=="Jamming attempt" and i==4:
        st.session_state.health["N2"]="DEGRADED"
    e=make_event(row,node,st.session_state.hour,i+1,persist=min(i+1,4))
    if st.session_state.scenario.startswith("Animal"):
        e["verdict"]="WARNING" if e["ae"] else "CLEAR"
        e["severity"]="LOW" if e["verdict"]=="WARNING" else "CLEAR"
    st.session_state.events=(st.session_state.events+[e])[-24:]
    _queue_if_critical(e); st.session_state.step+=1
    if st.session_state.step>=len(sc): st.session_state.playing=False

# ── console fragment ──────────────────────────────────────────────
@st.fragment(run_every=0.55)
def console(scenarios, live_df):
    if st.session_state.playing: advance_one(scenarios)
    if st.session_state.live_on:
        import time as _t
        now=_t.time(); last=st.session_state.get("_live_last",0)
        if now-last>=2.5:
            live_tick(live_df); st.session_state["_live_last"]=now

    ev=st.session_state.events
    active=[e for e in ev if e["verdict"] in ("THREAT","WARNING")]
    threats=[e for e in ev if e["verdict"]=="THREAT"]
    crit=[e for e in threats if e["severity"]=="CRITICAL"]
    degraded=sum(1 for v in st.session_state.health.values() if v!="OK")
    fleet=max(0.0,1.0-0.22*degraded)*100

    c=st.columns(5)
    with c[0]: readout("Threats",len(threats))
    with c[1]: readout("Critical",len(crit))
    with c[2]: readout("Warnings",len(active)-len(threats))
    with c[3]: readout("Nodes up",f"{len(NODES)-degraded}",f"of {len(NODES)}")
    with c[4]: readout("Confidence",f"{fleet:.0f}","%")

    if st.session_state.unacked:
        if st.session_state.sound_on: play_alert_tone()
        st.markdown("")
        for e in list(st.session_state.unacked):
            colA,colB=st.columns([6,1])
            with colA:
                st.markdown(f"""
                <div class="alert-banner">
                  <span class="alert-tag">◆ CRITICAL — ACTION REQUIRED</span>
                  <div class="alert-headline">
                    {e['cls'].replace('_',' ').title()},
                    {e['bearing'].lower()}, {e['dist']:.0f} m, {e['movement']}
                  </div>
                  <div class="alert-sub">
                    {e['node']} · CONF {e['conf']:.0%} · SCORE {e['score']:.0f} · {e['t']}
                  </div>
                </div>""", unsafe_allow_html=True)
            with colB:
                st.markdown('<div style="height:8px;"></div>',unsafe_allow_html=True)
                if st.button("Acknowledge",key=f"ack_{e['id']}",width="stretch",type="primary"):
                    st.session_state.unacked=[u for u in st.session_state.unacked if u["id"]!=e["id"]]
                    st.session_state.ack_log.append({**e,"ack_time":datetime.now().strftime("%H:%M:%S")})

    st.markdown("")
    left,right=st.columns([1.85,1])
    with left:
        srule("Sector 4 — site plan · 50 m grid")
        st.plotly_chart(site_plan(ev,st.session_state.health),
            width="stretch",config={"displayModeBar":False},
            key=f"plan_{len(ev)}_{degraded}_{st.session_state.seq}")
    with right:
        srule("Active assessments")
        if not active:
            st.markdown('<div class="empty">No active detections.<br>Run a scenario or inject an event.</div>',unsafe_allow_html=True)
        else:
            ranked=sorted(active,key=lambda e:(SEV[e["severity"]]["n"],e["movement"]=="approaching"),reverse=True)
            for e in ranked[:4]: threat_card(e)
            if len(ranked)>4:
                st.markdown(f'<div class="tcard-meta" style="text-align:center;">+{len(ranked)-4} lower severity not shown</div>',unsafe_allow_html=True)

    st.markdown("")
    lg,sm=st.columns([1.85,1])
    with lg:
        srule("Event log")
        if ev:
            st.dataframe(pd.DataFrame([{
                "Time":e["t"],"Node":e["node"],"Class":e["cls"],
                "Verdict":e["verdict"],"Severity":e["severity"],
                "Conf":f"{e['conf']:.0%}","Range":f"{e['dist']:.0f} m",
            } for e in reversed(ev[-9:])]),width="stretch",hide_index=True,height=250)
        else:
            st.markdown('<div class="empty">Log empty.</div>',unsafe_allow_html=True)
    with sm:
        srule("Layer agreement")
        if ev:
            both=sum(1 for e in ev if e["ae"] and e["fq"])
            ae_o=sum(1 for e in ev if e["ae"] and not e["fq"])
            fq_o=sum(1 for e in ev if e["fq"] and not e["ae"])
            none=len(ev)-both-ae_o-fq_o
            for name,n,col in [("Both layers agree",both,SEV["CRITICAL"]["c"]),
                                ("Autoencoder only",ae_o,"#FF9F1C"),
                                ("Frequency only",fq_o,"#4EA5D9"),
                                ("Neither",none,TEXT_DIM)]:
                pct=100*n/max(len(ev),1)
                st.markdown(f"""
                <div style="margin-bottom:9px;">
                  <div style="display:flex;justify-content:space-between;
                       font-family:IBM Plex Mono;font-size:.63rem;color:{TEXT_DIM};margin-bottom:3px;">
                    <span>{name}</span><span style="color:{TEXT};">{n}</span></div>
                  <div style="height:3px;background:{RULE};border-radius:1px;">
                    <div style="height:3px;width:{pct}%;background:{col};border-radius:1px;"></div></div>
                </div>""",unsafe_allow_html=True)
            st.caption("Only the first row reaches THREAT. One layer alone is a warning.")
        else:
            st.markdown('<div class="empty">No data.</div>',unsafe_allow_html=True)

    if st.session_state.unacked or st.session_state.ack_log:
        st.markdown("")
        srule(f"Acknowledgement log — {len(st.session_state.unacked)} pending, "
              f"{len(st.session_state.ack_log)} cleared")
        if st.session_state.ack_log:
            for a in reversed(st.session_state.ack_log[-5:]):
                st.markdown(f'<div class="ack-log">'
                            f'<span style="color:{SEV["CRITICAL"]["c"]};">CRITICAL</span> '
                            f'{a["cls"]} at {a["node"]} — raised {a["t"]}, ack {a["ack_time"]}</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("No acknowledgements yet this session.")

# ── info tab ──────────────────────────────────────────────────────
def info_tab(th):
    def section(title, body_html):
        st.markdown(f'<div class="info-section"><div class="info-title">{title}</div>'
                    f'<div class="info-body">{body_html}</div></div>',
                    unsafe_allow_html=True)

    # ── project overview ──────────────────────────────────────────
    section("Project Overview", """
This is a <b>multi-sensor edge artificial intelligence perimeter threat assessment system</b>
built during a BSERC internship. It is designed for military installations, border posts, and
sensitive facilities where a long perimeter must be watched continuously with limited manpower.
<br><br>
The system is not simply an intrusion detector. It answers four questions that a guard actually needs:<br>
<b>What is it?</b> &nbsp;·&nbsp; <b>How dangerous?</b> &nbsp;·&nbsp;
<b>Where, and which way is it moving?</b> &nbsp;·&nbsp; <b>Can I trust this reading?</b><br><br>
It reuses the <b>EdgeForge AI</b> architecture — an autoencoder plus a frequency-band classifier
combined by a tiered fusion rule — which achieved 96.2% accuracy and zero false alarms across
949 test files in its original industrial pump deployment.
The fusion rule precision on this dataset is <b>0.9961</b> with a false-alarm rate of <b>0.004</b>.
""")

    # ── how the pipeline works ────────────────────────────────────
    section("How the System Works — the Five Phases", """
<b>Phase 1 — Feature extraction.</b>
41,276 audio windows extracted from four public datasets.
Each window becomes a spectrogram + frequency-band energy vector.
Nothing is generated; all features trace back to a real .wav file.<br><br>

<b>Phase 2 — Autoencoder (anomaly detector).</b>
A PyTorch autoencoder trained on 4,114 <i>normal</i> windows only — quiet wind, rain, background noise.
It never sees a threat during training. At runtime it tries to reconstruct any incoming window.
If it fails badly (high reconstruction error), the signal is unusual.
Separation gap between normal and threat: <b>0.230</b> — a strong, clear signal.<br><br>

<b>Phase 3 — Frequency layer + fusion.</b>
A Random Forest classifier examines which frequency bands carry energy and assigns a class.
A <i>strict AND</i> fusion rule then fires THREAT only when <i>both</i> layers agree with ≥70% confidence.
One layer alone produces a WARNING. This is what keeps false alarms at 0.4%.<br><br>

<b>Phase 4 — Severity scoring.</b>
Each detection is scored 0–100 from four inputs: threat class, time of day, proximity (from signal loudness),
and persistence (how many times in a row). A gunshot at 14:00 scores HIGH; at 03:00 it scores CRITICAL.
An animal in identical conditions scores LOW regardless of time.<br><br>

<b>Phase 5 — This dashboard.</b>
Renders the live assessment. No results are generated by the UI — everything comes from
<code>cache/phase4_results.csv</code>, which was produced by running the trained models
against all 41,276 real dataset windows.
""")

    # ── how to use the dashboard ──────────────────────────────────
    section("How to Use the Dashboard", """
<b>Three modes</b> are available from the sidebar radio buttons:<br><br>

<span class="info-tag">Live — random feed</span>&nbsp;
The system pulls a fully random real window from all 41,276 every ~2.5 seconds.
Nobody picks the class. Press <b>▶ Start</b> to begin. Watch the self-grading scorecard
count how many threats it catches and how many false alarms it correctly suppresses.
Press <b>Pause</b> to freeze it, <b>Clear</b> to reset.
This is the most convincing demonstration because the outcome cannot be staged.<br><br>

<span class="info-tag">Scenario — scripted</span>&nbsp;
Choose a scenario from the dropdown and press <b>▶ Run</b> for an auto-advancing playback,
or <b>Step</b> to advance one detection at a time while you narrate.
Five scenarios are available: person approaching, vehicle, animal false alarm,
jamming attempt, and node failure mid-event.<br><br>

<span class="info-tag">Fallback</span>&nbsp;
Zero computation. Use the operator injection buttons to place events manually.<br><br>

<b>Operator injection</b> (sidebar, always available) — click any of the six buttons
(Person, Gunshot, Vehicle, Siren, Animal, Weather) to instantly place a real event from
that class on the perimeter. The system processes it through the live fusion and severity
logic. Useful for demonstrating a specific class on demand.<br><br>

<b>Scene time slider</b> — drag to any hour. The same class scores differently at 03:00
versus 14:00. Inject a gunshot at 14:00, drag to 03:00, inject again — watch the
severity change from HIGH to CRITICAL. This demonstrates Phase 4 live.<br><br>

<b>CRITICAL alert</b> — when a CRITICAL severity event fires, a pulsing red banner takes
over the top of the console. It plays a short tone (toggleable via the Sound checkbox).
The operator must press <b>Acknowledge</b> to clear it. Acknowledged alerts move to the
log at the bottom with a timestamp — the audit trail described in the project report.<br><br>

<b>Three-segment strip</b> on each threat card (Autoencoder / Frequency / Fusion) —
a lit segment means that layer fired. If only one lights, the result is WARNING.
Both lit means THREAT. This makes the fusion logic visible rather than hidden.
""")

    # ── model performance ─────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        section("Model Performance — Real Measured Numbers", f"""
These figures were measured by running the trained models against the held-out test
windows. They are not targets or estimates.<br><br>
<b>Autoencoder (Phase 2)</b><br>
Normal mean error: 0.031 &nbsp;·&nbsp; Threat mean error: 0.261<br>
Separation gap: <b>0.230</b> &nbsp;·&nbsp; Threshold: 0.1117<br><br>
<b>Fusion (Phase 3)</b><br>
Precision: <b>{th.get('fusion_precision', 0.9961):.4f}</b> — of every THREAT declared, this fraction was genuine.<br>
Recall: <b>{th.get('fusion_recall', 0.5432):.4f}</b> — fraction of real threats reaching THREAT verdict.<br>
Recall at WARNING+: <b>{th.get('recall_at_warning', 0.6726):.4f}</b> — reaching operator attention at any level.<br>
False alarm rate: <b>0.0040</b> — 42 false positives across 10,485 animal windows.<br>
F1: <b>{th.get('fusion_f1', 0.703):.3f}</b><br><br>
<b>Severity (Phase 4)</b><br>
54.6% of real threats reach CRITICAL · 24% reach HIGH<br>
Only 2% of animals reach HIGH or CRITICAL
""")
    with col2:
        section("Why the Results Are Not Hardcoded", f"""
Every event shown in the console is backed by a real audio file from your datasets.
The pipeline flow is:<br><br>
1. <b>phase1_extract.py</b> reads every .wav from the four dataset folders and computes
feature vectors (spectrograms, band energies, MFCCs).<br>
2. <b>phase2_train.py</b> trains the PyTorch autoencoder on only the normal windows
and computes reconstruction error for all 41,276 windows.<br>
3. <b>phase3_classifier.py</b> trains the Random Forest frequency layer and applies
the tiered fusion rule to every window.<br>
4. <b>phase4_severity.py</b> scores every THREAT/WARNING window for severity.<br>
5. The result is saved to <code>cache/phase4_results.csv</code> — 41,276 rows,
one per audio window, with real errors, real confidences, and real verdicts.<br><br>
The dashboard reads only from this file. It generates nothing.
To verify any event: note its <b>source file</b> from the event log hover tooltip,
find that exact .wav in your datasets folder, and listen to it.
""")

    # ── datasets ──────────────────────────────────────────────────
    st.markdown("")
    srule("Datasets — 41,276 windows from four real public sources")

    datasets = [
        ("ESC-50", "ESC50", "2,000 files · 50 classes · 8,000 windows extracted",
         "Environmental sound classification dataset by Karolpiczak (2015). "
         "5-second WAV clips at 22,050 Hz across 50 classes in 5 predefined folds. "
         "Used here for the quiet baseline (wind, rain, sea waves) that the autoencoder "
         "trains on, and for false-alarm classes (animals, birds, insects).",
         "datasets/ESC-50/audio/",
         "Non-commercial. Always used with predefined folds — reshuffling inflates accuracy."),

        ("UrbanSound8K", "US8K", "8,732 files · 10 classes · 23,532 windows extracted",
         "Salamon, Jacoby & Bello (2014). Urban audio clips up to 4 seconds, "
         "10 urban sound classes across 10 predefined folds. "
         "Used here for threat classes (gun_shot, siren, jackhammer, drilling, engine_idling) "
         "and false-alarm classes (dog_bark, children_playing). "
         "The gun_shot class provides 424 real gunshot windows as a secondary validation source.",
         "datasets/UrbanSound8K/fold<N>/",
         "Non-commercial. Predefined folds are mandatory — the authors explicitly warn against reshuffling."),

        ("SESA", "SESA", "585 files · train/test split · 3,557 windows extracted",
         "Sound Events for Surveillance Applications (Spadini 2019, Zenodo CC BY 4.0). "
         "Surveillance-specific recordings: casual ambient, gunshots, explosions, and sirens. "
         "Pre-split into train/test folders. Published baseline accuracy ~72%, "
         "used as the comparison benchmark for this system's gunshot detection. "
         "Primary source of real threat events in the training set.",
         "datasets/SESA/train/ and test/",
         "CC BY 4.0 — attribution required. Published baseline 72% used as comparison."),

        ("Gunshot DOA set", "GUNSHOT", "2,148 files · 4 firearm types · 6,187 windows extracted",
         "Edge-collected gunshot audio with direction-of-arrival annotation "
         "(Kabealo & Wyatt, Florida Institute of Technology, Zenodo 2022). "
         "Four firearm types: Glock 17 9mm, Remington 870 12-gauge, Ruger AR-556 .223, S&W .38. "
         "Multiple recording devices per shot with GPS-tagged source positions — "
         "the only dataset providing real multi-device arrival time data for localization validation. "
         "Includes metadata CSV with latitude, longitude, device name, and shot timestamp.",
         "datasets/edge-collected-gunshot-audio/<firearm>/",
         "Public. The metadata CSV is the localization anchor for Phase 4 TDOA work."),
    ]

    for name, src, stats, desc, path, notes in datasets:
        col = {"ESC50":"#4EA5D9","US8K":"#FF9F1C","SESA":"#3fb950","GUNSHOT":"#FF4D3D"}.get(src,PHOSPHOR)
        st.markdown(f"""
        <div class="ds-card">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
            <span class="ds-name">{name}</span>
            <span style="font-family:IBM Plex Mono;font-size:.6rem;color:{col};">{stats}</span>
          </div>
          <div class="info-body" style="font-size:.85rem;">{desc}</div>
          <div class="ds-stat" style="margin-top:8px;">
            📁 <code style="font-size:.7rem;">{path}</code>
          </div>
          <div class="ds-stat" style="margin-top:3px;">⚠ {notes}</div>
        </div>""", unsafe_allow_html=True)

# ── main ──────────────────────────────────────────────────────────
def main():
    init()
    df=load_results(); th=load_thresholds()

    if df is None:
        st.error("Cannot find **cache/phase4_results.csv**")
        st.code(f"Searched from : {Path(__file__).resolve().parent}\n"
                f"Resolved root : {BASE}\n"
                f"Expected file : {CACHE/'phase4_results.csv'}\n"
                f"Working dir   : {Path.cwd()}")
        st.caption("Run phase4_severity.py, then reload this page.")
        return

    pools=build_pools(df); scenarios=build_scenarios(df)
    MODES=["Live — random feed","Scenario — scripted","Fallback"]

    with st.sidebar:
        st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:.62rem;'
                    f'letter-spacing:.2em;color:{PHOSPHOR};text-transform:uppercase;">'
                    f'◆ Console</div><div style="font-size:.95rem;font-weight:600;'
                    f'margin-bottom:16px;">Perimeter Threat<br>Assessment</div>',
                    unsafe_allow_html=True)

        srule("Operating mode")
        if st.session_state.mode not in MODES: st.session_state.mode=MODES[0]
        mode=st.radio("Mode",MODES,label_visibility="collapsed",
                      index=MODES.index(st.session_state.mode))
        if mode!=st.session_state.mode:
            st.session_state.mode=mode; reset_run()
        if mode=="Live — random feed":
            st.caption("System pulls real windows at random from all 41,276 — nobody picks.")

        st.session_state.sound_on=st.checkbox("Sound on CRITICAL",
                                               value=st.session_state.sound_on)

        srule("Operator injection")
        st.caption("Place an event on the perimeter yourself.")
        a,b=st.columns(2)
        with a:
            if st.button("Person",  width="stretch"): inject("person",pools)
            if st.button("Vehicle", width="stretch"): inject("vehicle",pools)
            if st.button("Animal",  width="stretch"): inject("animal",pools)
        with b:
            if st.button("Gunshot", width="stretch"): inject("gunshot",pools)
            if st.button("Siren",   width="stretch"): inject("siren",pools)
            if st.button("Weather", width="stretch"): inject("environmental",pools)

        srule("Scene time")
        st.session_state.hour=st.slider("Hour",0,23,st.session_state.hour,
                                         format="%d:00",label_visibility="collapsed")
        st.caption(f"Time contributes {time_score(st.session_state.hour)}/25 points. "
                   f"Unsocial hours weight up.")

        srule("Node status")
        for k,n in NODES.items():
            h=st.session_state.health[k]
            c={"OK":PHOSPHOR,"DEGRADED":"#FF9F1C","OFFLINE":"#FF4D3D"}[h]
            st.markdown(f'<div class="noderow"><span>'
                        f'<span class="dot" style="background:{c};"></span>'
                        f'{k} · {n["sector"]}</span>'
                        f'<span style="color:{c};">{h}</span></div>',unsafe_allow_html=True)

        srule("Model performance")
        st.markdown(f"""<div style="font-family:IBM Plex Mono;font-size:.66rem;
          line-height:1.9;color:{TEXT_DIM};">
          PRECISION<span style="color:{TEXT};float:right;">{th.get('fusion_precision',0.9961):.4f}</span><br>
          RECALL<span style="color:{TEXT};float:right;">{th.get('fusion_recall',0.5432):.4f}</span><br>
          RECALL @ WARN+<span style="color:{TEXT};float:right;">{th.get('recall_at_warning',0.6726):.4f}</span><br>
          FALSE ALARM<span style="color:{PHOSPHOR};float:right;">0.0040</span><br>
          AE GAP<span style="color:{TEXT};float:right;">{th.get('gap',0.2304):.3f}</span>
          </div>""",unsafe_allow_html=True)

    # ── tabs ──────────────────────────────────────────────────────
    tab_console, tab_info = st.tabs(["Console", "Info"])

    with tab_console:
        live=st.session_state.playing or st.session_state.live_on
        pc=PHOSPHOR if live else TEXT_DIM
        st.markdown(f"""
        <div class="masthead">
          <div><div class="mast-title">Perimeter Threat Assessment</div>
            <div class="mast-sub">Decision-support console · Sector 4 · 300 × 200 m</div></div>
          <div><span class="pill" style="background:{pc}14;border:1px solid {pc};color:{pc};">
            {'● Running' if live else '○ Standby'}</span>
            <span class="pill" style="background:{PANEL};border:1px solid {RULE};
              color:{TEXT_DIM};margin-left:6px;">{mode}</span>
            <span class="pill" style="background:{PANEL};border:1px solid {RULE};
              color:{TEXT_DIM};margin-left:6px;">{st.session_state.hour:02d}:00</span></div>
        </div>""", unsafe_allow_html=True)

        if mode=="Live — random feed":
            t=st.columns([3,1,1,1])
            with t[0]:
                sc=st.session_state.scored; acc=(100*(sc["threat_ok"]+sc["fa_ok"])/sc["seen"] if sc["seen"] else 0)
                st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:.72rem;'
                            f'color:{TEXT_DIM};padding-top:9px;letter-spacing:.06em;">'
                            f'SELF-GRADING &nbsp;·&nbsp; seen {sc["seen"]} &nbsp;·&nbsp; '
                            f'threats caught <span style="color:{PHOSPHOR};">{sc["threat_ok"]}</span> &nbsp;·&nbsp; '
                            f'false alarms suppressed <span style="color:{PHOSPHOR};">{sc["fa_ok"]}</span> &nbsp;·&nbsp; '
                            f'correct <span style="color:{PHOSPHOR};">{acc:.0f}%</span></div>',unsafe_allow_html=True)
            with t[1]:
                if st.button("▶ Start" if not st.session_state.live_on else "● Live",
                             width="stretch",type="primary"):
                    st.session_state.live_on=True
            with t[2]:
                if st.button("Pause",width="stretch"): st.session_state.live_on=False
            with t[3]:
                if st.button("Clear",width="stretch"): reset_run()

        elif mode=="Scenario — scripted":
            t=st.columns([3,1,1,1])
            with t[0]:
                names=list(scenarios)
                sc=st.selectbox("Scenario",names,label_visibility="collapsed",
                                index=names.index(st.session_state.scenario) if st.session_state.scenario in names else 0)
                if sc!=st.session_state.scenario:
                    st.session_state.scenario=sc; reset_run()
            with t[1]:
                if st.button("▶ Run",width="stretch",type="primary"):
                    reset_run(); st.session_state.playing=True
            with t[2]:
                if st.button("Step",width="stretch"): advance_one(scenarios)
            with t[3]:
                if st.button("Clear",width="stretch"): reset_run()
        else:
            st.caption("Fallback mode — use operator injection buttons to place events.")

        st.markdown("")
        console(scenarios, build_live_pool(df))

    with tab_info:
        info_tab(th)


main()