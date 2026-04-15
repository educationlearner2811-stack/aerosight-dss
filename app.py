"""
Flight Delay Decision Support System
=====================================
Streamlit dashboard — loads trained model from notebook artifacts.
Usage:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Flight Delay DSS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1a365d; text-align: center; letter-spacing: -0.5px;
        padding-top: 0.5rem;
    }
    .sub-title {
        font-size: 1rem; color: #718096;
        text-align: center; margin-bottom: 1.5rem;
    }
.badge-delayed {
        background: #FED7D7; color: #C53030;
        padding: 0.6rem 1.4rem; border-radius: 10px;
        font-weight: 700; font-size: 1.3rem; display: block;
        text-align: center;
    }
    .badge-ontime {
        background: #C6F6D5; color: #276749;
        padding: 0.6rem 1.4rem; border-radius: 10px;
        font-weight: 700; font-size: 1.3rem; display: block;
        text-align: center;
    }
    .rec-box {
        background: #EBF8FF; border-left: 4px solid #3182CE;
        border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0;
        font-size: 0.9rem; color: #2C5282;
    }
    .rec-box-warn {
        background: #FFFBEB; border-left: 4px solid #D69E2E;
        border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0;
        font-size: 0.9rem; color: #744210;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #2d3748;
        border-left: 4px solid #3182ce; padding-left: 0.7rem;
        margin: 1rem 0 0.8rem 0;
    }
    .dark-card {
        background: #1e2a3a; border: 1px solid #2d3748;
        border-radius: 10px; padding: 14px; text-align: center;
    }
    .dark-card .label {
        font-size: 10px; color: #718096; font-weight: 600;
        letter-spacing: 0.06em; margin-bottom: 5px;
    }
    .dark-card .value {
        font-size: 20px; font-weight: 700; color: #e2e8f0;
    }
    .dark-card .sub {
        font-size: 11px; font-weight: 600; margin-top: 4px;
    }
    div[data-testid="metric-container"] {
        background: #F7FAFC; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
ARTIFACTS_DIR = 'model_artifacts'
DAY_NAMES   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

AIRLINE_FULL = {
    'AA':'American Airlines','DL':'Delta Air Lines','UA':'United Airlines',
    'WN':'Southwest Airlines','B6':'JetBlue Airways','AS':'Alaska Airlines',
    'NK':'Spirit Airlines','F9':'Frontier Airlines','G4':'Allegiant Air',
    'HA':'Hawaiian Airlines','MQ':'Envoy Air','OO':'SkyWest Airlines',
    '9E':'Endeavor Air','YX':'Republic Airways','OH':'PSA Airlines',
    'YV':'Mesa Airlines','QX':'Horizon Air','CP':'Compass Airlines',
}

FEATURE_LABELS = {
    'DepHour':                  'Departure Hour',
    'DayOfWeek':                'Day of Week',
    'Month':                    'Month',
    'Distance':                 'Route Distance',
    'IsPeakHour':               'Peak Hour',
    'IsWeekend':                'Weekend Flight',
    'Airline_enc':              'Airline',
    'Origin_enc':               'Origin Airport',
    'Dest_enc':                 'Destination Airport',
    'AvgWeatherDelay_Route':    'Avg Weather Delay (Route)',
    'AvgNASDelay_Origin':       'Avg NAS Delay (Origin)',
    'AvgLateAircraft_Airline':  'Avg Late Aircraft (Airline)',
    'AvgCarrierDelay_Airline':  'Avg Carrier Delay (Airline)',
}

# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def load_model_artifacts():
    model_path  = Path(ARTIFACTS_DIR) / 'model.pkl'
    meta_path   = Path(ARTIFACTS_DIR) / 'metadata.json'
    scaler_path = Path(ARTIFACTS_DIR) / 'scaler.pkl'
    if not model_path.exists() or not meta_path.exists():
        return None, None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    with open(meta_path, encoding='utf-8') as f:
        metadata = json.load(f)
    return model, scaler, metadata

@st.cache_data
def load_stats():
    stats = {}
    for name in ['airline_stats','route_stats','hourly_stats']:
        path = Path(ARTIFACTS_DIR) / f'{name}.csv'
        if path.exists():
            stats[name] = pd.read_csv(path)
    return stats

@st.cache_data
def load_route_lookup():
    lookup = {}
    for name, key in [
        ('route_lookup.csv',         'specific'),
        ('route_lookup_generic.csv', 'generic'),
        ('flight_times.csv',         'times'),
    ]:
        path = Path(ARTIFACTS_DIR) / name
        if path.exists():
            lookup[key] = pd.read_csv(path)
    return lookup

@st.cache_data
def load_explainability():
    fi_path  = Path(ARTIFACTS_DIR) / 'feature_importance.json'
    med_path = Path(ARTIFACTS_DIR) / 'feature_medians.json'
    fi, med = {}, {}
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)
    if med_path.exists():
        with open(med_path) as f:
            med = json.load(f)
    return fi, med

# ============================================================
# ROUTE HELPERS
# ============================================================
def get_route_info(airline, origin, dest, lookup):
    if 'specific' in lookup:
        m = lookup['specific']
        row = m[(m['Airline']==airline)&(m['Origin']==origin)&(m['Dest']==dest)]
        if not row.empty:
            return int(row.iloc[0]['typical_distance']), int(row.iloc[0]['typical_hour'])
    if 'generic' in lookup:
        m = lookup['generic']
        row = m[(m['Origin']==origin)&(m['Dest']==dest)]
        if not row.empty:
            return int(row.iloc[0]['typical_distance']), int(row.iloc[0]['typical_hour'])
    return 900, 9

def get_available_times(airline, origin, dest, lookup):
    if 'times' not in lookup:
        return None
    m = lookup['times']
    rows = m[(m['Airline']==airline)&(m['Origin']==origin)&(m['Dest']==dest)]
    times = sorted(rows['DepHour'].tolist())
    if not times:
        rows2 = m[(m['Origin']==origin)&(m['Dest']==dest)]
        times = sorted(rows2['DepHour'].unique().tolist())
    return times if times else None

def flight_count_for_time(airline, origin, dest, hour, lookup):
    if 'times' not in lookup:
        return 0
    m = lookup['times']
    row = m[(m['Airline']==airline)&(m['Origin']==origin)&
            (m['Dest']==dest)&(m['DepHour']==hour)]
    return int(row['flight_count'].sum()) if not row.empty else 0

def format_time(h):
    if h == 0:   return "12:00 AM (Midnight)"
    elif h < 12: return f"{h}:00 AM"
    elif h == 12: return "12:00 PM (Noon)"
    else:        return f"{h-12}:00 PM"

def get_time_risk(h):
    if 7 <= h <= 9:    return "🌅 Morning Peak",              "#d69e2e"
    elif 16 <= h <= 20: return "🌆 Evening Peak — highest risk","#e53e3e"
    elif 5 <= h <= 6:   return "🌄 Early Morning — lowest risk","#38a169"
    else:               return "🕐 Standard Window",            "#3182ce"

def get_dist_label(d):
    if d < 500:    return "Short Haul",    "#38a169"
    elif d < 1500: return "Medium Haul",   "#3182ce"
    elif d < 2500: return "Long Haul",     "#d69e2e"
    else:          return "Cross Country", "#e53e3e"

# ============================================================
# PREDICTION HELPERS
# ============================================================
def get_options(col, metadata):
    return metadata.get('encoders',{}).get(col,{}).get('classes',[])

def encode_value(col, value, metadata):
    mapping = metadata.get('encoders',{}).get(col,{}).get('mapping',{})
    return mapping.get(str(value), 0)

def build_input_df(airline, origin, dest, dep_hour, dow, month, distance, metadata):
    all_features = metadata['all_features']
    cat_features = metadata.get('cat_features', [])
    raw = {
        'DepHour':    dep_hour,
        'DayOfWeek':  dow,
        'Month':      month,
        'Distance':   distance,
        'IsPeakHour': 1 if (7 <= dep_hour <= 9 or 16 <= dep_hour <= 20) else 0,
        'IsWeekend':  1 if dow >= 5 else 0,
    }
    cat_vals = {'Airline': airline, 'Origin': origin, 'Dest': dest}
    for col in cat_features:
        raw[col + '_enc'] = encode_value(col, cat_vals.get(col,'Unknown'), metadata)
    return pd.DataFrame([{feat: raw.get(feat, 0) for feat in all_features}])

def run_prediction(input_df, model, scaler, metadata):
    num_features = metadata.get('num_features', [])
    use_scaled   = metadata.get('model_name') == 'Logistic Regression'
    df_pred = input_df.copy()
    if use_scaled and scaler is not None and num_features:
        present = [c for c in num_features if c in df_pred.columns]
        if present:
            df_pred[present] = scaler.transform(df_pred[present])
    prob    = model.predict_proba(df_pred)[0, 1]
    return float(prob), bool(prob >= 0.5)

def risk_level(prob):
    if prob < 0.25:    return "LOW RISK",      "#C6F6D5","#276749","🟢"
    elif prob < 0.40:  return "MODERATE RISK", "#FEFCBF","#744210","🟡"
    else:              return "HIGH RISK",      "#FED7D7","#C53030","🔴"

def airline_name(code):
    return AIRLINE_FULL.get(code, code)

def recommendations(prob, dep_hour, dow, distance):
    recs = []
    if prob >= 0.5:
        recs.append(("warn","Notify passengers proactively via app, SMS, and gate display."))
        recs.append(("warn","Pre-position gate staff 45 minutes ahead of standard boarding."))
        recs.append(("warn","Alert ground crew to expedite turnaround and baggage handling."))
        recs.append(("warn","Review downstream connecting flights on this tail number."))
    if dep_hour >= 17:
        recs.append(("info","Evening departure — network congestion compounds delays. Build a 15-min buffer."))
    if dow in [3,4,6]:
        recs.append(("info","Peak travel day (Thu/Fri/Sun) — consider proactive crew pre-positioning."))
    if distance > 2000:
        recs.append(("info","Long-haul route — small departure delays amplify significantly over distance."))
    if prob < 0.30:
        recs.append(("info","Low risk flight. Standard monitoring procedures are sufficient."))
    return recs

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        domain={'x':[0,1],'y':[0,1]},
        title={'text':"Delay Probability",'font':{'size':13}},
        number={'suffix':"%",'font':{'size':32}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickfont':{'size':10}},
            'bar':{'color':"#E53E3E" if prob>=0.4 else "#38A169",'thickness':0.3},
            'bgcolor':"white",'borderwidth':1,'bordercolor':"#E2E8F0",
            'steps':[
                {'range':[0,25],  'color':'#C6F6D5'},
                {'range':[25,40], 'color':'#FEFCBF'},
                {'range':[40,100],'color':'#FED7D7'},
            ],
            'threshold':{'line':{'color':"#2D3748",'width':3},'thickness':0.8,'value':40}
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40,b=0,l=0,r=0),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ============================================================
# EXPLAINABILITY CHART
# ============================================================
def make_explain_chart(input_df, feature_importance, feature_medians, all_features):
    """
    Show what drove this prediction using feature importances
    weighted by how far each input is from the typical (median) value.
    """
    if not feature_importance:
        return None

    scores = {}
    for feat in all_features:
        fi  = feature_importance.get(feat, 0)
        val = float(input_df[feat].iloc[0]) if feat in input_df.columns else 0
        med = feature_medians.get(feat, 0)
        # How different is this value from typical?
        diff = abs(val - med)
        max_range = max(abs(med * 2), 1)
        deviation = min(diff / max_range, 1.0)
        scores[feat] = fi * (0.5 + 0.5 * deviation)

    total = sum(scores.values()) or 1
    pcts  = {f: round(v / total * 100, 1) for f, v in scores.items()}

    top = sorted(pcts.items(), key=lambda x: x[1], reverse=True)[:6]
    labels = [FEATURE_LABELS.get(f, f) for f, _ in top]
    values = [v for _, v in top]

    colors = ['#E53E3E' if v == max(values) else
              '#D69E2E' if v >= sorted(values)[-2] else '#3182CE'
              for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:.0f}%" for v in values[::-1]],
        textposition='outside',
        textfont=dict(size=12)
    ))
    fig.update_layout(
        title="What drove this prediction?",
        xaxis_title="Contribution to prediction (%)",
        xaxis_range=[0, max(values)*1.35],
        height=260,
        margin=dict(t=40, b=20, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# ============================================================
# DARK INFO CARD HTML
# ============================================================
def dark_card(label, value, sub, sub_color):
    return f"""
    <div style='background:#1e2a3a;border:1px solid #2d3748;
    border-radius:10px;padding:14px;text-align:center;'>
        <div style='font-size:10px;color:#718096;font-weight:600;
        letter-spacing:0.06em;margin-bottom:5px;'>{label}</div>
        <div style='font-size:20px;font-weight:700;color:#e2e8f0;'>{value}</div>
        <div style='font-size:11px;font-weight:600;color:{sub_color};margin-top:4px;'>{sub}</div>
    </div>"""

# ============================================================
# LOAD EVERYTHING
# ============================================================
model, scaler, metadata    = load_model_artifacts()
stats                      = load_stats()
route_lookup               = load_route_lookup()
feature_importance, feat_medians = load_explainability()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ✈️ Flight Delay DSS")
    st.markdown("*Decision Support System*")
    st.divider()

    if model is not None:
        st.success("Model loaded")
        best_name = metadata.get('model_name','Unknown')
        perf = metadata.get('performance',{}).get(best_name,{})
        st.markdown(f"**Model:** `{best_name}`")
        st.markdown(f"**Dataset:** 7M US domestic flights (2024)")
        st.markdown(f"**Delay threshold:** ≥ 15 min (FAA standard)")
        st.divider()
        st.markdown("**Model Performance**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",  f"{perf.get('accuracy',0)*100:.1f}%")
        c2.metric("AUC",       f"{perf.get('roc_auc',0):.3f}")
        c1.metric("Precision", f"{perf.get('precision',0)*100:.1f}%")
        c2.metric("F1 Score",  f"{perf.get('f1',0):.3f}")
        st.divider()
        with st.expander("What do these metrics mean?"):
            st.markdown("""
**Accuracy** — % of all predictions correct. High because most flights are on time (~80%).

**AUC** — how well model separates delayed vs on-time. 0.5 = random, 1.0 = perfect.

**Precision** — when model predicts delayed, how often it's right.

**F1** — balance of precision and recall. Improved with `class_weight='balanced'` to handle the 80/20 imbalance.
            """)
    else:
        st.error("Model not found")
        st.info("Run the notebook first to train the model.")

    st.divider()
    st.caption("Random Forest · scikit-learn · Streamlit · 2024 BTS Data")

if model is None:
    st.error("**Model artifacts not found.** Run the Jupyter notebook first.")
    st.code("python generate_notebook.py\njupyter notebook flight_delay_pipeline.ipynb\nstreamlit run app.py", language="bash")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">✈️ Flight Delay Decision Support System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prescriptive Analytics · 7M Flights · 2024 US Domestic Operations</p>', unsafe_allow_html=True)

airlines = get_options('Airline', metadata) or ['AA','DL','UA','WN','B6','AS','NK','F9']
origins  = get_options('Origin',  metadata) or ['ATL','LAX','ORD','DFW','DEN','JFK','SFO','MIA']
dests    = get_options('Dest',    metadata) or origins

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict Delay",
    "🔄  Simulate & Recommend",
    "📊  Network Dashboard",
    "📈  Model Performance"
])

# ================================================================
# TAB 1 — PREDICT (new design: no sliders, auto-lookup, dropdown)
# ================================================================
with tab1:
    st.markdown('<p class="section-header">Enter Flight Details</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        airline = st.selectbox("Airline", airlines,
                               format_func=lambda x: f"{x} — {airline_name(x)}",
                               key='p_al')
    with c2:
        origin   = st.selectbox("Origin Airport (IATA)", origins, key='p_or')
        day_name = st.selectbox("Day of Week", DAY_NAMES, index=1, key='p_dw')
        dow_idx  = DAY_NAMES.index(day_name)
    with c3:
        dest_opts  = [d for d in dests if d != origin] or dests
        dest       = st.selectbox("Destination Airport (IATA)", dest_opts, key='p_de')
        month_name = st.selectbox("Month", MONTH_NAMES, index=5, key='p_mo')
        month_idx  = MONTH_NAMES.index(month_name) + 1

    # ---- Auto-lookup distance & departure times ----
    distance, default_hour = get_route_info(airline, origin, dest, route_lookup)
    available_times        = get_available_times(airline, origin, dest, route_lookup)

    st.divider()

    dep_col, time_card_col = st.columns([1, 2])

    with dep_col:
        if available_times:
            time_options = {format_time(h): h for h in available_times}
            default_label = format_time(default_hour)
            default_idx   = list(time_options.keys()).index(default_label) \
                            if default_label in time_options else 0
            selected_label = st.selectbox(
                "Scheduled Departure Time",
                options=list(time_options.keys()),
                index=default_idx,
                key='p_dep_time'
            )
            dep_hour = time_options[selected_label]
            count    = flight_count_for_time(airline, origin, dest, dep_hour, route_lookup)
            if count > 0:
                st.caption(f"📊 Based on {count:,} historical flights on this route")
        else:
            dep_hour = default_hour
            st.info(f"Typical departure: {format_time(dep_hour)}")

    with time_card_col:
        risk_text, risk_color = get_time_risk(dep_hour)
        if 7 <= dep_hour <= 9 or 16 <= dep_hour <= 20:
            risk_msg = "Consider earlier departure to reduce delay risk"
        elif 5 <= dep_hour <= 6:
            risk_msg = "Excellent choice — fewest delays at this hour"
        else:
            risk_msg = "Moderate delay risk at this hour"

        st.markdown(f"""
        <div style='background:#1e2a3a;border:1px solid #2d3748;
        border-radius:10px;padding:16px;margin-top:4px;'>
            <div style='font-size:24px;font-weight:700;color:#e2e8f0;margin-bottom:4px;'>
                {format_time(dep_hour)}
            </div>
            <div style='font-size:13px;font-weight:600;color:{risk_color};margin-bottom:6px;'>
                {risk_text}
            </div>
            <div style='font-size:12px;color:#718096;'>{risk_msg}</div>
        </div>
        """, unsafe_allow_html=True)

    # Info cards row
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    card1, card2, card3 = st.columns(3)

    dist_label, dist_color = get_dist_label(distance)

    with card1:
        st.markdown(dark_card("ROUTE DISTANCE", f"{distance:,} mi", dist_label, dist_color),
                    unsafe_allow_html=True)
    with card2:
        st.markdown(dark_card("ROUTE", f"{origin} → {dest}", f"{airline} operated", "#718096"),
                    unsafe_allow_html=True)
    with card3:
        st.markdown(dark_card("DEPARTING", day_name, f"{month_name} 2024", "#718096"),
                    unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:8px;font-size:11px;color:#4a5568;'>"
        "ℹ️ Departure times and distance are sourced from historical flight operations data for this route."
        "</div>", unsafe_allow_html=True
    )
    st.divider()

    predict_btn = st.button("🔮 Predict Delay Risk", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Running prediction..."):
            inp = build_input_df(airline, origin, dest, dep_hour,
                                 dow_idx, month_idx, distance, metadata)
            prob, delayed = run_prediction(inp, model, scaler, metadata)

        st.divider()

                # --- Result row ---
        r1, r2, r3 = st.columns([1.1, 1.2, 1.7])

        with r1:
            st.markdown("#### Prediction Result")
            if delayed:
                st.markdown('<div style="text-align:center"><span class="badge-delayed">⚠️ LIKELY DELAYED</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center"><span class="badge-ontime">✅ LIKELY ON TIME</span></div>',
                            unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"**Airline:** {airline} — {airline_name(airline)}")
            st.markdown(f"**Route:** {origin} → {dest}")
            st.markdown(f"**Departs:** {format_time(dep_hour)} on {day_name}")
            st.markdown(f"**Month:** {month_name}  |  **Distance:** {distance:,} mi")

        with r2:
            st.markdown("#### Delay Probability")
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
            try:
                if feature_importance and feat_medians:
                    all_feats = metadata.get('all_features', [])
                    scores = {}
                    for feat in all_feats:
                        fi = feature_importance.get(feat, 0)
                        val = float(inp[feat].iloc[0]) if feat in inp.columns else 0
                        med = feat_medians.get(feat, 0)
                        diff = abs(val - med)
                        max_range = max(abs(med * 2), 1)
                        deviation = min(diff / max_range, 1.0)
                        scores[feat] = fi * (0.5 + 0.5 * deviation)
                    total = sum(scores.values()) or 1
                    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    PLAIN_ENGLISH = {
                        'DepHour': lambda v, m: f"{'Late evening' if v >= 20 else 'Evening' if v >= 16 else 'Early morning' if v <= 6 else 'Morning' if v <= 9 else 'Afternoon'} departure ({int(v):02d}:00) — {'peak congestion window' if (7<=v<=9 or 16<=v<=20) else 'off-peak window'}",
                        'Month': lambda v, m: f"{'Summer' if v in [6,7,8] else 'Winter' if v in [12,1,2] else 'Spring/Fall'} season (Month {int(v)}) — {'historically high delay period' if v in [6,7,8,12] else 'moderate delay period'}",
                        'DayOfWeek': lambda v, m: f"{'Peak travel day (Friday)' if v == 4 else 'Peak travel day (Sunday)' if v == 6 else 'Mid-week departure' if v in [1,2] else 'Standard travel day'}",
                        'Distance': lambda v, m: f"{'Long haul' if v > 1500 else 'Medium haul' if v > 500 else 'Short haul'} route ({int(v):,} miles)",
                        'AvgNASDelay_Origin': lambda v, m: f"Origin airport NAS congestion avg {v:.1f} min — {'above average air traffic pressure' if v > m else 'below average air traffic pressure'}",
                        'AvgWeatherDelay_Route': lambda v, m: f"Route weather delay history avg {v:.1f} min — {'elevated weather risk on this route' if v > m else 'typical weather conditions for this route'}",
                        'AvgLateAircraft_Airline': lambda v, m: f"Airline late aircraft propagation avg {v:.1f} min — {'above average cascading delay risk' if v > m else 'below average cascading delay risk'}",
                        'AvgCarrierDelay_Airline': lambda v, m: f"Carrier operational delay avg {v:.1f} min — {'above average' if v > m else 'below average'} operational reliability",
                        'IsPeakHour': lambda v, m: f"{'Peak hour departure — historically high congestion period' if v == 1 else 'Off-peak departure — lower congestion risk'}",
                        'IsWeekend': lambda v, m: f"{'Weekend departure — elevated leisure travel demand' if v == 1 else 'Weekday departure — standard demand levels'}",
                        'Airline_enc': lambda v, m: f"Airline carrier operational history influences this prediction",
                        'Origin_enc': lambda v, m: f"Origin airport delay history influences this prediction",
                        'Dest_enc': lambda v, m: f"Destination airport delay history influences this prediction",
                    }
                    lines_html = ""
                    for feat, score in top3:
                        val = float(inp[feat].iloc[0]) if feat in inp.columns else 0
                        med = feat_medians.get(feat, 1)
                        if feat in PLAIN_ENGLISH:
                            try:
                                explanation = PLAIN_ENGLISH[feat](val, med)
                                lines_html += f"<div style='font-size:0.82rem;color:#CBD5E0;padding:0.25rem 0;border-bottom:1px solid rgba(255,255,255,0.1)'>• {explanation}</div>"
                            except Exception:
                                pass
                    if lines_html:
                        st.markdown(
                            f"<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);"
                            f"border:1px solid #3182CE;border-radius:12px;"
                            f"padding:1rem 1.25rem;"
                            f"box-shadow:0 4px 15px rgba(49,130,206,0.2)'>"
                            f"<div style='font-weight:700;font-size:0.9rem;color:#90CDF4;"
                            f"margin-bottom:0.6rem;letter-spacing:0.03em'>"
                            f"🔍 Top factors driving this prediction:</div>"
                            f"{lines_html}</div>",
                            unsafe_allow_html=True
                        )
            except Exception:
                pass

        with r3:
            label, bg, fg, icon = risk_level(prob)
            st.markdown("#### Operational Risk & Recommendations")
            st.markdown(
                f"<div style='background:{bg};color:{fg};padding:0.6rem 1rem;"
                f"border-radius:8px;font-weight:600;font-size:1rem;margin-bottom:0.8rem'>"
                f"{icon} {label} — {round(prob*100,1)}% chance of delay</div>",
                unsafe_allow_html=True
            )
            if 'airline_stats' in stats:
                net_avg = stats['airline_stats']['delay_rate'].mean() * 100
                delta   = round(prob*100 - net_avg, 1)
                delta_str = f"+{delta:.1f}pp above" if delta > 0 else f"{abs(delta):.1f}pp below"
                delta_col = "#C53030" if delta > 0 else "#276749"
                st.markdown(
                    f"<div style='font-size:12px;color:#718096;margin-bottom:0.6rem;'>"
                    f"Network average delay rate: <b>{net_avg:.1f}%</b> — "
                    f"<span style='color:{delta_col};font-weight:600;'>"
                    f"This flight is {delta_str} network average</span></div>",
                    unsafe_allow_html=True
                )
            recs = recommendations(prob, dep_hour, dow_idx, distance)
            for rtype, text in recs:
                css = "rec-box-warn" if rtype == "warn" else "rec-box"
                st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

        # --- Probability bar and explainability chart ---
        st.divider()
        ba, bb = st.columns(2)

        with ba:
            st.markdown("#### Probability Breakdown")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['On Time','Delayed'],
                y=[round((1-prob)*100,1), round(prob*100,1)],
                marker_color=['#38A169','#E53E3E'],
                text=[f"{round((1-prob)*100,1)}%", f"{round(prob*100,1)}%"],
                textposition='outside', textfont=dict(size=14)
            ))
            fig_bar.update_layout(
                height=240, yaxis_title='Probability (%)', yaxis_range=[0,115],
                showlegend=False, margin=dict(t=10,b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with bb:
            st.markdown("#### What drove this prediction?")
            explain_fig = make_explain_chart(inp, feature_importance,
                                             feat_medians,
                                             metadata.get('all_features',[]))
            if explain_fig:
                st.plotly_chart(explain_fig, use_container_width=True)
                st.caption("Shows which features the model weighted most for this specific flight. Red = highest influence, amber = second, blue = lower influence.")

        st.session_state['last'] = dict(
            airline=airline, origin=origin, dest=dest,
            dep_hour=dep_hour, dow_idx=dow_idx,
            month_idx=month_idx, distance=distance, prob=prob
        )

# ================================================================
# TAB 2 — SIMULATE (no sliders, real departure time dropdowns)
# ================================================================
with tab2:
    st.markdown('<p class="section-header">What-If Scenario Simulator</p>', unsafe_allow_html=True)
    st.markdown("Compare alternative operational decisions side-by-side to find the lowest-risk option.")

    last = st.session_state.get('last', {})
    def _idx(lst, val, default=0):
        try: return lst.index(val)
        except: return default

    cs1, cs2 = st.columns(2)

    with cs1:
        st.markdown("##### Base Flight")
        s_al  = st.selectbox("Airline", airlines, key='s_al',
                             index=_idx(airlines, last.get('airline', airlines[0])),
                             format_func=lambda x: f"{x} — {airline_name(x)}")
        s_or  = st.selectbox("Origin", origins, key='s_or',
                             index=_idx(origins, last.get('origin', origins[0])))
        s_de_opts = [d for d in dests if d != s_or] or dests
        s_de  = st.selectbox("Destination", s_de_opts, key='s_de',
                             index=_idx(s_de_opts, last.get('dest', s_de_opts[0])))
        s_dow = st.selectbox("Day of Week", range(7), format_func=lambda x: DAY_NAMES[x],
                             index=last.get('dow_idx',1), key='s_dow')
        s_mo  = st.selectbox("Month", range(1,13), format_func=lambda x: MONTH_NAMES[x-1],
                             index=last.get('month_idx',6)-1, key='s_mo')

    with cs2:
        st.markdown("##### Adjust Parameters")

        # Base departure time dropdown from real data
        s_dist, s_default_hr = get_route_info(s_al, s_or, s_de, route_lookup)
        base_times = get_available_times(s_al, s_or, s_de, route_lookup)

        if base_times:
            base_time_opts = {format_time(h): h for h in base_times}
            base_def_lbl   = format_time(s_default_hr)
            base_def_idx   = list(base_time_opts.keys()).index(base_def_lbl) \
                             if base_def_lbl in base_time_opts else 0
            sel_base = st.selectbox("Base Departure Time",
                                    options=list(base_time_opts.keys()),
                                    index=base_def_idx, key='s_base_time')
            s_hour = base_time_opts[sel_base]
            b_risk, b_color = get_time_risk(s_hour)
            st.markdown(f"<div style='font-size:11px;font-weight:600;color:{b_color};"
                        f"margin-bottom:8px;margin-top:-8px;'>{b_risk}</div>",
                        unsafe_allow_html=True)
        else:
            s_hour = s_default_hr
            st.info(f"Base time: {format_time(s_hour)}")

        # Alternative airline
        alt_al = st.selectbox("Try Different Airline", airlines, key='alt_al',
                              index=min(1, len(airlines)-1),
                              format_func=lambda x: f"{x} — {airline_name(x)}")

        # Alt departure time dropdown based on alt airline + same route
        _, alt_default_hr = get_route_info(alt_al, s_or, s_de, route_lookup)
        alt_times = get_available_times(alt_al, s_or, s_de, route_lookup)

        if alt_times:
            alt_time_opts = {format_time(h): h for h in alt_times}
            # Default to first time different from base
            alt_diff = [h for h in alt_times if h != s_hour]
            alt_def_h = alt_diff[0] if alt_diff else alt_times[0]
            alt_def_lbl = format_time(alt_def_h)
            alt_def_idx = list(alt_time_opts.keys()).index(alt_def_lbl) \
                          if alt_def_lbl in alt_time_opts else 0
            sel_alt = st.selectbox("Try Different Departure Time",
                                   options=list(alt_time_opts.keys()),
                                   index=alt_def_idx, key='s_alt_time')
            alt_hour = alt_time_opts[sel_alt]
            a_risk, a_color = get_time_risk(alt_hour)
            st.markdown(f"<div style='font-size:11px;font-weight:600;color:{a_color};"
                        f"margin-bottom:8px;margin-top:-8px;'>{a_risk}</div>",
                        unsafe_allow_html=True)
        else:
            alt_hour = alt_default_hr
            st.info(f"Alt time: {format_time(alt_hour)}")

        # Alternative day
        alt_dow = st.selectbox("Try Different Day", range(7),
                               format_func=lambda x: DAY_NAMES[x],
                               index=(last.get('dow_idx',1)+1)%7, key='alt_dow')

        st.markdown("")
        run_sim = st.button("🔄 Run All Scenarios", type="primary", use_container_width=True)

    if run_sim:
        hr_label = (f"Depart {format_time(alt_hour)}"
                    if alt_hour != s_hour else "Same Time")

        scenarios = {
            'Base scenario':              (s_al,  s_or, s_de, s_hour,    s_dow,   s_mo, s_dist),
            hr_label:                     (s_al,  s_or, s_de, alt_hour,  s_dow,   s_mo, s_dist),
            f'Switch to {alt_al}':        (alt_al,s_or, s_de, s_hour,    s_dow,   s_mo, s_dist),
            f'Fly on {DAY_NAMES[alt_dow]}':(s_al, s_or, s_de, s_hour,    alt_dow, s_mo, s_dist),
        }

        sim_results = {}
        for sc_name, params in scenarios.items():
            df_in = build_input_df(*params, metadata)
            p, d  = run_prediction(df_in, model, scaler, metadata)
            sim_results[sc_name] = {'prob': p, 'delayed': d}

        st.divider()
        st.markdown("#### Results")
        cols = st.columns(len(sim_results))
        base_prob = sim_results['Base scenario']['prob']

        for i, (sc, res) in enumerate(sim_results.items()):
            with cols[i]:
                delta = res['prob'] - base_prob if sc != 'Base scenario' else None
                st.metric(
                    label=sc,
                    value=f"{res['prob']*100:.1f}%",
                    delta=f"{delta*100:+.1f}pp" if delta is not None else None,
                    delta_color="inverse" if (delta is not None and delta < 0) else "off"
                )
                st.caption("⚠️ Delayed" if res['delayed'] else "✅ On Time")

        probs  = [r['prob']*100 for r in sim_results.values()]
        colors = ['#E53E3E' if p>=50 else '#D69E2E' if p>=30 else '#38A169' for p in probs]
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=list(sim_results.keys()), y=probs,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs], textposition='outside'
        ))
        fig_sim.add_hline(y=50, line_dash="dash", line_color="#E53E3E",
                          annotation_text="Decision threshold (50%)",
                          annotation_position="top right")
        fig_sim.update_layout(
            title="Delay Probability by Scenario",
            yaxis_title="Delay Probability (%)", yaxis_range=[0,115],
            height=380, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        best_sc = min(sim_results, key=lambda x: sim_results[x]['prob'])
        if best_sc != 'Base scenario':
            reduction = (base_prob - sim_results[best_sc]['prob']) * 100
            st.success(
                f"✅ **Recommended action:** '{best_sc}' reduces delay probability by "
                f"**{reduction:.1f} percentage points**."
            )
        else:
            st.info("The base scenario already has the lowest delay risk among the alternatives tested.")

# ================================================================
# TAB 3 — NETWORK DASHBOARD
# ================================================================
with tab3:
    st.markdown('<p class="section-header">Network Operations Dashboard</p>', unsafe_allow_html=True)

    if not stats:
        st.warning("No stats found. Run the notebook to generate statistics.")
        st.stop()

    if 'airline_stats' in stats:
        df_a = stats['airline_stats']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Airlines Tracked",    len(df_a))
        k2.metric("Network Delay Rate",  f"{df_a['delay_rate'].mean()*100:.1f}%")
        k3.metric("Best On-Time Carrier",df_a.loc[df_a['delay_rate'].idxmin(),'Airline'])
        k4.metric("Highest Delay Rate",  df_a.loc[df_a['delay_rate'].idxmax(),'Airline'])
        st.divider()

    if 'airline_stats' in stats:
        df_a2 = stats['airline_stats'].copy()
        df_a2['Airline Name'] = df_a2['Airline'].apply(lambda x: f"{x} ({airline_name(x)})")
        df_a2 = df_a2.sort_values('delay_rate', ascending=True)

        col_l, col_r = st.columns(2)
        with col_l:
            fig_al = px.bar(
                df_a2, x='delay_rate', y='Airline Name', orientation='h',
                color='delay_rate', color_continuous_scale='RdYlGn_r',
                labels={'delay_rate':'Delay Rate'},
                title='Delay Rate by Airline'
            )
            fig_al.update_layout(height=420, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            fig_al.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig_al, use_container_width=True)

        with col_r:
            fig_sc = px.scatter(
                df_a2, x='total_flights', y='avg_delay',
                size='total_flights', color='delay_rate',
                hover_data=['Airline'], color_continuous_scale='RdYlGn_r',
                labels={'total_flights':'Total Flights','avg_delay':'Avg Delay (min)'},
                title='Flight Volume vs Average Delay'
            )
            fig_sc.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sc, use_container_width=True)

    if 'hourly_stats' in stats:
        df_h = stats['hourly_stats']
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=df_h['DepHour'], y=df_h['delay_rate']*100,
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#E53E3E', width=2.5),
            marker=dict(size=7), fillcolor='rgba(229,62,62,0.1)'
        ))
        fig_h.add_vrect(x0=7, x1=9, fillcolor='orange', opacity=0.1,
                        annotation_text="AM peak", annotation_position="top left")
        fig_h.add_vrect(x0=16,x1=20,fillcolor='orange', opacity=0.1,
                        annotation_text="PM peak", annotation_position="top left")
        fig_h.update_layout(
            title='Delay Rate by Hour of Day',
            xaxis_title='Departure Hour', yaxis_title='Delay Rate (%)',
            height=320, xaxis=dict(tickvals=list(range(0,24,2))),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_h, use_container_width=True)

    if 'route_stats' in stats:
        df_r = stats['route_stats'].copy()
        df_r['Route'] = df_r['Origin'] + ' → ' + df_r['Dest']
        top_r = df_r.sort_values('delay_rate', ascending=False).head(15)
        fig_r = px.bar(
            top_r, x='Route', y='delay_rate',
            color='avg_delay', color_continuous_scale='YlOrRd',
            labels={'delay_rate':'Delay Rate','avg_delay':'Avg Delay (min)'},
            title='Top 15 Highest-Risk Routes'
        )
        fig_r.update_layout(height=380, xaxis_tickangle=-40, paper_bgcolor='rgba(0,0,0,0)')
        fig_r.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_r, use_container_width=True)

# ================================================================
# TAB 4 — MODEL PERFORMANCE
# ================================================================
with tab4:
    st.markdown('<p class="section-header">Model Evaluation & Justification</p>', unsafe_allow_html=True)
    st.markdown("Explains model choices, performance metrics, and why the results are valid.")

    perf_all  = metadata.get('performance', {})
    best_name = metadata.get('model_name', 'Random Forest')

    if perf_all:
        st.markdown("#### All Models Compared")
        rows = []
        for name, m in perf_all.items():
            rows.append({
                'Model':    name,
                'Accuracy': f"{m.get('accuracy',0)*100:.2f}%",
                'Precision':f"{m.get('precision',0)*100:.2f}%",
                'Recall':   f"{m.get('recall',0)*100:.2f}%",
                'F1 Score': f"{m.get('f1',0):.4f}",
                'ROC AUC':  f"{m.get('roc_auc',0):.4f}",
                'Selected': "✅ Best" if name == best_name else ""
            })
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    st.divider()
    st.markdown("#### Understanding the Metrics")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
**Why is Accuracy high but F1 low?**

This is a **class imbalance problem**. In 2024 US domestic flights:
- ~80% of flights arrived on time
- ~20% were delayed (≥15 minutes)

A model predicting "on time" for every flight achieves 80% accuracy while being useless. F1 penalises this — it requires the model to correctly identify both classes.

**How was this fixed?**
`class_weight='balanced'` was applied to all classifiers, making each delayed flight count proportionally more during training.
        """)

    with col_m2:
        st.markdown("""
**Metric interpretation:**

| Metric | Score | Meaning |
|---|---|---|
| Accuracy | ~66% | Reflects real data distribution |
| AUC | 0.699 | Good separation (0.5 = random) |
| Precision | ~33% | Acceptable given class imbalance |
| F1 | 0.431 | Improved with class balancing |

**Why Random Forest was selected:**
- Best AUC among all 3 models tested
- Built-in feature importance for explainability
- Handles non-linear scheduling patterns
- Incorporates historical weather and NAS delay patterns
- No feature scaling required
- Robust to outliers in delay durations
        """)

    st.divider()
    st.markdown("#### Feature Importance (Global)")
    if feature_importance:
        fi_df = pd.Series(feature_importance).sort_values(ascending=True)
        fi_df.index = [FEATURE_LABELS.get(i, i) for i in fi_df.index]
        fig_fi = go.Figure(go.Bar(
            x=fi_df.values, y=fi_df.index, orientation='h',
            marker_color='#3182CE',
            text=[f"{v*100:.1f}%" for v in fi_df.values],
            textposition='outside'
        ))
        fig_fi.update_layout(
            title="Global Feature Importance — Random Forest",
            xaxis_title="Importance Score", height=320,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=60, t=40, b=20)
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Shows which features the model relies on most across all 5.6M training examples.")

    st.divider()
    st.markdown("#### Dataset & Features")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flights",  "7,079,081")
    c2.metric("Features Used",  str(len(metadata.get('all_features',[]))))
    c3.metric("Training Split", "80%")
    c4.metric("Test Split",     "20%")

    features  = metadata.get('all_features', [])
    num_feats = metadata.get('num_features', [])
    feat_df = pd.DataFrame({
        'Feature':     features,
        'Type':        ['Numeric' if f in num_feats else 'Encoded Categorical' for f in features],
        'Description': [{
            'DepHour':    'Hour of scheduled departure (0–23)',
            'DayOfWeek':  'Day of week (1=Mon, 7=Sun) — BTS standard',
            'Month':      'Month of year (1–12)',
            'Distance':   'Route distance in miles',
            'IsPeakHour': '1 if departure in rush hour (7–9am or 4–8pm)',
            'IsWeekend':  '1 if Saturday or Sunday',
             'Airline_enc':             'Airline carrier code (label encoded)',
            'Origin_enc':              'Origin airport IATA code (label encoded)',
            'Dest_enc':                'Destination airport IATA code (label encoded)',
            'AvgWeatherDelay_Route':   'Historical avg weather delay for this route and month',
            'AvgNASDelay_Origin':      'Historical avg NAS delay at origin airport at this hour',
            'AvgLateAircraft_Airline': 'Historical avg late aircraft delay for this airline on this day',
            'AvgCarrierDelay_Airline': 'Historical avg carrier delay for this airline overall',
        }.get(f, f) for f in features]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
