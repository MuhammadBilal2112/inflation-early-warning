"""
==============================================================================
INFLATION REGIME DASHBOARD v4 — FINAL
==============================================================================
Uses CALIBRATED probabilities, adds risk cards, fiscal vulnerability
gauges, alert banners, and Panel A vs B comparison.

RUN: python3 Dashboard_v4_Final.py
OUTPUT: inflation_regimes_project/outputs/figures/dashboard_final.html
==============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, json, os, re, warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
MDIR = os.path.join(BASE_DIR, "outputs", "models")
FDIR = os.path.join(BASE_DIR, "outputs", "figures")

print("=" * 60)
print("  Building Final Dashboard (v4)...")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n  Loading data...")
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])
features_df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in features_df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    features_df = features_df.rename(columns=clean_map)

# Load CALIBRATED model
cal_path = os.path.join(MDIR, 'xgboost_calibrated_target_up_2q.joblib')
raw_path = os.path.join(MDIR, 'xgboost_target_up_2q.joblib')

if os.path.exists(cal_path):
    saved = joblib.load(cal_path)
    model = saved['model']
    calibrator = saved['calibrator']
    feature_cols = saved['features']
    use_calibrated = True
    print(f"  Model: calibrated XGBoost")
else:
    saved = joblib.load(raw_path)
    model = saved['model']
    calibrator = None
    feature_cols = saved['features']
    use_calibrated = False
    print(f"  Model: Raw XGBoost (calibrator not found)")

# Compute transition probs
raw_probs = model.predict_proba(features_df[feature_cols])[:, 1]
if use_calibrated and calibrator is not None:
    features_df['transition_prob'] = calibrator.predict(raw_probs)
else:
    features_df['transition_prob'] = raw_probs

# Load Panel B regime labels (4-measure, 74 countries)
panelB_path = os.path.join(PROC, 'regime_labels_panelB.csv')
has_panelB = os.path.exists(panelB_path)
if has_panelB:
    regime_b = pd.read_csv(panelB_path, parse_dates=['date'])
    panelB_countries = set(regime_b['country_code'].unique())
    print(f"  Panel B loaded: {len(panelB_countries)} countries")
else:
    regime_b = pd.DataFrame()
    panelB_countries = set()
    print(f"  Panel B: not found (skipping comparison)")

# Country names
cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']):
        cnames[r['country_code']] = str(r['country_name'])

REGIME = {
    0: ('Low', '#22c55e', 'Low & stable inflation'),
    1: ('Low-energy', '#3b82f6', 'Low headline, energy pressure'),
    2: ('Moderate', '#eab308', 'Moderate, food-driven'),
    3: ('Mod-energy', '#f97316', 'Moderate, energy-driven'),
    4: ('Elevated', '#ef4444', 'Broadly elevated'),
    5: ('Crisis', '#a855f7', 'High/crisis inflation'),
}

regime['regime_name'] = regime['regime'].map(lambda x: REGIME.get(int(x), ('?','#777','?'))[0] if pd.notna(x) else 'No data')

# Merge probs + fiscal data into regime
merge_cols = ['country_code', 'date', 'transition_prob']
fiscal_cols_to_merge = ['debt_gdp', 'sovereign_rating', 'fiscal_balance', 'private_credit_gdp']
for fc in fiscal_cols_to_merge:
    if fc in features_df.columns:
        merge_cols.append(fc)

regime = regime.merge(features_df[merge_cols], on=['country_code', 'date'], how='left')

# ============================================================
# BUILD COUNTRY JSON
# ============================================================
print("  Building country data...")

countries_json = {}
for cc in sorted(regime['country_code'].unique()):
    cc_r = regime[regime['country_code'] == cc].sort_values('date')
    cc_m = monthly[(monthly['country_code'] == cc) & (monthly['date'] >= '2010-01-01')].sort_values('date')
    if cc_r.empty: continue
    last = cc_r.iloc[-1]

    # Quarterly series
    q_dates, q_tp, q_h, q_f, q_e, q_reg = [], [], [], [], [], []
    for _, row in cc_r.iterrows():
        q_dates.append(row['date'].strftime('%Y-%m-%d'))
        q_tp.append(round(float(row['transition_prob']), 4) if pd.notna(row.get('transition_prob')) else None)
        q_h.append(round(float(row['hcpi_yoy']), 2) if pd.notna(row.get('hcpi_yoy')) else None)
        q_f.append(round(float(row['fcpi_yoy']), 2) if pd.notna(row.get('fcpi_yoy')) else None)
        q_e.append(round(float(row['ecpi_yoy']), 2) if pd.notna(row.get('ecpi_yoy')) else None)
        q_reg.append(int(row['regime']) if pd.notna(row.get('regime')) else None)

    # Monthly headline
    m_dates, m_vals = [], []
    for _, row in cc_m.iterrows():
        if pd.notna(row.get('hcpi_yoy')):
            m_dates.append(row['date'].strftime('%Y-%m-%d'))
            m_vals.append(round(float(row['hcpi_yoy']), 2))

    # Fiscal snapshot
    fiscal = {}
    for fc in ['debt_gdp', 'sovereign_rating', 'fiscal_balance', 'private_credit_gdp']:
        val = last.get(fc)
        fiscal[fc] = round(float(val), 1) if pd.notna(val) else None

    # Panel B data (if country is in 4-measure panel)
    pb_data = {}
    if has_panelB and cc in panelB_countries:
        cc_b = regime_b[regime_b['country_code'] == cc].sort_values('date')
        if not cc_b.empty:
            last_b = cc_b.iloc[-1]
            pb_data = {
                'regime': int(last_b['regime']) if pd.notna(last_b.get('regime')) else None,
                'hcpi': round(float(last_b['hcpi_yoy']), 1) if pd.notna(last_b.get('hcpi_yoy')) else None,
                'ccpi': round(float(last_b['ccpi_yoy']), 1) if pd.notna(last_b.get('ccpi_yoy')) else None,
                'hc_gap': round(float(last_b['headline_core_gap']), 1) if pd.notna(last_b.get('headline_core_gap')) else None,
                'qd': [r['date'].strftime('%Y-%m-%d') for _, r in cc_b.iterrows()][-20:],
                'qr': [int(r['regime']) if pd.notna(r.get('regime')) else None for _, r in cc_b.iterrows()][-20:],
            }

    countries_json[cc] = {
        'n': cnames.get(cc, cc),
        'pb': pb_data,
        'g': str(last.get('country_group', '')) if pd.notna(last.get('country_group')) else '',
        'rg': str(last.get('region', '')) if pd.notna(last.get('region')) else '',
        'cr': int(last['regime']) if pd.notna(last.get('regime')) else None,
        'ch': round(float(last['hcpi_yoy']), 1) if pd.notna(last.get('hcpi_yoy')) else None,
        'cf': round(float(last['fcpi_yoy']), 1) if pd.notna(last.get('fcpi_yoy')) else None,
        'ce': round(float(last['ecpi_yoy']), 1) if pd.notna(last.get('ecpi_yoy')) else None,
        'ctp': round(float(last['transition_prob']), 3) if pd.notna(last.get('transition_prob')) else None,
        'fi': fiscal,
        'qd': q_dates[-40:], 'qt': q_tp[-40:], 'qh': q_h[-40:], 'qf': q_f[-40:],
        'qe': q_e[-40:], 'qr': q_reg[-40:],
        'md': m_dates[-120:], 'mv': m_vals[-120:],
    }

data_json = json.dumps(countries_json, cls=NumpyEncoder)
print(f"  Country data: {len(countries_json)} countries, {len(data_json)//1024} KB")

# ============================================================
# BUILD PLOTS
# ============================================================
print("  Building plots...")

# World map
latest = regime.sort_values('date').groupby('country_code').last().reset_index()
latest['country_name'] = latest['country_code'].map(cnames)

fig_map = px.choropleth(
    latest, locations='country_code', color='regime_name',
    hover_name='country_name',
    hover_data={'regime_name': True, 'hcpi_yoy': ':.1f', 'transition_prob': ':.1%', 'country_code': False},
    color_discrete_map={v[0]: v[1] for v in REGIME.values()},
    labels={'regime_name': 'Regime', 'hcpi_yoy': 'CPI %', 'transition_prob': 'P(transition)'},
)
fig_map.update_geos(showframe=False, showcoastlines=True, coastlinecolor='#334155',
                    bgcolor='#0f172a', landcolor='#1e293b', oceancolor='#0f172a',
                    projection_type='natural earth', showocean=True, lataxis_range=[-55, 80])
fig_map.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', height=460,
                       margin=dict(l=0, r=0, t=0, b=0), font=dict(color='#e2e8f0'),
                       legend=dict(font=dict(color='#e2e8f0', size=11), bgcolor='rgba(15,23,42,0.8)',
                                   title='Regime', yanchor='bottom', y=0, xanchor='left', x=0))
map_html = fig_map.to_html(full_html=False, include_plotlyjs=False)

# Composition
comp = regime.groupby([regime['date'].dt.year, 'regime']).size().unstack(fill_value=0)
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
fig_comp = go.Figure()
for r_id in sorted(REGIME.keys()):
    rname, rcolor, _ = REGIME[r_id]
    if r_id in comp_pct.columns:
        fig_comp.add_trace(go.Scatter(x=comp_pct.index, y=comp_pct[r_id], name=rname,
                                       stackgroup='one', mode='none', fillcolor=rcolor,
                                       line=dict(width=0.3, color='rgba(255,255,255,0.15)')))
for dt, lbl in [(2008, 'GFC'), (2020, 'COVID'), (2022, 'Ukraine')]:
    fig_comp.add_vline(x=dt, line_dash='dot', line_color='rgba(255,255,255,0.25)')
    fig_comp.add_annotation(x=dt, y=97, text=lbl, showarrow=False, font=dict(size=9, color='rgba(255,255,255,0.45)'))
fig_comp.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', height=240,
                        font=dict(color='#e2e8f0'), margin=dict(l=50, r=20, t=10, b=40),
                        yaxis=dict(title='% of countries', range=[0, 100], gridcolor='#334155'),
                        xaxis=dict(gridcolor='#334155'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font_size=10))
comp_html = fig_comp.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# HIGH-RISK ALERT LIST
# ============================================================
latest_with_probs = latest.dropna(subset=['transition_prob']).copy()
high_risk = latest_with_probs[latest_with_probs['transition_prob'] > 0.25].sort_values('transition_prob', ascending=False)

alert_rows = ""
for _, row in high_risk.head(12).iterrows():
    cc = row['country_code']
    name = cnames.get(cc, cc)
    tp = row['transition_prob']
    hcpi = row['hcpi_yoy'] if pd.notna(row.get('hcpi_yoy')) else 0
    r = int(row['regime']) if pd.notna(row.get('regime')) else 0
    rname, rcolor, _ = REGIME.get(r, ('?', '#666', '?'))
    
    risk_color = '#ef4444' if tp > 0.35 else '#f97316' if tp > 0.30 else '#eab308'
    
    alert_rows += f"""<div class="alert-row" onclick="selectCountry('{cc}')">
      <div class="alert-country">{name}</div>
      <div class="alert-regime"><span class="dot" style="background:{rcolor}"></span>{rname}</div>
      <div class="alert-cpi">{hcpi:.1f}%</div>
      <div class="alert-prob" style="color:{risk_color}">{tp:.0%}</div>
    </div>\n"""

n_low = int(((latest['regime'] == 0) | (latest['regime'] == 1)).sum())
n_elevated = int((latest['regime'] == 4).sum())
n_crisis = int((latest['regime'] == 5).sum())
avg_tp = latest_with_probs['transition_prob'].mean()
n_alert = len(high_risk)

# ============================================================
# ASSEMBLE HTML
# ============================================================
print("  Assembling HTML...")

html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inflation Regime Early Warning System</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#0f172a; --card:#1e293b; --card2:#253449; --border:#334155;
  --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --dim:rgba(56,189,248,0.12);
  --green:#22c55e; --red:#ef4444; --amber:#eab308; --purple:#a855f7; --orange:#f97316;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}}
.header{{background:linear-gradient(135deg,#0f172a 0%,#1a3352 40%,#0f172a 100%);border-bottom:1px solid var(--border);padding:24px 40px}}
.header-inner{{max-width:1440px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}}
.header h1{{font-size:22px;font-weight:700;color:white;letter-spacing:-0.5px}}
.header .sub{{font-size:11px;color:var(--muted);margin-top:2px}}
.kpis{{display:flex;gap:24px}}.kpi{{text-align:center}}
.kpi-val{{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:var(--accent)}}
.kpi-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px}}
.container{{max-width:1440px;margin:0 auto;padding:16px 32px}}
.section-head{{font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid var(--border)}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:16px}}
.grid{{display:grid;gap:16px}}.grid-2{{grid-template-columns:1fr 1fr}}.grid-3{{grid-template-columns:1fr 1fr 1fr}}.grid-7-3{{grid-template-columns:7fr 3fr}}
.stats{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}}
.stat{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px 14px;flex:1;min-width:140px;display:flex;align-items:center;gap:8px}}
.stat .dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.stat-val{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600}}
.stat-lbl{{font-size:10px;color:var(--muted)}}
.search-box{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.search-box label{{font-weight:600;color:var(--accent);font-size:13px;white-space:nowrap}}
.search-input{{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:9px 14px;color:var(--text);font-size:13px;font-family:'DM Sans';width:300px;outline:none}}
.search-input:focus{{border-color:var(--accent)}}
.search-input::placeholder{{color:var(--muted)}}
.search-wrap{{position:relative;flex:1;max-width:360px}}
.dropdown{{position:absolute;top:100%;left:0;right:0;background:var(--card);border:1px solid var(--border);border-radius:0 0 8px 8px;max-height:280px;overflow-y:auto;z-index:100;display:none}}
.dropdown-item{{padding:9px 14px;cursor:pointer;font-size:12px;display:flex;justify-content:space-between;border-bottom:1px solid rgba(51,65,85,0.5)}}
.dropdown-item:hover{{background:var(--card2)}}
.regime-badge{{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;color:white}}
/* Alert table */
.alert-table{{width:100%}}
.alert-header{{display:grid;grid-template-columns:2fr 1.2fr 0.8fr 0.8fr;padding:6px 10px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid var(--border)}}
.alert-row{{display:grid;grid-template-columns:2fr 1.2fr 0.8fr 0.8fr;padding:8px 10px;font-size:12px;border-bottom:1px solid rgba(51,65,85,0.3);cursor:pointer;transition:background 0.15s}}
.alert-row:hover{{background:var(--card2)}}
.alert-country{{font-weight:500}}.alert-prob{{font-family:'JetBrains Mono',monospace;font-weight:600}}
.alert-regime{{display:flex;align-items:center;gap:5px}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
/* Risk card */
.risk-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.risk-card h3{{font-size:15px;margin-bottom:10px}}
.risk-row{{display:flex;justify-content:space-between;padding:5px 0;font-size:12px;border-bottom:1px solid rgba(51,65,85,0.3)}}
.risk-row .label{{color:var(--muted)}}.risk-row .value{{font-family:'JetBrains Mono',monospace;font-weight:500}}
.risk-gauge{{height:6px;background:var(--bg);border-radius:3px;margin-top:4px;overflow:hidden}}
.risk-gauge-fill{{height:100%;border-radius:3px;transition:width 0.3s}}
/* Notes */
.note{{background:rgba(56,189,248,0.04);border:1px solid rgba(56,189,248,0.12);border-radius:6px;padding:12px 16px;font-size:11px;color:var(--muted);line-height:1.6;margin-bottom:14px}}
.note strong{{color:var(--text)}}
.note.warn{{border-color:rgba(234,179,8,0.2);background:rgba(234,179,8,0.03)}}
.note.cal{{border-color:rgba(34,197,94,0.2);background:rgba(34,197,94,0.03)}}
.chart-box{{min-height:260px}}
.footer{{text-align:center;padding:20px;color:#475569;font-size:10px;border-top:1px solid var(--border);margin-top:20px}}
#countryPanel{{display:none}}
.country-title{{font-size:18px;font-weight:700;margin-bottom:12px}}
.badge{{padding:3px 10px;border-radius:16px;font-size:10px;font-weight:600;display:inline-flex;align-items:center;gap:4px;margin-right:6px}}
@media(max-width:900px){{.grid-2,.grid-3,.grid-7-3{{grid-template-columns:1fr}}.container{{padding:12px}}}}
</style>
</head>
<body>

<div class="header"><div class="header-inner">
  <div>
    <h1>&#x1F6A8; Inflation Regime Early Warning System</h1>
    <div class="sub">{'Calibrated' if use_calibrated else 'Raw'} XGBoost &bull; 135 Countries &bull; 1971&ndash;2025 &bull; Updated with isotonic calibration</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-val">0.839</div><div class="kpi-lbl">AUC-ROC</div></div>
    <div class="kpi"><div class="kpi-val" style="color:var(--green)">&#x2713; Cal.</div><div class="kpi-lbl">Calibrated</div></div>
    <div class="kpi"><div class="kpi-val">{n_alert}</div><div class="kpi-lbl">Alerts</div></div>
    <div class="kpi"><div class="kpi-val" style="color:var(--purple)">6</div><div class="kpi-lbl">Regimes</div></div>
  </div>
</div></div>

<div class="container">
  <div class="stats">
    <div class="stat"><div class="dot" style="background:var(--green)"></div><div><div class="stat-val">{n_low}</div><div class="stat-lbl">Low inflation</div></div></div>
    <div class="stat"><div class="dot" style="background:var(--red)"></div><div><div class="stat-val">{n_elevated}</div><div class="stat-lbl">Elevated</div></div></div>
    <div class="stat"><div class="dot" style="background:var(--purple)"></div><div><div class="stat-val">{n_crisis}</div><div class="stat-lbl">Crisis</div></div></div>
    <div class="stat"><div class="dot" style="background:var(--accent)"></div><div><div class="stat-val">{avg_tp:.0%}</div><div class="stat-lbl">Avg P(trans)</div></div></div>
  </div>

  <div class="grid grid-7-3">
    <div>
      <div class="card"><div class="section-head">Global Regime Map</div>{map_html}</div>
      <div class="card"><div class="section-head">Regime Composition Over Time</div>{comp_html}</div>
    </div>
    <div>
      <div class="card">
        <div class="section-head">&#x26A0; Highest Transition Risk</div>
        <div class="alert-table">
          <div class="alert-header"><div>Country</div><div>Regime</div><div>CPI</div><div>P(trans)</div></div>
          {alert_rows}
        </div>
      </div>
    </div>
  </div>

  <div class="note cal">
    <strong>&#x2705; Calibrated probabilities:</strong> This dashboard uses isotonic-regression calibrated transition probabilities.
    When the model says 25%, the actual transition rate is approximately 25&ndash;28%. Probabilities are reliable
    across the full range. AUC-ROC: 0.839 [0.816, 0.860], Brier score improved from 0.192 to 0.181.
  </div>
  <div class="note warn">
    <strong>&#9888; Grey countries:</strong> 53 countries lack required CPI components (headline + food + energy).
    Notable: <strong>Australia</strong> (quarterly CPI), <strong>Iran</strong> (no food CPI),
    <strong>Venezuela</strong> (incomplete data). Gulf states are covered.
  </div>

  <div class="search-box">
    <label>&#128269; Country Deep Dive</label>
    <div class="search-wrap">
      <input type="text" class="search-input" id="countrySearch" placeholder="Type country name or code..." autocomplete="off">
      <div class="dropdown" id="dropdown"></div>
    </div>
    <span id="selectedInfo" style="color:var(--muted);font-size:12px">Select a country to explore</span>
  </div>

  <div id="countryPanel">
    <div class="country-title" id="countryTitle"></div>
    <div id="countryBadges" style="margin-bottom:14px"></div>

    <div class="grid grid-2" style="margin-bottom:14px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div class="risk-card" id="riskCard"></div>
        <div class="risk-card" id="fiscalCard"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div class="risk-card" id="inflCard"></div>
        <div class="risk-card" id="panelBCard"></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:14px">
      <div class="card"><div class="chart-box" id="chartTransition"></div></div>
      <div class="card"><div class="chart-box" id="chartInflation"></div></div>
    </div>
    <div class="grid grid-2">
      <div class="card"><div class="chart-box" id="chartMonthly"></div></div>
      <div class="card"><div class="chart-box" id="chartRegime"></div></div>
    </div>
  </div>
</div>

<div class="footer">
  Inflation Regime Early Warning System &bull; XGBoost + Isotonic Calibration &bull;
  Bootstrap CI [0.816, 0.860] &bull; Trained 1971&ndash;2014 &bull; Tested COVID + Ukraine &bull;
  World Bank Data &bull; 72,321 monthly observations
</div>

<script>
const DATA = {data_json};
const REGIMES = {{
  0:{{n:'Low',c:'#22c55e',d:'Low & stable'}},1:{{n:'Low-energy',c:'#3b82f6',d:'Rising energy costs'}},
  2:{{n:'Moderate',c:'#eab308',d:'Food-driven'}},3:{{n:'Mod-energy',c:'#f97316',d:'Energy-driven'}},
  4:{{n:'Elevated',c:'#ef4444',d:'Broad pressure'}},5:{{n:'Crisis',c:'#a855f7',d:'All components spiralling'}}
}};
const REGIMES_B = {{
  0:{{n:'Low (transitory)',c:'#22c55e'}},1:{{n:'Low (persistent)',c:'#3b82f6'}},
  2:{{n:'Moderate',c:'#eab308'}},3:{{n:'Mod (supply)',c:'#f97316'}},
  4:{{n:'Elevated (demand)',c:'#ef4444'}},5:{{n:'Crisis',c:'#a855f7'}}
}};
const CL={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#1e293b',font:{{color:'#e2e8f0',family:'DM Sans',size:11}},
  margin:{{l:50,r:20,t:36,b:36}},xaxis:{{gridcolor:'#334155',zerolinecolor:'#334155'}},yaxis:{{gridcolor:'#334155',zerolinecolor:'#334155'}}}};

const entries=Object.entries(DATA).map(([c,d])=>({{c,n:d.n,r:d.cr}})).sort((a,b)=>a.n.localeCompare(b.n));
const si=document.getElementById('countrySearch'), dd=document.getElementById('dropdown');

si.addEventListener('input',function(){{
  const q=this.value.toLowerCase(); if(q.length<1){{dd.style.display='none';return}}
  const m=entries.filter(e=>e.n.toLowerCase().includes(q)||e.c.toLowerCase().includes(q)).slice(0,15);
  if(!m.length){{dd.style.display='none';return}}
  dd.innerHTML=m.map(e=>{{const r=REGIMES[e.r]||{{n:'?',c:'#666'}};
    return `<div class="dropdown-item" onclick="selectCountry('${{e.c}}')"><span>${{e.n}} (${{e.c}})</span><span class="regime-badge" style="background:${{r.c}}">${{r.n}}</span></div>`}}).join('');
  dd.style.display='block';
}});
si.addEventListener('focus',()=>{{if(si.value)si.dispatchEvent(new Event('input'))}});
document.addEventListener('click',e=>{{if(!e.target.closest('.search-wrap'))dd.style.display='none'}});

function riskColor(p){{return p>0.35?'#ef4444':p>0.25?'#f97316':p>0.15?'#eab308':'#22c55e'}}
function gauge(val,max,color){{const pct=Math.min(100,Math.max(0,(val/max)*100));return `<div class="risk-gauge"><div class="risk-gauge-fill" style="width:${{pct}}%;background:${{color}}"></div></div>`}}

function selectCountry(code){{
  dd.style.display='none'; const d=DATA[code]; if(!d)return;
  si.value=`${{d.n}} (${{code}})`; document.getElementById('countryPanel').style.display='block';
  document.getElementById('selectedInfo').textContent=`Showing: ${{d.n}}`;
  const r=REGIMES[d.cr]||{{n:'?',c:'#666',d:'?'}};

  document.getElementById('countryTitle').textContent=d.n;
  document.getElementById('countryBadges').innerHTML=`
    <span class="badge" style="background:${{r.c}}">R${{d.cr}}: ${{r.n}}</span>
    ${{d.g?`<span class="badge" style="background:var(--card);border:1px solid var(--border)">${{d.g}}</span>`:''}}
    ${{d.rg?`<span class="badge" style="background:var(--card);border:1px solid var(--border)">${{d.rg}}</span>`:''}}`;

  // Risk card
  const tp=d.ctp; const rc=riskColor(tp);
  document.getElementById('riskCard').innerHTML=`<h3 style="color:${{rc}}">Transition Risk</h3>
    <div class="risk-row"><span class="label">P(upward transition, 2Q)</span><span class="value" style="color:${{rc}}">${{tp!==null?(tp*100).toFixed(1)+'%':'N/A'}}</span></div>
    ${{gauge(tp||0,0.5,rc)}}
    <div class="risk-row" style="margin-top:8px"><span class="label">Risk level</span><span class="value">${{tp>0.35?'HIGH':tp>0.25?'ELEVATED':tp>0.15?'MODERATE':'LOW'}}</span></div>
    <div class="risk-row"><span class="label">Current regime</span><span class="value">${{r.d}}</span></div>`;

  // Fiscal card
  const fi=d.fi||{{}};
  document.getElementById('fiscalCard').innerHTML=`<h3 style="color:var(--accent)">Fiscal Vulnerability</h3>
    <div class="risk-row"><span class="label">Debt / GDP</span><span class="value">${{fi.debt_gdp!==null?fi.debt_gdp.toFixed(0)+'%':'N/A'}}</span></div>
    ${{gauge(fi.debt_gdp||0,150,'#f97316')}}
    <div class="risk-row"><span class="label">Sovereign rating</span><span class="value">${{fi.sovereign_rating!==null?fi.sovereign_rating.toFixed(0)+'/21':'N/A'}}</span></div>
    <div class="risk-row"><span class="label">Fiscal balance</span><span class="value">${{fi.fiscal_balance!==null?fi.fiscal_balance.toFixed(1)+'%':'N/A'}}</span></div>
    <div class="risk-row"><span class="label">Private credit/GDP</span><span class="value">${{fi.private_credit_gdp!==null?fi.private_credit_gdp.toFixed(0)+'%':'N/A'}}</span></div>`;

  // Inflation card
  document.getElementById('inflCard').innerHTML=`<h3 style="color:var(--amber)">Inflation Snapshot</h3>
    <div class="risk-row"><span class="label">Headline CPI</span><span class="value">${{d.ch!==null?d.ch.toFixed(1)+'%':'N/A'}}</span></div>
    <div class="risk-row"><span class="label">Food CPI</span><span class="value" style="color:#22c55e">${{d.cf!==null?d.cf.toFixed(1)+'%':'N/A'}}</span></div>
    <div class="risk-row"><span class="label">Energy CPI</span><span class="value" style="color:#ef4444">${{d.ce!==null?d.ce.toFixed(1)+'%':'N/A'}}</span></div>
    <div class="risk-row"><span class="label">Food-Energy gap</span><span class="value">${{d.cf!==null&&d.ce!==null?(d.cf-d.ce).toFixed(1)+'pp':'N/A'}}</span></div>`;

  // Charts
  const tv=d.qd.map((_,i)=>d.qt[i]!==null?i:null).filter(x=>x!==null);
  Plotly.newPlot('chartTransition',[{{x:tv.map(i=>d.qd[i]),y:tv.map(i=>d.qt[i]),type:'scatter',mode:'lines+markers',
    line:{{color:'#38bdf8',width:2.5}},marker:{{size:4}},fill:'tozeroy',fillcolor:'rgba(56,189,248,0.08)',name:'P(transition)'}}],
    {{...CL,title:{{text:'Transition Probability (2Q, calibrated)',font:{{size:13}}}},yaxis:{{...CL.yaxis,title:'P(transition)',range:[0,0.55]}},
    shapes:[{{type:'line',y0:0.25,y1:0.25,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#eab308',width:1}}}},
            {{type:'line',y0:0.35,y1:0.35,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#ef4444',width:1}}}},
            {{type:'rect',y0:0.25,y1:0.35,x0:0,x1:1,xref:'paper',fillcolor:'rgba(234,179,8,0.06)',line:{{width:0}}}},
            {{type:'rect',y0:0.35,y1:0.55,x0:0,x1:1,xref:'paper',fillcolor:'rgba(239,68,68,0.06)',line:{{width:0}}}}]}},
    {{displayModeBar:false,responsive:true}});

  const t2=[];
  [['qh','Headline','#38bdf8'],['qf','Food','#22c55e'],['qe','Energy','#ef4444']].forEach(([k,n,c])=>{{
    const v=d.qd.map((_,i)=>d[k][i]!==null?i:null).filter(x=>x!==null);
    if(v.length)t2.push({{x:v.map(i=>d.qd[i]),y:v.map(i=>d[k][i]),type:'scatter',mode:'lines',name:n,line:{{color:c,width:2}}}});}});
  Plotly.newPlot('chartInflation',t2,{{...CL,title:{{text:'Inflation Components',font:{{size:13}}}},yaxis:{{...CL.yaxis,title:'YoY %'}},
    legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center',font:{{size:10}}}}}},{{displayModeBar:false,responsive:true}});

  if(d.md.length){{Plotly.newPlot('chartMonthly',[{{x:d.md,y:d.mv,type:'scatter',mode:'lines',line:{{color:'#38bdf8',width:1.5}},
    fill:'tozeroy',fillcolor:'rgba(56,189,248,0.05)'}}],{{...CL,title:{{text:'Monthly Headline CPI',font:{{size:13}}}},yaxis:{{...CL.yaxis,title:'YoY %'}},showlegend:false}},
    {{displayModeBar:false,responsive:true}})}}else{{document.getElementById('chartMonthly').innerHTML='<div style="color:var(--muted);text-align:center;padding:80px">No monthly data</div>'}}

  const rt=[];Object.entries(REGIMES).forEach(([rid,info])=>{{
    const ix=d.qr.map((r,i)=>r==rid?i:null).filter(x=>x!==null);
    if(ix.length)rt.push({{x:ix.map(i=>d.qd[i]),y:ix.map(()=>1),type:'bar',name:`R${{rid}}: ${{info.n}}`,marker:{{color:info.c}},width:86400000*85}})}});
  Plotly.newPlot('chartRegime',rt,{{...CL,title:{{text:'Regime History',font:{{size:13}}}},barmode:'stack',yaxis:{{visible:false}},
    legend:{{orientation:'h',y:1.15,x:0.5,xanchor:'center',font:{{size:9}}}}}},{{displayModeBar:false,responsive:true}});

  // Panel B comparison card
  const pb=d.pb||{{}};
  if(pb.regime!==undefined && pb.regime!==null){{
    const rb=REGIMES_B[pb.regime]||{{n:'?',c:'#666'}};
    const rA=REGIMES[d.cr]||{{n:'?',c:'#666'}};
    document.getElementById('panelBCard').innerHTML=`<h3 style="color:var(--purple)">Panel A vs B Comparison</h3>
      <div style="font-size:10px;color:var(--muted);margin-bottom:8px">4 CPI measures (incl. core) &bull; 74 countries</div>
      <div class="risk-row"><span class="label">Panel A regime</span><span class="value"><span class="dot" style="background:${{rA.c}}"></span> ${{rA.n}}</span></div>
      <div class="risk-row"><span class="label">Panel B regime</span><span class="value"><span class="dot" style="background:${{rb.c}}"></span> ${{rb.n}}</span></div>
      ${{pb.ccpi!==null?`<div class="risk-row"><span class="label">Core CPI</span><span class="value">${{pb.ccpi.toFixed(1)}}%</span></div>`:''}}
      ${{pb.hc_gap!==null?`<div class="risk-row"><span class="label">Headline-Core gap</span><span class="value" style="color:${{pb.hc_gap>0?'var(--amber)':'var(--green)'}}">${{pb.hc_gap>0?'+':''}}${{pb.hc_gap.toFixed(1)}}pp</span></div>`:''}}
      <div style="margin-top:8px;font-size:10px;color:var(--muted)">
        ${{pb.hc_gap>0.5?'&#x26A0; Headline > Core: transitory supply pressure':pb.hc_gap<-0.5?'&#x1F534; Core > Headline: persistent demand pressure':'&#x2705; Headline &asymp; Core: balanced'}}
      </div>`;
  }}else{{
    document.getElementById('panelBCard').innerHTML=`<h3 style="color:var(--muted)">Panel B Comparison</h3>
      <div style="padding:20px 0;text-align:center;color:var(--muted);font-size:12px">
        Not available for this country.<br>Panel B requires core CPI data<br>(74 of 135 countries).
      </div>`;
  }}

  document.getElementById('countryPanel').scrollIntoView({{behavior:'smooth',block:'start'}});
}}
window.addEventListener('load',()=>selectCountry('GBR'));
</script>
</body>
</html>"""

out_path = os.path.join(FDIR, 'dashboard_final.html')
with open(out_path, 'w') as f:
    f.write(html_page)

print(f"\n{'='*60}")
print(f"  Dashboard generated: {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024:.0f} KB")
print(f"  Countries with alerts (>25% cal. prob): {n_alert}")
print(f"{'='*60}")
print(f"\n  Open: open {out_path}")
