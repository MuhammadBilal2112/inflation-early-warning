"""
==============================================================================
INFLATION REGIME DASHBOARD — FINAL REDESIGN
==============================================================================
"Bloomberg Terminal meets editorial data journalism"

Distinctive features:
  - Instrument Serif + Geist Mono typography
  - Warm charcoal + amber palette
  - Staggered entrance animations
  - Immersive full-width world map
  - Pulsing alert indicators
  - Circular transition probability gauge
  - Progressive card reveal on country select
  - Subtle grain texture background

Same data pipeline as v4 — only the HTML template is new.

RUN: python3 Dashboard_Redesign.py
OUTPUT: inflation_regimes_project/outputs/figures/dashboard_final.html
==============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib, json, os, re, warnings
warnings.filterwarnings('ignore')

class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
MDIR = os.path.join(BASE_DIR, "outputs", "models")
FDIR = os.path.join(BASE_DIR, "outputs", "figures")

print("=" * 60)
print("  Building Redesigned Dashboard...")
print("=" * 60)

# ============================================================
# DATA PIPELINE (identical to v4)
# ============================================================
print("\n  Loading data...")
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])
features_df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in features_df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    features_df = features_df.rename(columns=clean_map)

# Calibrated model
cal_path = os.path.join(MDIR, 'xgboost_calibrated_target_up_2q.joblib')
raw_path = os.path.join(MDIR, 'xgboost_target_up_2q.joblib')
if os.path.exists(cal_path):
    saved = joblib.load(cal_path)
    model, calibrator, feature_cols = saved['model'], saved['calibrator'], saved['features']
    raw_p = model.predict_proba(features_df[feature_cols])[:, 1]
    features_df['transition_prob'] = calibrator.predict(raw_p)
    cal_label = "Calibrated"
    print("  Model: CALIBRATED XGBoost")
else:
    saved = joblib.load(raw_path)
    model, calibrator, feature_cols = saved['model'], None, saved['features']
    features_df['transition_prob'] = model.predict_proba(features_df[feature_cols])[:, 1]
    cal_label = "Raw"

# Panel B
panelB_path = os.path.join(PROC, 'regime_labels_panelB.csv')
has_pB = os.path.exists(panelB_path)
if has_pB:
    regime_b = pd.read_csv(panelB_path, parse_dates=['date'])
    pB_cc = set(regime_b['country_code'].unique())
    print(f"  Panel B: {len(pB_cc)} countries")
else:
    regime_b, pB_cc = pd.DataFrame(), set()

cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']): cnames[r['country_code']] = str(r['country_name'])

REGIME = {
    0: ('Low', '#4ade80', 'Low & stable'),
    1: ('Low-energy', '#60a5fa', 'Rising energy, low headline'),
    2: ('Moderate', '#facc15', 'Food-driven moderate'),
    3: ('Mod-energy', '#fb923c', 'Energy-driven moderate'),
    4: ('Elevated', '#f87171', 'Broadly elevated'),
    5: ('Crisis', '#c084fc', 'All components spiralling'),
}

regime['regime_name'] = regime['regime'].map(lambda x: REGIME.get(int(x), ('?','#777','?'))[0] if pd.notna(x) else 'No data')

# Merge
merge_cols = ['country_code', 'date', 'transition_prob']
for fc in ['debt_gdp', 'sovereign_rating', 'fiscal_balance', 'private_credit_gdp']:
    if fc in features_df.columns: merge_cols.append(fc)
regime = regime.merge(features_df[merge_cols], on=['country_code', 'date'], how='left')

# ============================================================
# BUILD JSON
# ============================================================
print("  Building country data...")
countries_json = {}
for cc in sorted(regime['country_code'].unique()):
    cc_r = regime[regime['country_code'] == cc].sort_values('date')
    cc_m = monthly[(monthly['country_code'] == cc) & (monthly['date'] >= '2010-01-01')].sort_values('date')
    if cc_r.empty: continue
    last = cc_r.iloc[-1]

    qd, qt, qh, qf, qe, qr = [], [], [], [], [], []
    for _, row in cc_r.iterrows():
        qd.append(row['date'].strftime('%Y-%m-%d'))
        qt.append(round(float(row['transition_prob']), 4) if pd.notna(row.get('transition_prob')) else None)
        qh.append(round(float(row['hcpi_yoy']), 2) if pd.notna(row.get('hcpi_yoy')) else None)
        qf.append(round(float(row['fcpi_yoy']), 2) if pd.notna(row.get('fcpi_yoy')) else None)
        qe.append(round(float(row['ecpi_yoy']), 2) if pd.notna(row.get('ecpi_yoy')) else None)
        qr.append(int(row['regime']) if pd.notna(row.get('regime')) else None)

    md, mv = [], []
    for _, row in cc_m.iterrows():
        if pd.notna(row.get('hcpi_yoy')):
            md.append(row['date'].strftime('%Y-%m-%d'))
            mv.append(round(float(row['hcpi_yoy']), 2))

    fi = {}
    for fc in ['debt_gdp', 'sovereign_rating', 'fiscal_balance', 'private_credit_gdp']:
        v = last.get(fc); fi[fc] = round(float(v), 1) if pd.notna(v) else None

    pb = {}
    if has_pB and cc in pB_cc:
        cc_b = regime_b[regime_b['country_code'] == cc].sort_values('date')
        if not cc_b.empty:
            lb = cc_b.iloc[-1]
            pb = {
                'r': int(lb['regime']) if pd.notna(lb.get('regime')) else None,
                'cc': round(float(lb['ccpi_yoy']), 1) if pd.notna(lb.get('ccpi_yoy')) else None,
                'gap': round(float(lb['headline_core_gap']), 1) if pd.notna(lb.get('headline_core_gap')) else None,
            }

    countries_json[cc] = {
        'n': cnames.get(cc, cc),
        'g': str(last.get('country_group', '')) if pd.notna(last.get('country_group')) else '',
        'rg': str(last.get('region', '')) if pd.notna(last.get('region')) else '',
        'cr': int(last['regime']) if pd.notna(last.get('regime')) else None,
        'ch': round(float(last['hcpi_yoy']), 1) if pd.notna(last.get('hcpi_yoy')) else None,
        'cf': round(float(last['fcpi_yoy']), 1) if pd.notna(last.get('fcpi_yoy')) else None,
        'ce': round(float(last['ecpi_yoy']), 1) if pd.notna(last.get('ecpi_yoy')) else None,
        'ctp': round(float(last['transition_prob']), 3) if pd.notna(last.get('transition_prob')) else None,
        'fi': fi, 'pb': pb,
        'qd': qd[-40:], 'qt': qt[-40:], 'qh': qh[-40:], 'qf': qf[-40:],
        'qe': qe[-40:], 'qr': qr[-40:], 'md': md[-120:], 'mv': mv[-120:],
    }

data_json = json.dumps(countries_json, cls=NpEnc)
print(f"  {len(countries_json)} countries, {len(data_json)//1024} KB")

# ============================================================
# PLOTLY CHARTS
# ============================================================
print("  Building charts...")

latest = regime.sort_values('date').groupby('country_code').last().reset_index()
latest['country_name'] = latest['country_code'].map(cnames)

# World map
fig_map = px.choropleth(
    latest, locations='country_code', color='regime_name',
    hover_name='country_name',
    hover_data={'regime_name': True, 'hcpi_yoy': ':.1f', 'transition_prob': ':.1%', 'country_code': False},
    color_discrete_map={v[0]: v[1] for v in REGIME.values()},
    labels={'regime_name': 'Regime', 'hcpi_yoy': 'CPI %', 'transition_prob': 'P(trans)'},
)
fig_map.update_geos(showframe=False, showcoastlines=True, coastlinecolor='#3a3a3a',
                    bgcolor='rgba(0,0,0,0)', landcolor='#2a2a2a', oceancolor='#1a1a1a',
                    projection_type='natural earth', showocean=True, lataxis_range=[-55, 80])
fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=480,
                       margin=dict(l=0, r=0, t=0, b=0), font=dict(color='#d4d4d4'),
                       legend=dict(font=dict(color='#d4d4d4', size=11), bgcolor='rgba(26,26,26,0.85)',
                                   title='', yanchor='bottom', y=0.02, xanchor='left', x=0.02))
map_html = fig_map.to_html(full_html=False, include_plotlyjs=False)

# Composition
comp = regime.groupby([regime['date'].dt.year, 'regime']).size().unstack(fill_value=0)
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
fig_comp = go.Figure()
for r_id in sorted(REGIME.keys()):
    rn, rc, _ = REGIME[r_id]
    if r_id in comp_pct.columns:
        fig_comp.add_trace(go.Scatter(x=comp_pct.index, y=comp_pct[r_id], name=rn,
                                       stackgroup='one', mode='none', fillcolor=rc,
                                       line=dict(width=0.3, color='rgba(255,255,255,0.1)')))
for dt, lbl in [(2008, 'GFC'), (2020, 'COVID'), (2022, 'Ukraine')]:
    fig_comp.add_vline(x=dt, line_dash='dot', line_color='rgba(255,255,255,0.2)')
    fig_comp.add_annotation(x=dt, y=97, text=lbl, showarrow=False, font=dict(size=9, color='rgba(255,255,255,0.35)'))
fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,30,0.5)', height=220,
                        font=dict(color='#a3a3a3'), margin=dict(l=45, r=15, t=5, b=35),
                        yaxis=dict(title='% of countries', range=[0, 100], gridcolor='#333'),
                        xaxis=dict(gridcolor='#333'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font_size=9))
comp_html = fig_comp.to_html(full_html=False, include_plotlyjs=False)

# Alerts
lp = latest.dropna(subset=['transition_prob'])
high_risk = lp[lp['transition_prob'] > 0.20].sort_values('transition_prob', ascending=False)

alert_html = ""
for _, row in high_risk.head(10).iterrows():
    cc = row['country_code']
    tp = row['transition_prob']
    hcpi = row['hcpi_yoy'] if pd.notna(row.get('hcpi_yoy')) else 0
    r = int(row['regime']) if pd.notna(row.get('regime')) else 0
    rn, rc, _ = REGIME.get(r, ('?', '#555', '?'))
    pulse = ' pulse' if tp > 0.30 else ''
    alert_html += f"""<div class="al-row" onclick="selectCountry('{cc}')">
      <span class="al-dot{pulse}" style="background:{rc}"></span>
      <span class="al-name">{cnames.get(cc, cc)}</span>
      <span class="al-val">{tp:.0%}</span>
    </div>\n"""

n_low = int(((latest['regime'] == 0) | (latest['regime'] == 1)).sum())
n_elev = int((latest['regime'] == 4).sum())
n_crisis = int((latest['regime'] == 5).sum())
avg_tp = lp['transition_prob'].mean()

# ============================================================
# HTML TEMPLATE
# ============================================================
print("  Assembling HTML...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Inflation Regime Early Warning System</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Geist+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes pulse-ring{{0%{{box-shadow:0 0 0 0 rgba(248,113,113,0.5)}}70%{{box-shadow:0 0 0 6px rgba(248,113,113,0)}}100%{{box-shadow:0 0 0 0 rgba(248,113,113,0)}}}}
@keyframes slideIn{{from{{opacity:0;transform:translateX(-12px)}}to{{opacity:1;transform:translateX(0)}}}}

:root{{
  --bg:#1a1a1a;--bg2:#222;--card:#262626;--card-h:#2e2e2e;--border:#383838;
  --text:#e5e5e5;--muted:#737373;--accent:#f59e0b;--accent2:#d97706;
  --green:#4ade80;--red:#f87171;--amber:#facc15;--purple:#c084fc;--blue:#60a5fa;--orange:#fb923c;
  --serif:'Instrument Serif',Georgia,serif;
  --sans:'Outfit',system-ui,sans-serif;
  --mono:'Geist Mono','JetBrains Mono',monospace;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-weight:400;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
}}

/* HEADER */
.hero{{padding:48px 0 0;text-align:center;animation:fadeIn 0.8s ease}}
.hero h1{{font-family:var(--serif);font-size:42px;font-weight:400;color:white;letter-spacing:-1px}}
.hero .sub{{font-size:13px;color:var(--muted);margin:6px 0 24px;font-weight:300;letter-spacing:0.5px}}
.kpi-bar{{display:flex;justify-content:center;gap:48px;padding:18px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);margin-bottom:0}}
.kpi{{text-align:center}}.kpi-v{{font-family:var(--mono);font-size:26px;font-weight:600;color:var(--accent)}}
.kpi-l{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:2px}}

/* MAP */
.map-wrap{{width:100%;background:#1a1a1a;padding:0;animation:fadeUp 1s ease 0.2s both}}

/* MAIN */
.main{{max-width:1400px;margin:0 auto;padding:24px 40px}}
.section{{margin-bottom:24px;animation:fadeUp 0.6s ease both}}
.s1{{animation-delay:0.3s}}.s2{{animation-delay:0.4s}}.s3{{animation-delay:0.5s}}.s4{{animation-delay:0.6s}}
.sec-title{{font-family:var(--serif);font-size:20px;color:var(--accent);margin-bottom:12px;font-weight:400}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;transition:border-color 0.2s,transform 0.2s}}
.card:hover{{border-color:#555;transform:translateY(-1px)}}
.grid{{display:grid;gap:16px}}.g2{{grid-template-columns:1fr 1fr}}.g3{{grid-template-columns:1fr 1fr 1fr}}.g73{{grid-template-columns:7fr 3fr}}

/* ALERTS */
.al-panel{{max-height:440px;overflow-y:auto}}
.al-row{{display:flex;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid rgba(56,56,56,0.5);cursor:pointer;transition:background 0.15s;animation:slideIn 0.3s ease both}}
.al-row:hover{{background:var(--card-h)}}
.al-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.al-dot.pulse{{animation:pulse-ring 1.5s infinite}}
.al-name{{flex:1;font-size:13px;font-weight:400}}.al-val{{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--accent)}}

/* SEARCH */
.search-area{{display:flex;align-items:center;gap:14px;margin-bottom:20px;flex-wrap:wrap}}
.search-area label{{font-family:var(--serif);font-size:18px;color:var(--accent)}}
.s-input{{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 16px;color:var(--text);font-size:14px;font-family:var(--sans);width:340px;outline:none;transition:border-color 0.2s}}
.s-input:focus{{border-color:var(--accent)}}
.s-input::placeholder{{color:#555}}
.s-wrap{{position:relative}}.dd{{position:absolute;top:100%;left:0;right:0;background:var(--card);border:1px solid var(--border);border-radius:0 0 10px 10px;max-height:280px;overflow-y:auto;z-index:100;display:none}}
.dd-item{{padding:10px 14px;cursor:pointer;font-size:13px;display:flex;justify-content:space-between;border-bottom:1px solid rgba(56,56,56,0.3);transition:background 0.12s}}
.dd-item:hover{{background:var(--card-h)}}
.rbadge{{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;color:white;font-family:var(--mono)}}

/* COUNTRY */
#countryPanel{{display:none;animation:fadeUp 0.5s ease}}
.c-title{{font-family:var(--serif);font-size:28px;color:white;margin-bottom:4px}}
.c-badges{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
.badge{{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500;border:1px solid var(--border);display:inline-flex;align-items:center;gap:5px}}

/* RISK CARDS */
.rcard{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;transition:border-color 0.2s}}
.rcard:hover{{border-color:#555}}
.rcard h3{{font-family:var(--serif);font-size:16px;font-weight:400;margin-bottom:10px}}
.rrow{{display:flex;justify-content:space-between;padding:6px 0;font-size:12px;border-bottom:1px solid rgba(56,56,56,0.4)}}
.rrow .rl{{color:var(--muted)}}.rrow .rv{{font-family:var(--mono);font-weight:500}}

/* GAUGE */
.gauge-wrap{{width:100px;height:100px;margin:0 auto 8px;position:relative}}
.gauge-bg{{fill:none;stroke:#333;stroke-width:8}}
.gauge-fill{{fill:none;stroke-width:8;stroke-linecap:round;transition:stroke-dashoffset 0.8s ease}}
.gauge-text{{font-family:var(--mono);font-size:18px;font-weight:600;fill:var(--text);text-anchor:middle;dominant-baseline:middle}}
.gauge-label{{font-size:10px;fill:var(--muted);text-anchor:middle}}

.chart-box{{min-height:250px}}

/* NOTES */
.note{{border-radius:8px;padding:12px 16px;font-size:11px;color:var(--muted);line-height:1.6;margin-bottom:14px;border:1px solid}}
.note strong{{color:var(--text)}}.note.ok{{border-color:rgba(74,222,128,0.2);background:rgba(74,222,128,0.03)}}
.note.warn{{border-color:rgba(250,204,21,0.2);background:rgba(250,204,21,0.03)}}

.footer{{text-align:center;padding:32px;color:#525252;font-size:10px;border-top:1px solid var(--border);margin-top:32px;font-family:var(--mono)}}

@media(max-width:900px){{.g2,.g3,.g73{{grid-template-columns:1fr}}.main{{padding:16px}}.hero h1{{font-size:28px}}}}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <h1>Inflation Regime Early Warning</h1>
  <div class="sub">{cal_label} XGBoost &middot; 135 Countries &middot; 1971&ndash;2025 &middot; World Bank Data</div>
  <div class="kpi-bar">
    <div class="kpi"><div class="kpi-v">0.839</div><div class="kpi-l">AUC-ROC</div></div>
    <div class="kpi"><div class="kpi-v" style="color:var(--green)">{cal_label}</div><div class="kpi-l">Probabilities</div></div>
    <div class="kpi"><div class="kpi-v">{n_low}</div><div class="kpi-l">Low inflation</div></div>
    <div class="kpi"><div class="kpi-v" style="color:var(--red)">{n_elev + n_crisis}</div><div class="kpi-l">Elevated + Crisis</div></div>
    <div class="kpi"><div class="kpi-v">{avg_tp:.0%}</div><div class="kpi-l">Avg P(trans)</div></div>
  </div>
</div>

<!-- MAP (full width, immersive) -->
<div class="map-wrap">{map_html}</div>

<div class="main">

  <!-- MAP + ALERTS side by side -->
  <div class="grid g73 section s1">
    <div class="card">
      <div class="sec-title">Regime Composition Over Time</div>
      {comp_html}
    </div>
    <div class="card">
      <div class="sec-title">&#x26A0; Highest Risk</div>
      <div class="al-panel">{alert_html}</div>
    </div>
  </div>

  <!-- NOTES -->
  <div class="note ok section s2">
    <strong>&#x2705; Calibrated probabilities.</strong> Isotonic regression applied. When the model says 25%,
    the actual transition rate is ~25&ndash;28%. Brier score: 0.181.
  </div>
  <div class="note warn section s2">
    <strong>&#9888; 53 grey countries</strong> lack CPI component data. Notable: Australia (quarterly only),
    Iran (no food CPI), Venezuela (incomplete). Gulf states are covered.
  </div>

  <!-- SEARCH -->
  <div class="search-area section s3">
    <label>Explore a country</label>
    <div class="s-wrap">
      <input type="text" class="s-input" id="si" placeholder="Type name or ISO code..." autocomplete="off">
      <div class="dd" id="dd"></div>
    </div>
    <span id="selInfo" style="color:var(--muted);font-size:12px"></span>
  </div>

  <!-- COUNTRY PANEL -->
  <div id="countryPanel">
    <div class="c-title" id="cTitle"></div>
    <div class="c-badges" id="cBadges"></div>

    <div class="grid g3" style="margin-bottom:16px">
      <div class="rcard" id="rcRisk"></div>
      <div class="rcard" id="rcFiscal"></div>
      <div class="rcard" id="rcPanelB"></div>
    </div>

    <div class="grid g2" style="margin-bottom:16px">
      <div class="card"><div class="chart-box" id="chTrans"></div></div>
      <div class="card"><div class="chart-box" id="chInfl"></div></div>
    </div>
    <div class="grid g2">
      <div class="card"><div class="chart-box" id="chMonthly"></div></div>
      <div class="card"><div class="chart-box" id="chRegime"></div></div>
    </div>
  </div>

</div>

<div class="footer">
  Inflation Regime Early Warning System &bull; XGBoost + Isotonic Calibration &bull;
  CI [0.816, 0.860] &bull; Train 1971&ndash;2014 &bull; Test: COVID + Ukraine &bull;
  72,321 observations &bull; World Bank CMO + Fiscal Space
</div>

<script>
const D={data_json};
const R={{0:{{n:'Low',c:'#4ade80',d:'Low & stable'}},1:{{n:'Low-energy',c:'#60a5fa',d:'Rising energy'}},
  2:{{n:'Moderate',c:'#facc15',d:'Food-driven'}},3:{{n:'Mod-energy',c:'#fb923c',d:'Energy-driven'}},
  4:{{n:'Elevated',c:'#f87171',d:'Broad pressure'}},5:{{n:'Crisis',c:'#c084fc',d:'Spiralling'}}}};
const RB={{0:{{n:'Low (transitory)',c:'#4ade80'}},1:{{n:'Low (persistent)',c:'#60a5fa'}},
  2:{{n:'Moderate',c:'#facc15'}},3:{{n:'Mod (supply)',c:'#fb923c'}},
  4:{{n:'Elevated (demand)',c:'#f87171'}},5:{{n:'Crisis',c:'#c084fc'}}}};
const CL={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(30,30,30,0.5)',
  font:{{color:'#a3a3a3',family:'Outfit,sans-serif',size:11}},margin:{{l:48,r:16,t:34,b:34}},
  xaxis:{{gridcolor:'#333',zerolinecolor:'#333'}},yaxis:{{gridcolor:'#333',zerolinecolor:'#333'}}}};

const entries=Object.entries(D).map(([c,d])=>({{c,n:d.n,r:d.cr}})).sort((a,b)=>a.n.localeCompare(b.n));
const si=document.getElementById('si'),dd=document.getElementById('dd');
si.addEventListener('input',function(){{const q=this.value.toLowerCase();if(q.length<1){{dd.style.display='none';return}}
  const m=entries.filter(e=>e.n.toLowerCase().includes(q)||e.c.toLowerCase().includes(q)).slice(0,12);
  if(!m.length){{dd.style.display='none';return}}
  dd.innerHTML=m.map(e=>{{const r=R[e.r]||{{n:'?',c:'#555'}};
    return `<div class="dd-item" onclick="selectCountry('${{e.c}}')"><span>${{e.n}} (${{e.c}})</span><span class="rbadge" style="background:${{r.c}}">${{r.n}}</span></div>`}}).join('');
  dd.style.display='block'}});
si.addEventListener('focus',()=>{{if(si.value)si.dispatchEvent(new Event('input'))}});
document.addEventListener('click',e=>{{if(!e.target.closest('.s-wrap'))dd.style.display='none'}});

function svgGauge(val,max,color){{
  const pct=Math.min(1,Math.max(0,val/max));const r=42;const circ=2*Math.PI*r;const offset=circ*(1-pct*0.75);
  const riskLbl=val>0.35?'HIGH':val>0.25?'ELEVATED':val>0.15?'MODERATE':'LOW';
  return `<svg class="gauge-wrap" viewBox="0 0 100 100" style="width:110px;height:110px;display:block;margin:0 auto 8px">
    <circle cx="50" cy="50" r="${{r}}" class="gauge-bg" transform="rotate(135,50,50)" stroke-dasharray="${{circ*0.75}} ${{circ*0.25}}"/>
    <circle cx="50" cy="50" r="${{r}}" class="gauge-fill" style="stroke:${{color}}" transform="rotate(135,50,50)" stroke-dasharray="${{circ*0.75}} ${{circ*0.25}}" stroke-dashoffset="${{offset}}"/>
    <text x="50" y="46" class="gauge-text" style="fill:${{color}}">${{(val*100).toFixed(0)}}%</text>
    <text x="50" y="62" class="gauge-label">${{riskLbl}}</text>
  </svg>`}}

function rc(p){{return p>0.35?'#f87171':p>0.25?'#fb923c':p>0.15?'#facc15':'#4ade80'}}

function selectCountry(code){{
  dd.style.display='none';const d=D[code];if(!d)return;
  si.value=`${{d.n}} (${{code}})`;document.getElementById('countryPanel').style.display='block';
  document.getElementById('selInfo').textContent=d.n;
  const r=R[d.cr]||{{n:'?',c:'#555',d:'?'}};const tp=d.ctp||0;const color=rc(tp);

  document.getElementById('cTitle').textContent=d.n;
  document.getElementById('cBadges').innerHTML=`
    <span class="badge" style="border-color:${{r.c}};color:${{r.c}}"><span class="al-dot" style="background:${{r.c}}"></span>R${{d.cr}}: ${{r.n}}</span>
    ${{d.g?`<span class="badge">${{d.g}}</span>`:''}}
    ${{d.rg?`<span class="badge">${{d.rg}}</span>`:''}}`;

  // Risk gauge card
  document.getElementById('rcRisk').innerHTML=`<h3 style="color:${{color}}">Transition Risk</h3>
    ${{svgGauge(tp,0.5,color)}}
    <div class="rrow"><span class="rl">P(upward, 2Q)</span><span class="rv" style="color:${{color}}">${{(tp*100).toFixed(1)}}%</span></div>
    <div class="rrow"><span class="rl">Headline CPI</span><span class="rv">${{d.ch!==null?d.ch.toFixed(1)+'%':'—'}}</span></div>
    <div class="rrow"><span class="rl">Food CPI</span><span class="rv" style="color:var(--green)">${{d.cf!==null?d.cf.toFixed(1)+'%':'—'}}</span></div>
    <div class="rrow"><span class="rl">Energy CPI</span><span class="rv" style="color:var(--red)">${{d.ce!==null?d.ce.toFixed(1)+'%':'—'}}</span></div>
    <div class="rrow"><span class="rl">Regime</span><span class="rv">${{r.d}}</span></div>`;

  // Fiscal card
  const fi=d.fi||{{}};
  const dg=fi.debt_gdp;const dgColor=dg>100?'var(--red)':dg>70?'var(--orange)':dg>50?'var(--amber)':'var(--green)';
  document.getElementById('rcFiscal').innerHTML=`<h3 style="color:var(--amber)">Fiscal Profile</h3>
    <div class="rrow"><span class="rl">Debt / GDP</span><span class="rv" style="color:${{dg?dgColor:'var(--muted)'}}">${{dg!==null?dg.toFixed(0)+'%':'—'}}</span></div>
    <div style="height:5px;background:#333;border-radius:3px;margin:4px 0 8px;overflow:hidden"><div style="height:100%;width:${{Math.min(100,(dg||0)/1.5)}}%;background:${{dg?dgColor:'#333'}};border-radius:3px"></div></div>
    <div class="rrow"><span class="rl">Sovereign rating</span><span class="rv">${{fi.sovereign_rating!==null?fi.sovereign_rating.toFixed(0)+' / 21':'—'}}</span></div>
    <div class="rrow"><span class="rl">Fiscal balance</span><span class="rv">${{fi.fiscal_balance!==null?(fi.fiscal_balance>0?'+':'')+fi.fiscal_balance.toFixed(1)+'%':'—'}}</span></div>
    <div class="rrow"><span class="rl">Private credit/GDP</span><span class="rv">${{fi.private_credit_gdp!==null?fi.private_credit_gdp.toFixed(0)+'%':'—'}}</span></div>`;

  // Panel B card
  const pb=d.pb||{{}};
  if(pb.r!==undefined&&pb.r!==null){{
    const rb=RB[pb.r]||{{n:'?',c:'#555'}};const rA=R[d.cr]||{{n:'?',c:'#555'}};
    const gapSign=pb.gap>0?'+':'';const gapColor=pb.gap>0.5?'var(--amber)':pb.gap<-0.5?'var(--red)':'var(--green)';
    const gapMsg=pb.gap>0.5?'Supply pressure (transitory)':pb.gap<-0.5?'Demand pressure (persistent)':'Balanced';
    document.getElementById('rcPanelB').innerHTML=`<h3 style="color:var(--purple)">Panel A vs B</h3>
      <div style="font-size:10px;color:var(--muted);margin-bottom:8px">4 CPI measures &bull; 74 countries</div>
      <div class="rrow"><span class="rl">Panel A</span><span class="rv"><span class="al-dot" style="background:${{rA.c}}"></span> ${{rA.n}}</span></div>
      <div class="rrow"><span class="rl">Panel B</span><span class="rv"><span class="al-dot" style="background:${{rb.c}}"></span> ${{rb.n}}</span></div>
      ${{pb.cc!==null?`<div class="rrow"><span class="rl">Core CPI</span><span class="rv">${{pb.cc.toFixed(1)}}%</span></div>`:''}}
      ${{pb.gap!==null?`<div class="rrow"><span class="rl">H&ndash;C gap</span><span class="rv" style="color:${{gapColor}}">${{gapSign}}${{pb.gap.toFixed(1)}}pp</span></div>`:''}}
      <div style="margin-top:8px;font-size:10px;color:${{gapColor}};font-style:italic">${{gapMsg}}</div>`;
  }}else{{
    document.getElementById('rcPanelB').innerHTML=`<h3 style="color:var(--muted)">Panel A vs B</h3>
      <div style="padding:30px 0;text-align:center;color:#525252;font-size:12px;line-height:1.8">
        Not available.<br>Requires core CPI data<br><span style="font-size:10px">(74 of 135 countries)</span></div>`;
  }}

  // CHARTS
  const tv=d.qd.map((_,i)=>d.qt[i]!==null?i:null).filter(x=>x!==null);
  Plotly.newPlot('chTrans',[{{x:tv.map(i=>d.qd[i]),y:tv.map(i=>d.qt[i]),type:'scatter',mode:'lines+markers',
    line:{{color:'#f59e0b',width:2.5}},marker:{{size:4}},fill:'tozeroy',fillcolor:'rgba(245,158,11,0.06)'}}],
    {{...CL,title:{{text:'Transition Probability (2Q, calibrated)',font:{{size:13,family:'Instrument Serif'}}}},
    yaxis:{{...CL.yaxis,title:'P(transition)',range:[0,0.55]}},
    shapes:[{{type:'line',y0:0.25,y1:0.25,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#facc15',width:1}}}},
            {{type:'line',y0:0.35,y1:0.35,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#f87171',width:1}}}},
            {{type:'rect',y0:0.25,y1:0.35,x0:0,x1:1,xref:'paper',fillcolor:'rgba(250,204,21,0.04)',line:{{width:0}}}},
            {{type:'rect',y0:0.35,y1:0.55,x0:0,x1:1,xref:'paper',fillcolor:'rgba(248,113,113,0.04)',line:{{width:0}}}}]}},
    {{displayModeBar:false,responsive:true}});

  const t2=[];[['qh','Headline','#f59e0b'],['qf','Food','#4ade80'],['qe','Energy','#f87171']].forEach(([k,n,c])=>{{
    const v=d.qd.map((_,i)=>d[k][i]!==null?i:null).filter(x=>x!==null);
    if(v.length)t2.push({{x:v.map(i=>d.qd[i]),y:v.map(i=>d[k][i]),type:'scatter',mode:'lines',name:n,line:{{color:c,width:2}}}});}});
  Plotly.newPlot('chInfl',t2,{{...CL,title:{{text:'Inflation Components',font:{{size:13,family:'Instrument Serif'}}}},yaxis:{{...CL.yaxis,title:'YoY %'}},
    legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center',font:{{size:10}}}}}},{{displayModeBar:false,responsive:true}});

  if(d.md.length){{Plotly.newPlot('chMonthly',[{{x:d.md,y:d.mv,type:'scatter',mode:'lines',line:{{color:'#f59e0b',width:1.5}},
    fill:'tozeroy',fillcolor:'rgba(245,158,11,0.04)'}}],{{...CL,title:{{text:'Monthly CPI',font:{{size:13,family:'Instrument Serif'}}}},
    yaxis:{{...CL.yaxis,title:'YoY %'}},showlegend:false}},{{displayModeBar:false,responsive:true}})}}
  else{{document.getElementById('chMonthly').innerHTML='<div style="color:#525252;text-align:center;padding:80px">No monthly data</div>'}}

  const rt=[];Object.entries(R).forEach(([rid,info])=>{{const ix=d.qr.map((r,i)=>r==rid?i:null).filter(x=>x!==null);
    if(ix.length)rt.push({{x:ix.map(i=>d.qd[i]),y:ix.map(()=>1),type:'bar',name:info.n,marker:{{color:info.c}},width:86400000*85}})}});
  Plotly.newPlot('chRegime',rt,{{...CL,title:{{text:'Regime History',font:{{size:13,family:'Instrument Serif'}}}},barmode:'stack',
    yaxis:{{visible:false}},legend:{{orientation:'h',y:1.15,x:0.5,xanchor:'center',font:{{size:9}}}}}},{{displayModeBar:false,responsive:true}});

  document.getElementById('countryPanel').scrollIntoView({{behavior:'smooth',block:'start'}});
}}
window.addEventListener('load',()=>selectCountry('GBR'));
</script>
</body>
</html>"""

out = os.path.join(FDIR, 'dashboard_final.html')
with open(out, 'w') as f:
    f.write(html)

print(f"\n{'='*60}")
print(f"  Dashboard: {out}")
print(f"  Size: {os.path.getsize(out)/1024:.0f} KB")
print(f"{'='*60}")
print(f"\n  open {out}")
