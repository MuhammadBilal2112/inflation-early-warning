"""
==============================================================================
INFLATION REGIME DASHBOARD — INTERACTIVE (v3)
==============================================================================
Generates a self-contained HTML dashboard with:
  - World map (Plotly choropleth)
  - Country SEARCH/FILTER (type-ahead, click on map)
  - Per-country: transition prob, inflation components, regime history
  - Global composition over time
  - All interactive via embedded JavaScript — NO server needed

RUN: python3 Dashboard_Interactive.py
OUTPUT: inflation_regimes_project/outputs/figures/dashboard_interactive.html
==============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, json, os, re, warnings
warnings.filterwarnings('ignore')

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
MDIR = os.path.join(BASE_DIR, "outputs", "models")
FDIR = os.path.join(BASE_DIR, "outputs", "figures")

print("=" * 60)
print("  Building Interactive Dashboard...")
print("=" * 60)

# ============================================================
# 1. LOAD AND PREPARE ALL DATA
# ============================================================
print("\n  Loading data...")
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])
features_df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in features_df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    features_df = features_df.rename(columns=clean_map)

# Load model & compute transition probabilities
saved = joblib.load(os.path.join(MDIR, 'xgboost_target_up_2q.joblib'))
model = saved['model']
feature_cols = saved['features']
features_df['transition_prob'] = model.predict_proba(features_df[feature_cols])[:, 1]

# Merge probs into regime
regime = regime.merge(features_df[['country_code', 'date', 'transition_prob']],
                       on=['country_code', 'date'], how='left')

# Country names
cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']):
        cnames[r['country_code']] = str(r['country_name'])

REGIME = {
    0: ('Low', '#22c55e'),
    1: ('Low-energy', '#3b82f6'),
    2: ('Moderate', '#eab308'),
    3: ('Mod-energy', '#f97316'),
    4: ('Elevated', '#ef4444'),
    5: ('Crisis', '#a855f7'),
}

regime['regime_name'] = regime['regime'].map(
    lambda x: REGIME.get(int(x), ('?', '#777'))[0] if pd.notna(x) else 'No data')

# ============================================================
# 2. BUILD PER-COUNTRY JSON FOR JAVASCRIPT
# ============================================================
print("  Building country data...")

countries_json = {}
for cc in sorted(regime['country_code'].unique()):
    cc_r = regime[regime['country_code'] == cc].sort_values('date')
    cc_m = monthly[(monthly['country_code'] == cc) & (monthly['date'] >= '2010-01-01')].sort_values('date')
    if cc_r.empty:
        continue
    last = cc_r.iloc[-1]

    # Quarterly series
    q_dates, q_tp, q_h, q_f, q_e, q_reg = [], [], [], [], [], []
    for _, row in cc_r.iterrows():
        q_dates.append(row['date'].strftime('%Y-%m-%d'))
        q_tp.append(round(row['transition_prob'], 4) if pd.notna(row.get('transition_prob')) else None)
        q_h.append(round(row['hcpi_yoy'], 2) if pd.notna(row.get('hcpi_yoy')) else None)
        q_f.append(round(row['fcpi_yoy'], 2) if pd.notna(row.get('fcpi_yoy')) else None)
        q_e.append(round(row['ecpi_yoy'], 2) if pd.notna(row.get('ecpi_yoy')) else None)
        q_reg.append(int(row['regime']) if pd.notna(row.get('regime')) else None)

    # Monthly headline
    m_dates, m_vals = [], []
    for _, row in cc_m.iterrows():
        if pd.notna(row.get('hcpi_yoy')):
            m_dates.append(row['date'].strftime('%Y-%m-%d'))
            m_vals.append(round(row['hcpi_yoy'], 2))

    countries_json[cc] = {
        'n': cnames.get(cc, cc),
        'g': str(last.get('country_group', '')) if pd.notna(last.get('country_group')) else '',
        'rg': str(last.get('region', '')) if pd.notna(last.get('region')) else '',
        'cr': int(last['regime']) if pd.notna(last.get('regime')) else None,
        'ch': round(last['hcpi_yoy'], 1) if pd.notna(last.get('hcpi_yoy')) else None,
        'ctp': round(last['transition_prob'], 3) if pd.notna(last.get('transition_prob')) else None,
        'qd': q_dates[-40:], 'qt': q_tp[-40:], 'qh': q_h[-40:], 'qf': q_f[-40:],
        'qe': q_e[-40:], 'qr': q_reg[-40:],
        'md': m_dates[-120:], 'mv': m_vals[-120:],
    }

# Custom encoder to handle numpy float32/int64 types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

data_json = json.dumps(countries_json, cls=NumpyEncoder)
print(f"  Country data: {len(countries_json)} countries, {len(data_json)//1024} KB")

# ============================================================
# 3. BUILD WORLD MAP (Plotly)
# ============================================================
print("  Building world map...")
latest = regime.sort_values('date').groupby('country_code').last().reset_index()
latest['country_name'] = latest['country_code'].map(cnames)

fig_map = px.choropleth(
    latest, locations='country_code', color='regime_name',
    hover_name='country_name',
    hover_data={'regime_name': True, 'hcpi_yoy': ':.1f', 'transition_prob': ':.1%', 'country_code': False},
    color_discrete_map={v[0]: v[1] for v in REGIME.values()},
    labels={'regime_name': 'Regime', 'hcpi_yoy': 'Headline CPI %', 'transition_prob': 'P(transition)'},
)
fig_map.update_geos(
    showframe=False, showcoastlines=True, coastlinecolor='#334155',
    bgcolor='#0f172a', landcolor='#1e293b', oceancolor='#0f172a',
    projection_type='natural earth', showocean=True, lataxis_range=[-55, 80],
)
fig_map.update_layout(
    paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', height=460,
    margin=dict(l=0, r=0, t=0, b=0), font=dict(color='#e2e8f0'),
    legend=dict(font=dict(color='#e2e8f0', size=11), bgcolor='rgba(15,23,42,0.8)',
                title='Regime', yanchor='bottom', y=0, xanchor='left', x=0),
)
map_html = fig_map.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# 4. BUILD COMPOSITION CHART (Plotly)
# ============================================================
print("  Building composition chart...")
comp = regime.groupby([regime['date'].dt.year, 'regime']).size().unstack(fill_value=0)
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100

fig_comp = go.Figure()
for r_id in sorted(REGIME.keys()):
    rname, rcolor = REGIME[r_id]
    if r_id in comp_pct.columns:
        fig_comp.add_trace(go.Scatter(
            x=comp_pct.index, y=comp_pct[r_id], name=rname,
            stackgroup='one', mode='none', fillcolor=rcolor,
            line=dict(width=0.3, color='rgba(255,255,255,0.15)'),
        ))
for dt, lbl in [(2008, 'GFC'), (2020, 'COVID'), (2022, 'Ukraine')]:
    fig_comp.add_vline(x=dt, line_dash='dot', line_color='rgba(255,255,255,0.25)')
    fig_comp.add_annotation(x=dt, y=97, text=lbl, showarrow=False,
                             font=dict(size=9, color='rgba(255,255,255,0.45)'))
fig_comp.update_layout(
    paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', height=240,
    font=dict(color='#e2e8f0'), margin=dict(l=50, r=20, t=10, b=40),
    yaxis=dict(title='% of countries', range=[0, 100], gridcolor='#334155'),
    xaxis=dict(gridcolor='#334155'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font_size=10),
)
comp_html = fig_comp.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# 5. COMPUTE SUMMARY STATS
# ============================================================
n_low = int(((latest['regime'] == 0) | (latest['regime'] == 1)).sum())
n_elevated = int((latest['regime'] == 4).sum())
n_crisis = int((latest['regime'] == 5).sum())
avg_tp = latest['transition_prob'].mean()

# ============================================================
# 6. ASSEMBLE HTML WITH JAVASCRIPT INTERACTIVITY
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
  --bg: #0f172a; --card: #1e293b; --card-hover: #253449; --border: #334155;
  --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --accent-dim: rgba(56,189,248,0.12);
  --green: #22c55e; --red: #ef4444; --amber: #eab308; --purple: #a855f7; --orange: #f97316;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; }}

.header {{
  background: linear-gradient(135deg, #0f172a 0%, #1a3352 40%, #0f172a 100%);
  border-bottom: 1px solid var(--border); padding: 28px 40px;
}}
.header-inner {{ max-width:1440px; margin:0 auto; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; }}
.header h1 {{ font-size:24px; font-weight:700; color:white; letter-spacing:-0.5px; }}
.header .sub {{ font-size:12px; color:var(--muted); margin-top:3px; }}
.kpis {{ display:flex; gap:28px; }}
.kpi {{ text-align:center; }}
.kpi-val {{ font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:700; color:var(--accent); }}
.kpi-lbl {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.8px; }}

.container {{ max-width:1440px; margin:0 auto; padding:20px 40px; }}
.section {{ margin-bottom:24px; }}
.section-head {{ font-size:13px; font-weight:600; color:var(--accent); text-transform:uppercase;
  letter-spacing:1.2px; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid var(--border); }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
.grid {{ display:grid; gap:20px; }}
.grid-2 {{ grid-template-columns:1fr 1fr; }}
.grid-3 {{ grid-template-columns:1fr 1fr 1fr; }}

/* SEARCH */
.search-container {{
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:20px 24px; margin-bottom:24px; display:flex; align-items:center; gap:16px; flex-wrap:wrap;
}}
.search-container label {{ font-weight:600; color:var(--accent); font-size:14px; white-space:nowrap; }}
.search-input {{
  background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px 16px;
  color:var(--text); font-size:14px; font-family:'DM Sans',sans-serif; width:320px; outline:none;
  transition: border-color 0.2s;
}}
.search-input:focus {{ border-color:var(--accent); }}
.search-input::placeholder {{ color:var(--muted); }}

.dropdown {{
  position:absolute; top:100%; left:0; right:0; background:var(--card); border:1px solid var(--border);
  border-radius:0 0 8px 8px; max-height:300px; overflow-y:auto; z-index:100; display:none;
}}
.dropdown-item {{
  padding:10px 16px; cursor:pointer; font-size:13px; display:flex; justify-content:space-between;
  border-bottom:1px solid rgba(51,65,85,0.5); transition:background 0.15s;
}}
.dropdown-item:hover {{ background:var(--card-hover); }}
.dropdown-item .regime-badge {{
  padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; color:white;
}}
.search-wrap {{ position:relative; flex:1; max-width:400px; }}

/* COUNTRY PANEL */
.country-header {{
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;
  margin-bottom:16px;
}}
.country-title {{ font-size:20px; font-weight:700; }}
.country-badges {{ display:flex; gap:8px; }}
.badge {{
  padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600;
  display:inline-flex; align-items:center; gap:4px;
}}

/* CHART CONTAINERS */
.chart-box {{ min-height:280px; position:relative; }}
.chart-box .placeholder {{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  color:var(--muted); font-size:14px;
}}

/* NOTES */
.note {{
  background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.15);
  border-radius:6px; padding:14px 18px; font-size:12px; color:var(--muted); line-height:1.7;
  margin-bottom:20px;
}}
.note strong {{ color:var(--text); }}
.note.warn {{ border-color:rgba(234,179,8,0.2); background:rgba(234,179,8,0.04); }}

.footer {{
  text-align:center; padding:24px; color:#475569; font-size:11px;
  border-top:1px solid var(--border); margin-top:24px;
}}

/* Stats strip */
.stats {{ display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }}
.stat {{
  background:var(--card); border:1px solid var(--border); border-radius:6px;
  padding:12px 18px; flex:1; min-width:160px; display:flex; align-items:center; gap:10px;
}}
.stat-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.stat-val {{ font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:600; }}
.stat-lbl {{ font-size:11px; color:var(--muted); }}

@media (max-width:900px) {{
  .grid-2, .grid-3 {{ grid-template-columns:1fr; }}
  .container {{ padding:16px; }}
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header"><div class="header-inner">
  <div>
    <h1>Inflation Regime Early Warning System</h1>
    <div class="sub">Machine Learning Prediction of Regime Transitions &mdash; 135 Countries &mdash; 1971&ndash;2025</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-val">0.839</div><div class="kpi-lbl">AUC-ROC</div></div>
    <div class="kpi"><div class="kpi-val" style="color:var(--green)">51%</div><div class="kpi-lbl">Detection</div></div>
    <div class="kpi"><div class="kpi-val">135</div><div class="kpi-lbl">Countries</div></div>
    <div class="kpi"><div class="kpi-val" style="color:var(--purple)">6</div><div class="kpi-lbl">Regimes</div></div>
  </div>
</div></div>

<div class="container">

  <!-- STATS -->
  <div class="stats">
    <div class="stat"><div class="stat-dot" style="background:var(--green)"></div>
      <div><div class="stat-val">{n_low}</div><div class="stat-lbl">Low inflation</div></div></div>
    <div class="stat"><div class="stat-dot" style="background:var(--red)"></div>
      <div><div class="stat-val">{n_elevated}</div><div class="stat-lbl">Elevated</div></div></div>
    <div class="stat"><div class="stat-dot" style="background:var(--purple)"></div>
      <div><div class="stat-val">{n_crisis}</div><div class="stat-lbl">Crisis</div></div></div>
    <div class="stat"><div class="stat-dot" style="background:var(--accent)"></div>
      <div><div class="stat-val">{avg_tp:.0%}</div><div class="stat-lbl">Avg transition P</div></div></div>
  </div>

  <!-- MAP -->
  <div class="section">
    <div class="section-head">Global Regime Map</div>
    <div class="card">{map_html}</div>
  </div>

  <!-- NOTES -->
  <div class="note">
    <strong>How to read:</strong> Countries are classified into 6 inflation regimes via Gaussian Mixture Models.
    The <strong>transition probability</strong> is the XGBoost model's estimate of an upward regime shift within 2 quarters.
    Trained on 1971&ndash;2014; tested on COVID &amp; Ukraine crises. Alert: 30%+. High risk: 50%+.
  </div>
  <div class="note warn">
    <strong>&#9888; Grey countries:</strong> 53 countries lack required CPI components.
    Notable: <strong>Australia</strong> (quarterly CPI only), <strong>Iran</strong> (no food CPI),
    <strong>Venezuela &amp; Zimbabwe</strong> (incomplete data). Gulf states (Saudi Arabia, UAE, Qatar) are covered.
  </div>

  <!-- COMPOSITION -->
  <div class="section">
    <div class="section-head">Regime Composition Over Time</div>
    <div class="card">{comp_html}</div>
  </div>

  <!-- COUNTRY SEARCH -->
  <div class="search-container">
    <label>&#128269; Country Deep Dive</label>
    <div class="search-wrap">
      <input type="text" class="search-input" id="countrySearch"
             placeholder="Type a country name or code..." autocomplete="off">
      <div class="dropdown" id="dropdown"></div>
    </div>
    <span id="selectedInfo" style="color:var(--muted); font-size:13px;">Select a country to explore</span>
  </div>

  <!-- COUNTRY PANEL (populated by JS) -->
  <div id="countryPanel" style="display:none;">
    <div class="country-header">
      <div class="country-title" id="countryTitle"></div>
      <div class="country-badges" id="countryBadges"></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:20px;">
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
  Dissertation: <em>Predicting Inflation Regimes Using Machine Learning: A Global Panel Approach</em><br>
  XGBoost &bull; Bootstrap CI [0.816, 0.860] &bull; p&lt;0.001 vs Logistic Regression &bull;
  World Bank CMO &amp; Fiscal Space data &bull; 72,321 monthly observations
</div>

<script>
// ============================================================
// EMBEDDED COUNTRY DATA
// ============================================================
const DATA = {data_json};

const REGIMES = {{
  0: {{name:'Low', color:'#22c55e'}},
  1: {{name:'Low-energy', color:'#3b82f6'}},
  2: {{name:'Moderate', color:'#eab308'}},
  3: {{name:'Mod-energy', color:'#f97316'}},
  4: {{name:'Elevated', color:'#ef4444'}},
  5: {{name:'Crisis', color:'#a855f7'}},
}};

const CHART_LAYOUT = {{
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#1e293b',
  font:{{color:'#e2e8f0', family:'DM Sans, sans-serif', size:11}},
  margin:{{l:50, r:20, t:36, b:36}},
  xaxis:{{gridcolor:'#334155', zerolinecolor:'#334155'}},
  yaxis:{{gridcolor:'#334155', zerolinecolor:'#334155'}},
}};

// ============================================================
// SEARCH FUNCTIONALITY
// ============================================================
const searchInput = document.getElementById('countrySearch');
const dropdown = document.getElementById('dropdown');
const entries = Object.entries(DATA).map(([code, d]) => ({{code, name:d.n, regime:d.cr}}));
entries.sort((a,b) => a.name.localeCompare(b.name));

searchInput.addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  if (q.length < 1) {{ dropdown.style.display='none'; return; }}

  const matches = entries.filter(e =>
    e.name.toLowerCase().includes(q) || e.code.toLowerCase().includes(q)
  ).slice(0, 15);

  if (matches.length === 0) {{ dropdown.style.display='none'; return; }}

  dropdown.innerHTML = matches.map(m => {{
    const r = REGIMES[m.regime] || {{name:'?', color:'#666'}};
    return `<div class="dropdown-item" onclick="selectCountry('${{m.code}}')">
      <span>${{m.name}} (${{m.code}})</span>
      <span class="regime-badge" style="background:${{r.color}}">${{r.name}}</span>
    </div>`;
  }}).join('');
  dropdown.style.display='block';
}});

searchInput.addEventListener('focus', () => {{ if(searchInput.value) searchInput.dispatchEvent(new Event('input')); }});
document.addEventListener('click', (e) => {{ if(!e.target.closest('.search-wrap')) dropdown.style.display='none'; }});

// ============================================================
// RENDER COUNTRY
// ============================================================
function selectCountry(code) {{
  dropdown.style.display = 'none';
  const d = DATA[code];
  if (!d) return;

  searchInput.value = `${{d.n}} (${{code}})`;
  document.getElementById('countryPanel').style.display = 'block';

  // Header
  const r = REGIMES[d.cr] || {{name:'?', color:'#666'}};
  document.getElementById('countryTitle').textContent = d.n;
  document.getElementById('countryBadges').innerHTML = `
    <span class="badge" style="background:${{r.color}}">R${{d.cr}}: ${{r.name}}</span>
    ${{d.ch !== null ? `<span class="badge" style="background:var(--card);border:1px solid var(--border)">CPI: ${{d.ch}}%</span>` : ''}}
    ${{d.ctp !== null ? `<span class="badge" style="background:${{d.ctp > 0.5 ? 'var(--red)' : d.ctp > 0.3 ? 'var(--orange)' : 'var(--card)'}};border:1px solid var(--border)">P(trans): ${{(d.ctp*100).toFixed(0)}}%</span>` : ''}}
    <span class="badge" style="background:var(--card);border:1px solid var(--border)">${{d.g || ''}}</span>
  `;
  document.getElementById('selectedInfo').textContent = `Showing: ${{d.n}}`;

  // 1. Transition probability chart
  const tpValid = d.qd.map((dt, i) => d.qt[i] !== null ? i : null).filter(x => x !== null);
  Plotly.newPlot('chartTransition', [{{
    x: tpValid.map(i => d.qd[i]),
    y: tpValid.map(i => d.qt[i]),
    type:'scatter', mode:'lines+markers',
    line:{{color:'#38bdf8', width:2.5}}, marker:{{size:4}},
    fill:'tozeroy', fillcolor:'rgba(56,189,248,0.08)',
    name:'P(transition)',
  }}], {{
    ...CHART_LAYOUT,
    title:{{text:'Transition Probability (2Q ahead)', font:{{size:13}}}},
    yaxis:{{...CHART_LAYOUT.yaxis, title:'P(upward transition)', range:[0,1.02]}},
    shapes:[
      {{type:'line',y0:0.3,y1:0.3,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#eab308',width:1}}}},
      {{type:'line',y0:0.5,y1:0.5,x0:0,x1:1,xref:'paper',line:{{dash:'dash',color:'#ef4444',width:1}}}},
      {{type:'rect',y0:0.3,y1:0.5,x0:0,x1:1,xref:'paper',fillcolor:'rgba(234,179,8,0.06)',line:{{width:0}}}},
      {{type:'rect',y0:0.5,y1:1,x0:0,x1:1,xref:'paper',fillcolor:'rgba(239,68,68,0.06)',line:{{width:0}}}},
    ],
  }}, {{displayModeBar:false, responsive:true}});

  // 2. Inflation components
  const traces2 = [];
  const cols = [['qh','Headline','#38bdf8'],['qf','Food','#22c55e'],['qe','Energy','#ef4444']];
  cols.forEach(([key,name,color]) => {{
    const valid = d.qd.map((dt,i) => d[key][i] !== null ? i : null).filter(x=>x!==null);
    if(valid.length > 0) {{
      traces2.push({{
        x:valid.map(i=>d.qd[i]), y:valid.map(i=>d[key][i]),
        type:'scatter', mode:'lines', name, line:{{color, width:2}},
      }});
    }}
  }});
  Plotly.newPlot('chartInflation', traces2, {{
    ...CHART_LAYOUT,
    title:{{text:'Inflation Components (quarterly)', font:{{size:13}}}},
    yaxis:{{...CHART_LAYOUT.yaxis, title:'YoY %'}},
    legend:{{orientation:'h', y:1.12, x:0.5, xanchor:'center', font:{{size:10}}}},
    shapes:[{{type:'line',y0:0,y1:0,x0:0,x1:1,xref:'paper',line:{{color:'rgba(255,255,255,0.15)',width:0.5}}}}],
  }}, {{displayModeBar:false, responsive:true}});

  // 3. Monthly headline
  if (d.md.length > 0) {{
    Plotly.newPlot('chartMonthly', [{{
      x:d.md, y:d.mv, type:'scatter', mode:'lines',
      line:{{color:'#38bdf8', width:1.5}},
      fill:'tozeroy', fillcolor:'rgba(56,189,248,0.05)',
    }}], {{
      ...CHART_LAYOUT,
      title:{{text:'Monthly Headline CPI', font:{{size:13}}}},
      yaxis:{{...CHART_LAYOUT.yaxis, title:'YoY %'}},
      showlegend:false,
    }}, {{displayModeBar:false, responsive:true}});
  }} else {{
    document.getElementById('chartMonthly').innerHTML = '<div class="placeholder">No monthly data available</div>';
  }}

  // 4. Regime history (color blocks)
  const regTraces = [];
  Object.entries(REGIMES).forEach(([rid, info]) => {{
    const indices = d.qr.map((r,i) => r == rid ? i : null).filter(x=>x!==null);
    if(indices.length > 0) {{
      regTraces.push({{
        x:indices.map(i=>d.qd[i]), y:indices.map(()=>1),
        type:'bar', name:`R${{rid}}: ${{info.name}}`,
        marker:{{color:info.color}}, width: 86400000*85,
      }});
    }}
  }});
  Plotly.newPlot('chartRegime', regTraces, {{
    ...CHART_LAYOUT,
    title:{{text:'Regime History', font:{{size:13}}}},
    barmode:'stack', yaxis:{{visible:false}},
    legend:{{orientation:'h', y:1.15, x:0.5, xanchor:'center', font:{{size:9}}}},
  }}, {{displayModeBar:false, responsive:true}});

  // Scroll to panel
  document.getElementById('countryPanel').scrollIntoView({{behavior:'smooth', block:'start'}});
}}

// Auto-load UK on page open
window.addEventListener('load', () => selectCountry('GBR'));
</script>
</body>
</html>"""

out_path = os.path.join(FDIR, 'dashboard_interactive.html')
with open(out_path, 'w') as f:
    f.write(html_page)

print(f"\n{'='*60}")
print(f"  Dashboard generated!")
print(f"  File: {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024:.0f} KB")
print(f"{'='*60}")
print(f"\n  Open in browser:")
print(f"  open {out_path}")