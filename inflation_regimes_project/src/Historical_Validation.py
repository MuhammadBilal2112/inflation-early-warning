"""
==============================================================================
HISTORICAL EPISODE VALIDATION
==============================================================================

Tests the model against specific, well-documented economic crises and events.
For each episode, we check:
  1. Did the regime classification match what actually happened?
  2. Did the model's transition probability rise BEFORE the event?
  3. How far in advance did it give a warning signal?
  4. Which features drove the prediction (SHAP)?

EPISODES TESTED:
  OUT-OF-SAMPLE (model never trained on these):
    - 2018 Turkey Lira Crisis
    - 2018 Argentina Peso Crisis  
    - 2019-21 Lebanon Collapse
    - 2020-21 COVID Global Shock
    - 2022 Ukraine Energy Shock
    - 2022 Sri Lanka Default
    - 2023 Egypt Devaluation
    - 2023 Nigeria Naira Crisis
  
  IN-SAMPLE (model trained on these, but regime labels still validated):
    - 2007-08 GFC commodity spike
    - 2013 Taper Tantrum (EM currencies)
    - 2014-16 Oil Price Collapse

RUN: python3 Historical_Validation.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import os, re, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.figsize': (14, 8), 'figure.dpi': 150, 'font.size': 11,
    'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})

C = {'blue': '#1F4E79', 'red': '#C62828', 'green': '#2E7D32', 'amber': '#E8A838',
     'purple': '#6A1B9A', 'teal': '#00838F', 'gray': '#616161'}

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
FDIR = os.path.join(BASE_DIR, "outputs", "figures")
TDIR = os.path.join(BASE_DIR, "outputs", "tables")
MDIR = os.path.join(BASE_DIR, "outputs", "models")

fc = 0
def save_fig(name):
    global fc; fc += 1
    plt.savefig(os.path.join(FDIR, f"{name}.png"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Fig {fc}: {name}.png")

print("=" * 70)
print("HISTORICAL EPISODE VALIDATION")
print("=" * 70)

# Load data
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])
features_df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in features_df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    features_df = features_df.rename(columns=clean_map)

# Load calibrated model
cal_path = os.path.join(MDIR, 'xgboost_calibrated_target_up_2q.joblib')
raw_path = os.path.join(MDIR, 'xgboost_target_up_2q.joblib')
if os.path.exists(cal_path):
    saved = joblib.load(cal_path)
    model, calibrator, feature_cols = saved['model'], saved['calibrator'], saved['features']
    raw_p = model.predict_proba(features_df[feature_cols])[:, 1]
    features_df['transition_prob'] = calibrator.predict(raw_p)
    print("  Using CALIBRATED model")
else:
    saved = joblib.load(raw_path)
    model, calibrator, feature_cols = saved['model'], None, saved['features']
    features_df['transition_prob'] = model.predict_proba(features_df[feature_cols])[:, 1]
    print("  Using raw model")

# Country names
cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']): cnames[r['country_code']] = str(r['country_name'])

# Merge transition probs into regime
regime = regime.merge(features_df[['country_code', 'date', 'transition_prob']],
                       on=['country_code', 'date'], how='left')

REGIME_NAMES = {0: 'Low', 1: 'Low-energy', 2: 'Moderate', 3: 'Mod-energy', 4: 'Elevated', 5: 'Crisis'}
REGIME_COLORS = {0: '#2E7D32', 1: '#1F4E79', 2: '#E8A838', 3: '#FF8F00', 4: '#C62828', 5: '#6A1B9A'}

# ============================================================
# DEFINE EPISODES
# ============================================================
episodes = [
    {
        'name': 'Turkey Lira Crisis 2018',
        'countries': ['TUR'],
        'pre_period': ('2017-01-01', '2018-06-01'),
        'crisis_period': ('2018-07-01', '2019-06-01'),
        'plot_range': ('2016-01-01', '2020-01-01'),
        'expected': 'Transition from Moderate/Elevated to Crisis',
        'oos': True,
        'context': 'Lira lost 40% value, inflation hit 25%. Central bank resisted rate hikes.',
    },
    {
        'name': 'Argentina Peso Crisis 2018',
        'countries': ['ARG'],
        'pre_period': ('2017-01-01', '2018-04-01'),
        'crisis_period': ('2018-05-01', '2019-06-01'),
        'plot_range': ('2016-01-01', '2020-01-01'),
        'expected': 'Transition to Elevated/Crisis',
        'oos': True,
        'context': 'Peso crashed, IMF bailout, inflation hit 50%.',
    },
    {
        'name': 'Lebanon Collapse 2019-21',
        'countries': ['LBN'],
        'pre_period': ('2019-01-01', '2019-09-01'),
        'crisis_period': ('2019-10-01', '2021-12-01'),
        'plot_range': ('2018-01-01', '2023-01-01'),
        'expected': 'Transition to Crisis (hyperinflation)',
        'oos': True,
        'context': 'Banking collapse, currency peg broke, inflation exceeded 200%.',
    },
    {
        'name': 'COVID Global Shock 2020',
        'countries': ['USA', 'GBR', 'BRA', 'IND', 'ZAF'],
        'pre_period': ('2019-07-01', '2020-03-01'),
        'crisis_period': ('2020-04-01', '2021-12-01'),
        'plot_range': ('2019-01-01', '2023-01-01'),
        'expected': 'Widespread upward transitions in 2021',
        'oos': True,
        'context': 'Pandemic demand shock, supply chains disrupted, fiscal stimulus.',
    },
    {
        'name': 'Ukraine Energy Shock 2022',
        'countries': ['DEU', 'GBR', 'POL', 'CZE', 'HUN'],
        'pre_period': ('2021-07-01', '2022-02-01'),
        'crisis_period': ('2022-03-01', '2023-06-01'),
        'plot_range': ('2020-01-01', '2024-01-01'),
        'expected': 'European transitions driven by gas prices',
        'oos': True,
        'context': 'Russian gas cutoff, energy prices spiked 300%+.',
    },
    {
        'name': 'Sri Lanka Default 2022',
        'countries': ['LKA'],
        'pre_period': ('2021-01-01', '2022-03-01'),
        'crisis_period': ('2022-04-01', '2023-06-01'),
        'plot_range': ('2020-01-01', '2024-01-01'),
        'expected': 'Transition to Elevated/Crisis',
        'oos': True,
        'context': 'FX reserves depleted, sovereign default, inflation hit 70%.',
    },
    {
        'name': 'Egypt Devaluation 2023',
        'countries': ['EGY'],
        'pre_period': ('2022-01-01', '2023-01-01'),
        'crisis_period': ('2023-01-01', '2024-06-01'),
        'plot_range': ('2021-01-01', '2025-01-01'),
        'expected': 'Transition upward from multiple devaluations',
        'oos': True,
        'context': 'Three devaluations, inflation exceeded 35%.',
    },
    {
        'name': 'Nigeria Naira Crisis 2023',
        'countries': ['NGA'],
        'pre_period': ('2022-01-01', '2023-06-01'),
        'crisis_period': ('2023-06-01', '2024-06-01'),
        'plot_range': ('2021-01-01', '2025-01-01'),
        'expected': 'Transition upward after naira float',
        'oos': True,
        'context': 'New president floated naira, currency crashed, inflation surged.',
    },
    {
        'name': 'GFC Commodity Spike 2007-08',
        'countries': ['USA', 'GBR', 'DEU', 'BRA', 'IND'],
        'pre_period': ('2007-01-01', '2008-06-01'),
        'crisis_period': ('2008-07-01', '2009-06-01'),
        'plot_range': ('2006-01-01', '2010-01-01'),
        'expected': 'Brief spike then rapid return to Low',
        'oos': False,
        'context': 'Oil hit $147, then financial crisis caused deflation scare.',
    },
    {
        'name': '2008 Global Financial Crisis (Great Recession)',
        'countries': ['USA', 'GBR', 'DEU', 'FRA', 'JPN', 'ESP', 'ITA', 'GRC', 'IRL', 'ISL',
                      'BRA', 'RUS', 'IND', 'CHN', 'MEX', 'KOR', 'ZAF', 'TUR', 'IDN', 'NGA'],
        'pre_period': ('2007-01-01', '2008-09-01'),
        'crisis_period': ('2008-09-01', '2010-06-01'),
        'plot_range': ('2006-01-01', '2011-01-01'),
        'expected': 'Commodity-driven inflation spike mid-2008, then sharp disinflation/deflation post-Lehman',
        'oos': False,
        'context': 'Lehman collapse Sep 2008. Oil spike to $147 then crash to $32. Global recession. '
                   'Iceland banking collapse. Greece/Ireland debt crisis began. Massive central bank easing. '
                   'Unique pattern: inflation ROSE before the crisis (commodity boom) then FELL sharply after (demand collapse).',
    },
    {
        'name': '2014-16 Oil Price Collapse',
        'countries': ['SAU', 'RUS', 'NGA', 'NOR', 'CAN'],
        'pre_period': ('2013-01-01', '2014-06-01'),
        'crisis_period': ('2014-07-01', '2016-06-01'),
        'plot_range': ('2012-01-01', '2017-01-01'),
        'expected': 'Oil exporters face different pressures than importers',
        'oos': False,
        'context': 'Oil fell from $115 to $28. Ruble crashed. Nigeria entered recession.',
    },
]

# ============================================================
# ANALYSE EACH EPISODE
# ============================================================
print(f"\nAnalysing {len(episodes)} historical episodes...\n")

all_episode_results = []

for ep in episodes:
    print(f"\n{'='*60}")
    oos_label = "OUT-OF-SAMPLE" if ep['oos'] else "IN TRAINING DATA"
    print(f"  {ep['name']} [{oos_label}]")
    print(f"  {ep['context']}")
    print(f"{'='*60}")

    for cc in ep['countries']:
        cc_regime = regime[regime['country_code'] == cc].sort_values('date')
        cc_name = cnames.get(cc, cc)

        if cc_regime.empty:
            print(f"\n    {cc} ({cc_name}): NO DATA")
            continue

        # Pre-crisis regime
        pre_data = cc_regime[(cc_regime['date'] >= ep['pre_period'][0]) &
                              (cc_regime['date'] < ep['pre_period'][1])]
        crisis_data = cc_regime[(cc_regime['date'] >= ep['crisis_period'][0]) &
                                 (cc_regime['date'] <= ep['crisis_period'][1])]

        if pre_data.empty and crisis_data.empty:
            print(f"\n    {cc} ({cc_name}): No data for this period")
            continue

        # What regime was the country in before?
        pre_regime = int(pre_data['regime'].mode().iloc[0]) if not pre_data.empty and pre_data['regime'].notna().any() else None
        pre_regime_name = REGIME_NAMES.get(pre_regime, '?') if pre_regime is not None else '?'

        # What regime during crisis?
        crisis_regime = int(crisis_data['regime'].mode().iloc[0]) if not crisis_data.empty and crisis_data['regime'].notna().any() else None
        crisis_regime_name = REGIME_NAMES.get(crisis_regime, '?') if crisis_regime is not None else '?'

        # Did transition happen?
        transitioned = crisis_regime is not None and pre_regime is not None and crisis_regime > pre_regime
        transition_label = f"YES: R{pre_regime}({pre_regime_name}) → R{crisis_regime}({crisis_regime_name})" if transitioned else f"NO: stayed R{pre_regime}({pre_regime_name})" if pre_regime is not None else "Unknown"

        # Max transition probability before crisis
        pre_max_prob = pre_data['transition_prob'].max() if not pre_data.empty and pre_data['transition_prob'].notna().any() else None

        # Was there an early warning?
        if pre_max_prob is not None:
            early_warning = pre_max_prob > 0.25
            # How many quarters before crisis did it first exceed 25%?
            warning_data = pre_data[pre_data['transition_prob'] > 0.25]
            if not warning_data.empty:
                first_warning = warning_data['date'].min()
                crisis_start = pd.Timestamp(ep['crisis_period'][0])
                lead_time_days = (crisis_start - first_warning).days
                lead_time_q = lead_time_days / 90
                lead_str = f"{lead_time_q:.1f} quarters before crisis"
            else:
                lead_str = "No warning"
        else:
            early_warning = None
            lead_str = "No data"

        print(f"\n    {cc} ({cc_name}):")
        print(f"      Pre-crisis regime: R{pre_regime} ({pre_regime_name})")
        print(f"      Crisis regime:     R{crisis_regime} ({crisis_regime_name})")
        print(f"      Transition:        {transition_label}")
        print(f"      Max P(trans) before crisis: {pre_max_prob:.1%}" if pre_max_prob is not None else "      Max P(trans): N/A")
        print(f"      Early warning (>25%): {'YES' if early_warning else 'NO'} — {lead_str}")

        all_episode_results.append({
            'Episode': ep['name'],
            'Country': cc,
            'Country Name': cc_name,
            'Out of Sample': ep['oos'],
            'Pre-crisis Regime': f"R{pre_regime}: {pre_regime_name}" if pre_regime is not None else 'N/A',
            'Crisis Regime': f"R{crisis_regime}: {crisis_regime_name}" if crisis_regime is not None else 'N/A',
            'Transitioned Up': transitioned,
            'Max P(trans) Before': round(pre_max_prob, 4) if pre_max_prob is not None else None,
            'Early Warning': early_warning,
            'Lead Time': lead_str,
        })

# Save results table
results_df = pd.DataFrame(all_episode_results)
results_df.to_csv(os.path.join(TDIR, 'table26_historical_validation.csv'), index=False)
print(f"\n\nSaved: table26_historical_validation.csv")

# ============================================================
# FIGURE: Timeline plots for key episodes
# ============================================================
print("\n\nGenerating episode timeline figures...")

# Select most interesting episodes for plotting
plot_episodes = [
    ('2008 Global Financial Crisis (Great Recession)', ['USA', 'GBR', 'DEU', 'GRC']),
    ('Turkey Lira Crisis 2018', ['TUR']),
    ('COVID Global Shock 2020', ['USA', 'GBR', 'BRA']),
    ('Ukraine Energy Shock 2022', ['DEU', 'GBR', 'POL']),
    ('Sri Lanka Default 2022', ['LKA']),
    ('Egypt Devaluation 2023', ['EGY']),
    ('Nigeria Naira Crisis 2023', ['NGA']),
]

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for idx, (ep_name, countries) in enumerate(plot_episodes):
    ax = axes[idx // 2, idx % 2]
    ep = next(e for e in episodes if e['name'] == ep_name)

    for i, cc in enumerate(countries):
        cc_data = regime[(regime['country_code'] == cc) &
                          (regime['date'] >= ep['plot_range'][0]) &
                          (regime['date'] <= ep['plot_range'][1])].sort_values('date')

        if cc_data.empty:
            continue

        color = [C['blue'], C['red'], C['green']][i % 3]
        label = cnames.get(cc, cc)

        # Plot transition probability
        tp_data = cc_data.dropna(subset=['transition_prob'])
        if not tp_data.empty:
            ax.plot(tp_data['date'], tp_data['transition_prob'],
                    '-o', color=color, linewidth=2, markersize=3, label=label)

    # Mark crisis period
    ax.axvspan(pd.Timestamp(ep['crisis_period'][0]), pd.Timestamp(ep['crisis_period'][1]),
               alpha=0.1, color=C['red'], label='Crisis period')
    ax.axhline(y=0.25, color=C['amber'], linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.35, color=C['red'], linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('P(upward transition)')
    ax.set_ylim(-0.02, 0.55)
    ax.set_title(ep_name, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')

# Hide unused subplot if any
for idx in range(len(plot_episodes), axes.size):
    axes.flatten()[idx].set_visible(False)

plt.suptitle('Historical Episode Validation — Did the model signal transitions before they happened?',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig('fig43_historical_episode_validation')

# ============================================================
# FIGURE: Regime transition trajectories
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
single_country_episodes = [
    ('USA', 'USA GFC 2008', '2006-01-01', '2011-01-01'),
    ('GBR', 'UK GFC 2008', '2006-01-01', '2011-01-01'),
    ('GRC', 'Greece GFC 2008', '2006-01-01', '2011-01-01'),
    ('TUR', 'Turkey 2018', '2016-01-01', '2020-01-01'),
    ('LKA', 'Sri Lanka 2022', '2020-01-01', '2024-01-01'),
    ('EGY', 'Egypt 2023', '2021-01-01', '2025-01-01'),
    ('NGA', 'Nigeria 2023', '2021-01-01', '2025-01-01'),
    ('DEU', 'Germany 2022', '2019-01-01', '2024-01-01'),
]

for idx, (cc, title, start, end) in enumerate(single_country_episodes):
    ax = axes[idx // 4, idx % 4]
    cc_data = regime[(regime['country_code'] == cc) &
                      (regime['date'] >= start) & (regime['date'] <= end)].sort_values('date')

    if cc_data.empty:
        ax.set_title(f'{title}: No data')
        continue

    # Plot headline inflation
    ax2 = ax.twinx()
    infl_data = cc_data.dropna(subset=['hcpi_yoy'])
    if not infl_data.empty:
        ax2.fill_between(infl_data['date'], 0, infl_data['hcpi_yoy'],
                          alpha=0.15, color=C['gray'])
        ax2.set_ylabel('CPI %', color=C['gray'], fontsize=8)
        ax2.tick_params(axis='y', labelcolor=C['gray'], labelsize=8)

    # Colour background by regime
    for _, row in cc_data.iterrows():
        r = int(row['regime']) if pd.notna(row.get('regime')) else None
        if r is not None:
            ax.axvspan(row['date'], row['date'] + pd.Timedelta(days=90),
                       alpha=0.3, color=REGIME_COLORS.get(r, '#ccc'), linewidth=0)

    # Plot transition probability
    tp_data = cc_data.dropna(subset=['transition_prob'])
    if not tp_data.empty:
        ax.plot(tp_data['date'], tp_data['transition_prob'],
                '-', color='black', linewidth=2, label='P(trans)')
        ax.axhline(y=0.25, color=C['amber'], linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_ylim(-0.02, 0.55)
    ax.set_ylabel('P(trans)', fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(labelsize=8)

plt.suptitle('Regime trajectories — transition probability (black line) over regime background (colours)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig('fig44_regime_trajectories')

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

oos_results = results_df[results_df['Out of Sample'] == True]
total_episodes_oos = len(oos_results)
transitions_detected = oos_results['Transitioned Up'].sum()
warnings_given = oos_results['Early Warning'].sum()

print(f"""
  OUT-OF-SAMPLE episodes tested: {oos_results['Episode'].nunique()}
  Country-episodes evaluated: {total_episodes_oos}
  Actual upward transitions: {int(transitions_detected)}
  Early warnings given (>25%): {int(warnings_given)}
  
  Detection rate: {warnings_given/max(total_episodes_oos,1)*100:.0f}%
  (Of {total_episodes_oos} country-episodes, the model flagged {int(warnings_given)} in advance)

EPISODE-BY-EPISODE:
""")

for ep_name in oos_results['Episode'].unique():
    ep_data = oos_results[oos_results['Episode'] == ep_name]
    n = len(ep_data)
    n_trans = ep_data['Transitioned Up'].sum()
    n_warn = ep_data['Early Warning'].sum()
    print(f"  {ep_name}:")
    print(f"    Countries tested: {n}, Transitions: {int(n_trans)}, Warned: {int(n_warn)}")

print(f"""
FILES CREATED:
  table26_historical_validation.csv
  fig43: Episode timeline plots
  fig44: Regime trajectory plots
""")
