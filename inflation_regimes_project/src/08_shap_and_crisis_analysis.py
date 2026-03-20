"""
==============================================================================
STEPS 8+9: SHAP INTERPRETABILITY AND CRISIS EPISODE ANALYSIS
==============================================================================

Computes SHAP interpretability and crisis episode analysis.

  Part A — SHAP Analysis (Step 9):
    1. Computes SHAP values for the best model (XGBoost, 2Q horizon)
    2. Global feature importance ranking (SHAP-based, not Gini)
    3. SHAP summary plot (beeswarm) showing direction of effects
    4. SHAP by country group (AE vs EMDE comparison)
    5. SHAP dependence plots for key variables (non-linear effects)
    6. Individual country waterfall plots (per-prediction explanations)
  
  Part B — Crisis Episode Analysis (Step 8):
    7. COVID vs Ukraine episode comparison using SHAP
    8. Transition probability timelines for selected countries
    9. "Fingerprint" comparison: what drove each crisis?
    10. Pseudo-real-time early warning evaluation

OUTPUT:
  - outputs/figures/fig26-fig33 (8 figures)
  - outputs/tables/table15-table18 (4 tables)

REQUIRES: Steps 2-7 Revised completed (models must be saved)
RUN: python3 Step09_SHAP_and_Crisis_Analysis.py
TIME: 10-20 minutes (SHAP computation is intensive)
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os, warnings, time, re
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 150, 'font.size': 11,
    'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})

C = {'blue': '#1F4E79', 'red': '#C62828', 'green': '#2E7D32', 'amber': '#E8A838',
     'purple': '#6A1B9A', 'teal': '#00838F', 'gray': '#616161'}

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
FDIR = os.path.join(BASE_DIR, "outputs", "figures")
TDIR = os.path.join(BASE_DIR, "outputs", "tables")
MDIR = os.path.join(BASE_DIR, "outputs", "models")
for d in [FDIR, TDIR, MDIR]:
    os.makedirs(d, exist_ok=True)

fc = 0
def save_fig(name):
    global fc; fc += 1
    plt.savefig(os.path.join(FDIR, f"{name}.png"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Fig {fc}: {name}.png")

total_start = time.time()

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
print("Loading data and models...")

df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df = df.rename(columns=clean_map)

# Load the best model (XGBoost, 2Q horizon from revised Step 7)
model_path = os.path.join(MDIR, 'xgboost_target_up_2q.joblib')
saved = joblib.load(model_path)
model = saved['model']
feature_cols = saved['features']
imputer = saved['imputer']

print(f"  Data: {df.shape}")
print(f"  Model: {type(model).__name__} with {len(feature_cols)} features")

# Prepare test data
target_col = 'target_up_2q'
test_mask = df['split'].isin(['test_covid', 'test_ukraine'])
test_data = df[test_mask].dropna(subset=[target_col]).copy()
X_test = test_data[feature_cols].copy()
y_test = test_data[target_col].values

# For tree-based models (XGBoost), we don't need to impute/scale
# but SHAP TreeExplainer handles NaN natively
print(f"  Test data: {len(test_data)} obs ({test_data['country_code'].nunique()} countries)")


# ============================================================
# PART A: SHAP ANALYSIS (STEP 9)
# ============================================================
print("\n" + "=" * 70)
print("PART A: SHAP ANALYSIS")
print("=" * 70)

# ============================================================
# 1. Compute SHAP values
# ============================================================
print("\n  Computing SHAP values (this takes a few minutes)...")
t0 = time.time()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values might be a list [class_0, class_1]
# or a single array. We want class 1 (positive = upward transition)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Class 1 (upward transition)
else:
    shap_vals = shap_values

print(f"  SHAP computation done in {time.time()-t0:.1f}s")
print(f"  SHAP values shape: {shap_vals.shape}")
print(f"  Expected value (base rate): {explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]:.4f}")


# ============================================================
# 2. FIGURE 26: Global SHAP importance (mean |SHAP|)
# ============================================================
print("\n  Generating SHAP figures...")

# Compute mean absolute SHAP per feature
mean_shap = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

# Color by feature group
def get_group_color(feat):
    if any(x in feat for x in ['hcpi', 'fcpi', 'ecpi', 'ccpi', 'ppi', 'headline', 'food_energy', 'component_disp']):
        return C['blue'], 'Inflation'
    elif any(x in feat for x in ['energy_index', 'food_commodity', 'metals', 'fertiliser', 'oil_price']):
        return C['red'], 'Commodity'
    elif feat in ['debt_gdp', 'primary_balance', 'fiscal_balance', 'ext_debt_gdp',
                   'st_debt_reserves', 'private_credit_gdp', 'sovereign_rating', 'concessional_share']:
        return C['green'], 'Fiscal'
    elif feat.startswith('regime'):
        return C['purple'], 'Regime state'
    else:
        return C['gray'], 'Country'

top25 = mean_shap.head(25)
colors = [get_group_color(f)[0] for f in top25['feature']]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top25)), top25['mean_abs_shap'].values, color=colors, alpha=0.85)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value| (impact on prediction)')
ax.set_title('Figure 26: SHAP-based feature importance (XGBoost, 2Q ahead)')

from matplotlib.patches import Patch
legend_elements = [Patch(fc=C['blue'], label='Inflation'), Patch(fc=C['red'], label='Commodity'),
                    Patch(fc=C['green'], label='Fiscal'), Patch(fc=C['purple'], label='Regime state'),
                    Patch(fc=C['gray'], label='Country')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
save_fig('fig26_shap_global_importance')

# Save SHAP importance table
mean_shap['group'] = [get_group_color(f)[1] for f in mean_shap['feature']]
mean_shap.to_csv(os.path.join(TDIR, 'table15_shap_importance.csv'), index=False)
print(f"    Saved: table15_shap_importance.csv")


# ============================================================
# 3. FIGURE 27: SHAP Beeswarm (summary plot)
# ============================================================
shap.summary_plot(shap_vals, X_test, feature_names=feature_cols,
                  max_display=20, show=False, plot_size=(12, 9))
plt.title('Figure 27: SHAP summary — feature value vs impact (top 20)', fontsize=13)
plt.tight_layout()
save_fig('fig27_shap_beeswarm')


# ============================================================
# 4. FIGURE 28: SHAP by country group (AE vs EMDE)
# ============================================================
# Do the same features matter for both groups?

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, (group, group_label) in enumerate([('Advanced economies', 'Advanced Economies'),
                                            ('EMDEs', 'EMDEs')]):
    ax = axes[i]
    group_mask = test_data['country_group'].values == group
    
    if group_mask.sum() < 20:
        ax.set_title(f'{group_label}: insufficient data')
        continue
    
    group_shap = np.abs(shap_vals[group_mask]).mean(axis=0)
    group_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': group_shap
    }).sort_values('mean_abs_shap', ascending=False).head(15)
    
    colors_g = [get_group_color(f)[0] for f in group_importance['feature']]
    ax.barh(range(len(group_importance)), group_importance['mean_abs_shap'].values,
            color=colors_g, alpha=0.85)
    ax.set_yticks(range(len(group_importance)))
    ax.set_yticklabels(group_importance['feature'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'Figure 28{"ab"[i]}: {group_label}', fontweight='bold')

plt.suptitle('Figure 28: SHAP importance by country group — what drives transitions?',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('fig28_shap_ae_vs_emde')


# ============================================================
# 5. FIGURE 29: SHAP dependence plots for key variables
# ============================================================
# These show NON-LINEAR effects — the relationship between a
# feature's value and its impact on the prediction.

key_features = ['hcpi_yoy', 'oil_price_12m_chg', 'debt_gdp', 'sovereign_rating',
                'energy_index_12m_chg', 'food_commodity_index_12m_chg']
key_features = [f for f in key_features if f in feature_cols]

n_plots = len(key_features)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

for idx, feat in enumerate(key_features):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col]
    
    feat_idx = feature_cols.index(feat)
    feat_vals = X_test[feat].values
    feat_shap = shap_vals[:, feat_idx]
    
    # Remove NaN for plotting
    valid = ~np.isnan(feat_vals)
    
    ax.scatter(feat_vals[valid], feat_shap[valid], alpha=0.15, s=8, color=C['blue'])
    
    # Add smoothed trend line
    from scipy.ndimage import uniform_filter1d
    if valid.sum() > 50:
        sort_idx = np.argsort(feat_vals[valid])
        x_sorted = feat_vals[valid][sort_idx]
        y_sorted = feat_shap[valid][sort_idx]
        # Moving average
        window = max(len(x_sorted) // 20, 10)
        y_smooth = uniform_filter1d(y_sorted.astype(float), size=window)
        ax.plot(x_sorted, y_smooth, color=C['red'], linewidth=2)
    
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(feat)
    ax.set_ylabel('SHAP value')
    ax.set_title(feat, fontsize=10, fontweight='bold')

# Hide empty subplots
for idx in range(len(key_features), n_rows * n_cols):
    row, col = idx // n_cols, idx % n_cols
    axes[row, col].set_visible(False)

plt.suptitle('Figure 29: SHAP dependence plots — non-linear effects of key variables',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig29_shap_dependence_plots')


# ============================================================
# 6. FIGURE 30: Individual country SHAP waterfall plots
# ============================================================
# Show exactly which features pushed a specific prediction
# toward or away from "upward transition"

case_studies = [
    ('TUR', '2021-07-01', 'Turkey Q3 2021 — pre-crisis acceleration'),
    ('GBR', '2022-04-01', 'UK Q2 2022 — energy shock transmission'),
    ('USA', '2022-04-01', 'US Q2 2022 — post-COVID inflation peak'),
    ('NGA', '2023-01-01', 'Nigeria Q1 2023 — FX-driven inflation'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, (cc, date_str, title) in enumerate(case_studies):
    ax = axes[idx // 2, idx % 2]
    
    # Find the observation
    target_date = pd.Timestamp(date_str)
    obs_mask = (test_data['country_code'] == cc) & (test_data['date'] == target_date)
    
    if obs_mask.sum() == 0:
        # Try closest date
        cc_data = test_data[test_data['country_code'] == cc]
        if len(cc_data) > 0:
            closest_idx = (cc_data['date'] - target_date).abs().idxmin()
            obs_mask = test_data.index == closest_idx
            actual_date = test_data.loc[closest_idx, 'date']
            title = title.split('—')[0] + f"({actual_date.strftime('%Y-Q%q')}) — " + title.split('—')[1] if '—' in title else title
    
    if obs_mask.sum() == 0:
        ax.text(0.5, 0.5, f'{cc}: No data in test period', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue
    
    obs_idx = np.where(obs_mask.values)[0][0]
    obs_shap = shap_vals[obs_idx]
    obs_features = X_test.iloc[obs_idx]
    
    # Get predicted probability
    y_pred_prob = model.predict_proba(X_test.iloc[[obs_idx]])[:, 1][0]
    actual = y_test[obs_idx]
    
    # Top 10 features by absolute SHAP
    top_idx = np.argsort(np.abs(obs_shap))[-10:][::-1]
    top_features = [feature_cols[j] for j in top_idx]
    top_shap_vals = obs_shap[top_idx]
    top_feat_vals = [obs_features.iloc[j] for j in top_idx]
    
    # Waterfall-style horizontal bar plot
    colors_w = [C['red'] if v > 0 else C['blue'] for v in top_shap_vals]
    
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_shap_vals, color=colors_w, alpha=0.8)
    
    labels = [f"{f} = {v:.1f}" if not np.isnan(v) else f"{f} = NaN"
              for f, v in zip(top_features, top_feat_vals)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP value (→ increases transition probability)')
    
    pred_label = f"P(transition)={y_pred_prob:.1%}"
    actual_label = "Actually transitioned" if actual == 1 else "Did not transition"
    ax.set_title(f'{title}\n{pred_label} | {actual_label}', fontsize=10)

plt.suptitle('Figure 30: SHAP waterfall — what drove specific predictions?',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig('fig30_shap_waterfall_cases')


# ============================================================
# PART B: CRISIS EPISODE ANALYSIS (STEP 8)
# ============================================================
print("\n" + "=" * 70)
print("PART B: CRISIS EPISODE ANALYSIS")
print("=" * 70)

# ============================================================
# 7. FIGURE 31: COVID vs Ukraine SHAP fingerprints
# ============================================================
# Compare which features drove transitions in each crisis

covid_mask = test_data['split'].values == 'test_covid'
ukraine_mask = test_data['split'].values == 'test_ukraine'

# Only look at observations where an upward transition actually happened
covid_trans_mask = covid_mask & (y_test == 1)
ukraine_trans_mask = ukraine_mask & (y_test == 1)

print(f"\n  COVID transitions: {covid_trans_mask.sum()}")
print(f"  Ukraine transitions: {ukraine_trans_mask.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, (mask, period_label) in enumerate([(covid_trans_mask, 'COVID (2019-20)'),
                                            (ukraine_trans_mask, 'Ukraine (2021+)')]):
    ax = axes[i]
    
    if mask.sum() < 5:
        ax.text(0.5, 0.5, f'{period_label}: Too few transitions', ha='center', transform=ax.transAxes)
        continue
    
    period_shap = np.abs(shap_vals[mask]).mean(axis=0)
    period_imp = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': period_shap
    }).sort_values('mean_abs_shap', ascending=False).head(15)
    
    colors_p = [get_group_color(f)[0] for f in period_imp['feature']]
    ax.barh(range(len(period_imp)), period_imp['mean_abs_shap'].values,
            color=colors_p, alpha=0.85)
    ax.set_yticks(range(len(period_imp)))
    ax.set_yticklabels(period_imp['feature'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| during transitions')
    ax.set_title(f'Figure 31{"ab"[i]}: {period_label}', fontweight='bold')

plt.suptitle('Figure 31: Crisis fingerprints — what drove transitions in each episode?',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('fig31_crisis_fingerprints')

# Save fingerprint comparison table
fingerprint_rows = []
for mask, period in [(covid_trans_mask, 'COVID'), (ukraine_trans_mask, 'Ukraine')]:
    if mask.sum() >= 5:
        period_shap_vals = np.abs(shap_vals[mask]).mean(axis=0)
        for feat, shap_v in zip(feature_cols, period_shap_vals):
            fingerprint_rows.append({'Period': period, 'Feature': feat, 'Mean_abs_SHAP': round(shap_v, 6)})

fingerprint_df = pd.DataFrame(fingerprint_rows)
if not fingerprint_df.empty:
    pivot = fingerprint_df.pivot(index='Feature', columns='Period', values='Mean_abs_SHAP')
    if 'COVID' in pivot.columns and 'Ukraine' in pivot.columns:
        pivot['Ratio_Ukraine_vs_COVID'] = pivot['Ukraine'] / pivot['COVID'].clip(lower=0.0001)
    pivot = pivot.sort_values('Ukraine' if 'Ukraine' in pivot.columns else pivot.columns[0], ascending=False).head(20)
    pivot.to_csv(os.path.join(TDIR, 'table16_crisis_fingerprints.csv'))
    print(f"    Saved: table16_crisis_fingerprints.csv")


# ============================================================
# 8. FIGURE 32: Transition probability timelines
# ============================================================
# For selected countries, show how P(upward transition) evolved
# over 2019-2025, with regime background coloring

timeline_countries = {
    'GBR': 'United Kingdom', 'USA': 'United States', 'TUR': 'Turkey',
    'BRA': 'Brazil', 'DEU': 'Germany', 'NGA': 'Nigeria',
}

# Get predictions for all test observations
y_prob_all = model.predict_proba(X_test)[:, 1]
test_data_with_probs = test_data.copy()
test_data_with_probs['transition_prob'] = y_prob_all

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for idx, (cc, name) in enumerate(timeline_countries.items()):
    ax = axes[idx // 2, idx % 2]
    
    cc_data = test_data_with_probs[test_data_with_probs['country_code'] == cc].sort_values('date')
    
    if cc_data.empty:
        ax.set_title(f'{name}: No data')
        continue
    
    # Plot transition probability
    ax.plot(cc_data['date'], cc_data['transition_prob'], 'o-',
            color=C['blue'], linewidth=1.5, markersize=4, label='P(upward transition)')
    
    # Mark actual transitions
    if 'regime' in cc_data.columns:
        actual_trans = cc_data[cc_data[target_col] == 1]
        if not actual_trans.empty:
            ax.scatter(actual_trans['date'], actual_trans['transition_prob'],
                      color=C['red'], s=60, zorder=5, marker='^', label='Actual transition')
    
    # Add threshold line
    ax.axhline(y=0.3, color=C['amber'], linestyle='--', linewidth=1, alpha=0.7, label='Alert threshold (0.3)')
    
    # Mark crisis events
    ax.axvline(x=pd.Timestamp('2020-03-01'), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=pd.Timestamp('2022-02-01'), color='gray', linestyle=':', alpha=0.5)
    
    ax.set_ylabel('P(upward transition)')
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper left')

plt.suptitle('Figure 32: Transition probability timelines (2Q ahead, XGBoost)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig('fig32_transition_timelines')


# ============================================================
# 9. FIGURE 33: Early warning evaluation
# ============================================================
# Did the model flag transitions BEFORE they happened?

print("\n  Evaluating early warning performance...")

ew_results = []
for cc, name in timeline_countries.items():
    cc_data = test_data_with_probs[test_data_with_probs['country_code'] == cc].sort_values('date')
    
    if cc_data.empty or target_col not in cc_data.columns:
        continue
    
    # For each actual transition, was the probability elevated beforehand?
    transitions = cc_data[cc_data[target_col] == 1]
    
    for _, trans_row in transitions.iterrows():
        trans_date = trans_row['date']
        
        # Look at predictions 1-2 quarters before
        before = cc_data[(cc_data['date'] < trans_date) &
                          (cc_data['date'] >= trans_date - pd.Timedelta(days=180))]
        
        if len(before) > 0:
            max_prob_before = before['transition_prob'].max()
            flagged = max_prob_before > 0.3
            
            ew_results.append({
                'Country': name,
                'Transition date': (f"{trans_date.year}-Q{(trans_date.month - 1) // 3 + 1}"
                                    if hasattr(trans_date, 'month') else str(trans_date)),
                'Max P(trans) before': round(max_prob_before, 3),
                'Flagged (>0.3)': 'Yes' if flagged else 'No',
            })

if ew_results:
    ew_df = pd.DataFrame(ew_results)
    print(f"\n  Early warning results:")
    print(f"    Total transitions evaluated: {len(ew_df)}")
    print(f"    Flagged in advance: {(ew_df['Flagged (>0.3)'] == 'Yes').sum()} ({(ew_df['Flagged (>0.3)'] == 'Yes').mean()*100:.0f}%)")
    ew_df.to_csv(os.path.join(TDIR, 'table17_early_warning_evaluation.csv'), index=False)
    print(f"    Saved: table17_early_warning_evaluation.csv")


# ============================================================
# 10. FIGURE 34: Feature importance by group (summary)
# ============================================================

# Aggregate SHAP importance by feature group
group_shap = {}
for feat, shap_mean in zip(feature_cols, np.abs(shap_vals).mean(axis=0)):
    _, group = get_group_color(feat)
    group_shap[group] = group_shap.get(group, 0) + shap_mean

group_df = pd.DataFrame(list(group_shap.items()), columns=['Group', 'Total SHAP'])
group_df = group_df.sort_values('Total SHAP', ascending=False)
group_df['Percentage'] = group_df['Total SHAP'] / group_df['Total SHAP'].sum() * 100

fig, ax = plt.subplots(figsize=(8, 5))
group_colors_map = {'Inflation': C['blue'], 'Commodity': C['red'], 'Fiscal': C['green'],
                     'Regime state': C['purple'], 'Country': C['gray']}
bar_colors = [group_colors_map.get(g, C['gray']) for g in group_df['Group']]
ax.barh(range(len(group_df)), group_df['Percentage'].values, color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(group_df)))
ax.set_yticklabels(group_df['Group'].values)
ax.invert_yaxis()
ax.set_xlabel('Share of total SHAP importance (%)')
ax.set_title('Figure 33: Feature group contributions to predictions')

# Add percentage labels
for i, pct in enumerate(group_df['Percentage'].values):
    ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=10)

plt.tight_layout()
save_fig('fig33_feature_group_shares')

# Save as table
group_df.to_csv(os.path.join(TDIR, 'table18_feature_group_importance.csv'), index=False)
print(f"    Saved: table18_feature_group_importance.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"STEPS 8+9 COMPLETE — SHAP & CRISIS ANALYSIS")
print(f"{'='*70}")

print(f"""
SHAP ANALYSIS:
  {len(X_test):,} test observations
  {len(key_features)} dependence plots
  {len(case_studies)} waterfall case studies

CRISIS ANALYSIS:
  COVID vs Ukraine fingerprint comparison
  Timelines for {len(timeline_countries)} countries

FIGURES: {fc} generated (fig26-fig33)
TABLES: table15-table18
TOTAL TIME: {total_time/60:.1f} minutes
""")
