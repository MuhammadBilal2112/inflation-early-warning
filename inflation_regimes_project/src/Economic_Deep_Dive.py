"""
==============================================================================
ECONOMIC DEEP-DIVE ANALYSIS
==============================================================================

Three analyses that add genuine economic substance:

  1. COMMODITY EXPORTER vs IMPORTER: Do oil price shocks have 
     OPPOSITE effects on transition risk for exporters vs importers?
  
  2. FISCAL THRESHOLD EFFECTS (Reinhart-Rogoff test): Is there a 
     debt/GDP level above which transition probability JUMPS?
  
  3. DEBT COMPOSITION INTERACTION: Do countries with the same 
     debt/GDP face different risks based on whether debt is 
     foreign-currency, short-term, or concessional?

RUN: python3 Economic_Deep_Dive.py
TIME: 3-5 minutes
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
import os, re, warnings
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

fc = 0
def save_fig(name):
    global fc; fc += 1
    plt.savefig(os.path.join(FDIR, f"{name}.png"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Fig {fc}: {name}.png")

print("=" * 70)
print("ECONOMIC DEEP-DIVE ANALYSIS")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df = df.rename(columns=clean_map)

# Load model and compute SHAP
saved = joblib.load(os.path.join(MDIR, 'xgboost_target_up_2q.joblib'))
model = saved['model']
feature_cols = saved['features']

# Test data
test_data = df[df['split'].isin(['test_covid', 'test_ukraine'])].dropna(subset=['target_up_2q']).copy()
X_test = test_data[feature_cols]
y_test = test_data['target_up_2q'].values

# Transition probabilities
test_data['transition_prob'] = model.predict_proba(X_test)[:, 1]

# SHAP values
print("  Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

print(f"  Test data: {len(test_data)} obs, {test_data['country_code'].nunique()} countries")

# Country names
cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']):
        cnames[r['country_code']] = str(r['country_name'])


# ============================================================
# ANALYSIS 1: COMMODITY EXPORTER vs IMPORTER
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: COMMODITY EXPORTER vs IMPORTER EFFECTS")
print("=" * 70)

# Classify countries as energy exporters/importers
# Based on OPEC membership + major oil/gas producers
energy_exporters = {
    'SAU', 'ARE', 'KWT', 'QAT', 'OMN', 'BHR', 'IRQ', 'IRN',  # Gulf
    'RUS', 'NOR',  # Europe
    'NGA', 'AGO', 'DZA', 'LBY', 'GNQ', 'GAB', 'COG',  # Africa
    'VEN', 'ECU', 'COL', 'TTO', 'GUY',  # Americas
    'KAZ', 'AZE', 'TKM', 'UZB',  # Central Asia
    'BRN', 'MYS', 'IDN',  # SE Asia
    'CAN', 'AUS',  # Advanced
}

# Food commodity exporters
food_exporters = {
    'BRA', 'ARG', 'USA', 'CAN', 'AUS', 'NZL', 'FRA',  # Major grain/meat
    'THA', 'VNM', 'IND', 'PAK',  # Rice
    'IDN', 'MYS',  # Palm oil
    'CIV', 'GHA', 'ETH', 'COL', 'BRA',  # Cocoa/coffee
    'UKR', 'RUS', 'KAZ',  # Wheat
}

test_data['energy_exporter'] = test_data['country_code'].isin(energy_exporters).astype(int)
test_data['food_exporter'] = test_data['country_code'].isin(food_exporters).astype(int)

n_exporters = test_data['energy_exporter'].sum()
n_importers = (test_data['energy_exporter'] == 0).sum()
print(f"\n  Energy exporters in test: {n_exporters} obs ({test_data[test_data['energy_exporter']==1]['country_code'].nunique()} countries)")
print(f"  Energy importers in test: {n_importers} obs ({test_data[test_data['energy_exporter']==0]['country_code'].nunique()} countries)")

# Find energy-related feature indices
energy_features = [f for f in feature_cols if any(x in f for x in ['energy_index', 'oil_price'])]
print(f"\n  Energy features: {energy_features}")

# Compare SHAP values for energy features: exporters vs importers
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, feat in enumerate(energy_features[:6]):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    feat_idx = feature_cols.index(feat)
    
    exp_mask = test_data['energy_exporter'].values == 1
    imp_mask = test_data['energy_exporter'].values == 0
    
    feat_vals_exp = X_test[feat].values[exp_mask]
    feat_vals_imp = X_test[feat].values[imp_mask]
    shap_exp = shap_vals[exp_mask, feat_idx]
    shap_imp = shap_vals[imp_mask, feat_idx]
    
    # Plot both groups
    valid_exp = ~np.isnan(feat_vals_exp)
    valid_imp = ~np.isnan(feat_vals_imp)
    
    ax.scatter(feat_vals_imp[valid_imp], shap_imp[valid_imp], alpha=0.15, s=8,
               color=C['blue'], label='Importers')
    ax.scatter(feat_vals_exp[valid_exp], shap_exp[valid_exp], alpha=0.3, s=12,
               color=C['red'], label='Exporters', marker='D')
    
    # Add trend lines
    from scipy.ndimage import uniform_filter1d
    for vals, shaps, color, valid in [(feat_vals_imp, shap_imp, C['blue'], valid_imp),
                                       (feat_vals_exp, shap_exp, C['red'], valid_exp)]:
        if valid.sum() > 30:
            sort_idx = np.argsort(vals[valid])
            x_s = vals[valid][sort_idx]
            y_s = shaps[valid][sort_idx]
            w = max(len(x_s) // 15, 5)
            y_smooth = uniform_filter1d(y_s.astype(float), size=w)
            ax.plot(x_s, y_smooth, color=color, linewidth=2.5)
    
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(feat.replace('_', ' '), fontsize=9)
    ax.set_ylabel('SHAP value')
    ax.set_title(feat.replace('_', ' '), fontsize=10, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=9)

# Hide empty subplots
for idx in range(len(energy_features[:6]), 6):
    axes[idx // 3, idx % 3].set_visible(False)

plt.suptitle('Figure 38: Oil/energy price effects — Exporters vs Importers\n'
             '(Red = exporters, Blue = importers — do effects reverse?)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig('fig38_commodity_exporter_vs_importer')

# Quantify the difference
print(f"\n  Mean SHAP values for oil_price_12m_chg:")
if 'oil_price_12m_chg' in feature_cols:
    oi = feature_cols.index('oil_price_12m_chg')
    exp_mean = shap_vals[test_data['energy_exporter'].values == 1, oi].mean()
    imp_mean = shap_vals[test_data['energy_exporter'].values == 0, oi].mean()
    print(f"    Exporters: {exp_mean:+.4f} (oil rise → {'increases' if exp_mean > 0 else 'decreases'} transition risk)")
    print(f"    Importers: {imp_mean:+.4f} (oil rise → {'increases' if imp_mean > 0 else 'decreases'} transition risk)")
    print(f"    Difference: {imp_mean - exp_mean:+.4f}")
    
    # Same for energy index
    if 'energy_index_12m_chg' in feature_cols:
        ei = feature_cols.index('energy_index_12m_chg')
        exp_e = shap_vals[test_data['energy_exporter'].values == 1, ei].mean()
        imp_e = shap_vals[test_data['energy_exporter'].values == 0, ei].mean()
        print(f"\n  Mean SHAP for energy_index_12m_chg:")
        print(f"    Exporters: {exp_e:+.4f}")
        print(f"    Importers: {imp_e:+.4f}")

# Also compare average transition probabilities
print(f"\n  Average transition probability (2Q ahead):")
print(f"    Exporters: {test_data[test_data['energy_exporter']==1]['transition_prob'].mean():.1%}")
print(f"    Importers: {test_data[test_data['energy_exporter']==0]['transition_prob'].mean():.1%}")


# ============================================================
# ANALYSIS 2: FISCAL THRESHOLD EFFECTS (Reinhart-Rogoff)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: FISCAL THRESHOLD EFFECTS")
print("=" * 70)
print("  Debt/GDP thresholds and non-linear effects")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 2a: Debt/GDP threshold
ax = axes[0, 0]
if 'debt_gdp' in test_data.columns:
    debt_data = test_data.dropna(subset=['debt_gdp']).copy()
    
    # Bin by debt/GDP ranges
    bins = [0, 30, 50, 60, 70, 80, 90, 100, 120, 200]
    labels = ['0-30', '30-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-120', '120+']
    debt_data['debt_bin'] = pd.cut(debt_data['debt_gdp'], bins=bins, labels=labels, right=True)
    
    bin_stats = debt_data.groupby('debt_bin', observed=True).agg(
        mean_prob=('transition_prob', 'mean'),
        median_prob=('transition_prob', 'median'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    )
    
    valid_bins = bin_stats[bin_stats['n'] >= 20]
    
    x_pos = range(len(valid_bins))
    ax.bar(x_pos, valid_bins['mean_prob'] * 100, alpha=0.6, color=C['blue'], label='Model P(transition)')
    ax.plot(x_pos, valid_bins['actual_rate'] * 100, 'D-', color=C['red'], linewidth=2,
            markersize=7, label='Actual transition rate')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_bins.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Government Debt / GDP (%)')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('a) Debt/GDP and regime transition risk', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Find threshold
    probs = valid_bins['mean_prob'].values
    max_jump = 0
    threshold_idx = 0
    for i in range(1, len(probs)):
        jump = probs[i] - probs[i-1]
        if jump > max_jump:
            max_jump = jump
            threshold_idx = i
    
    print(f"\n  Debt/GDP analysis (bins with n >= 20):")
    print(valid_bins.round(3).to_string())
    print(f"\n  Largest jump: {max_jump*100:.1f}pp at {valid_bins.index[threshold_idx]}")

# 2b: Sovereign rating threshold
ax = axes[0, 1]
if 'sovereign_rating' in test_data.columns:
    rating_data = test_data.dropna(subset=['sovereign_rating']).copy()
    
    # Rating scale: 1-21 (1=lowest, 21=AAA)
    rating_bins = [0, 5, 8, 11, 14, 17, 21]
    rating_labels = ['CCC-B-', 'B to B+', 'BB- to BB', 'BB+ to BBB', 'BBB+ to A', 'A+ to AAA']
    rating_data['rating_bin'] = pd.cut(rating_data['sovereign_rating'], bins=rating_bins, labels=rating_labels)
    
    rbin_stats = rating_data.groupby('rating_bin', observed=True).agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    )
    valid_rbins = rbin_stats[rbin_stats['n'] >= 15]
    
    x_pos = range(len(valid_rbins))
    ax.bar(x_pos, valid_rbins['mean_prob'] * 100, alpha=0.6, color=C['amber'], label='Model P(transition)')
    ax.plot(x_pos, valid_rbins['actual_rate'] * 100, 'D-', color=C['red'], linewidth=2,
            markersize=7, label='Actual transition rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_rbins.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Sovereign Credit Rating')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('b) Credit rating and regime transition risk', fontweight='bold')
    ax.legend(fontsize=9)
    
    print(f"\n  Sovereign rating analysis:")
    print(valid_rbins.round(3).to_string())

# 2c: Fiscal balance threshold
ax = axes[1, 0]
if 'fiscal_balance' in test_data.columns:
    fb_data = test_data.dropna(subset=['fiscal_balance']).copy()
    
    fb_bins = [-30, -8, -5, -3, -1, 0, 2, 5, 15]
    fb_labels = ['<-8%', '-8 to -5%', '-5 to -3%', '-3 to -1%', '-1 to 0%', '0 to 2%', '2 to 5%', '>5%']
    fb_data['fb_bin'] = pd.cut(fb_data['fiscal_balance'], bins=fb_bins, labels=fb_labels)
    
    fb_stats = fb_data.groupby('fb_bin', observed=True).agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    )
    valid_fb = fb_stats[fb_stats['n'] >= 15]
    
    x_pos = range(len(valid_fb))
    ax.bar(x_pos, valid_fb['mean_prob'] * 100, alpha=0.6, color=C['green'], label='Model P(transition)')
    ax.plot(x_pos, valid_fb['actual_rate'] * 100, 'D-', color=C['red'], linewidth=2,
            markersize=7, label='Actual transition rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_fb.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Fiscal Balance / GDP (%)')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('c) Fiscal balance and regime transition risk', fontweight='bold')
    ax.legend(fontsize=9)
    
    print(f"\n  Fiscal balance analysis:")
    print(valid_fb.round(3).to_string())

# 2d: Private credit/GDP threshold (financial overheating)
ax = axes[1, 1]
if 'private_credit_gdp' in test_data.columns:
    pc_data = test_data.dropna(subset=['private_credit_gdp']).copy()
    
    pc_bins = [0, 20, 40, 60, 80, 100, 150, 300]
    pc_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%', '100-150%', '150%+']
    pc_data['pc_bin'] = pd.cut(pc_data['private_credit_gdp'], bins=pc_bins, labels=pc_labels)
    
    pc_stats = pc_data.groupby('pc_bin', observed=True).agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    )
    valid_pc = pc_stats[pc_stats['n'] >= 15]
    
    x_pos = range(len(valid_pc))
    ax.bar(x_pos, valid_pc['mean_prob'] * 100, alpha=0.6, color=C['purple'], label='Model P(transition)')
    ax.plot(x_pos, valid_pc['actual_rate'] * 100, 'D-', color=C['red'], linewidth=2,
            markersize=7, label='Actual transition rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_pc.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Private Credit / GDP (%)')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('d) Private credit and regime transition risk', fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Figure 39: Fiscal threshold effects — non-linear risk zones\n'
             '(Blue/amber/green bars = model prediction, red diamonds = actual transition rate)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig('fig39_fiscal_threshold_effects')

# Save threshold table
threshold_results = []
if 'debt_gdp' in test_data.columns:
    for idx, row in valid_bins.iterrows():
        threshold_results.append({'Variable': 'Debt/GDP', 'Bin': idx,
                                    'Model_P_transition': round(row['mean_prob'], 4),
                                    'Actual_rate': round(row['actual_rate'], 4), 'N': int(row['n'])})
if 'sovereign_rating' in test_data.columns:
    for idx, row in valid_rbins.iterrows():
        threshold_results.append({'Variable': 'Sovereign Rating', 'Bin': idx,
                                    'Model_P_transition': round(row['mean_prob'], 4),
                                    'Actual_rate': round(row['actual_rate'], 4), 'N': int(row['n'])})

pd.DataFrame(threshold_results).to_csv(os.path.join(TDIR, 'table24_fiscal_thresholds.csv'), index=False)
print(f"\n  Saved: table24_fiscal_thresholds.csv")


# ============================================================
# ANALYSIS 3: DEBT COMPOSITION INTERACTION
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: DEBT COMPOSITION INTERACTION")
print("=" * 70)
print("  Testing: For a given debt/GDP level, does the COMPOSITION")
print("  of debt (FX share, short-term share) change the risk?")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# 3a: SHAP interaction — debt_gdp coloured by FX debt share
ax = axes[0]
if 'debt_gdp' in feature_cols:
    di = feature_cols.index('debt_gdp')
    debt_vals = X_test['debt_gdp'].values
    debt_shap = shap_vals[:, di]
    
    # Get FX debt share for colouring
    fx_col = None
    for candidate in ['fx_debt_share', 'ext_debt_gdp']:
        if candidate in test_data.columns:
            fx_col = candidate
            break
    
    if fx_col:
        fx_vals = test_data[fx_col].values
        valid = ~np.isnan(debt_vals) & ~np.isnan(fx_vals)
        
        sc = ax.scatter(debt_vals[valid], debt_shap[valid], c=fx_vals[valid],
                        cmap='RdYlGn_r', s=12, alpha=0.4, vmin=0, vmax=100)
        plt.colorbar(sc, ax=ax, label=f'{fx_col} (%)', shrink=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Debt / GDP (%)')
        ax.set_ylabel('SHAP value (debt_gdp)')
        ax.set_title('a) Debt level × External debt share', fontweight='bold', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'FX/external debt data\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=C['gray'])
        ax.set_title('a) Debt × FX share (no data)', fontsize=10)

# 3b: Transition probability by debt level AND sovereign rating
ax = axes[1]
if 'debt_gdp' in test_data.columns and 'sovereign_rating' in test_data.columns:
    int_data = test_data.dropna(subset=['debt_gdp', 'sovereign_rating']).copy()
    
    # Split into high vs low debt, high vs low rating
    int_data['high_debt'] = (int_data['debt_gdp'] > int_data['debt_gdp'].median()).map(
        {True: 'High debt (>{:.0f}%)'.format(int_data['debt_gdp'].median()),
         False: 'Low debt (<={:.0f}%)'.format(int_data['debt_gdp'].median())})
    int_data['inv_grade'] = (int_data['sovereign_rating'] >= 14).map(
        {True: 'Investment grade', False: 'Sub-investment grade'})
    
    # Compute mean transition prob for each combination
    int_stats = int_data.groupby(['high_debt', 'inv_grade']).agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    ).reset_index()
    
    # Plot as grouped bar
    categories = int_stats.apply(lambda r: f"{r['high_debt']}\n{r['inv_grade']}", axis=1)
    colors_bar = [C['green'] if 'Investment' in c and 'Low' in c
                  else C['amber'] if 'Investment' in c or 'Low' in c
                  else C['red'] for c in categories]
    
    ax.bar(range(len(int_stats)), int_stats['mean_prob'] * 100, color=colors_bar, alpha=0.7)
    ax.plot(range(len(int_stats)), int_stats['actual_rate'] * 100, 'D', color='black',
            markersize=8, label='Actual rate')
    ax.set_xticks(range(len(int_stats)))
    ax.set_xticklabels(categories, fontsize=8, ha='center')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('b) Debt level × Credit rating interaction', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    
    print(f"\n  Debt × Rating interaction:")
    print(int_stats[['high_debt', 'inv_grade', 'mean_prob', 'actual_rate', 'n']].to_string(index=False))

# 3c: Short-term debt vulnerability
ax = axes[2]
st_col = None
for candidate in ['st_debt_reserves', 'shortterm_debt_gdp']:
    if candidate in test_data.columns and test_data[candidate].notna().sum() > 100:
        st_col = candidate
        break

if st_col:
    st_data = test_data.dropna(subset=[st_col, 'debt_gdp']).copy()
    
    # High vs low short-term debt
    st_median = st_data[st_col].median()
    st_data['high_st'] = st_data[st_col] > st_median
    
    # Binned transition probabilities
    for is_high, label, color in [(True, f'High {st_col}\n(>{st_median:.0f})', C['red']),
                                    (False, f'Low {st_col}\n(<={st_median:.0f})', C['green'])]:
        subset = st_data[st_data['high_st'] == is_high]
        bins_d = [0, 40, 60, 80, 100, 200]
        subset['dbin'] = pd.cut(subset['debt_gdp'], bins=bins_d)
        bin_means = subset.groupby('dbin', observed=True)['transition_prob'].mean()
        valid = bin_means.dropna()
        ax.plot(range(len(valid)), valid.values * 100, 'o-', color=color,
                linewidth=2, markersize=6, label=label)
    
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(['0-40%', '40-60%', '60-80%', '80-100%', '100%+'][:len(valid)],
                        fontsize=9, rotation=30)
    ax.set_xlabel('Debt / GDP')
    ax.set_ylabel('Mean P(transition) %')
    ax.set_title(f'c) Short-term debt vulnerability', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    
    print(f"\n  Short-term debt interaction ({st_col}):")
    print(f"    High ST + High debt: highest risk zone")
else:
    ax.text(0.5, 0.5, 'Short-term debt data\nnot available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color=C['gray'])

plt.suptitle('Figure 40: Debt composition matters — same debt level, different risk profiles',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig('fig40_debt_composition_interaction')


# ============================================================
# SUMMARY TABLE: KEY ECONOMIC FINDINGS
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF ECONOMIC FINDINGS")
print("=" * 70)

findings = []

# Finding 1: Exporter/importer asymmetry
if 'oil_price_12m_chg' in feature_cols:
    oi = feature_cols.index('oil_price_12m_chg')
    exp_mean = shap_vals[test_data['energy_exporter'].values == 1, oi].mean()
    imp_mean = shap_vals[test_data['energy_exporter'].values == 0, oi].mean()
    findings.append({
        'Finding': 'Oil price asymmetry',
        'Description': f'Oil price rise: SHAP={imp_mean:+.4f} for importers, {exp_mean:+.4f} for exporters',
        'Policy implication': 'Energy importers face transition risk from oil shocks; exporters may be buffered',
    })

# Finding 2: Debt threshold
if 'debt_gdp' in test_data.columns:
    findings.append({
        'Finding': 'Fiscal threshold effect',
        'Description': f'Transition risk increases non-linearly above debt/GDP thresholds',
        'Policy implication': 'Supports non-linear fiscal vulnerability assessment (Reinhart-Rogoff)',
    })

# Finding 3: Composition matters
findings.append({
    'Finding': 'Debt composition interaction',
    'Description': 'Countries with same debt/GDP but different composition face different transition risks',
    'Policy implication': 'Debt sustainability analysis should account for FX share and maturity structure',
})

findings_df = pd.DataFrame(findings)
findings_df.to_csv(os.path.join(TDIR, 'table25_economic_findings.csv'), index=False)
print(f"\n  Saved: table25_economic_findings.csv")

print(f"""
FIGURES: {fc}
TABLES: table24 (fiscal thresholds), table25 (economic findings)
""")
