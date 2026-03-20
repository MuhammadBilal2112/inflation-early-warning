"""
==============================================================================
STEP 5: INFLATION REGIME DISCOVERY (UNSUPERVISED LEARNING)
==============================================================================

Uses unsupervised machine learning to discover inflation regimes from the data.

  1. Prepares multi-dimensional inflation features (level, dispersion, momentum)
  2. Fits Gaussian Mixture Models (GMM) with K=2–6 components
  3. Selects optimal K using BIC
  4. Labels every country-quarter with its regime
  5. Validates regimes against known economic events
  6. Produces regime maps and transition matrices

OUTPUT FILES:
  - data/processed/regime_labels.csv          (regime for every country-quarter)
  - data/processed/regime_characteristics.csv (what each regime looks like)
  - data/processed/transition_matrix.csv      (regime transition probabilities)
  - outputs/figures/fig14-fig20                (regime visualisations)

REQUIRES: Steps 2-3 completed
RUN: python3 Step05_Regime_Discovery.py
TIME: 3-5 minutes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.figsize':(12,6),'figure.dpi':150,'font.size':11,
    'font.family':'sans-serif','figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False})

C = {'blue':'#1F4E79','red':'#C62828','green':'#2E7D32','amber':'#E8A838',
     'purple':'#6A1B9A','teal':'#00838F','gray':'#616161'}

# Regime colours (will be assigned after discovery)
REGIME_COLORS = ['#2E7D32', '#1F4E79', '#E8A838', '#C62828']  # green, blue, amber, red
REGIME_NAMES = {}  # Will be filled after regime discovery

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
FDIR = os.path.join(BASE_DIR, "outputs", "figures"); os.makedirs(FDIR, exist_ok=True)
TDIR = os.path.join(BASE_DIR, "outputs", "tables"); os.makedirs(TDIR, exist_ok=True)

fc = 0
def save_fig(name):
    global fc; fc += 1
    plt.savefig(os.path.join(FDIR, f"{name}.png"), bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  Fig {fc}: {name}.png")

print("Loading data...")
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])
print(f"  Monthly panel: {monthly.shape}")


# ============================================================
# Aggregate to quarterly and build regime features
# ============================================================
# Quarterly frequency reduces monthly noise while capturing regime transitions

print("\n" + "=" * 60)
print("STEP 5a: PREPARING REGIME FEATURES")
print("=" * 60)

# Aggregate monthly to quarterly (mean within each quarter)
monthly['quarter'] = monthly['date'].dt.to_period('Q')

quarterly = monthly.groupby(['country_code', 'quarter']).agg({
    'hcpi_yoy': 'mean',
    'fcpi_yoy': 'mean',
    'ecpi_yoy': 'mean',
    'ccpi_yoy': 'mean',
    'ppi_yoy': 'mean',
    'country_name': 'first',
    'country_group': 'first',
    'income_group': 'first',
    'region': 'first',
}).reset_index()

# Convert quarter period to datetime (first day of quarter)
quarterly['date'] = quarterly['quarter'].dt.to_timestamp()
quarterly['year'] = quarterly['date'].dt.year

print(f"  Quarterly panel: {quarterly.shape}")

# Filter to observations where we have the THREE core measures
# (headline + food + energy CPI)
has_core = quarterly[['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']].notna().all(axis=1)
regime_data = quarterly[has_core].copy()
print(f"  With all 3 core measures: {len(regime_data):,} obs, {regime_data['country_code'].nunique()} countries")

# ============================================================
# Build feature vector
# ============================================================

# Levels
regime_data['hcpi_level'] = regime_data['hcpi_yoy']
regime_data['fcpi_level'] = regime_data['fcpi_yoy']
regime_data['ecpi_level'] = regime_data['ecpi_yoy']

# Cross-component dispersion
regime_data['component_dispersion'] = regime_data[['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']].std(axis=1)

# Momentum (2-quarter diff)
regime_data = regime_data.sort_values(['country_code', 'date'])
regime_data['hcpi_momentum'] = regime_data.groupby('country_code')['hcpi_yoy'].diff(2)

# Food-energy gap
regime_data['food_energy_gap'] = regime_data['fcpi_yoy'] - regime_data['ecpi_yoy']

# Rolling volatility (4-quarter std)
regime_data['hcpi_volatility'] = regime_data.groupby('country_code')['hcpi_yoy'].transform(
    lambda x: x.rolling(4, min_periods=2).std()
)

# Drop rows with NaN in any feature (from momentum/volatility computation)
feature_cols = ['hcpi_level', 'fcpi_level', 'ecpi_level', 'component_dispersion',
                'hcpi_momentum', 'food_energy_gap', 'hcpi_volatility']

regime_ready = regime_data.dropna(subset=feature_cols).copy()
print(f"  After feature engineering: {len(regime_ready):,} obs, {regime_ready['country_code'].nunique()} countries")

# Winsorise extreme values (cap at 1st/99th percentile)
# This prevents hyperinflation episodes from dominating the clustering
for col in feature_cols:
    p01 = regime_ready[col].quantile(0.01)
    p99 = regime_ready[col].quantile(0.99)
    regime_ready[col] = regime_ready[col].clip(p01, p99)

# Standardise features (mean=0, std=1)
scaler = StandardScaler()
X = scaler.fit_transform(regime_ready[feature_cols])

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Features used:")
for i, col in enumerate(feature_cols):
    raw = regime_ready[col]
    print(f"    {i+1}. {col:25s}  range: [{raw.min():.1f}, {raw.max():.1f}]  mean: {raw.mean():.2f}")


# ============================================================
# Fit GMMs with different numbers of components
# ============================================================

print("\n" + "=" * 60)
print("STEP 5b: FITTING GAUSSIAN MIXTURE MODELS")
print("=" * 60)

# Try K = 2, 3, 4, 5, 6 components
K_range = range(2, 7)
gmm_results = {}

for k in K_range:
    print(f"  Fitting GMM with K={k}...", end="", flush=True)
    
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',     # Full covariance (components can be correlated)
        n_init=10,                  # Run 10 times, keep best (avoids bad local optima)
        max_iter=300,
        random_state=42,
        reg_covar=1e-5              # Regularisation to prevent singular covariance
    )
    gmm.fit(X)
    
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    log_likelihood = gmm.score(X) * len(X)
    
    # Silhouette score (how well-separated are the clusters?)
    labels = gmm.predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    
    gmm_results[k] = {
        'model': gmm,
        'bic': bic,
        'aic': aic,
        'log_likelihood': log_likelihood,
        'silhouette': sil,
        'labels': labels,
        'probabilities': gmm.predict_proba(X),
    }
    
    print(f"  BIC: {bic:,.0f}  AIC: {aic:,.0f}  Silhouette: {sil:.3f}")

# Select best K (lowest BIC)
best_k = min(gmm_results.keys(), key=lambda k: gmm_results[k]['bic'])
print(f"\n  Best K by BIC: {best_k}")


# ============================================================
# FIGURE 14: Model selection plot (BIC, AIC, Silhouette)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ks = list(K_range)
bics = [gmm_results[k]['bic'] for k in ks]
aics = [gmm_results[k]['aic'] for k in ks]
sils = [gmm_results[k]['silhouette'] for k in ks]

ax1.plot(ks, bics, 'o-', color=C['blue'], linewidth=2, markersize=8, label='BIC')
ax1.plot(ks, aics, 's--', color=C['red'], linewidth=1.5, markersize=7, label='AIC')
ax1.axvline(x=best_k, color=C['green'], linestyle=':', linewidth=1.5, label=f'Best K={best_k}')
ax1.set_xlabel('Number of regimes (K)')
ax1.set_ylabel('Information criterion (lower = better)')
ax1.set_title('Figure 14a: Model selection - BIC and AIC')
ax1.legend()

ax2.plot(ks, sils, 'D-', color=C['purple'], linewidth=2, markersize=8)
ax2.axvline(x=best_k, color=C['green'], linestyle=':', linewidth=1.5, label=f'Best K={best_k}')
ax2.set_xlabel('Number of regimes (K)')
ax2.set_ylabel('Silhouette score (higher = better)')
ax2.set_title('Figure 14b: Cluster separation')
ax2.legend()

plt.tight_layout()
save_fig('fig14_model_selection_bic_aic')


# ============================================================
# Characterise regimes
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5c: CHARACTERISING REGIMES (K={best_k})")
print("=" * 60)

best_gmm = gmm_results[best_k]['model']
labels = gmm_results[best_k]['labels']
probs = gmm_results[best_k]['probabilities']

# Assign labels to data
regime_ready['regime_raw'] = labels
for i in range(best_k):
    regime_ready[f'regime_prob_{i}'] = probs[:, i]

# Characterise each regime by its mean feature values
print(f"\n  Raw regime characteristics (mean values):")
regime_chars = regime_ready.groupby('regime_raw')[feature_cols].mean()
print(regime_chars.round(2).to_string())

# SORT regimes by headline inflation level (Regime 0 = lowest, Regime K-1 = highest)
# This ensures regime labels are economically interpretable
regime_order = regime_chars['hcpi_level'].sort_values().index.tolist()
label_map = {old: new for new, old in enumerate(regime_order)}
regime_ready['regime'] = regime_ready['regime_raw'].map(label_map)

# Re-map probability columns
new_probs = np.zeros_like(probs)
for old, new in label_map.items():
    new_probs[:, new] = probs[:, old]
for i in range(best_k):
    regime_ready[f'regime_prob_{i}'] = new_probs[:, i]

# Name the regimes based on their characteristics
regime_chars_sorted = regime_ready.groupby('regime')[feature_cols + ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']].mean()

print(f"\n  SORTED regime characteristics (Regime 0 = lowest inflation):")
print(regime_chars_sorted[['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy', 'component_dispersion', 'hcpi_volatility']].round(2).to_string())

# Auto-generate regime names based on headline inflation level
regime_counts = regime_ready['regime'].value_counts().sort_index()
print(f"\n  Regime sizes:")
for r in range(best_k):
    mean_h = regime_chars_sorted.loc[r, 'hcpi_yoy']
    n = regime_counts.get(r, 0)
    pct = n / len(regime_ready) * 100
    
    # Name based on inflation level
    if mean_h < 3:
        name = "Low & stable"
    elif mean_h < 7:
        name = "Moderate"
    elif mean_h < 15:
        name = "Elevated"
    else:
        name = "High / Crisis"
    
    REGIME_NAMES[r] = name
    print(f"    Regime {r} ({name:15s}): {n:,} obs ({pct:.1f}%), mean headline: {mean_h:.1f}%")


# ============================================================
# FIGURE 15: Regime characteristics radar/bar chart
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel a: Mean inflation by regime and component
x = np.arange(best_k)
width = 0.25
for i, (col, label) in enumerate([('hcpi_yoy', 'Headline'), ('fcpi_yoy', 'Food'), ('ecpi_yoy', 'Energy')]):
    vals = [regime_chars_sorted.loc[r, col] for r in range(best_k)]
    axes[0].bar(x + i*width, vals, width, label=label, 
                color=[C['blue'], C['green'], C['red']][i], alpha=0.8)

axes[0].set_xticks(x + width)
axes[0].set_xticklabels([f"Regime {r}\n({REGIME_NAMES[r]})" for r in range(best_k)], fontsize=9)
axes[0].set_ylabel('Mean inflation rate (%)')
axes[0].set_title('Figure 15a: Mean inflation by regime and component')
axes[0].legend()

# Panel b: Regime dispersion and volatility
disp = [regime_chars_sorted.loc[r, 'component_dispersion'] for r in range(best_k)]
vol = [regime_chars_sorted.loc[r, 'hcpi_volatility'] for r in range(best_k)]
axes[1].bar(x - 0.15, disp, 0.3, label='Component dispersion', color=C['amber'], alpha=0.8)
axes[1].bar(x + 0.15, vol, 0.3, label='Headline volatility', color=C['purple'], alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"R{r}: {REGIME_NAMES[r]}" for r in range(best_k)], fontsize=9)
axes[1].set_ylabel('Value')
axes[1].set_title('Figure 15b: Dispersion and volatility by regime')
axes[1].legend()

plt.tight_layout()
save_fig('fig15_regime_characteristics')


# ============================================================
# Compute transition matrix
# ============================================================

print("\n" + "=" * 60)
print("STEP 5d: COMPUTING TRANSITION MATRIX")
print("=" * 60)

# For each country, look at consecutive quarters and count transitions
regime_ready = regime_ready.sort_values(['country_code', 'date'])
regime_ready['regime_next'] = regime_ready.groupby('country_code')['regime'].shift(-1)

# Only count transitions within the same country (not across countries)
transitions = regime_ready.dropna(subset=['regime_next']).copy()
transitions['regime_next'] = transitions['regime_next'].astype(int)

# Build transition count matrix
trans_matrix = pd.crosstab(transitions['regime'], transitions['regime_next'], normalize='index')
trans_matrix.index = [f"R{r}: {REGIME_NAMES[r]}" for r in range(best_k)]
trans_matrix.columns = [f"R{r}: {REGIME_NAMES[r]}" for r in range(best_k)]

print("\n  Transition matrix (probability of moving from row to column):")
print(trans_matrix.round(3).to_string())

# Key insight: diagonal values = persistence (probability of staying in same regime)
print(f"\n  Regime persistence (probability of staying):")
for r in range(best_k):
    persistence = trans_matrix.iloc[r, r]
    print(f"    {REGIME_NAMES[r]:15s}: {persistence:.1%}")


# ============================================================
# FIGURE 16: Transition matrix heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Transition probability'})
ax.set_title(f'Figure 16: Regime transition matrix ({best_k} regimes)')
ax.set_xlabel('To regime (next quarter)')
ax.set_ylabel('From regime (current quarter)')
plt.tight_layout()
save_fig('fig16_transition_matrix')


# ============================================================
# FIGURE 17: Regime timeline for selected countries
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
countries_to_show = ['USA', 'GBR', 'TUR', 'BRA', 'IND', 'NGA']
country_labels = ['United States', 'United Kingdom', 'Turkey', 'Brazil', 'India', 'Nigeria']

for idx, (cc, label) in enumerate(zip(countries_to_show, country_labels)):
    ax = axes.flatten()[idx]
    cdata = regime_ready[regime_ready['country_code'] == cc].sort_values('date')
    
    if cdata.empty:
        ax.set_title(f'{label}: No data')
        continue
    
    # Plot inflation line
    ax.plot(cdata['date'], cdata['hcpi_yoy'], color='black', linewidth=0.8, alpha=0.7)
    
    # Color background by regime
    for _, row in cdata.iterrows():
        r = int(row['regime'])
        color = REGIME_COLORS[min(r, len(REGIME_COLORS)-1)]
        ax.axvspan(row['date'], row['date'] + pd.Timedelta(days=90), 
                   alpha=0.25, color=color, linewidth=0)
    
    ax.set_title(f'{label}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Inflation (%)')
    
    # Limit y-axis for readability (except Turkey)
    if cc != 'TUR':
        ax.set_ylim(-5, max(30, cdata['hcpi_yoy'].quantile(0.95) * 1.2))

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=REGIME_COLORS[min(r, len(REGIME_COLORS)-1)], alpha=0.4,
                          label=f'R{r}: {REGIME_NAMES[r]}') for r in range(best_k)]
fig.legend(handles=legend_elements, loc='lower center', ncol=best_k, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Figure 17: Inflation regime timelines for selected countries', 
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
save_fig('fig17_regime_timelines')


# ============================================================
# FIGURE 18: Global regime composition over time
# ============================================================
# What share of countries are in each regime at each point in time?

regime_time = regime_ready.groupby(['date', 'regime']).size().unstack(fill_value=0)
regime_time_pct = regime_time.div(regime_time.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
regime_time_pct.plot.area(ax=ax, color=REGIME_COLORS[:best_k], alpha=0.7, linewidth=0.5)
ax.set_ylabel('Share of countries (%)')
ax.set_title('Figure 18: Global regime composition over time')
ax.legend([f'R{r}: {REGIME_NAMES[r]}' for r in range(best_k)], 
          loc='upper right', fontsize=9)
ax.set_ylim(0, 100)
ax.set_xlim(regime_time_pct.index.min(), regime_time_pct.index.max())

# Mark key events
for date_str, label in [('2008-07-01', 'GFC'), ('2020-01-01', 'COVID'), ('2022-01-01', 'Ukraine')]:
    ax.axvline(x=pd.Timestamp(date_str), color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.annotate(label, xy=(pd.Timestamp(date_str), 95), fontsize=8, ha='center', color='black')

save_fig('fig18_global_regime_composition')


# ============================================================
# FIGURE 19: Regime distribution by country group
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, group in enumerate(['Advanced economies', 'EMDEs']):
    grp_data = regime_ready[regime_ready['country_group'] == group]
    grp_time = grp_data.groupby(['date', 'regime']).size().unstack(fill_value=0)
    grp_time_pct = grp_time.div(grp_time.sum(axis=1), axis=0) * 100
    
    grp_time_pct.plot.area(ax=axes[i], color=REGIME_COLORS[:best_k], alpha=0.7, linewidth=0.5)
    axes[i].set_title(f'Figure 19{"ab"[i]}: {group}', fontweight='bold')
    axes[i].set_ylabel('Share of countries (%)')
    axes[i].set_ylim(0, 100)
    axes[i].legend([f'R{r}: {REGIME_NAMES[r]}' for r in range(best_k)], fontsize=7, loc='upper right')

plt.tight_layout()
save_fig('fig19_regime_by_country_group')


# ============================================================
# Validation against known events
# ============================================================

print("\n" + "=" * 60)
print("STEP 5e: VALIDATION AGAINST KNOWN EVENTS")
print("=" * 60)

def check_regime(country_code, year, expected_regime_name, tolerance=1):
    """Check if a country is in the expected regime type during a given year."""
    data = regime_ready[(regime_ready['country_code'] == country_code) & 
                         (regime_ready['year'] == year)]
    if data.empty:
        return f"  {country_code} {year}: NO DATA"
    
    modal_regime = data['regime'].mode().iloc[0]
    regime_name = REGIME_NAMES[modal_regime]
    
    status = "OK" if expected_regime_name.lower() in regime_name.lower() else "CHECK"
    return f"  {country_code} {year}: Regime {modal_regime} ({regime_name}) [{status}] - expected: {expected_regime_name}"

# Known facts to validate against:
print("\n--- Advanced economies should be 'Low' pre-2021 ---")
for cc in ['USA', 'GBR', 'DEU', 'JPN']:
    print(check_regime(cc, 2019, 'Low'))

print("\n--- 2022 post-Ukraine: Advanced economies should move up ---")
for cc in ['USA', 'GBR', 'DEU']:
    print(check_regime(cc, 2022, 'Moderate'))

print("\n--- Turkey: should be high/crisis by 2022 ---")
print(check_regime('TUR', 2019, 'Moderate'))
print(check_regime('TUR', 2022, 'High'))

print("\n--- EMDEs with known high inflation ---")
for cc, yr in [('VEN', 2018), ('ARG', 2023), ('LBN', 2021), ('NGA', 2023)]:
    print(check_regime(cc, yr, 'High'))

print("\n--- 2008 GFC period: brief spike ---")
for cc in ['USA', 'GBR']:
    print(check_regime(cc, 2008, 'Low'))


# ============================================================
# Save outputs
# ============================================================

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Save regime labels (the key output)
output_cols = ['country_code', 'country_name', 'quarter', 'date', 'year',
               'country_group', 'income_group', 'region',
               'hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy',
               'regime'] + [f'regime_prob_{i}' for i in range(best_k)]

# Add feature columns
for col in feature_cols:
    if col in regime_ready.columns:
        output_cols.append(col)

regime_output = regime_ready[[c for c in output_cols if c in regime_ready.columns]].copy()
regime_output['quarter'] = regime_output['quarter'].astype(str)

regime_path = os.path.join(PROC, 'regime_labels.csv')
regime_output.to_csv(regime_path, index=False)
print(f"  Saved: regime_labels.csv ({len(regime_output):,} rows, {regime_output['country_code'].nunique()} countries)")

# Save regime characteristics
chars_path = os.path.join(TDIR, 'table04_regime_characteristics.csv')
regime_chars_final = regime_ready.groupby('regime').agg({
    'hcpi_yoy': ['mean', 'median', 'std', 'count'],
    'fcpi_yoy': ['mean', 'median'],
    'ecpi_yoy': ['mean', 'median'],
    'component_dispersion': 'mean',
    'hcpi_volatility': 'mean',
}).round(2)
regime_chars_final.to_csv(chars_path)
print(f"  Saved: table04_regime_characteristics.csv")

# Save transition matrix
trans_path = os.path.join(TDIR, 'table05_transition_matrix.csv')
trans_matrix.to_csv(trans_path)
print(f"  Saved: table05_transition_matrix.csv")

# Save model selection results
model_sel = pd.DataFrame([{
    'K': k,
    'BIC': gmm_results[k]['bic'],
    'AIC': gmm_results[k]['aic'],
    'Silhouette': gmm_results[k]['silhouette'],
    'Selected': 'YES' if k == best_k else ''
} for k in K_range])
model_sel.to_csv(os.path.join(TDIR, 'table06_model_selection.csv'), index=False)
print(f"  Saved: table06_model_selection.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"STEP 5 COMPLETE - REGIME DISCOVERY SUMMARY")
print(f"{'='*60}")

print(f"""
REGIMES DISCOVERED: {best_k}
  Selected by: Bayesian Information Criterion (BIC)

REGIME DEFINITIONS:""")

for r in range(best_k):
    n = regime_counts.get(r, 0)
    pct = n / len(regime_ready) * 100
    mean_h = regime_chars_sorted.loc[r, 'hcpi_yoy']
    mean_f = regime_chars_sorted.loc[r, 'fcpi_yoy']
    mean_e = regime_chars_sorted.loc[r, 'ecpi_yoy']
    disp = regime_chars_sorted.loc[r, 'component_dispersion']
    vol = regime_chars_sorted.loc[r, 'hcpi_volatility']
    persist = trans_matrix.iloc[r, r]
    
    print(f"""
  Regime {r}: {REGIME_NAMES[r]}
    Observations: {n:,} ({pct:.1f}%)
    Mean headline CPI: {mean_h:.1f}%  |  Food: {mean_f:.1f}%  |  Energy: {mean_e:.1f}%
    Component dispersion: {disp:.1f}  |  Volatility: {vol:.1f}
    Persistence: {persist:.1%} (probability of staying in this regime next quarter)""")

print(f"""
FILES CREATED:
  data/processed/regime_labels.csv          ({len(regime_output):,} country-quarter observations)
  outputs/tables/table04_regime_characteristics.csv
  outputs/tables/table05_transition_matrix.csv
  outputs/tables/table06_model_selection.csv
  outputs/figures/fig14-fig19              (6 regime figures)

FIGURES: {fc} total

""")
