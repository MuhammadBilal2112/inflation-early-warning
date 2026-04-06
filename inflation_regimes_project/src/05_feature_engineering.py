"""
==============================================================================
STEP 6: FEATURE ENGINEERING FOR PREDICTION
==============================================================================

Builds the complete feature matrix and target variables for supervised ML.
All features are constructed using only information available at time t
to predict what happens at time t+h.

Produces:
  1. Target variables: upward regime transitions at 1, 2, and 4 quarters ahead
  2. Inflation features: lagged levels, momentum, volatility, cross-component gaps
  3. Commodity features: price changes and volatility
  4. Fiscal features: carried forward from last annual observation
  5. Regime state features: current regime probabilities from GMM
  6. Country characteristics: AE/EMDE, income group, region (encoded)
  7. Train/validation/test split flags

OUTPUT: data/processed/features_and_targets.csv

REQUIRES: Steps 2-5 completed
RUN: python3 Step06_Feature_Engineering.py
TIME: 1-2 minutes
==============================================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

print("Loading data...")

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
TDIR = os.path.join(BASE_DIR, "outputs", "tables"); os.makedirs(TDIR, exist_ok=True)

# Load regime labels (quarterly, from Step 5)
regime = pd.read_csv(os.path.join(PROC, 'regime_labels.csv'), parse_dates=['date'])
print(f"  Regime labels: {regime.shape} ({regime['country_code'].nunique()} countries)")

# Load master monthly panel (for computing monthly-based features)
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])
print(f"  Monthly panel: {monthly.shape}")


# ============================================================
# Construct target variables
# ============================================================
# Binary target: will this country move to a higher inflation regime within h quarters?

print("\n" + "=" * 60)
print("STEP 6a: CONSTRUCTING TARGET VARIABLES")
print("=" * 60)

regime = regime.sort_values(['country_code', 'date']).reset_index(drop=True)

# For each forecast horizon, create the target
horizons = {'1q': 1, '2q': 2, '4q': 4}  # 1 quarter, 2 quarters, 4 quarters ahead

for label, h in horizons.items():
    # What regime will this country be in h quarters from now?
    regime[f'regime_future_{label}'] = regime.groupby('country_code')['regime'].shift(-h)
    
    # Binary target: did the country move to a STRICTLY higher regime?
    regime[f'target_up_{label}'] = (
        regime[f'regime_future_{label}'] > regime['regime']
    ).astype(float)
    
    # Multi-class target: what regime will it be in?
    # (already have this as regime_future_{label})
    
    # Set to NaN where we don't have the future data
    mask = regime[f'regime_future_{label}'].isna()
    regime.loc[mask, f'target_up_{label}'] = np.nan

# Alternative target: entry into elevated/crisis regimes (R4 or R5) specifically.
# Only fires when the country starts from a non-elevated state (R0-R3).
# Tests whether the ordinal assumption (any upward move = positive)
# matters for the model — if AUC is similar, the assumption is robust.
for label, h in horizons.items():
    future_col = f'regime_future_{label}'
    regime[f'target_crisis_entry_{label}'] = np.where(
        regime[future_col].isna(), np.nan,
        ((regime['regime'] <= 3) & (regime[future_col] >= 4)).astype(float)
    )

# Print target statistics
print("\n  Target variable statistics:")
print(f"  {'Target':<28s} {'Positive rate':>15s} {'N valid':>10s} {'N positive':>10s}")
print(f"  {'-'*63}")
for label in horizons:
    for prefix in ['target_up', 'target_crisis_entry']:
        col = f'{prefix}_{label}'
        valid = regime[col].dropna()
        pos_rate = valid.mean() * 100
        n_pos = int(valid.sum())
        print(f"  {col:<28s} {pos_rate:>14.1f}% {len(valid):>10,} {n_pos:>10,}")


# ============================================================
# Compute inflation features from monthly data
# ============================================================
# Aggregate monthly features to quarterly

print("\n" + "=" * 60)
print("STEP 6b: COMPUTING INFLATION FEATURES")
print("=" * 60)

monthly = monthly.sort_values(['country_code', 'date']).reset_index(drop=True)

# Compute additional monthly features before aggregating
for col in ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']:
    base = col.split('_')[0]
    
    # Month-on-month change in YoY inflation (acceleration)
    monthly[f'{base}_mom_change'] = monthly.groupby('country_code')[col].diff(1)
    
    # 6-month rolling volatility
    monthly[f'{base}_6m_vol'] = monthly.groupby('country_code')[col].transform(
        lambda x: x.rolling(6, min_periods=3).std()
    )
    
    # 12-month rolling volatility
    monthly[f'{base}_12m_vol'] = monthly.groupby('country_code')[col].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )

# Cross-component features
monthly['headline_food_gap'] = monthly['hcpi_yoy'] - monthly['fcpi_yoy']
monthly['headline_energy_gap'] = monthly['hcpi_yoy'] - monthly['ecpi_yoy']
monthly['food_energy_gap_m'] = monthly['fcpi_yoy'] - monthly['ecpi_yoy']

# Component dispersion (monthly)
monthly['component_disp_m'] = monthly[['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']].std(axis=1)

# Now aggregate to quarterly (mean of last 3 months of each quarter)
monthly['quarter'] = monthly['date'].dt.to_period('Q')

# Define which monthly columns to aggregate to quarterly
agg_cols = {
    # Inflation levels (current quarter average)
    'hcpi_yoy': 'mean', 'fcpi_yoy': 'mean', 'ecpi_yoy': 'mean',
    'ccpi_yoy': 'mean', 'ppi_yoy': 'mean',
    
    # Acceleration (average of monthly changes within quarter)
    'hcpi_mom_change': 'mean', 'fcpi_mom_change': 'mean', 'ecpi_mom_change': 'mean',
    
    # Volatility (end-of-quarter value, most recent)
    'hcpi_6m_vol': 'last', 'fcpi_6m_vol': 'last', 'ecpi_6m_vol': 'last',
    'hcpi_12m_vol': 'last',
    
    # Cross-component gaps
    'headline_food_gap': 'mean', 'headline_energy_gap': 'mean',
    'food_energy_gap_m': 'mean', 'component_disp_m': 'mean',
    
    # Commodity features (end-of-quarter)
    'energy_index': 'last', 'food_commodity_index': 'last',
    'metals_index': 'last', 'fertiliser_index': 'last',
    'oil_price': 'last',
    'energy_index_3m_chg': 'last', 'energy_index_12m_chg': 'last',
    'energy_index_6m_vol': 'last',
    'food_commodity_index_3m_chg': 'last', 'food_commodity_index_12m_chg': 'last',
    'food_commodity_index_6m_vol': 'last',
    'metals_index_12m_chg': 'last',
    'fertiliser_index_12m_chg': 'last',
    'oil_price_12m_chg': 'last', 'oil_price_6m_vol': 'last',
    
    # Fiscal features (carried forward - take last available in quarter)
    'debt_gdp': 'last', 'primary_balance': 'last', 'fiscal_balance': 'last',
    'ext_debt_gdp': 'last', 'st_debt_reserves': 'last',
    'private_credit_gdp': 'last', 'sovereign_rating': 'last',
    'concessional_share': 'last',
    
    # Country classifications
    'country_group': 'first', 'income_group': 'first', 'region': 'first',
}

quarterly_features = monthly.groupby(['country_code', 'quarter']).agg(agg_cols).reset_index()
quarterly_features['date'] = quarterly_features['quarter'].dt.to_timestamp()

print(f"  Quarterly features computed: {quarterly_features.shape}")


# ============================================================
# Add lagged inflation features
# ============================================================
# 1Q and 2Q lags

print("\n  Adding lagged features...")

quarterly_features = quarterly_features.sort_values(['country_code', 'date']).reset_index(drop=True)

lag_cols = ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy', 'component_disp_m', 'hcpi_12m_vol']
for col in lag_cols:
    if col in quarterly_features.columns:
        # 1-quarter lag
        quarterly_features[f'{col}_lag1'] = quarterly_features.groupby('country_code')[col].shift(1)
        # 2-quarter lag
        quarterly_features[f'{col}_lag2'] = quarterly_features.groupby('country_code')[col].shift(2)

# Inflation CHANGE over past 2 quarters (momentum)
for col in ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']:
    if col in quarterly_features.columns and f'{col}_lag2' in quarterly_features.columns:
        quarterly_features[f'{col}_2q_change'] = quarterly_features[col] - quarterly_features[f'{col}_lag2']

print(f"  Features after adding lags: {quarterly_features.shape[1]} columns")


# ============================================================
# Merge features with regime data
# ============================================================

print("\n" + "=" * 60)
print("STEP 6c: MERGING FEATURES WITH TARGETS")
print("=" * 60)

# Select regime columns to merge
regime_cols = ['country_code', 'date', 'regime',
               'regime_prob_0', 'regime_prob_1', 'regime_prob_2',
               'regime_prob_3', 'regime_prob_4', 'regime_prob_5',
               'target_up_1q', 'target_up_2q', 'target_up_4q',
               'regime_future_1q', 'regime_future_2q', 'regime_future_4q']
regime_cols = [c for c in regime_cols if c in regime.columns]

# Merge on country_code + date
dataset = quarterly_features.merge(
    regime[regime_cols],
    on=['country_code', 'date'],
    how='inner'  # Only keep observations that have both features AND regime labels
)

print(f"  Merged dataset: {dataset.shape}")
print(f"  Countries: {dataset['country_code'].nunique()}")
print(f"  Date range: {dataset['date'].min().strftime('%Y-Q%q')} to {dataset['date'].max().strftime('%Y-Q%q')}")


# ============================================================
# Encode categorical variables
# ============================================================

print("\n  Encoding categorical variables...")

# Country group: binary (0 = AE, 1 = EMDE)
dataset['is_emde'] = (dataset['country_group'] == 'EMDEs').astype(float)
dataset.loc[dataset['country_group'].isna(), 'is_emde'] = np.nan

# Income group: ordinal encoding (economic ordering)
income_map = {
    'High income': 4, 'Upper middle income': 3, 
    'Lower middle income': 2, 'Low income': 1,
    'Not classified': np.nan
}
dataset['income_encoded'] = dataset['income_group'].map(income_map)

# Region: one-hot encoding (no natural ordering)
if 'region' in dataset.columns:
    region_dummies = pd.get_dummies(dataset['region'], prefix='region', dtype=float)
    dataset = pd.concat([dataset, region_dummies], axis=1)

print(f"  Encoded dataset: {dataset.shape}")


# ============================================================
# Define train/validation/test splits
# ============================================================

print("\n" + "=" * 60)
print("STEP 6d: DEFINING TIME-BASED SPLITS")
print("=" * 60)

dataset['split'] = 'train'
dataset.loc[dataset['date'].dt.year.between(2015, 2018), 'split'] = 'validation'
dataset.loc[dataset['date'].dt.year.between(2019, 2020), 'split'] = 'test_covid'
dataset.loc[dataset['date'].dt.year >= 2021, 'split'] = 'test_ukraine'

# Also mark a "full_test" that combines both test periods
dataset['is_test'] = dataset['split'].isin(['test_covid', 'test_ukraine']).astype(int)


# ============================================================
# Country-demeaned features (within-country fixed effects)
# ============================================================
# Subtract each country's TRAIN-period mean from key inflation features.
# This removes time-invariant country baselines (e.g. Turkey having
# structurally higher inflation than Germany) so the model learns
# deviations from each country's norm, not country-level averages.
# Train-period means are computed on pre-2015 data only, then applied
# to all periods to avoid leakage from validation/test.

print("\n  Adding country-demeaned features (within-country FE)...")

demean_cols = [c for c in [
    'hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy',
    'hcpi_yoy_lag1', 'hcpi_yoy_lag2',
    'fcpi_yoy_lag1', 'ecpi_yoy_lag1',
    'hcpi_yoy_2q_change', 'hcpi_12m_vol',
    'component_disp_m',
] if c in dataset.columns]

# Country means computed on TRAIN set only
train_means = (
    dataset[dataset['split'] == 'train']
    .groupby('country_code')[demean_cols]
    .mean()
    .rename(columns={c: f'_cm_{c}' for c in demean_cols})
)
dataset = dataset.merge(train_means, on='country_code', how='left')

for col in demean_cols:
    dataset[f'{col}_dm'] = dataset[col] - dataset[f'_cm_{col}']

dataset.drop(columns=[f'_cm_{c}' for c in demean_cols], inplace=True)
print(f"  Added {len(demean_cols)} country-demeaned features")


print("\n  Split distribution:")
split_stats = dataset.groupby('split').agg(
    n_obs=('country_code', 'size'),
    n_countries=('country_code', 'nunique'),
    years=('date', lambda x: f"{x.dt.year.min()}-{x.dt.year.max()}"),
    up_rate_1q=('target_up_1q', lambda x: f"{x.mean()*100:.1f}%")
).reindex(['train', 'validation', 'test_covid', 'test_ukraine'])
print(split_stats.to_string())


# ============================================================
# Define the final feature list
# ============================================================

print("\n" + "=" * 60)
print("STEP 6e: FINAL FEATURE INVENTORY")
print("=" * 60)

# Categorise all features
target_cols = ['target_up_1q', 'target_up_2q', 'target_up_4q',
               'target_crisis_entry_1q', 'target_crisis_entry_2q', 'target_crisis_entry_4q',
               'regime_future_1q', 'regime_future_2q', 'regime_future_4q']
id_cols = ['country_code', 'country_name', 'quarter', 'date', 'year', 'split', 'is_test',
           'country_group', 'income_group', 'region']
regime_state_cols = ['regime', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2',
                     'regime_prob_3', 'regime_prob_4', 'regime_prob_5']

# Everything else is a potential feature
all_cols = set(dataset.columns)
non_feature_cols = set(target_cols + id_cols + regime_state_cols + 
                        [c for c in dataset.columns if 'regime_future' in c])

# Feature columns (everything that's not ID, target, or regime state we keep regime probs as features)
feature_candidates = sorted(all_cols - non_feature_cols - {'country_name', 'quarter'})

# Separate into groups for reporting
inflation_features = [c for c in feature_candidates if any(x in c for x in ['hcpi','fcpi','ecpi','ccpi','ppi','headline','food_energy','component_disp'])]
commodity_features = [c for c in feature_candidates if any(x in c for x in ['energy_index','food_commodity','metals_index','fertiliser_index','oil_price','wheat','rice'])]
fiscal_features = [c for c in feature_candidates if c in ['debt_gdp','primary_balance','fiscal_balance','ext_debt_gdp','st_debt_reserves','private_credit_gdp','sovereign_rating','concessional_share']]
country_features = [c for c in feature_candidates if c.startswith('region_') or c in ['is_emde','income_encoded']]
regime_features = [c for c in feature_candidates if 'regime_prob' in c or c == 'regime']

# The actual feature list for modelling
feature_list = inflation_features + commodity_features + fiscal_features + country_features + regime_features

print(f"\n  FEATURE GROUPS:")
print(f"    Inflation features:  {len(inflation_features)}")
for f in inflation_features:
    pct_valid = dataset[f].notna().mean()*100
    print(f"      {f:35s} ({pct_valid:.0f}% non-null)")

print(f"\n    Commodity features:  {len(commodity_features)}")
for f in commodity_features[:5]:
    print(f"      {f}")
if len(commodity_features) > 5:
    print(f"      ... and {len(commodity_features)-5} more")

print(f"\n    Fiscal features:     {len(fiscal_features)}")
for f in fiscal_features:
    pct_valid = dataset[f].notna().mean()*100
    print(f"      {f:35s} ({pct_valid:.0f}% non-null)")

print(f"\n    Country features:    {len(country_features)}")
for f in country_features:
    print(f"      {f}")

print(f"\n    Regime state:        {len(regime_features)}")
for f in regime_features:
    print(f"      {f}")

print(f"\n  TOTAL FEATURES: {len(feature_list)}")


# ============================================================
# Save outputs
# ============================================================

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Save the complete dataset
output_path = os.path.join(PROC, 'features_and_targets.csv')
dataset['quarter'] = dataset['quarter'].astype(str)
dataset.to_csv(output_path, index=False)
size_mb = os.path.getsize(output_path) / (1024*1024)
print(f"  Saved: features_and_targets.csv ({len(dataset):,} rows, {dataset.shape[1]} cols, {size_mb:.1f} MB)")

# Save the feature list (so Step 7 knows exactly which columns to use)
feature_info = pd.DataFrame({
    'feature': feature_list,
    'group': ['inflation']*len(inflation_features) + ['commodity']*len(commodity_features) + 
             ['fiscal']*len(fiscal_features) + ['country']*len(country_features) + 
             ['regime']*len(regime_features),
    'pct_non_null': [dataset[f].notna().mean()*100 if f in dataset.columns else 0 for f in feature_list]
})
feature_list_path = os.path.join(TDIR, 'table07_feature_list.csv')
feature_info.to_csv(feature_list_path, index=False)
print(f"  Saved: table07_feature_list.csv ({len(feature_info)} features)")

# Save split summary
split_summary_path = os.path.join(TDIR, 'table08_data_splits.csv')
split_detail = dataset.groupby('split').agg(
    n_observations=('country_code', 'size'),
    n_countries=('country_code', 'nunique'),
    year_start=('date', lambda x: x.dt.year.min()),
    year_end=('date', lambda x: x.dt.year.max()),
    target_up_1q_rate=('target_up_1q', 'mean'),
    target_up_2q_rate=('target_up_2q', 'mean'),
    target_up_4q_rate=('target_up_4q', 'mean'),
).reindex(['train', 'validation', 'test_covid', 'test_ukraine'])
split_detail.to_csv(split_summary_path)
print(f"  Saved: table08_data_splits.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"STEP 6 COMPLETE - FEATURE ENGINEERING SUMMARY")
print(f"{'='*60}")

print(f"""
DATASET: {output_path}
  Rows: {len(dataset):,}  (country-quarter observations)
  Countries: {dataset['country_code'].nunique()}
  Period: {dataset['date'].min().strftime('%Y-Q1')} to {dataset['date'].max().strftime('%Y-Q1')}
  Total columns: {dataset.shape[1]}

FEATURES: {len(feature_list)} predictor variables
  Inflation features:   {len(inflation_features):3d}  (levels, lags, momentum, volatility, gaps, demeaned)
  Commodity features:   {len(commodity_features):3d}  (prices, changes, volatility)
  Fiscal features:      {len(fiscal_features):3d}  (debt, balance, external debt, rating)
  Country features:     {len(country_features):3d}  (AE/EMDE, income, region dummies)
  Regime state:         {len(regime_features):3d}  (current regime + GMM probabilities)
  Note: *_dm features are country-demeaned (train-mean subtracted per country)

TARGET VARIABLES: 3 primary + 3 sensitivity targets
  target_up_1q:           Any upward transition, 1Q  ({dataset['target_up_1q'].mean()*100:.1f}% positive)
  target_up_2q:           Any upward transition, 2Q  ({dataset['target_up_2q'].mean()*100:.1f}% positive)
  target_up_4q:           Any upward transition, 4Q  ({dataset['target_up_4q'].mean()*100:.1f}% positive)
  target_crisis_entry_1q: Entry into R4/R5, 1Q       ({dataset['target_crisis_entry_1q'].mean()*100:.1f}% positive)
  target_crisis_entry_2q: Entry into R4/R5, 2Q       ({dataset['target_crisis_entry_2q'].mean()*100:.1f}% positive)
  target_crisis_entry_4q: Entry into R4/R5, 4Q       ({dataset['target_crisis_entry_4q'].mean()*100:.1f}% positive)

TIME-BASED SPLITS (no data leakage):
  Train:          {(dataset['split']=='train').sum():,} obs  (2000-2014)
  Validation:     {(dataset['split']=='validation').sum():,} obs  (2015-2018)
  Test (COVID):   {(dataset['split']=='test_covid').sum():,} obs  (2019-2020)
  Test (Ukraine): {(dataset['split']=='test_ukraine').sum():,} obs  (2021-2023+)

""")
