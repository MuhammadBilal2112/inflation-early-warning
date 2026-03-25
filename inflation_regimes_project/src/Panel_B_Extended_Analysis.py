"""
==============================================================================
PANEL B: EXTENDED ANALYSIS WITH 4 CPI MEASURES
==============================================================================

This script runs a PARALLEL analysis using headline + food + energy + CORE CPI
on a subset of 74 countries, then compares results against the main Panel A
(3 measures, 135 countries).

The comparison answers: "Does adding core CPI change the regime structure
or improve prediction accuracy?"

PIPELINE:
  Part 1: Build 4-measure regime features (10 features including core gaps)
  Part 2: GMM regime discovery on 74-country subsample
  Part 3: Feature engineering with core CPI features
  Part 4: Train models (same specs as Panel A)
  Part 5: Head-to-head comparison (Panel A vs Panel B)
  Part 6: SHAP comparison (do different features matter?)

RUN: python3 Panel_B_Extended_Analysis.py
TIME: 10-15 minutes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time, re, joblib
warnings.filterwarnings('ignore')

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              accuracy_score, brier_score_loss, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import shap

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

# Load data
print("=" * 70)
print("PANEL B: EXTENDED ANALYSIS (4 CPI MEASURES)")
print("=" * 70)

print("\nLoading data...")
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])
print(f"  Monthly panel: {monthly.shape}")


# ============================================================
# PART 1: BUILD 4-MEASURE REGIME FEATURES
# ============================================================
print("\n" + "=" * 70)
print("PART 1: BUILDING 4-MEASURE REGIME FEATURES")
print("=" * 70)

# Filter to observations with all 4 measures
four_measures = ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy', 'ccpi_yoy']
has_4 = monthly[four_measures].notna().all(axis=1)
monthly_4 = monthly[has_4].copy()

print(f"  4-measure observations: {has_4.sum():,}")
print(f"  Countries: {monthly_4['country_code'].nunique()}")

# Aggregate to quarterly
monthly_4['quarter'] = monthly_4['date'].dt.to_period('Q')

quarterly = monthly_4.groupby(['country_code', 'quarter']).agg({
    'hcpi_yoy': 'mean', 'fcpi_yoy': 'mean', 'ecpi_yoy': 'mean', 'ccpi_yoy': 'mean',
    'ppi_yoy': 'mean',
    'country_name': 'first', 'country_group': 'first',
    'income_group': 'first', 'region': 'first',
}).reset_index()

quarterly['date'] = quarterly['quarter'].dt.to_timestamp()
quarterly['year'] = quarterly['date'].dt.year

print(f"  Quarterly panel: {quarterly.shape}")

# Build regime features (10 features — more than Panel A's 7)
quarterly = quarterly.sort_values(['country_code', 'date']).reset_index(drop=True)

# Levels (4 measures)
quarterly['hcpi_level'] = quarterly['hcpi_yoy']
quarterly['fcpi_level'] = quarterly['fcpi_yoy']
quarterly['ecpi_level'] = quarterly['ecpi_yoy']
quarterly['ccpi_level'] = quarterly['ccpi_yoy']

# Component dispersion (now across 4 measures, not 3)
quarterly['component_dispersion'] = quarterly[four_measures].std(axis=1)

# Headline momentum
quarterly['hcpi_momentum'] = quarterly.groupby('country_code')['hcpi_yoy'].diff(2)

# Food-energy gap
quarterly['food_energy_gap'] = quarterly['fcpi_yoy'] - quarterly['ecpi_yoy']

# Headline-core gap
quarterly['headline_core_gap'] = quarterly['hcpi_yoy'] - quarterly['ccpi_yoy']

# Headline volatility
quarterly['hcpi_volatility'] = quarterly.groupby('country_code')['hcpi_yoy'].transform(
    lambda x: x.rolling(4, min_periods=2).std()
)

# Core CPI momentum
quarterly['ccpi_momentum'] = quarterly.groupby('country_code')['ccpi_yoy'].diff(2)

# Feature list for GMM (10 features)
gmm_features_b = ['hcpi_level', 'fcpi_level', 'ecpi_level', 'ccpi_level',
                    'component_dispersion', 'hcpi_momentum', 'food_energy_gap',
                    'headline_core_gap', 'hcpi_volatility', 'ccpi_momentum']

regime_ready = quarterly.dropna(subset=gmm_features_b).copy()
print(f"  After feature engineering: {len(regime_ready):,} obs, {regime_ready['country_code'].nunique()} countries")
print(f"  GMM features: {len(gmm_features_b)} (vs 7 in Panel A)")


# ============================================================
# PART 2: GMM REGIME DISCOVERY (TRAIN-ONLY)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: GMM REGIME DISCOVERY (4 MEASURES, TRAIN-ONLY)")
print("=" * 70)

# Split: train on pre-2015 only
train_cutoff = pd.Timestamp('2015-01-01')
train_data = regime_ready[regime_ready['date'] < train_cutoff].copy()
future_data = regime_ready[regime_ready['date'] >= train_cutoff].copy()

print(f"  Train data: {len(train_data):,} obs")
print(f"  Future data: {len(future_data):,} obs")

# Winsorise and standardise train data
X_train_raw = train_data[gmm_features_b].values.copy()
for i in range(X_train_raw.shape[1]):
    p01, p99 = np.percentile(X_train_raw[:, i], [1, 99])
    X_train_raw[:, i] = np.clip(X_train_raw[:, i], p01, p99)

gmm_scaler = StandardScaler()
X_train_scaled = gmm_scaler.fit_transform(X_train_raw)

# Try K=2 to K=7
print(f"\n  Fitting GMMs...")
best_bic = np.inf
best_k = 4
for k in range(2, 8):
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=10,
                            max_iter=300, random_state=42, reg_covar=1e-5)
    gmm.fit(X_train_scaled)
    bic = gmm.bic(X_train_scaled)
    print(f"    K={k}: BIC={bic:,.0f}")
    if bic < best_bic:
        best_bic = bic
        best_k = k

print(f"\n  Best K by BIC: {best_k}")

# Fit final GMM with best K
gmm_b = GaussianMixture(n_components=best_k, covariance_type='full', n_init=10,
                          max_iter=300, random_state=42, reg_covar=1e-5)
gmm_b.fit(X_train_scaled)

# Assign labels to all data
# Train
train_labels = gmm_b.predict(X_train_scaled)
train_probs = gmm_b.predict_proba(X_train_scaled)

# Future
X_future_raw = future_data[gmm_features_b].values.copy()
for i in range(X_future_raw.shape[1]):
    p01, p99 = np.percentile(X_train_raw[:, i], [1, 99])  # Use TRAIN percentiles
    X_future_raw[:, i] = np.clip(X_future_raw[:, i], p01, p99)
X_future_scaled = gmm_scaler.transform(X_future_raw)
future_labels = gmm_b.predict(X_future_scaled)
future_probs = gmm_b.predict_proba(X_future_scaled)

# Sort regimes by headline inflation (train-only ordering)
train_regime_means = pd.DataFrame({'label': train_labels, 'hcpi': X_train_raw[:, 0]})
order = train_regime_means.groupby('label')['hcpi'].mean().sort_values().index.tolist()
label_map = {old: new for new, old in enumerate(order)}

train_labels_sorted = np.array([label_map[l] for l in train_labels])
future_labels_sorted = np.array([label_map[l] for l in future_labels])

# Remap probabilities
train_probs_sorted = np.zeros_like(train_probs)
future_probs_sorted = np.zeros_like(future_probs)
for old, new in label_map.items():
    train_probs_sorted[:, new] = train_probs[:, old]
    future_probs_sorted[:, new] = future_probs[:, old]

# Assign to data
train_data['regime'] = train_labels_sorted
future_data['regime'] = future_labels_sorted
for i in range(best_k):
    train_data[f'regime_prob_{i}'] = train_probs_sorted[:, i]
    future_data[f'regime_prob_{i}'] = future_probs_sorted[:, i]

regime_b = pd.concat([train_data, future_data]).sort_values(['country_code', 'date']).reset_index(drop=True)

# Print regime characteristics
print(f"\n  Panel B regime characteristics ({best_k} regimes):")
chars = regime_b.groupby('regime')[four_measures + ['headline_core_gap']].mean()
for r in range(best_k):
    if r in chars.index:
        h = chars.loc[r, 'hcpi_yoy']
        core = chars.loc[r, 'ccpi_yoy']
        gap = chars.loc[r, 'headline_core_gap']
        n = (regime_b['regime'] == r).sum()
        print(f"    R{r}: headline={h:.1f}%, core={core:.1f}%, h-c gap={gap:.1f}pp (n={n:,})")

# Save
regime_b_path = os.path.join(PROC, 'regime_labels_panelB.csv')
regime_b['quarter'] = regime_b['quarter'].astype(str)
regime_b.to_csv(regime_b_path, index=False)
print(f"\n  Saved: regime_labels_panelB.csv")


# ============================================================
# PART 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("PART 3: FEATURE ENGINEERING (PANEL B)")
print("=" * 70)

# Build prediction features from monthly data (only 4-measure countries)
countries_b = regime_b['country_code'].unique()
monthly_b = monthly[monthly['country_code'].isin(countries_b)].copy()

# Compute monthly features
for col in ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy', 'ccpi_yoy']:
    base = col.split('_')[0]
    monthly_b[f'{base}_mom_change'] = monthly_b.groupby('country_code')[col].diff(1)
    monthly_b[f'{base}_6m_vol'] = monthly_b.groupby('country_code')[col].transform(
        lambda x: x.rolling(6, min_periods=3).std())

# Cross-component features (now including core)
monthly_b['headline_food_gap'] = monthly_b['hcpi_yoy'] - monthly_b['fcpi_yoy']
monthly_b['headline_energy_gap'] = monthly_b['hcpi_yoy'] - monthly_b['ecpi_yoy']
monthly_b['headline_core_gap_m'] = monthly_b['hcpi_yoy'] - monthly_b['ccpi_yoy']
monthly_b['food_energy_gap_m'] = monthly_b['fcpi_yoy'] - monthly_b['ecpi_yoy']
monthly_b['component_disp_m'] = monthly_b[four_measures].std(axis=1)

# Aggregate to quarterly
monthly_b['quarter'] = monthly_b['date'].dt.to_period('Q')

agg_cols = {
    'hcpi_yoy': 'mean', 'fcpi_yoy': 'mean', 'ecpi_yoy': 'mean', 'ccpi_yoy': 'mean',
    'ppi_yoy': 'mean',
    'hcpi_mom_change': 'mean', 'fcpi_mom_change': 'mean', 'ecpi_mom_change': 'mean',
    'ccpi_mom_change': 'mean',
    'hcpi_6m_vol': 'last', 'fcpi_6m_vol': 'last', 'ecpi_6m_vol': 'last', 'ccpi_6m_vol': 'last',
    'headline_food_gap': 'mean', 'headline_energy_gap': 'mean',
    'headline_core_gap_m': 'mean', 'food_energy_gap_m': 'mean',
    'component_disp_m': 'mean',
    # Commodity features
    'energy_index': 'last', 'food_commodity_index': 'last',
    'energy_index_3m_chg': 'last', 'energy_index_12m_chg': 'last',
    'food_commodity_index_12m_chg': 'last',
    'oil_price_12m_chg': 'last', 'oil_price_6m_vol': 'last',
    # Fiscal features
    'debt_gdp': 'last', 'primary_balance': 'last', 'fiscal_balance': 'last',
    'sovereign_rating': 'last', 'private_credit_gdp': 'last',
    'country_group': 'first', 'income_group': 'first', 'region': 'first',
}
agg_cols = {k: v for k, v in agg_cols.items() if k in monthly_b.columns}

qf = monthly_b.groupby(['country_code', 'quarter']).agg(agg_cols).reset_index()
qf['date'] = qf['quarter'].dt.to_timestamp()

# Add lags
qf = qf.sort_values(['country_code', 'date']).reset_index(drop=True)
for col in ['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy', 'ccpi_yoy']:
    if col in qf.columns:
        qf[f'{col}_lag1'] = qf.groupby('country_code')[col].shift(1)
        qf[f'{col}_lag2'] = qf.groupby('country_code')[col].shift(2)

# Merge with regime labels
regime_merge = regime_b[['country_code', 'date', 'regime'] +
                         [f'regime_prob_{i}' for i in range(best_k)]].copy()
df_b = qf.merge(regime_merge, on=['country_code', 'date'], how='inner')

# Build targets
df_b = df_b.sort_values(['country_code', 'date']).reset_index(drop=True)
for label, h in {'1q': 1, '2q': 2, '4q': 4}.items():
    df_b[f'regime_future_{label}'] = df_b.groupby('country_code')['regime'].shift(-h)
    df_b[f'target_up_{label}'] = (df_b[f'regime_future_{label}'] > df_b['regime']).astype(float)
    df_b.loc[df_b[f'regime_future_{label}'].isna(), f'target_up_{label}'] = np.nan

# Encode categoricals
df_b['is_emde'] = (df_b['country_group'] == 'EMDEs').astype(float)
income_map = {'High income': 4, 'Upper middle income': 3, 'Lower middle income': 2, 'Low income': 1}
df_b['income_encoded'] = df_b['income_group'].map(income_map)
if 'region' in df_b.columns:
    dummies = pd.get_dummies(df_b['region'], prefix='region', dtype=float)
    dummies.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in dummies.columns]
    df_b = pd.concat([df_b, dummies], axis=1)

# Clean column names
clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df_b.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df_b = df_b.rename(columns=clean_map)

# Define splits
df_b['split'] = 'train'
df_b.loc[df_b['date'].dt.year.between(2015, 2018), 'split'] = 'validation'
df_b.loc[df_b['date'].dt.year.between(2019, 2020), 'split'] = 'test_covid'
df_b.loc[df_b['date'].dt.year >= 2021, 'split'] = 'test_ukraine'

# Define feature columns
exclude = {'country_code', 'country_name', 'quarter', 'date', 'year', 'split',
           'country_group', 'income_group', 'region',
           'target_up_1q', 'target_up_2q', 'target_up_4q',
           'regime_future_1q', 'regime_future_2q', 'regime_future_4q'}
feature_cols_b = [c for c in df_b.columns if c not in exclude]
feature_cols_b = [c for c in feature_cols_b if df_b.loc[df_b['split'] == 'train', c].notna().mean() > 0.40]

print(f"  Feature matrix: {df_b.shape}")
print(f"  Features: {len(feature_cols_b)}")
print(f"  Target rates:")
for t in ['target_up_1q', 'target_up_2q', 'target_up_4q']:
    v = df_b[t].dropna()
    print(f"    {t}: {v.mean()*100:.1f}% (n={len(v):,})")

print(f"\n  Split sizes:")
for s in ['train', 'validation', 'test_covid', 'test_ukraine']:
    print(f"    {s}: {(df_b['split']==s).sum():,}")


# ============================================================
# PART 4: TRAIN MODELS (SAME SPECS AS PANEL A)
# ============================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING MODELS (PANEL B)")
print("=" * 70)

targets = {'1Q ahead': 'target_up_1q', '2Q ahead': 'target_up_2q', '4Q ahead': 'target_up_4q'}

# Quick HP tuning for XGBoost on Panel B
Xt_tune = df_b[df_b['split'] == 'train'].dropna(subset=['target_up_2q'])
yt_tune = Xt_tune['target_up_2q'].values
Xt_tune = Xt_tune[feature_cols_b]
avg_pos = yt_tune.mean()
cw = (1 - avg_pos) / avg_pos

tscv = TimeSeriesSplit(n_splits=4)
best_xgb_score = -1
best_xgb_params = {}

print("\n  Tuning XGBoost on Panel B...")
for depth in [4, 6, 8]:
    for lr in [0.03, 0.05]:
        fold_scores = []
        params = {'n_estimators': 300, 'max_depth': depth, 'learning_rate': lr,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': cw,
                  'eval_metric': 'logloss', 'random_state': 42, 'verbosity': 0, 'tree_method': 'hist'}
        for tr_idx, val_idx in tscv.split(Xt_tune):
            X_tr, X_val = Xt_tune.iloc[tr_idx], Xt_tune.iloc[val_idx]
            y_tr, y_val = yt_tune[tr_idx], yt_tune[val_idx]
            if y_val.sum() == 0: continue
            mdl = xgb.XGBClassifier(**params)
            mdl.fit(X_tr, y_tr)
            try:
                fold_scores.append(roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1]))
            except: pass
        if fold_scores:
            mean_s = np.mean(fold_scores)
            if mean_s > best_xgb_score:
                best_xgb_score = mean_s
                best_xgb_params = params

print(f"    Best CV AUC: {best_xgb_score:.4f}")

# Train all models
models_b = {
    'Logistic Regression': {
        'cls': LogisticRegression,
        'p': {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42, 'solver': 'lbfgs'},
        'imp': True, 'sc': True
    },
    'Random Forest': {
        'cls': RandomForestClassifier,
        'p': {'n_estimators': 500, 'max_depth': 12, 'min_samples_leaf': 20, 'max_features': 'sqrt',
              'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'imp': True, 'sc': False
    },
    'XGBoost': {
        'cls': xgb.XGBClassifier,
        'p': best_xgb_params,
        'imp': False, 'sc': False
    },
}

results_b = []
best_model_b = None
best_model_features = None

for hn, tc in targets.items():
    print(f"\n  {hn}:")
    train = df_b[df_b['split'] == 'train'].dropna(subset=[tc])
    Xtr = train[feature_cols_b]; ytr = train[tc].values
    
    imp = SimpleImputer(strategy='median'); sc = StandardScaler()
    Xtr_imp = pd.DataFrame(imp.fit_transform(Xtr), columns=feature_cols_b, index=Xtr.index)
    Xtr_sc = pd.DataFrame(sc.fit_transform(Xtr_imp), columns=feature_cols_b, index=Xtr.index)
    
    for mn, cfg in models_b.items():
        Xtr_use = Xtr_sc if cfg['sc'] else (Xtr_imp if cfg['imp'] else Xtr)
        mdl = cfg['cls'](**cfg['p'])
        mdl.fit(Xtr_use, ytr)
        
        for pn, sn in [('Validation', 'validation'), ('COVID', 'test_covid'), ('Ukraine', 'test_ukraine')]:
            test = df_b[df_b['split'] == sn].dropna(subset=[tc])
            if len(test) == 0 or test[tc].sum() == 0: continue
            Xte = test[feature_cols_b]; yte = test[tc].values
            
            if cfg['sc']:
                Xte_i = pd.DataFrame(imp.transform(Xte), columns=feature_cols_b, index=Xte.index)
                Xte_use = pd.DataFrame(sc.transform(Xte_i), columns=feature_cols_b, index=Xte.index)
            elif cfg['imp']:
                Xte_use = pd.DataFrame(imp.transform(Xte), columns=feature_cols_b, index=Xte.index)
            else:
                Xte_use = Xte
            
            yp = mdl.predict_proba(Xte_use)[:, 1]
            try: auc = roc_auc_score(yte, yp)
            except: auc = np.nan
            try: prauc = average_precision_score(yte, yp)
            except: prauc = np.nan
            brier = brier_score_loss(yte, yp)
            
            results_b.append({
                'Panel': 'B (4 measures, 74 countries)',
                'Horizon': hn, 'Model': mn, 'Period': pn,
                'AUC-ROC': round(auc, 4), 'PR-AUC': round(prauc, 4),
                'Brier': round(brier, 4), 'N': len(yte),
            })
            print(f"    {mn:22s} {pn:12s}: AUC={auc:.3f}")
        
        # Save best model for SHAP
        if hn == '2Q ahead' and mn == 'XGBoost':
            best_model_b = mdl
            best_model_features = feature_cols_b
            joblib.dump({'model': mdl, 'imputer': imp, 'scaler': sc, 'features': feature_cols_b},
                        os.path.join(MDIR, 'xgboost_panelB_target_up_2q.joblib'))

rdf_b = pd.DataFrame(results_b)


# ============================================================
# PART 5: HEAD-TO-HEAD COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("PART 5: PANEL A vs PANEL B COMPARISON")
print("=" * 70)

# Load Panel A results
panel_a_path = os.path.join(TDIR, 'table09_model_comparison_revised.csv')
if os.path.exists(panel_a_path):
    rdf_a = pd.read_csv(panel_a_path)
    rdf_a['Panel'] = 'A (3 measures, 135 countries)'
    
    # Compare on Ukraine test (both panels)
    print("\n  XGBoost AUC-ROC on Ukraine test:")
    print(f"  {'Horizon':<12s} {'Panel A (3m, 135c)':<22s} {'Panel B (4m, 74c)':<22s} {'Difference':<12s}")
    print(f"  {'-'*68}")
    
    comparison_rows = []
    for hn in targets:
        a_row = rdf_a[(rdf_a['Horizon'] == hn) & (rdf_a['Model'] == 'XGBoost') &
                       (rdf_a['Period'].str.contains('Ukraine'))]['AUC-ROC']
        b_row = rdf_b[(rdf_b['Horizon'] == hn) & (rdf_b['Model'] == 'XGBoost') &
                       (rdf_b['Period'] == 'Ukraine')]['AUC-ROC']
        
        a_val = a_row.values[0] if len(a_row) > 0 else np.nan
        b_val = b_row.values[0] if len(b_row) > 0 else np.nan
        diff = b_val - a_val if not (np.isnan(a_val) or np.isnan(b_val)) else np.nan
        
        better = 'B' if diff > 0.005 else ('A' if diff < -0.005 else 'Similar')
        print(f"  {hn:<12s} {a_val:<22.4f} {b_val:<22.4f} {diff:+.4f} ({better})")
        
        comparison_rows.append({
            'Horizon': hn, 'Panel_A_AUC': a_val, 'Panel_B_AUC': b_val,
            'Difference': round(diff, 4) if not np.isnan(diff) else None,
            'Better': better,
        })
    
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(TDIR, 'table22_panel_comparison.csv'), index=False)
    print(f"\n  Saved: table22_panel_comparison.csv")
else:
    print("  WARNING: Panel A results not found for comparison")

# Regime comparison
print(f"\n  Regime structure comparison:")
print(f"  Panel A: 6 regimes, 135 countries, 3 CPI measures")
print(f"  Panel B: {best_k} regimes, 74 countries, 4 CPI measures (+core)")
print(f"  Panel B adds headline-core gap as regime-defining feature")


# ============================================================
# PART 6: SHAP COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("PART 6: SHAP ANALYSIS (PANEL B)")
print("=" * 70)

if best_model_b is not None:
    test_b = df_b[df_b['split'].isin(['test_covid', 'test_ukraine'])].dropna(subset=['target_up_2q'])
    X_test_b = test_b[feature_cols_b]
    
    print(f"  Computing SHAP values ({len(X_test_b)} obs)...")
    explainer = shap.TreeExplainer(best_model_b)
    shap_vals = explainer.shap_values(X_test_b)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    
    # SHAP importance
    mean_shap = pd.DataFrame({
        'feature': feature_cols_b,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\n  Top 15 features (Panel B):")
    for i, row in mean_shap.head(15).iterrows():
        print(f"    {row['feature']:35s} {row['mean_abs_shap']:.4f}")
    
    # Check: where do core CPI features rank?
    core_features = [f for f in feature_cols_b if 'ccpi' in f or 'core' in f.lower()]
    print(f"\n  Core CPI feature rankings:")
    for cf in core_features:
        rank = mean_shap[mean_shap['feature'] == cf].index[0] + 1 if cf in mean_shap['feature'].values else '?'
        shap_val = mean_shap[mean_shap['feature'] == cf]['mean_abs_shap'].values[0] if cf in mean_shap['feature'].values else 0
        total_rank = (mean_shap['feature'] == cf).idxmax() + 1
        print(f"    {cf:35s} rank={list(mean_shap['feature']).index(cf)+1}/{len(feature_cols_b)}, SHAP={shap_val:.4f}")
    
    # Save SHAP
    mean_shap.to_csv(os.path.join(TDIR, 'table23_shap_importance_panelB.csv'), index=False)
    
    # Figure: Panel A vs Panel B feature importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel B SHAP
    top20 = mean_shap.head(20)
    def gc(f):
        if 'ccpi' in f or 'core' in f.lower(): return C['teal']  # Core CPI features in teal
        elif any(x in f for x in ['hcpi', 'fcpi', 'ecpi', 'headline', 'food_energy', 'component']): return C['blue']
        elif any(x in f for x in ['energy_index', 'food_commodity', 'oil_price']): return C['red']
        elif f in ['debt_gdp', 'primary_balance', 'fiscal_balance', 'sovereign_rating', 'private_credit_gdp']: return C['green']
        elif 'regime' in f: return C['purple']
        else: return C['gray']
    
    axes[1].barh(range(len(top20)), top20['mean_abs_shap'].values,
                  color=[gc(f) for f in top20['feature']], alpha=0.85)
    axes[1].set_yticks(range(len(top20)))
    axes[1].set_yticklabels(top20['feature'].values, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean |SHAP|')
    axes[1].set_title(f'Panel B: 4 measures, 74 countries', fontweight='bold')
    
    # Panel A SHAP (load from saved)
    shap_a_path = os.path.join(TDIR, 'table15_shap_importance.csv')
    if os.path.exists(shap_a_path):
        shap_a = pd.read_csv(shap_a_path).head(20)
        axes[0].barh(range(len(shap_a)), shap_a['mean_abs_shap'].values,
                      color=[gc(f) for f in shap_a['feature']], alpha=0.85)
        axes[0].set_yticks(range(len(shap_a)))
        axes[0].set_yticklabels(shap_a['feature'].values, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Mean |SHAP|')
        axes[0].set_title(f'Panel A: 3 measures, 135 countries', fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=C['blue'], label='Inflation'), Patch(fc=C['teal'], label='Core CPI (new)'),
                        Patch(fc=C['red'], label='Commodity'), Patch(fc=C['green'], label='Fiscal'),
                        Patch(fc=C['purple'], label='Regime'), Patch(fc=C['gray'], label='Country')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9)
    
    plt.suptitle('Figure 37: Feature importance comparison — Panel A vs Panel B',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_fig('fig37_panel_comparison_shap')


# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"PANEL B ANALYSIS COMPLETE")
print(f"{'='*70}")

print(f"""
PANEL B COMPLETE:
  Countries: 74 (34 AE + 40 EMDE), Regimes: {best_k}, Features: {len(feature_cols_b)}

FILES CREATED:
  data/processed/regime_labels_panelB.csv
  outputs/tables/table22_panel_comparison.csv
  outputs/tables/table23_shap_importance_panelB.csv
  outputs/figures/fig37_panel_comparison_shap.png

TOTAL TIME: {total_time/60:.1f} minutes
""")
