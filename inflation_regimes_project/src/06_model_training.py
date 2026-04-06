"""
==============================================================================
STEP 7 REVISED: RIGOROUS MODEL TRAINING AND EVALUATION
==============================================================================

  - GMM re-fitted on train data only (pre-2015), no look-ahead bias
  - Hyperparameter tuning via TimeSeriesSplit cross-validation
  - Bootstrap confidence intervals (1000 samples) for AUC-ROC
  - Expanding window validation (2012-2018)
  - Brier score + calibration curves
  - Threshold search 0.05-0.70
  - Bootstrap DeLong test for model comparison

RUN: python3 Step07_Revised_Model_Training.py
TIME: 20-40 minutes (hyperparameter tuning is the slow part)
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time, joblib, re
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              accuracy_score, roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

plt.rcParams.update({'figure.figsize':(12,6),'figure.dpi':150,'font.size':11,
    'font.family':'sans-serif','figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False})

C = {'blue':'#1F4E79','red':'#C62828','green':'#2E7D32','amber':'#E8A838',
     'purple':'#6A1B9A','teal':'#00838F','gray':'#616161'}
MC = {'Logistic Regression':C['gray'],'Random Forest':C['green'],
      'XGBoost':C['blue'],'LightGBM':C['red']}

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
# PART 1: FIX GMM LEAKAGE — Refit on train data only
# ============================================================
print("=" * 70)
print("PART 1: FIXING GMM LEAKAGE — Refitting on train data only")
print("=" * 70)

# Load the regime data (has the GMM features already computed)
regime_full = pd.read_csv(os.path.join(PROC, 'regime_labels.csv'), parse_dates=['date'])
monthly = pd.read_csv(os.path.join(PROC, 'master_panel_monthly.csv'), parse_dates=['date'])

gmm_features = ['hcpi_level', 'fcpi_level', 'ecpi_level', 'component_dispersion',
                 'hcpi_momentum', 'food_energy_gap', 'hcpi_volatility']

# Split: train-only for GMM fitting
train_cutoff = pd.Timestamp('2015-01-01')
regime_train = regime_full[regime_full['date'] < train_cutoff].copy()
regime_future = regime_full[regime_full['date'] >= train_cutoff].copy()

# Prepare train features for GMM
train_gmm_data = regime_train.dropna(subset=gmm_features).copy()
X_gmm_train_raw = train_gmm_data[gmm_features].values

# Winsorise (same as Step 5)
for i, col in enumerate(gmm_features):
    p01 = np.percentile(X_gmm_train_raw[:, i], 1)
    p99 = np.percentile(X_gmm_train_raw[:, i], 99)
    X_gmm_train_raw[:, i] = np.clip(X_gmm_train_raw[:, i], p01, p99)

# Standardise (fit on train only)
gmm_scaler = StandardScaler()
X_gmm_train = gmm_scaler.fit_transform(X_gmm_train_raw)

# Fit GMM on train data only (use K=6 as discovered in Step 5)
print(f"\n  Fitting GMM (K=6) on train data only ({len(X_gmm_train):,} obs)...")
gmm_train = GaussianMixture(
    n_components=6, covariance_type='full', n_init=10,
    max_iter=300, random_state=42, reg_covar=1e-5
)
gmm_train.fit(X_gmm_train)
print(f"  GMM converged: {gmm_train.converged_}")

# Assign regimes to TRAIN data
train_labels = gmm_train.predict(X_gmm_train)
train_probs = gmm_train.predict_proba(X_gmm_train)

# Now assign regimes to FUTURE data using the TRAIN-fitted GMM
future_gmm_data = regime_future.dropna(subset=gmm_features).copy()
X_gmm_future_raw = future_gmm_data[gmm_features].values

# Winsorise using TRAIN percentiles
for i, col in enumerate(gmm_features):
    p01 = np.percentile(regime_train.dropna(subset=gmm_features)[col].values, 1)
    p99 = np.percentile(regime_train.dropna(subset=gmm_features)[col].values, 99)
    X_gmm_future_raw[:, i] = np.clip(X_gmm_future_raw[:, i], p01, p99)

# Transform using TRAIN-fitted scaler
X_gmm_future = gmm_scaler.transform(X_gmm_future_raw)

# Predict using TRAIN-fitted GMM
future_labels = gmm_train.predict(X_gmm_future)
future_probs = gmm_train.predict_proba(X_gmm_future)

# Sort regimes by headline inflation (same logic as Step 5)
all_labels_combined = np.concatenate([train_labels, future_labels])
all_features_combined = np.concatenate([X_gmm_train_raw, X_gmm_future_raw])
# Use hcpi_level (column 0) to sort
regime_means = pd.DataFrame({'label': all_labels_combined, 'hcpi': all_features_combined[:, 0]})
order = regime_means.groupby('label')['hcpi'].mean().sort_values().index.tolist()
label_map = {old: new for new, old in enumerate(order)}

# Re-map all labels
train_labels_sorted = np.array([label_map[l] for l in train_labels])
future_labels_sorted = np.array([label_map[l] for l in future_labels])

train_probs_sorted = np.zeros_like(train_probs)
future_probs_sorted = np.zeros_like(future_probs)
for old, new in label_map.items():
    train_probs_sorted[:, new] = train_probs[:, old]
    future_probs_sorted[:, new] = future_probs[:, old]

# Update the regime data
train_gmm_data['regime'] = train_labels_sorted
future_gmm_data['regime'] = future_labels_sorted
for i in range(6):
    train_gmm_data[f'regime_prob_{i}'] = train_probs_sorted[:, i]
    future_gmm_data[f'regime_prob_{i}'] = future_probs_sorted[:, i]

# Combine back
regime_fixed = pd.concat([train_gmm_data, future_gmm_data]).sort_values(['country_code', 'date']).reset_index(drop=True)

# Print regime characteristics
print(f"\n  Train-fitted GMM regime characteristics:")
chars = regime_fixed.groupby('regime')[['hcpi_yoy', 'fcpi_yoy', 'ecpi_yoy']].mean()
for r in range(6):
    if r in chars.index:
        h = chars.loc[r, 'hcpi_yoy']
        n = (regime_fixed['regime'] == r).sum()
        name = "Low" if h < 3 else ("Low-energy" if h < 4 else ("Moderate" if h < 7 else ("Moderate-energy" if h < 10 else ("Elevated" if h < 20 else "Crisis"))))
        print(f"    R{r}: {name:15s} headline={h:.1f}% (n={n:,})")

print(f"\n  Total regime-labelled obs: {len(regime_fixed):,}")
print(f"  GMM fitted on train data only.")

# Save fixed regime labels
regime_fixed_path = os.path.join(PROC, 'regime_labels_fixed.csv')
regime_fixed['quarter'] = regime_fixed['quarter'].astype(str) if 'quarter' in regime_fixed.columns else regime_fixed['date'].dt.to_period('Q').astype(str)
regime_fixed.to_csv(regime_fixed_path, index=False)
print(f"  Saved: regime_labels_fixed.csv")


# ============================================================
# PART 2: Rebuild features with fixed regime labels
# ============================================================
print("\n" + "=" * 70)
print("PART 2: REBUILDING FEATURE MATRIX WITH FIXED REGIMES")
print("=" * 70)

# Load the original features (non-regime columns are fine)
df_orig = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])

# Clean column names for LightGBM
clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df_orig.columns if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df_orig = df_orig.rename(columns=clean_map)

# Replace regime columns with fixed versions
regime_cols_to_replace = ['regime', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2',
                           'regime_prob_3', 'regime_prob_4', 'regime_prob_5']

# Drop old regime columns
df = df_orig.drop(columns=[c for c in regime_cols_to_replace if c in df_orig.columns], errors='ignore')

# Merge fixed regime columns
regime_merge = regime_fixed[['country_code', 'date'] + regime_cols_to_replace].copy()
df = df.merge(regime_merge, on=['country_code', 'date'], how='left')

# Rebuild targets using FIXED regime labels
df = df.sort_values(['country_code', 'date']).reset_index(drop=True)
for label, h in {'1q': 1, '2q': 2, '4q': 4}.items():
    df[f'regime_future_{label}'] = df.groupby('country_code')['regime'].shift(-h)
    df[f'target_up_{label}'] = (df[f'regime_future_{label}'] > df['regime']).astype(float)
    df.loc[df[f'regime_future_{label}'].isna(), f'target_up_{label}'] = np.nan

print(f"  Feature matrix rebuilt: {df.shape}")
print(f"  Target rates (with fixed regimes):")
for t in ['target_up_1q', 'target_up_2q', 'target_up_4q']:
    v = df[t].dropna()
    print(f"    {t}: {v.mean()*100:.1f}% positive (n={len(v):,})")


# ============================================================
# PART 3: Define features and splits
# ============================================================
print("\n" + "=" * 70)
print("PART 3: FEATURE SELECTION AND SPLITS")
print("=" * 70)

exclude_cols = {'country_code', 'country_name', 'quarter', 'date', 'year',
                'split', 'is_test', 'country_group', 'income_group', 'region',
                'target_up_1q', 'target_up_2q', 'target_up_4q',
                'regime_future_1q', 'regime_future_2q', 'regime_future_4q'}

feature_cols = [c for c in df.columns if c not in exclude_cols]
train_mask = df['split'] == 'train'
feature_cols = [c for c in feature_cols if df.loc[train_mask, c].notna().mean() > 0.40]
print(f"  Features: {len(feature_cols)}")

targets = {'1Q ahead': 'target_up_1q', '2Q ahead': 'target_up_2q', '4Q ahead': 'target_up_4q'}
eval_periods = {'Validation (2015-18)': 'validation', 'Test: COVID (2019-20)': 'test_covid',
                'Test: Ukraine (2021+)': 'test_ukraine'}

# Prepare splits
def get_split(df, fcols, tcol, sname):
    m = df['split'] == sname
    s = df[m].dropna(subset=[tcol])
    return s[fcols].copy(), s[tcol].values

splits = {}
for hn, tc in targets.items():
    splits[hn] = {'train': get_split(df, feature_cols, tc, 'train')}
    for pn, sn in eval_periods.items():
        splits[hn][pn] = get_split(df, feature_cols, tc, sn)
    Xt, yt = splits[hn]['train']
    print(f"  {hn}: Train={len(yt):,} (pos={yt.mean()*100:.1f}%)")


# ============================================================
# PART 4: HYPERPARAMETER TUNING (FIX 2)
# ============================================================
print("\n" + "=" * 70)
print("PART 4: HYPERPARAMETER TUNING WITH TimeSeriesSplit")
print("=" * 70)

# Use 2Q horizon for tuning (middle ground)
Xt_tune, yt_tune = splits['2Q ahead']['train']
avg_pos = yt_tune.mean()
cw = (1 - avg_pos) / avg_pos

# TimeSeriesSplit: 4 folds within the training period
tscv = TimeSeriesSplit(n_splits=4)

def tune_model(model_class, param_grid, X, y, cv, model_name):
    """Simple grid search with TimeSeriesSplit."""
    print(f"\n  Tuning {model_name}...")
    best_score = -1
    best_params = None
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"    Searching {n_combos} parameter combinations x {cv.n_splits} folds")
    
    # Generate all parameter combinations
    import itertools
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            if y_val.sum() == 0:
                continue
            
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            try:
                score = roc_auc_score(y_val, y_prob)
                fold_scores.append(score)
            except:
                pass
        
        if fold_scores:
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
    
    print(f"    Best CV AUC-ROC: {best_score:.4f}")
    print(f"    Best params: {best_params}")
    return best_params, best_score

# XGBoost tuning
xgb_grid = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [cw],
    'eval_metric': ['logloss'],
    'random_state': [42],
    'verbosity': [0],
    'tree_method': ['hist'],
}
xgb_best_params, xgb_cv_score = tune_model(xgb.XGBClassifier, xgb_grid, Xt_tune, yt_tune, tscv, "XGBoost")

# LightGBM tuning
lgb_grid = {
    'n_estimators': [300, 500],
    'max_depth': [6, 8, -1],
    'learning_rate': [0.03, 0.05, 0.1],
    'num_leaves': [31, 63],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'is_unbalance': [True],
    'random_state': [42],
    'verbose': [-1],
}
lgb_best_params, lgb_cv_score = tune_model(lgb.LGBMClassifier, lgb_grid, Xt_tune, yt_tune, tscv, "LightGBM")

print(f"\n  Hyperparameter tuning complete.")
print(f"    XGBoost best CV: {xgb_cv_score:.4f}")
print(f"    LightGBM best CV: {lgb_cv_score:.4f}")


# ============================================================
# PART 5: TRAIN FINAL MODELS AND EVALUATE
# ============================================================
print("\n" + "=" * 70)
print("PART 5: TRAINING FINAL MODELS WITH TUNED PARAMETERS")
print("=" * 70)

models_cfg = {
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
        'p': xgb_best_params,
        'imp': False, 'sc': False
    },
    'LightGBM': {
        'cls': lgb.LGBMClassifier,
        'p': lgb_best_params,
        'imp': False, 'sc': False
    },
}

all_results = []
all_roc = {}
all_imp = {}
all_probs = {}  # Store predictions for bootstrap CI

for hn, tc in targets.items():
    print(f"\n{'='*50}")
    print(f"  HORIZON: {hn}")
    print(f"{'='*50}")
    
    Xtr_raw, ytr = splits[hn]['train']
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    Xtr_imp = pd.DataFrame(imp.fit_transform(Xtr_raw), columns=feature_cols, index=Xtr_raw.index)
    Xtr_sc = pd.DataFrame(sc.fit_transform(Xtr_imp), columns=feature_cols, index=Xtr_raw.index)
    
    for mn, cfg in models_cfg.items():
        t1 = time.time()
        print(f"\n  Training {mn}...", end="", flush=True)
        
        Xtr_use = Xtr_sc if cfg['sc'] else (Xtr_imp if cfg['imp'] else Xtr_raw)
        mdl = cfg['cls'](**cfg['p'])
        mdl.fit(Xtr_use, ytr)
        print(f" done ({time.time()-t1:.1f}s)")

        if hasattr(mdl, 'feature_importances_'):
            all_imp[f"{mn}_{hn}"] = dict(zip(feature_cols, mdl.feature_importances_))
        elif hasattr(mdl, 'coef_'):
            all_imp[f"{mn}_{hn}"] = dict(zip(feature_cols, np.abs(mdl.coef_[0])))

        # Find best threshold on VALIDATION SET only — freeze it for all test periods
        Xv_raw, yv = splits[hn]['Validation (2015-18)']
        if cfg['sc']:
            Xv_i = pd.DataFrame(imp.transform(Xv_raw), columns=feature_cols, index=Xv_raw.index)
            Xv_use = pd.DataFrame(sc.transform(Xv_i), columns=feature_cols, index=Xv_raw.index)
        elif cfg['imp']:
            Xv_use = pd.DataFrame(imp.transform(Xv_raw), columns=feature_cols, index=Xv_raw.index)
        else:
            Xv_use = Xv_raw
        yv_prob = mdl.predict_proba(Xv_use)[:, 1]
        best_thresh, best_thresh_f1 = 0.5, 0.0
        if len(yv) > 0 and yv.sum() > 0:
            for t in np.arange(0.05, 0.75, 0.05):
                ft = f1_score(yv, (yv_prob >= t).astype(int), zero_division=0)
                if ft > best_thresh_f1:
                    best_thresh_f1, best_thresh = ft, t
        print(f"    Threshold tuned on validation: {best_thresh:.2f} (val F1={best_thresh_f1:.3f})")

        for pn, sn in eval_periods.items():
            Xe_raw, ye = splits[hn][pn]
            if len(ye) == 0 or ye.sum() == 0:
                continue
            
            if cfg['sc']:
                Xe_i = pd.DataFrame(imp.transform(Xe_raw), columns=feature_cols, index=Xe_raw.index)
                Xe_use = pd.DataFrame(sc.transform(Xe_i), columns=feature_cols, index=Xe_raw.index)
            elif cfg['imp']:
                Xe_use = pd.DataFrame(imp.transform(Xe_raw), columns=feature_cols, index=Xe_raw.index)
            else:
                Xe_use = Xe_raw
            
            yp = mdl.predict_proba(Xe_use)[:, 1]
            
            # Store for bootstrap CI
            all_probs[f"{mn}_{hn}_{pn}"] = (ye, yp)
            
            # Compute all metrics including BRIER SCORE (FIX 5)
            try: auc = roc_auc_score(ye, yp)
            except: auc = np.nan
            try: prauc = average_precision_score(ye, yp)
            except: prauc = np.nan
            
            brier = brier_score_loss(ye, yp)

            # Apply validation-tuned threshold (same for all periods — no per-period search)
            bf1 = f1_score(ye, (yp >= best_thresh).astype(int), zero_division=0)
            ypred = (yp >= 0.5).astype(int)
            f1v = f1_score(ye, ypred, zero_division=0)
            acc = accuracy_score(ye, ypred)

            all_results.append({
                'Horizon': hn, 'Model': mn, 'Period': pn,
                'AUC-ROC': round(auc, 4), 'PR-AUC': round(prauc, 4),
                'Brier': round(brier, 4),
                'F1 (0.5)': round(f1v, 4), 'Best F1': round(bf1, 4),
                'Best Threshold': round(best_thresh, 2),
                'Accuracy': round(acc, 4),
                'N': len(ye), 'Positive Rate': round(ye.mean(), 3),
            })
            
            # ROC data for plotting
            key = f"{hn}_{pn}"
            if key not in all_roc:
                all_roc[key] = {}
            try:
                fpr, tpr, _ = roc_curve(ye, yp)
                all_roc[key][mn] = (fpr, tpr, auc)
            except:
                pass
            
            print(f"    {pn}: AUC={auc:.3f}, PR-AUC={prauc:.3f}, Brier={brier:.3f}, BestF1={bf1:.3f}")
        
        joblib.dump({'model': mdl, 'imputer': imp, 'scaler': sc, 'features': feature_cols},
                    os.path.join(MDIR, f"{mn.replace(' ','_').lower()}_{tc}.joblib"))

rdf = pd.DataFrame(all_results)


# ============================================================
# PART 6: BOOTSTRAP CONFIDENCE INTERVALS (FIX 3)
# ============================================================
print("\n" + "=" * 70)
print("PART 6: BOOTSTRAP CONFIDENCE INTERVALS (1000 samples)")
print("=" * 70)

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, alpha=0.05):
    """Compute bootstrap CI for AUC-ROC."""
    rng = np.random.RandomState(42)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except:
            pass
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return np.mean(aucs), lo, hi

def bootstrap_auc_difference_test(y_true, y_prob_a, y_prob_b, n_boot=1000):
    """Test if AUC difference is significant (bootstrap version of DeLong)."""
    rng = np.random.RandomState(42)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        try:
            auc_a = roc_auc_score(y_true[idx], y_prob_a[idx])
            auc_b = roc_auc_score(y_true[idx], y_prob_b[idx])
            diffs.append(auc_a - auc_b)
        except:
            pass
    diffs = np.array(diffs)
    p_value = np.mean(diffs <= 0)  # P(model_a <= model_b)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), p_value

# Compute CIs for Ukraine test period
ci_results = []
for hn in targets:
    print(f"\n  {hn}:")
    for mn in models_cfg:
        key = f"{mn}_{hn}_Test: Ukraine (2021+)"
        if key in all_probs:
            ye, yp = all_probs[key]
            mean_auc, lo, hi = bootstrap_auc_ci(ye, yp)
            ci_results.append({
                'Horizon': hn, 'Model': mn,
                'AUC-ROC': round(mean_auc, 4),
                'CI Lower (95%)': round(lo, 4),
                'CI Upper (95%)': round(hi, 4),
                'CI Width': round(hi - lo, 4),
            })
            print(f"    {mn:22s}: AUC={mean_auc:.3f} [{lo:.3f}, {hi:.3f}]")

ci_df = pd.DataFrame(ci_results)
ci_df.to_csv(os.path.join(TDIR, 'table12_bootstrap_confidence_intervals.csv'), index=False)
print(f"\n  Saved: table12_bootstrap_confidence_intervals.csv")

# Statistical comparison: Best ML vs LogReg (FIX 3 - DeLong equivalent)
print("\n  SIGNIFICANCE TESTS (Bootstrap DeLong equivalent):")
print(f"  H0: Best ML model AUC <= Logistic Regression AUC")
sig_results = []
for hn in targets:
    best_mn = rdf[(rdf['Horizon'] == hn) & (rdf['Period'] == 'Test: Ukraine (2021+)')].sort_values('AUC-ROC', ascending=False).iloc[0]['Model']
    
    key_best = f"{best_mn}_{hn}_Test: Ukraine (2021+)"
    key_lr = f"Logistic Regression_{hn}_Test: Ukraine (2021+)"
    
    if key_best in all_probs and key_lr in all_probs:
        ye = all_probs[key_best][0]
        yp_best = all_probs[key_best][1]
        yp_lr = all_probs[key_lr][1]
        
        mean_diff, lo_diff, hi_diff, p_val = bootstrap_auc_difference_test(ye, yp_best, yp_lr)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "ns"))
        
        sig_results.append({
            'Horizon': hn, 'Model A': best_mn, 'Model B': 'Logistic Regression',
            'AUC Diff': round(mean_diff, 4), 'CI Lower': round(lo_diff, 4),
            'CI Upper': round(hi_diff, 4), 'p-value': round(p_val, 4), 'Significance': sig
        })
        print(f"  {hn}: {best_mn} vs LogReg: diff={mean_diff:+.3f} [{lo_diff:+.3f}, {hi_diff:+.3f}], p={p_val:.4f} {sig}")

pd.DataFrame(sig_results).to_csv(os.path.join(TDIR, 'table13_significance_tests.csv'), index=False)


# ============================================================
# PART 7: EXPANDING WINDOW VALIDATION (FIX 4)
# ============================================================
print("\n" + "=" * 70)
print("PART 7: EXPANDING WINDOW VALIDATION")
print("=" * 70)

# Use best model (LightGBM with tuned params) and 2Q horizon
ew_results = []
target_col = 'target_up_2q'

for train_end_year in [2012, 2013, 2014, 2015, 2016, 2017]:
    test_year = train_end_year + 1
    
    # Train on all data through train_end_year
    ew_train = df[df['date'].dt.year <= train_end_year].dropna(subset=[target_col])
    ew_test = df[df['date'].dt.year == test_year].dropna(subset=[target_col])
    
    if len(ew_test) < 20 or ew_test[target_col].sum() < 5:
        continue
    
    Xtr = ew_train[feature_cols]
    ytr = ew_train[target_col].values
    Xte = ew_test[feature_cols]
    yte = ew_test[target_col].values
    
    # Train LightGBM with tuned params
    mdl = lgb.LGBMClassifier(**lgb_best_params)
    mdl.fit(Xtr, ytr)
    
    yp = mdl.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, yp)
    except:
        auc = np.nan
    
    ew_results.append({
        'Train through': train_end_year,
        'Test year': test_year,
        'Train N': len(ytr),
        'Test N': len(yte),
        'AUC-ROC': round(auc, 4),
        'Positive Rate': round(yte.mean(), 3),
    })
    print(f"  Train through {train_end_year}, test {test_year}: AUC={auc:.3f} (n={len(yte)}, pos={yte.mean()*100:.1f}%)")

ew_df = pd.DataFrame(ew_results)
ew_df.to_csv(os.path.join(TDIR, 'table14_expanding_window.csv'), index=False)
print(f"\n  Mean expanding-window AUC: {ew_df['AUC-ROC'].mean():.3f}")
print(f"  Std: {ew_df['AUC-ROC'].std():.3f}")
print(f"  Expanding window validation complete.")


# ============================================================
# PART 8: FIGURES
# ============================================================
print("\n" + "=" * 70)
print("PART 8: GENERATING FIGURES")
print("=" * 70)

# Fig 20: ROC curves Ukraine
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (hn, tc) in enumerate(targets.items()):
    ax = axes[i]; key = f"{hn}_Test: Ukraine (2021+)"
    if key in all_roc:
        for mn, (fpr, tpr, av) in all_roc[key].items():
            ax.plot(fpr, tpr, label=f'{mn} ({av:.3f})', color=MC.get(mn, 'gray'), linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{"abc"[i]}) {hn}'); ax.legend(fontsize=7, loc='lower right')
plt.suptitle('Figure 20: ROC Curves — Ukraine Test (2021-2025) [Tuned models, fixed GMM]', fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('fig20_roc_curves_ukraine_revised')

# Fig 21: ROC curves COVID
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (hn, tc) in enumerate(targets.items()):
    ax = axes[i]; key = f"{hn}_Test: COVID (2019-20)"
    if key in all_roc:
        for mn, (fpr, tpr, av) in all_roc[key].items():
            ax.plot(fpr, tpr, label=f'{mn} ({av:.3f})', color=MC.get(mn, 'gray'), linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{"abc"[i]}) {hn}'); ax.legend(fontsize=7, loc='lower right')
plt.suptitle('Figure 21: ROC Curves — COVID Test (2019-2020) [Tuned models, fixed GMM]', fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('fig21_roc_curves_covid_revised')

# Fig 22: Bar chart comparison with error bars from bootstrap
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, pn in enumerate(eval_periods):
    ax = axes[i]; pd_ = rdf[rdf['Period'] == pn]; x = np.arange(len(targets)); w = 0.18
    for j, mn in enumerate(models_cfg):
        md = pd_[pd_['Model'] == mn]
        aucs = [md[md['Horizon'] == h]['AUC-ROC'].values[0] if len(md[md['Horizon'] == h]) > 0 else 0 for h in targets]
        ax.bar(x + j * w, aucs, w, label=mn, color=MC.get(mn, 'gray'), alpha=0.85)
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(list(targets.keys()), fontsize=9)
    ax.set_ylabel('AUC-ROC'); ax.set_title(pn, fontweight='bold', fontsize=10)
    ax.set_ylim(0.4, 0.95); ax.axhline(y=0.5, color='black', ls='--', alpha=0.3)
    if i == 0: ax.legend(fontsize=7)
plt.suptitle('Figure 22: Model comparison [Tuned hyperparameters, fixed GMM]', fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('fig22_model_comparison_revised')

# Fig 23: Feature importance
br = rdf[(rdf['Period'] == 'Test: Ukraine (2021+)') & (rdf['Horizon'] == '2Q ahead')].sort_values('AUC-ROC', ascending=False).iloc[0]
bmn = br['Model']
ik = f"{bmn}_2Q ahead"
if ik in all_imp:
    idf = pd.DataFrame({'feature': list(all_imp[ik].keys()), 'importance': list(all_imp[ik].values())}).sort_values('importance', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    def gc(f):
        if any(x in f for x in ['hcpi', 'fcpi', 'ecpi', 'ccpi', 'ppi', 'headline', 'food_energy', 'component_disp']): return C['blue']
        elif any(x in f for x in ['energy_index', 'food_commodity', 'metals', 'fertiliser', 'oil_price']): return C['red']
        elif f in ['debt_gdp', 'primary_balance', 'fiscal_balance', 'ext_debt_gdp', 'st_debt_reserves', 'private_credit_gdp', 'sovereign_rating', 'concessional_share']: return C['green']
        elif f.startswith('regime'): return C['purple']
        else: return C['gray']
    ax.barh(range(len(idf)), idf['importance'].values, color=[gc(f) for f in idf['feature']], alpha=0.85)
    ax.set_yticks(range(len(idf))); ax.set_yticklabels(idf['feature'].values, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel('Feature Importance'); ax.set_title(f'Figure 23: Top 20 features — {bmn} (2Q, tuned)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C['blue'], alpha=0.85, label='Inflation'), Patch(fc=C['red'], alpha=0.85, label='Commodity'),
                        Patch(fc=C['green'], alpha=0.85, label='Fiscal'), Patch(fc=C['purple'], alpha=0.85, label='Regime'),
                        Patch(fc=C['gray'], alpha=0.85, label='Country')], loc='lower right', fontsize=8)
    plt.tight_layout(); save_fig('fig23_feature_importance_revised')
    pd.DataFrame({'feature': list(all_imp[ik].keys()), 'importance': list(all_imp[ik].values())}).sort_values('importance', ascending=False).to_csv(os.path.join(TDIR, 'table10_feature_importance_revised.csv'), index=False)

# Fig 24: Calibration curves (FIX 7)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, hn in enumerate(targets):
    ax = axes[i]
    for mn in models_cfg:
        key = f"{mn}_{hn}_Test: Ukraine (2021+)"
        if key in all_probs:
            ye, yp = all_probs[key]
            prob_true, prob_pred = calibration_curve(ye, yp, n_bins=8, strategy='uniform')
            ax.plot(prob_pred, prob_true, 'o-', label=mn, color=MC.get(mn, 'gray'), linewidth=1.2, markersize=4)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
    ax.set_title(f'{"abc"[i]}) {hn}'); ax.legend(fontsize=7); ax.set_xlim(0, 0.7); ax.set_ylim(0, 0.7)
plt.suptitle('Figure 24: Calibration curves — Ukraine Test (2021+)', fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('fig24_calibration_curves')

# Fig 25: Expanding window stability
if len(ew_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ew_df['Test year'], ew_df['AUC-ROC'], 'o-', color=C['blue'], linewidth=2, markersize=8)
    ax.fill_between(ew_df['Test year'],
                     ew_df['AUC-ROC'] - ew_df['AUC-ROC'].std(),
                     ew_df['AUC-ROC'] + ew_df['AUC-ROC'].std(),
                     alpha=0.2, color=C['blue'])
    ax.axhline(y=0.5, color='black', ls='--', alpha=0.3)
    ax.set_xlabel('Test Year'); ax.set_ylabel('AUC-ROC')
    ax.set_title('Figure 25: Expanding window validation — LightGBM (2Q ahead)')
    ax.set_ylim(0.4, 1.0)
    save_fig('fig25_expanding_window_stability')


# ============================================================
# PART 9: SAVE RESULTS AND SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PART 9: SAVING ALL OUTPUTS")
print("=" * 70)

# Main results table (with Brier score)
rdf.to_csv(os.path.join(TDIR, 'table09_model_comparison_revised.csv'), index=False)
print(f"  Saved: table09_model_comparison_revised.csv")

# Print results table
print("\n  RESULTS TABLE (Ukraine test):")
for h in targets:
    print(f"\n  --- {h} ---")
    hr = rdf[(rdf['Horizon'] == h) & (rdf['Period'] == 'Test: Ukraine (2021+)')][['Model', 'AUC-ROC', 'PR-AUC', 'Brier', 'Best F1']]
    print(hr.sort_values('AUC-ROC', ascending=False).to_string(index=False))

total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"STEP 7 REVISED — COMPLETE")
print(f"{'='*70}")

# Best models
print(f"\nBEST MODELS (Ukraine test, tuned, fixed GMM):")
for h in targets:
    best = rdf[(rdf['Horizon'] == h) & (rdf['Period'] == 'Test: Ukraine (2021+)')].sort_values('AUC-ROC', ascending=False).iloc[0]
    ci_row = ci_df[(ci_df['Horizon'] == h) & (ci_df['Model'] == best['Model'])]
    ci_str = ""
    if len(ci_row) > 0:
        ci_str = f" [{ci_row.iloc[0]['CI Lower (95%)']:.3f}, {ci_row.iloc[0]['CI Upper (95%)']:.3f}]"
    print(f"  {h:12s}: {best['Model']:22s} AUC={best['AUC-ROC']:.3f}{ci_str} Brier={best['Brier']:.3f}")

print(f"""
FILES CREATED:
  table09_model_comparison_revised.csv
  table10_feature_importance_revised.csv
  table12_bootstrap_confidence_intervals.csv
  table13_significance_tests.csv
  table14_expanding_window.csv
  fig20-fig25 (6 figures)
  Models saved to outputs/models/

TOTAL TIME: {total_time/60:.1f} minutes | {fc} figures
""")
