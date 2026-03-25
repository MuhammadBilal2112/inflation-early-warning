"""
==============================================================================
MODEL CALIBRATION FIX + CONDITIONAL FISCAL ANALYSIS
==============================================================================

Two targeted fixes:
  1. Probability calibration (isotonic regression) — fixes overconfidence
  2. Conditional fiscal thresholds — removes ceiling effect by filtering
     to countries currently in low/moderate regimes

RUN: python3 Calibration_Fix.py
TIME: 1-2 minutes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os, re, warnings
warnings.filterwarnings('ignore')

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

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
print("FIX 1: PROBABILITY CALIBRATION")
print("=" * 70)

# Load data
df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df = df.rename(columns=clean_map)

# Load model
saved = joblib.load(os.path.join(MDIR, 'xgboost_target_up_2q.joblib'))
model = saved['model']
feature_cols = saved['features']

# Get validation set (2015-2018) — used to FIT the calibrator
val_data = df[df['split'] == 'validation'].dropna(subset=['target_up_2q']).copy()
X_val = val_data[feature_cols]
y_val = val_data['target_up_2q'].values
raw_probs_val = model.predict_proba(X_val)[:, 1]

# Get test set — used to EVALUATE
test_data = df[df['split'].isin(['test_covid', 'test_ukraine'])].dropna(subset=['target_up_2q']).copy()
X_test = test_data[feature_cols]
y_test = test_data['target_up_2q'].values
raw_probs_test = model.predict_proba(X_test)[:, 1]

# ============================================================
# Fit isotonic regression calibrator on VALIDATION set
# ============================================================
print("\n  Fitting isotonic regression on validation set...")
calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrator.fit(raw_probs_val, y_val)

# Apply to test set
calibrated_probs_test = calibrator.predict(raw_probs_test)

# Also calibrate validation (in-sample check)
calibrated_probs_val = calibrator.predict(raw_probs_val)

# ============================================================
# Compare before vs after calibration
# ============================================================
print("\n  BEFORE calibration (test set):")
brier_before = brier_score_loss(y_test, raw_probs_test)
auc_before = roc_auc_score(y_test, raw_probs_test)
print(f"    Brier score: {brier_before:.4f}")
print(f"    AUC-ROC: {auc_before:.4f}")

print(f"\n  AFTER calibration (test set):")
brier_after = brier_score_loss(y_test, calibrated_probs_test)
auc_after = roc_auc_score(y_test, calibrated_probs_test)
print(f"    Brier score: {brier_after:.4f} ({'improved' if brier_after < brier_before else 'worsened'}: {brier_before - brier_after:+.4f})")
print(f"    AUC-ROC: {auc_after:.4f} (should be unchanged: {auc_after - auc_before:+.4f})")

# ============================================================
# Calibration check: binned predicted vs actual
# ============================================================
print(f"\n  Calibration table (BEFORE vs AFTER):")
print(f"  {'Bin':>12s} {'Raw P':>8s} {'Cal P':>8s} {'Actual':>8s} {'N':>6s} {'Before':>10s} {'After':>10s}")
print(f"  {'-'*62}")

prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(prob_bins) - 1):
    lo, hi = prob_bins[i], prob_bins[i + 1]
    mask = (raw_probs_test >= lo) & (raw_probs_test < hi)
    if mask.sum() < 10:
        continue
    
    raw_mean = raw_probs_test[mask].mean()
    cal_mean = calibrated_probs_test[mask].mean()
    actual_mean = y_test[mask].mean()
    n = mask.sum()
    
    err_before = abs(raw_mean - actual_mean)
    err_after = abs(cal_mean - actual_mean)
    
    b_status = f"{err_before:.3f}"
    a_status = f"{err_after:.3f}"
    
    print(f"  {lo:.1f}-{hi:.1f}     {raw_mean:>7.1%} {cal_mean:>7.1%} {actual_mean:>7.1%} {n:>5d} {b_status:>10s} {a_status:>10s}")

# ============================================================
# Figure 41: Before vs After calibration curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel a: Calibration curves
ax = axes[0]
prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probs_test, n_bins=10, strategy='uniform')
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs_test, n_bins=10, strategy='uniform')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
ax.plot(prob_pred_raw, prob_true_raw, 'o-', color=C['red'], linewidth=2, markersize=6,
        label=f'Before (Brier={brier_before:.3f})')
ax.plot(prob_pred_cal, prob_true_cal, 's-', color=C['green'], linewidth=2, markersize=6,
        label=f'After (Brier={brier_after:.3f})')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('a) Calibration curves', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Panel b: Probability distribution shift
ax = axes[1]
ax.hist(raw_probs_test, bins=50, alpha=0.5, color=C['red'], label='Before (raw)', density=True)
ax.hist(calibrated_probs_test, bins=50, alpha=0.5, color=C['green'], label='After (calibrated)', density=True)
ax.axvline(x=y_test.mean(), color='black', linestyle='--', linewidth=1, label=f'Base rate ({y_test.mean():.1%})')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title('b) Probability distributions', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('Figure 41: Probability calibration — before vs after isotonic regression',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('fig41_calibration_before_after')

# ============================================================
# Save calibrated model
# ============================================================
saved_calibrated = {
    'model': model,
    'calibrator': calibrator,
    'imputer': saved['imputer'],
    'scaler': saved['scaler'],
    'features': feature_cols,
}
cal_model_path = os.path.join(MDIR, 'xgboost_calibrated_target_up_2q.joblib')
joblib.dump(saved_calibrated, cal_model_path)
print(f"\n  Saved calibrated model: {cal_model_path}")

# Store calibrated probs for use in Fix 2
test_data['raw_prob'] = raw_probs_test
test_data['calibrated_prob'] = calibrated_probs_test


# ============================================================
# FIX 2: CONDITIONAL FISCAL THRESHOLDS
# ============================================================
print("\n" + "=" * 70)
print("FIX 2: CONDITIONAL FISCAL THRESHOLDS")
print("=" * 70)
print("  (Filtering to countries in LOW/MODERATE regimes only,")
print("   removing the ceiling effect from already-crisis countries)")

# Merge current regime info
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])
test_with_regime = test_data.merge(
    regime[['country_code', 'date', 'regime']],
    on=['country_code', 'date'], how='left', suffixes=('', '_current')
)

# Filter: only countries currently in regimes 0, 1, 2, or 3 (not already elevated/crisis)
low_mod_mask = test_with_regime['regime_current'].isin([0, 1, 2, 3])
conditional_data = test_with_regime[low_mod_mask].copy()

n_total = len(test_with_regime)
n_filtered = len(conditional_data)
print(f"\n  Total test obs: {n_total}")
print(f"  In low/moderate regimes: {n_filtered} ({n_filtered/n_total*100:.0f}%)")
print(f"  Excluded (already elevated/crisis): {n_total - n_filtered}")

# Re-do fiscal threshold analysis on filtered data
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel a: Debt/GDP (conditional, using CALIBRATED probabilities)
ax = axes[0]
if 'debt_gdp' in conditional_data.columns:
    debt_data = conditional_data.dropna(subset=['debt_gdp']).copy()
    
    bins = [0, 30, 50, 60, 70, 80, 100, 200]
    labels = ['0-30%', '30-50%', '50-60%', '60-70%', '70-80%', '80-100%', '100%+']
    debt_data['debt_bin'] = pd.cut(debt_data['debt_gdp'], bins=bins, labels=labels)
    
    bin_stats = debt_data.groupby('debt_bin', observed=True).agg(
        cal_prob=('calibrated_prob', 'mean'),
        raw_prob=('raw_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        n=('target_up_2q', 'size'),
    )
    valid = bin_stats[bin_stats['n'] >= 15]
    
    x = range(len(valid))
    ax.bar(x, valid['cal_prob'] * 100, alpha=0.6, color=C['blue'], label='Calibrated P(transition)')
    ax.plot(x, valid['actual_rate'] * 100, 'D-', color=C['red'], linewidth=2, markersize=7,
            label='Actual transition rate')
    ax.set_xticks(x)
    ax.set_xticklabels(valid.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Government Debt / GDP')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('a) Debt/GDP threshold\n(low/moderate regimes only, calibrated)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    
    print(f"\n  CONDITIONAL debt/GDP analysis (excluding elevated/crisis):")
    print(f"  {'Bin':>10s} {'Cal P':>8s} {'Actual':>8s} {'N':>5s}")
    for idx, row in valid.iterrows():
        print(f"  {idx:>10s} {row['cal_prob']:>7.1%} {row['actual_rate']:>7.1%} {int(row['n']):>5d}")

# Panel b: By country group (conditional, calibrated)
ax = axes[1]
if 'country_group' in conditional_data.columns and 'debt_gdp' in conditional_data.columns:
    for i, (group, color, marker) in enumerate([('Advanced economies', C['blue'], 'o'),
                                                   ('EMDEs', C['red'], 's')]):
        g_data = conditional_data[(conditional_data['country_group'] == group)].dropna(subset=['debt_gdp'])
        if len(g_data) < 30:
            continue
        
        bins_g = [0, 40, 60, 80, 100, 200]
        labels_g = ['0-40%', '40-60%', '60-80%', '80-100%', '100%+']
        g_data['dbin'] = pd.cut(g_data['debt_gdp'], bins=bins_g, labels=labels_g)
        
        g_stats = g_data.groupby('dbin', observed=True).agg(
            cal_prob=('calibrated_prob', 'mean'),
            actual_rate=('target_up_2q', 'mean'),
            n=('target_up_2q', 'size'),
        )
        valid_g = g_stats[g_stats['n'] >= 10]
        
        ax.plot(range(len(valid_g)), valid_g['cal_prob'].values * 100, f'{marker}-',
                color=color, linewidth=2, markersize=7, label=f'{group} (model)')
        ax.plot(range(len(valid_g)), valid_g['actual_rate'].values * 100, f'{marker}--',
                color=color, linewidth=1, markersize=5, alpha=0.6, label=f'{group} (actual)')
    
    ax.set_xticks(range(len(labels_g)))
    ax.set_xticklabels(labels_g, fontsize=9)
    ax.set_xlabel('Debt / GDP')
    ax.set_ylabel('Transition probability (%)')
    ax.set_title('b) By country group\n(low/moderate regimes only, calibrated)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle('Figure 42: Fiscal thresholds — conditional analysis with calibrated probabilities',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig('fig42_fiscal_thresholds_conditional')


# ============================================================
# Re-check debt composition with calibrated probs
# ============================================================
print("\n" + "=" * 70)
print("DEBT COMPOSITION (RE-CHECKED WITH CALIBRATED PROBS)")
print("=" * 70)

if 'debt_gdp' in test_data.columns and 'sovereign_rating' in test_data.columns:
    int_data = test_data.dropna(subset=['debt_gdp', 'sovereign_rating']).copy()
    debt_median = int_data['debt_gdp'].median()
    int_data['high_debt'] = int_data['debt_gdp'] > debt_median
    int_data['inv_grade'] = int_data['sovereign_rating'] >= 14
    
    print(f"\n  {'Group':>35s} {'Cal P':>8s} {'Raw P':>8s} {'Actual':>8s} {'N':>5s}")
    print(f"  {'-'*65}")
    for hd_label, hd_val in [('High debt', True), ('Low debt', False)]:
        for ig_label, ig_val in [('Inv grade', True), ('Sub-inv', False)]:
            subset = int_data[(int_data['high_debt'] == hd_val) & (int_data['inv_grade'] == ig_val)]
            if len(subset) > 20:
                cal = subset['calibrated_prob'].mean()
                raw = subset['raw_prob'].mean()
                actual = subset['target_up_2q'].mean()
                n = len(subset)
                print(f"  {hd_label + ' + ' + ig_label:>35s} {cal:>7.1%} {raw:>7.1%} {actual:>7.1%} {n:>5d}")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"FIXES APPLIED")
print(f"{'='*70}")
print(f"""
  Brier score: {brier_before:.4f} → {brier_after:.4f}
  AUC-ROC: {auc_after:.4f}

FIGURES: {fc}
  fig41: Calibration before vs after
  fig42: Conditional fiscal thresholds (calibrated)
""")
