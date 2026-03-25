"""
==============================================================================
VERIFICATION OF ECONOMIC DEEP-DIVE FINDINGS
==============================================================================
For each finding, we check:
  - Which specific countries drive the result
  - Whether the pattern holds in both test periods (COVID + Ukraine)
  - Whether it holds for AEs and EMDEs separately
  - Sanity checks against known economic facts
==============================================================================
"""

import pandas as pd
import numpy as np
import joblib, os, re, warnings
warnings.filterwarnings('ignore')

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR, "data", "processed")
MDIR = os.path.join(BASE_DIR, "outputs", "models")

# Load data
df = pd.read_csv(os.path.join(PROC, 'features_and_targets.csv'), parse_dates=['date'])
regime = pd.read_csv(os.path.join(PROC, 'regime_labels_fixed.csv'), parse_dates=['date'])

clean_map = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns
             if c != re.sub(r'[^A-Za-z0-9_]', '_', c)}
if clean_map:
    df = df.rename(columns=clean_map)

# Load model
saved = joblib.load(os.path.join(MDIR, 'xgboost_target_up_2q.joblib'))
model = saved['model']
feature_cols = saved['features']

# Test data with predictions
test = df[df['split'].isin(['test_covid', 'test_ukraine'])].dropna(subset=['target_up_2q']).copy()
test['transition_prob'] = model.predict_proba(test[feature_cols])[:, 1]

# Country names
cnames = {}
for _, r in regime[['country_code', 'country_name']].drop_duplicates().iterrows():
    if pd.notna(r['country_name']):
        cnames[r['country_code']] = str(r['country_name'])

# Energy exporters
energy_exporters = {
    'SAU', 'ARE', 'KWT', 'QAT', 'OMN', 'BHR', 'IRQ', 'IRN',
    'RUS', 'NOR', 'NGA', 'AGO', 'DZA', 'LBY', 'GNQ', 'GAB', 'COG',
    'VEN', 'ECU', 'COL', 'TTO', 'GUY',
    'KAZ', 'AZE', 'TKM', 'UZB',
    'BRN', 'MYS', 'IDN',
    'CAN', 'AUS',
}
test['energy_exporter'] = test['country_code'].isin(energy_exporters).astype(int)

print("=" * 70)
print("VERIFICATION OF ECONOMIC FINDINGS")
print("=" * 70)


# ============================================================
# CHECK 1: COMMODITY EXPORTER/IMPORTER
# ============================================================
print("\n" + "=" * 60)
print("CHECK 1: COMMODITY EXPORTERS vs IMPORTERS")
print("=" * 60)

# Which exporters are in our data?
exp_countries = test[test['energy_exporter'] == 1]['country_code'].unique()
print(f"\n  Exporters in test ({len(exp_countries)}):")
for cc in sorted(exp_countries):
    n = len(test[(test['country_code'] == cc) & (test['energy_exporter'] == 1)])
    tp = test[(test['country_code'] == cc)]['transition_prob'].mean()
    actual = test[(test['country_code'] == cc)]['target_up_2q'].mean()
    print(f"    {cc} ({cnames.get(cc, cc):25s}): P(trans)={tp:.1%}, actual rate={actual:.1%}, n={n}")

# Split by test period
print(f"\n  Transition prob by group and period:")
for period, split in [('COVID (2019-20)', 'test_covid'), ('Ukraine (2021+)', 'test_ukraine')]:
    p_data = test[test['split'] == split]
    if len(p_data) == 0:
        continue
    exp_p = p_data[p_data['energy_exporter'] == 1]['transition_prob'].mean()
    imp_p = p_data[p_data['energy_exporter'] == 0]['transition_prob'].mean()
    exp_actual = p_data[p_data['energy_exporter'] == 1]['target_up_2q'].mean()
    imp_actual = p_data[p_data['energy_exporter'] == 0]['target_up_2q'].mean()
    print(f"    {period}:")
    print(f"      Exporters: P(trans)={exp_p:.1%}, actual={exp_actual:.1%} (n={p_data['energy_exporter'].sum()})")
    n_imp = len(p_data) - int(p_data['energy_exporter'].sum())
    print(f"      Importers: P(trans)={imp_p:.1%}, actual={imp_actual:.1%} (n={n_imp})")

# Known facts check
print(f"\n  SANITY CHECK — Known 2022 outcomes:")
known_checks = [
    ('NOR', 'Norway — oil exporter, moderate inflation'),
    ('SAU', 'Saudi Arabia — oil exporter, low inflation'),
    ('RUS', 'Russia — oil exporter but sanctioned'),
    ('TUR', 'Turkey — oil importer, very high inflation'),
    ('GBR', 'UK — energy importer, high 2022 inflation'),
    ('IND', 'India — oil importer, moderate inflation'),
    ('NGA', 'Nigeria — oil exporter but also food importer'),
]
for cc, desc in known_checks:
    cc_data = test[(test['country_code'] == cc) & (test['date'].dt.year == 2022)]
    if len(cc_data) > 0:
        tp = cc_data['transition_prob'].mean()
        actual = cc_data['target_up_2q'].mean()
        print(f"    {desc}")
        print(f"      2022: P(trans)={tp:.1%}, actual transition rate={actual:.1%}")


# ============================================================
# CHECK 2: FISCAL THRESHOLDS
# ============================================================
print("\n" + "=" * 60)
print("CHECK 2: FISCAL THRESHOLD VERIFICATION")
print("=" * 60)

if 'debt_gdp' in test.columns:
    # Which countries are in each debt bin?
    debt_data = test.dropna(subset=['debt_gdp']).copy()
    
    # The key finding: 50-60% has highest risk, 80-90% drops
    print(f"\n  Countries driving the 50-60% peak (highest risk bin):")
    peak_bin = debt_data[(debt_data['debt_gdp'] >= 50) & (debt_data['debt_gdp'] < 60)]
    peak_countries = peak_bin.groupby('country_code').agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        debt=('debt_gdp', 'mean'),
        n=('target_up_2q', 'size'),
    ).sort_values('n', ascending=False)
    
    for cc, row in peak_countries.head(10).iterrows():
        group = debt_data[debt_data['country_code'] == cc]['country_group'].iloc[0] if 'country_group' in debt_data.columns else '?'
        print(f"    {cc} ({cnames.get(cc, cc):25s}) [{group:5s}]: debt={row['debt']:.0f}%, P(trans)={row['mean_prob']:.1%}, actual={row['actual_rate']:.1%}, n={int(row['n'])}")
    
    print(f"\n  Countries driving the 80-90% dip (lowest risk bin):")
    dip_bin = debt_data[(debt_data['debt_gdp'] >= 80) & (debt_data['debt_gdp'] < 90)]
    dip_countries = dip_bin.groupby('country_code').agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        debt=('debt_gdp', 'mean'),
        n=('target_up_2q', 'size'),
    ).sort_values('n', ascending=False)
    
    for cc, row in dip_countries.head(10).iterrows():
        group = debt_data[debt_data['country_code'] == cc]['country_group'].iloc[0] if 'country_group' in debt_data.columns else '?'
        print(f"    {cc} ({cnames.get(cc, cc):25s}) [{group:5s}]: debt={row['debt']:.0f}%, P(trans)={row['mean_prob']:.1%}, actual={row['actual_rate']:.1%}, n={int(row['n'])}")
    
    # Check if pattern holds separately for AEs and EMDEs
    print(f"\n  DEBT THRESHOLD BY COUNTRY GROUP:")
    for group in ['Advanced economies', 'EMDEs']:
        g_data = debt_data[debt_data['country_group'] == group] if 'country_group' in debt_data.columns else pd.DataFrame()
        if len(g_data) < 50:
            continue
        
        bins = [0, 40, 60, 80, 100, 200]
        labels = ['0-40%', '40-60%', '60-80%', '80-100%', '100%+']
        g_data['dbin'] = pd.cut(g_data['debt_gdp'], bins=bins, labels=labels)
        
        bin_stats = g_data.groupby('dbin', observed=True).agg(
            mean_prob=('transition_prob', 'mean'),
            actual_rate=('target_up_2q', 'mean'),
            n=('target_up_2q', 'size'),
        )
        valid = bin_stats[bin_stats['n'] >= 15]
        
        print(f"\n    {group}:")
        for idx, row in valid.iterrows():
            print(f"      {idx}: P(trans)={row['mean_prob']:.1%}, actual={row['actual_rate']:.1%}, n={int(row['n'])}")


# ============================================================
# CHECK 3: DEBT COMPOSITION
# ============================================================
print("\n" + "=" * 60)
print("CHECK 3: DEBT COMPOSITION VERIFICATION")
print("=" * 60)

if 'debt_gdp' in test.columns and 'sovereign_rating' in test.columns:
    int_data = test.dropna(subset=['debt_gdp', 'sovereign_rating']).copy()
    
    debt_median = int_data['debt_gdp'].median()
    int_data['high_debt'] = int_data['debt_gdp'] > debt_median
    int_data['inv_grade'] = int_data['sovereign_rating'] >= 14
    
    # The surprising finding: high debt + investment grade = highest risk
    print(f"\n  Debt median: {debt_median:.0f}%")
    print(f"\n  Countries in HIGH DEBT + INVESTMENT GRADE (highest risk group):")
    hd_ig = int_data[(int_data['high_debt']) & (int_data['inv_grade'])]
    hd_ig_countries = hd_ig.groupby('country_code').agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        debt=('debt_gdp', 'mean'),
        rating=('sovereign_rating', 'mean'),
        n=('target_up_2q', 'size'),
    ).sort_values('n', ascending=False)
    
    for cc, row in hd_ig_countries.head(12).iterrows():
        print(f"    {cc} ({cnames.get(cc, cc):25s}): debt={row['debt']:.0f}%, rating={row['rating']:.0f}, P(trans)={row['mean_prob']:.1%}, actual={row['actual_rate']:.1%}")
    
    print(f"\n  Countries in HIGH DEBT + SUB-INVESTMENT (lower risk — surprising):")
    hd_si = int_data[(int_data['high_debt']) & (~int_data['inv_grade'])]
    hd_si_countries = hd_si.groupby('country_code').agg(
        mean_prob=('transition_prob', 'mean'),
        actual_rate=('target_up_2q', 'mean'),
        debt=('debt_gdp', 'mean'),
        n=('target_up_2q', 'size'),
    ).sort_values('n', ascending=False)
    
    for cc, row in hd_si_countries.head(12).iterrows():
        print(f"    {cc} ({cnames.get(cc, cc):25s}): debt={row['debt']:.0f}%, P(trans)={row['mean_prob']:.1%}, actual={row['actual_rate']:.1%}")

    # KEY CHECK: Is the actual transition rate also higher for HI+IG?
    print(f"\n  ACTUAL TRANSITION RATES (ground truth):")
    for hd_label, hd_val in [('High debt', True), ('Low debt', False)]:
        for ig_label, ig_val in [('Inv grade', True), ('Sub-inv', False)]:
            subset = int_data[(int_data['high_debt'] == hd_val) & (int_data['inv_grade'] == ig_val)]
            if len(subset) > 20:
                actual = subset['target_up_2q'].mean()
                model_p = subset['transition_prob'].mean()
                n = len(subset)
                print(f"    {hd_label:10s} + {ig_label:10s}: ACTUAL={actual:.1%}, MODEL={model_p:.1%}, n={n}")


# ============================================================
# CHECK 4: OVERALL MODEL CALIBRATION CHECK
# ============================================================
print("\n" + "=" * 60)
print("CHECK 4: ARE MODEL PROBABILITIES CALIBRATED?")
print("=" * 60)

# If model says 30% probability, does the event happen ~30% of the time?
prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test['prob_bin'] = pd.cut(test['transition_prob'], bins=prob_bins)

cal_check = test.groupby('prob_bin', observed=True).agg(
    mean_predicted=('transition_prob', 'mean'),
    actual_rate=('target_up_2q', 'mean'),
    n=('target_up_2q', 'size'),
)

print(f"\n  {'Predicted P':>15s} {'Actual rate':>15s} {'N':>8s} {'Calibration':>15s}")
print(f"  {'-'*55}")
for idx, row in cal_check.iterrows():
    diff = abs(row['mean_predicted'] - row['actual_rate'])
    status = 'GOOD' if diff < 0.10 else ('OK' if diff < 0.15 else 'POOR')
    print(f"  {row['mean_predicted']:>14.1%} {row['actual_rate']:>14.1%} {int(row['n']):>8d} {status:>15s}")

print(f"""
======================================================================
VERIFICATION SUMMARY
======================================================================

Review the output above to confirm each finding is robust.
""")