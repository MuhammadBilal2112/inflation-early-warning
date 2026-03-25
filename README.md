# Inflation Regime Early Warning System

A machine learning pipeline that classifies countries into inflation regimes and predicts upward regime transitions — providing actionable early warnings for policymakers and researchers.

Built on World Bank data covering **135 countries from 1971–2025**, using Gaussian Mixture Models for regime discovery and gradient boosting for transition prediction.

**Validated against 11 real-world crises including COVID-19, the Ukraine energy shock, Turkey's lira crisis, Sri Lanka's sovereign default, and the 2008 Global Financial Crisis — with an 81% out-of-sample detection rate and approximately one year of average lead time.**

---

## Key Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost (tuned via TimeSeriesSplit CV) |
| AUC-ROC (1Q ahead) | **0.839** [0.816, 0.860] |
| AUC-ROC (2Q ahead) | **0.813** [0.791, 0.835] |
| AUC-ROC (4Q ahead) | **0.779** [0.754, 0.802] |
| ML vs Linear baseline | p < 0.001 (bootstrap DeLong test) |
| Historical crisis detection | **81%** (13/16 out-of-sample episodes) |
| Average early warning lead time | **~1 year** (3.9 quarters) |
| Countries covered | 135 |
| Inflation regimes | 6 (GMM, BIC-selected) |
| Probability calibration | Isotonic regression, Brier: 0.181 |

## Historical Validation Highlights

The model was trained on pre-2015 data and tested against crises it never saw:

| Crisis | Country | Warning Signal | Lead Time | Outcome |
|--------|---------|---------------|-----------|---------|
| Lira Crisis 2018 | Turkey | 39.4% | 4.1 quarters | Correctly predicted R1→R3 |
| Banking Collapse 2019 | Lebanon | 45.8% | 3.0 quarters | Correctly predicted R1→R5 |
| COVID Shock 2020 | USA | 45.8% | 3.1 quarters | Correctly predicted R1→R2 |
| COVID Shock 2020 | Brazil | 41.7% | 3.1 quarters | Correctly predicted R1→R3 |
| Energy Shock 2022 | UK | 34.7% | 2.7 quarters | Correctly predicted R2→R4 |
| Sovereign Default 2022 | Sri Lanka | 45.8% | 5.1 quarters | Correctly predicted R1→R5 |
| Devaluation 2023 | Egypt | 29.0% | 4.1 quarters | Correctly predicted R3→R5 |
| GFC 2008 | USA | Low signal | — | Correctly identified as deflationary shock |

The model has zero knowledge of geopolitics, pandemics, or wars. It detects vulnerability from numerical patterns alone — because inflation crises are preceded by measurable preconditions in economic data.

## Interactive Dashboard

The pipeline generates a self-contained interactive HTML dashboard featuring:
- World choropleth map coloured by current inflation regime
- Type-ahead country search across all 135 countries
- Calibrated transition probability with circular gauge
- Fiscal vulnerability profile (debt/GDP, sovereign rating, fiscal balance)
- Inflation component breakdown (headline, food, energy CPI)
- Panel A vs Panel B regime comparison (3 vs 4 CPI measures)
- Monthly CPI time series and regime history timeline
- High-risk alert panel with pulsing indicators

## What Are Inflation Regimes?

Rather than using arbitrary thresholds (e.g., "high" = above 10%), the project uses unsupervised learning to discover natural clusters in multi-dimensional inflation data:

| Regime | Headline CPI | Key Characteristic | Example |
|--------|-------------|-------------------|---------|
| R0: Low | ~0.2% | Stable, falling energy prices | Japan 2010s |
| R1: Low-energy | ~3.2% | Low headline, rising energy | Europe 2017-2019 |
| R2: Moderate | ~3.8% | Food-driven inflation | Sub-Saharan Africa droughts |
| R3: Mod-energy | ~7.8% | Energy-driven inflation | 2022 energy crisis |
| R4: Elevated | ~11.3% | Broad inflation pressure | Turkey 2019-2020 |
| R5: Crisis | ~71.2% | All components spiralling | Argentina 2023, Lebanon 2021 |

Key insight: the data distinguishes between **food-driven** and **energy-driven** inflation at every level. A country at 6% headline where food=12% and energy=2% is in a different regime than one where all components are at 6%. These require different policy responses.

## Dual-Panel Analysis (Novel Finding)

The pipeline runs two parallel analyses and compares them:

| | Panel A | Panel B |
|--|---------|---------|
| CPI Measures | 3 (headline + food + energy) | 4 (+ core CPI) |
| Countries | 135 | 74 |
| Best AUC, 1Q | **0.839** (wins) | 0.808 |
| Best AUC, 4Q | 0.776 | **0.842** (wins) |
| Key advantage | More countries, more training data | Captures persistent vs transitory inflation |

Panel B reveals a hidden dimension: two regimes with identical 2.5% headline inflation but opposite headline-core gaps — one transitory (supply pressure), one persistent (demand overheating). Central bankers would respond completely differently.

**Policy implication:** Use Panel A for quarterly alerts (maximum coverage), Panel B for annual vulnerability assessments (richer measurement).

## Methodology

### Regime Discovery
- Gaussian Mixture Model (K=6, BIC-selected), fitted on **training data only** (pre-2015)
- 7 features: headline/food/energy CPI, component dispersion, momentum, food-energy gap, volatility
- Robustness: HMM confirms regime structure (ARI = 0.687)

### Prediction Models
| Model | Tuning | Key Hyperparameters |
|-------|--------|-------------------|
| Logistic Regression | Fixed (balanced class weights) | C=1.0, L2 penalty |
| Random Forest | Conservative defaults (Breiman 2001) | 500 trees, depth=12 |
| XGBoost | **TimeSeriesSplit CV (4-fold)** | n_est=300, depth=8, lr=0.03 |
| LightGBM | **TimeSeriesSplit CV (4-fold)** | n_est=300, depth=6, lr=0.03 |

### Evaluation
- **Time-based splits:** Train (1971–2014), Validation (2015–2018), Test-COVID (2019–2020), Test-Ukraine (2021–2025)
- **No data leakage:** GMM, scaler, imputer all fitted on training data only
- **Metrics:** AUC-ROC, PR-AUC, Brier score, F1, TPR@5%FPR, TPR@10%FPR
- **Statistical significance:** Bootstrap CIs (1000 samples) + DeLong-equivalent test
- **Temporal stability:** Expanding window validation (6 windows, mean AUC=0.804, std=0.035)
- **Calibration:** Isotonic regression (Brier: 0.192 → 0.181)
- **Interpretability:** TreeSHAP values, partial dependence plots, crisis fingerprints
- **Historical validation:** 11 episodes, 81% detection rate, ~1 year lead time

### Interpretability (SHAP)
- Global feature importance ranking
- AE vs EMDE driver comparison
- COVID vs Ukraine crisis fingerprints
- Individual country waterfall plots (Turkey, UK, USA, Nigeria)
- Non-linear dependence analysis (fiscal thresholds, commodity effects)

### Economic Deep-Dive
- Fiscal threshold effects: transition risk rises monotonically with debt/GDP (27.6% at low debt → 35.2% at 100%+)
- Commodity exporter vs importer: importers face modestly higher risk (31.9% vs 28.9%)
- Debt composition: debt level matters more than composition after calibration

## Project Structure

```
inflation_regimes_project/
├── src/
│   ├── setup/                              # Environment setup scripts
│   ├── 01_load_inflation_data.py           # World Bank CPI data processing
│   ├── 02_load_fiscal_commodity_data.py    # Fiscal & commodity data, panel merge
│   ├── 03_exploratory_analysis.py          # EDA: 13 figures, 3 tables
│   ├── 04_regime_discovery.py              # GMM regime classification (K=6)
│   ├── 05_feature_engineering.py           # 67 features + 3 target variables
│   ├── 06_model_training.py               # 4 models, HP tuning, bootstrap CIs
│   ├── 07_patch_regime_ordering.py         # Train-only regime label ordering
│   ├── 08_shap_and_crisis_analysis.py      # SHAP + COVID/Ukraine deep-dive
│   ├── 09_robustness_checks.py            # HMM, DTW, PDPs, multi-class, maps
│   ├── 10_generate_dashboard.py            # Interactive HTML dashboard
│   ├── 11_panel_b_extended_analysis.py     # 4-measure analysis (74 countries)
│   ├── 12_economic_deep_dive.py            # Fiscal thresholds, commodity effects
│   ├── 13_verify_economic_findings.py      # Verification of economic results
│   ├── 14_calibration_fix.py               # Isotonic regression calibration
│   └── 15_historical_validation.py         # 11-episode crisis validation
│
├── data/
│   ├── raw/                                # World Bank source files (not included)
│   └── processed/                          # Generated analysis-ready CSVs
│
├── outputs/
│   ├── figures/                            # ~45 figures + interactive dashboard
│   ├── tables/                             # ~26 results tables
│   └── models/                             # Trained model files (.joblib)
│
├── requirements.txt
├── run_all.sh                              # Run full pipeline (~20 min)
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.12+
- macOS or Linux

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/inflation-regime-early-warning.git
cd inflation-regime-early-warning/inflation_regimes_project

# Install dependencies
pip3 install -r requirements.txt

# Place World Bank data in data/raw/
# (see Data Sources below for download links)
```

### Run the Full Pipeline
```bash
# Core pipeline (~20 minutes)
bash run_all.sh

# Additional analyses
python3 src/11_panel_b_extended_analysis.py    # Panel B (4 measures)
python3 src/12_economic_deep_dive.py           # Fiscal thresholds
python3 src/13_verify_economic_findings.py     # Verification
python3 src/14_calibration_fix.py              # Probability calibration
python3 src/15_historical_validation.py        # Crisis validation

# Open the dashboard
open outputs/figures/dashboard_final.html
```

### Run Individual Steps
```bash
python3 src/01_load_inflation_data.py          # ~1 min
python3 src/02_load_fiscal_commodity_data.py    # ~1 min
python3 src/03_exploratory_analysis.py         # ~2 min
python3 src/04_regime_discovery.py             # ~3 min
python3 src/05_feature_engineering.py          # ~1 min
python3 src/06_model_training.py               # ~7 min (hyperparameter tuning)
python3 src/07_patch_regime_ordering.py        # <1 min
python3 src/08_shap_and_crisis_analysis.py     # ~1 min
python3 src/09_robustness_checks.py            # ~2 min
python3 src/10_generate_dashboard.py           # ~1 min
```

## Data Sources

All data is publicly available from the World Bank:

- **[Inflation Database](https://www.worldbank.org/en/research/brief/inflation-database)** — CPI measures for 209 countries, 1970–2025
- **[Fiscal Space Database](https://www.worldbank.org/en/research/brief/fiscal-space)** — 23 fiscal indicators for 204 countries, 1990–2024
- **[Commodity Markets Outlook (Pink Sheet)](https://www.worldbank.org/en/research/commodity-markets)** — 71 commodity price series, monthly/annual

Download the Excel files and place them in `data/raw/` before running the pipeline.

## Audit Trail

This codebase was subject to a rigorous independent code audit addressing 9 methodological concerns. All were resolved:

| Concern | Resolution |
|---------|-----------|
| GMM look-ahead bias | Refitted on training data only |
| Hyperparameter tuning | TimeSeriesSplit CV for XGBoost/LightGBM |
| Statistical significance | Bootstrap CIs + DeLong-equivalent test |
| Expanding window validation | 6 windows, mean AUC=0.804, std=0.035 |
| Brier score | Added; improved from 0.192 to 0.181 with calibration |
| Calibration curves | Isotonic regression applied |
| SHAP values | TreeSHAP with global, group, episode, and individual analysis |
| RF parameter justification | Breiman (2001) conservative defaults documented |
| Regime ordering | Train-only label assignment verified |

## Technical Notes

**Why not LSTM?** Tree-based models outperform deep learning on tabular data ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)). Our data is a tabular panel (67 features per country-quarter), not a sequence prediction problem.

**Probability calibration:** Raw XGBoost probabilities above 50% were systematically overconfident. Isotonic regression calibration fitted on the validation set corrects this. Calibrated model is saved separately.

**67 features** spanning: inflation levels/lags/momentum/volatility (29), commodity prices/changes (15), fiscal indicators (8), country characteristics (9), and GMM regime probabilities (6).

## Built With

Python 3.12 · XGBoost · LightGBM · scikit-learn · SHAP · Plotly · pandas · geopandas · hmmlearn · tslearn

## License

MIT License — see [LICENSE](LICENSE) for details.
