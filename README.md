# Inflation Regime Early Warning System

A machine learning pipeline that classifies countries into inflation regimes and predicts upward regime transitions — providing actionable early warnings for policymakers and researchers.

Built on World Bank data covering **135 countries from 1971–2025**, using Gaussian Mixture Models for regime discovery and gradient boosting for transition prediction.

## Key Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost (tuned via TimeSeriesSplit CV) |
| AUC-ROC (1Q ahead) | **0.839** [0.816, 0.860] |
| AUC-ROC (2Q ahead) | **0.813** [0.791, 0.835] |
| AUC-ROC (4Q ahead) | **0.779** [0.754, 0.802] |
| ML vs Linear baseline | p < 0.001 (bootstrap DeLong test) |
| Early warning detection | 51% at 0.3 alert threshold |
| Countries covered | 135 |
| Inflation regimes | 6 (GMM, BIC-selected) |

## Interactive Dashboard

The pipeline generates a self-contained interactive HTML dashboard with:
- 🗺️ World choropleth map coloured by current regime
- 🔍 Type-ahead country search across all 135 countries
- 📈 Transition probability timelines with alert zones
- 📊 Inflation component breakdown (headline, food, energy)
- 🎨 Regime history visualisation

## What Are Inflation Regimes?

Rather than using arbitrary thresholds (e.g., "high" = above 10%), this project uses unsupervised learning to discover natural clusters in multi-dimensional inflation data. The GMM identifies 6 regimes based on the **joint behaviour** of headline, food, and energy CPI:

| Regime | Headline CPI | Key Characteristic |
|--------|-------------|-------------------|
| R0: Low | ~0.2% | Stable, falling energy prices |
| R1: Low-energy | ~3.2% | Low headline, rising energy |
| R2: Moderate | ~3.8% | Food-driven inflation |
| R3: Mod-energy | ~7.8% | Energy-driven inflation |
| R4: Elevated | ~11.3% | Broad inflation pressure |
| R5: Crisis | ~71.2% | All components spiralling |

This distinction matters: a country with 8% headline where food=15% and energy=2% (supply shock) is in a **different regime** than one where all components are at 8%.

## Methodology

### Regime Discovery
- Gaussian Mixture Model (K=6, selected by BIC)
- 7 features: headline/food/energy CPI, component dispersion, momentum, food-energy gap, volatility
- **GMM fitted on training data only** (pre-2015) to prevent look-ahead bias
- Robustness: HMM confirms regime structure (Adjusted Rand Index = 0.687)

### Prediction Models
Four models predict binary upward regime transitions at 1, 2, and 4 quarter horizons:

| Model | Tuning | AUC-ROC (1Q, Ukraine test) |
|-------|--------|---------------------------|
| Logistic Regression | Balanced class weights | 0.726 |
| Random Forest | Breiman (2001) defaults | 0.820 |
| **XGBoost** | **TimeSeriesSplit CV** | **0.839** |
| LightGBM | TimeSeriesSplit CV | 0.836 |

### Rigorous Evaluation
- **Time-based splits**: Train (1971–2014) → Validation (2015–18) → Test-COVID (2019–20) → Test-Ukraine (2021–25)
- **No data leakage**: GMM, scaler, imputer all fitted on training data only
- **Bootstrap CIs**: 1000-sample confidence intervals for all AUC estimates
- **Expanding window**: 6 temporal windows, mean AUC = 0.804 (σ = 0.035)
- **Calibration**: Brier score + calibration curves
- **Interpretability**: TreeSHAP values, partial dependence plots, crisis fingerprints

## Project Structure

```
inflation_regimes_project/
├── src/
│   ├── setup/                          # Environment setup
│   ├── 01_load_inflation_data.py       # World Bank CPI data processing
│   ├── 02_load_fiscal_commodity_data.py # Fiscal & commodity data, panel merge
│   ├── 03_exploratory_analysis.py      # EDA: 13 figures, 3 tables
│   ├── 04_regime_discovery.py          # GMM regime classification
│   ├── 05_feature_engineering.py       # 67 features + 3 target variables
│   ├── 06_model_training.py            # 4 models, HP tuning, bootstrap CIs
│   ├── 07_patch_regime_ordering.py     # Train-only regime label ordering
│   ├── 08_shap_and_crisis_analysis.py  # SHAP + COVID/Ukraine deep-dive
│   ├── 09_robustness_checks.py         # HMM, DTW, PDPs, multi-class, world maps
│   └── 10_generate_dashboard.py        # Interactive HTML dashboard
│
├── data/
│   ├── raw/                            # World Bank source files (not included)
│   └── processed/                      # Generated analysis-ready CSVs
│
├── outputs/
│   ├── figures/                        # ~40 figures + interactive dashboard
│   ├── tables/                         # ~21 results tables
│   └── models/                         # Trained model files (.joblib)
│
├── requirements.txt
├── run_all.sh                          # Run full pipeline
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/inflation-regime-early-warning.git
cd inflation-regime-early-warning/inflation_regimes_project

# Install dependencies
pip3 install -r requirements.txt

# Place World Bank data in data/raw/ (see Data Sources below)

# Run the full pipeline (~20 minutes)
bash run_all.sh

# Open the dashboard
open outputs/figures/dashboard_interactive.html
```

### Run Individual Steps
```bash
python3 src/01_load_inflation_data.py        # ~1 min
python3 src/02_load_fiscal_commodity_data.py  # ~1 min
python3 src/03_exploratory_analysis.py       # ~2 min
python3 src/04_regime_discovery.py           # ~3 min
python3 src/05_feature_engineering.py        # ~1 min
python3 src/06_model_training.py             # ~7 min (hyperparameter tuning)
python3 src/07_patch_regime_ordering.py      # <1 min
python3 src/08_shap_and_crisis_analysis.py   # ~1 min
python3 src/09_robustness_checks.py          # ~2 min
python3 src/10_generate_dashboard.py         # ~1 min
```

## Data Sources

All data is publicly available from the World Bank:

- **[Inflation Database](https://www.worldbank.org/en/research/brief/inflation-database)**: CPI measures for 209 countries, 1970–2025
- **[Fiscal Space Database](https://www.worldbank.org/en/research/brief/fiscal-space)**: 23 fiscal indicators for 204 countries, 1990–2024
- **[Commodity Markets Outlook](https://www.worldbank.org/en/research/commodity-markets)**: 71 commodity price series, monthly/annual

Download the Excel files and place them in `data/raw/` before running the pipeline.

## Technical Notes

**Why not LSTM?** Tree-based models consistently outperform deep learning on tabular data ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)). Our data is a tabular panel (67 features per country-quarter), not a sequence prediction problem. See `outputs/tables/note_lstm_justification.txt` for detailed rationale.

**Regime ordering**: Regime labels (0–5) are assigned using GMM means computed from **training data only**, ensuring no look-ahead bias even in label numbering.

**67 features** spanning: inflation levels/lags/momentum/volatility (29), commodity prices/changes (15), fiscal indicators (8), country characteristics (9), and GMM regime probabilities (6).

## Built With

Python 3.12 · XGBoost · LightGBM · scikit-learn · SHAP · Plotly · pandas · geopandas

## License

MIT License — see [LICENSE](LICENSE) for details.
