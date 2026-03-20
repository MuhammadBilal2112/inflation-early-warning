#!/bin/bash
# ==============================================================
# INFLATION REGIME EARLY WARNING SYSTEM
# Master execution script — runs the full pipeline
# ==============================================================
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Prerequisites:
#   - Python 3.12+
#   - pip3 install -r requirements.txt
#   - Raw data files in data/raw/
# ==============================================================

set -e  # Exit on any error

echo "=============================================="
echo "  Inflation Regime Pipeline — Full Run"
echo "=============================================="

cd "$(dirname "$0")"

echo ""
echo "[1/10] Loading inflation data..."
python3 src/01_load_inflation_data.py

echo ""
echo "[2/10] Loading fiscal and commodity data..."
python3 src/02_load_fiscal_commodity_data.py

echo ""
echo "[3/10] Exploratory analysis..."
python3 src/03_exploratory_analysis.py

echo ""
echo "[4/10] Regime discovery (GMM)..."
python3 src/04_regime_discovery.py

echo ""
echo "[5/10] Feature engineering..."
python3 src/05_feature_engineering.py

echo ""
echo "[6/10] Model training (with hyperparameter tuning)..."
echo "       This step takes 5-10 minutes..."
python3 src/06_model_training.py

echo ""
echo "[7/10] Patching regime ordering..."
python3 src/07_patch_regime_ordering.py

echo ""
echo "[8/10] SHAP and crisis analysis..."
python3 src/08_shap_and_crisis_analysis.py

echo ""
echo "[9/10] Robustness checks..."
python3 src/09_robustness_checks.py

echo ""
echo "[10/10] Generating dashboard..."
python3 src/10_generate_dashboard.py

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Dashboard: outputs/figures/dashboard_interactive.html"
echo "=============================================="
