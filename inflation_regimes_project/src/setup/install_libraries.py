"""
STEP 1B: Install All Required Libraries
========================================
Run this AFTER setup_project.py

HOW TO RUN:
  python install_libraries.py

This will install every Python library needed for the entire project.
It may take 5-10 minutes depending on your internet speed.

If you get errors about permissions, try:
  pip install --user [library_name]

If you're using Anaconda, some libraries install differently:
  conda install -c conda-forge [library_name]
"""

import subprocess
import sys

def install_libraries():
    """Install all required libraries for the project."""
    
    print("=" * 60)
    print("  INSTALLING REQUIRED LIBRARIES")
    print("=" * 60)
    print()
    print(f"  Using Python: {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print()
    
    # Libraries grouped by purpose
    # Each tuple: (pip_name, import_name, description)
    libraries = [
        # === CORE DATA HANDLING ===
        ("numpy", "numpy", "Mathematical arrays and operations"),
        ("pandas", "pandas", "Data tables and manipulation"),
        ("openpyxl", "openpyxl", "Reading Excel .xlsx files"),
        ("scipy", "scipy", "Scientific computing utilities"),
        
        # === VISUALIZATION ===
        ("matplotlib", "matplotlib", "Static charts and plots"),
        ("seaborn", "seaborn", "Statistical data visualisation"),
        ("plotly", "plotly", "Interactive charts and maps"),
        
        # === CORE MACHINE LEARNING ===
        ("scikit-learn", "sklearn", "ML algorithms, metrics, preprocessing"),
        ("xgboost", "xgboost", "Gradient boosted trees (top performer)"),
        ("lightgbm", "lightgbm", "Fast gradient boosting"),
        
        # === SPECIALISED ML ===
        ("hmmlearn", "hmmlearn", "Hidden Markov Models for regimes"),
        ("tslearn", "tslearn", "Time series clustering (DTW)"),
        ("shap", "shap", "Model interpretability (SHAP values)"),
        ("statsmodels", "statsmodels", "Statistical models and tests"),
        
        # === GEOSPATIAL ===
        ("geopandas", "geopandas", "Maps coloured by data values"),
        
        # === DEVELOPMENT TOOLS ===
        ("jupyter", "jupyter", "Interactive notebooks"),
        ("notebook", "notebook", "Jupyter notebook server"),
        ("ipykernel", "ipykernel", "Python kernel for Jupyter"),
    ]
    
    results = {"success": [], "failed": [], "already": []}
    
    for pip_name, import_name, description in libraries:
        # First check if already installed
        try:
            __import__(import_name)
            results["already"].append(pip_name)
            print(f"  [ALREADY INSTALLED] {pip_name:20s} - {description}")
            continue
        except ImportError:
            pass
        
        # Try to install
        print(f"  [INSTALLING]        {pip_name:20s} - {description}...", end="", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            results["success"].append(pip_name)
            print(" OK")
        except subprocess.CalledProcessError:
            results["failed"].append(pip_name)
            print(" FAILED")
    
    # Summary
    print()
    print("=" * 60)
    print("  INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"  Already installed: {len(results['already'])}")
    print(f"  Newly installed:   {len(results['success'])}")
    print(f"  Failed:            {len(results['failed'])}")
    
    if results["failed"]:
        print()
        print("  FAILED LIBRARIES (try installing manually):")
        for lib in results["failed"]:
            print(f"    pip install {lib}")
        print()
        print("  If pip fails, try conda:")
        for lib in results["failed"]:
            print(f"    conda install -c conda-forge {lib}")
    else:
        print()
        print("  ALL LIBRARIES INSTALLED SUCCESSFULLY!")
    
    print()
    print("  NEXT STEP: Run python verify_setup.py")
    print()

if __name__ == "__main__":
    install_libraries()
