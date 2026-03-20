"""
STEP 1A: Project Folder Setup
==============================
Run this script FIRST to create your project folder structure.

HOW TO RUN:
1. Save this file somewhere on your computer (e.g., Desktop)
2. Open a terminal / command prompt
3. Navigate to where you saved it: cd Desktop
4. Run: python setup_project.py

It will create the entire project folder structure for you.
"""

import os
import sys

def create_project():
    """Create the full project folder structure."""
    
    # The project will be created in your current directory
    project_name = "inflation_regimes_project"
    
    # Define all folders we need
    folders = [
        f"{project_name}/data/raw",              # Your original Excel files go here
        f"{project_name}/data/processed",         # Cleaned datasets will go here
        f"{project_name}/data/interim",           # Intermediate files during processing
        f"{project_name}/notebooks",              # Jupyter notebooks (one per step)
        f"{project_name}/src",                    # Reusable Python functions
        f"{project_name}/outputs/figures",        # Charts and plots
        f"{project_name}/outputs/tables",         # Results tables (CSV)
        f"{project_name}/outputs/models",         # Saved trained models
        f"{project_name}/papers",                 # Your PDF research papers
    ]
    
    print("=" * 60)
    print("  INFLATION REGIMES PROJECT - FOLDER SETUP")
    print("=" * 60)
    print()
    
    # Create each folder
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  Created: {folder}/")
    
    print()
    
    # Create a README in the project root
    readme_content = """# Predicting Inflation Regimes Using Machine Learning

## Project Structure

```
inflation_regimes_project/
│
├── data/
│   ├── raw/                  # Original Excel files (DO NOT MODIFY)
│   │   ├── Inflationdata.xlsx
│   │   ├── Fiscalspacedata.xlsx
│   │   ├── CMOHistoricalDataMonthly.xlsx
│   │   └── CMOHistoricalDataAnnual.xlsx
│   │
│   ├── processed/            # Clean, analysis-ready datasets
│   │   ├── inflation_panel_monthly.csv
│   │   ├── inflation_panel_annual.csv
│   │   ├── master_panel_monthly.csv
│   │   └── master_panel_annual.csv
│   │
│   └── interim/              # Intermediate processing files
│
├── notebooks/                # Jupyter notebooks (run in order)
│   ├── Step02_Load_Inflation_Data.ipynb
│   ├── Step03_Load_Fiscal_Commodity_Data.ipynb
│   ├── Step04_Exploratory_Analysis.ipynb
│   ├── Step05_Regime_Discovery.ipynb
│   ├── Step06_Feature_Engineering.ipynb
│   ├── Step07_Model_Training.ipynb
│   ├── Step08_Crisis_Analysis.ipynb
│   ├── Step09_SHAP_Interpretability.ipynb
│   └── Step10_Dashboard.ipynb
│
├── src/                      # Reusable Python code
│   ├── data_loading.py       # Functions to load each dataset
│   ├── feature_engineering.py
│   ├── regime_models.py
│   └── evaluation.py
│
├── outputs/
│   ├── figures/              # All charts and maps
│   ├── tables/               # Results in CSV format
│   └── models/               # Saved model files
│
├── papers/                   # Research papers for reference
│   ├── machinelearningatcentralbanks.pdf
│   └── ssrn2679090.pdf
│
└── README.md                 # This file
```

## How to Run

1. Place your data files in `data/raw/`
2. Open notebooks in order (Step02, Step03, etc.)
3. Run each cell top-to-bottom
4. Outputs are saved automatically to `outputs/`

## Data Sources

- **Inflation Database**: World Bank, 209 countries, 1970-2025
- **Fiscal Space Database**: World Bank, 204 countries, 1990-2024  
- **Commodity Prices (Pink Sheet)**: World Bank, 71 series, 1960-2025
"""
    
    readme_path = os.path.join(project_name, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  Created: {readme_path}")
    
    # Create an empty __init__.py in src/ so Python treats it as a package
    init_path = os.path.join(project_name, "src", "__init__.py")
    with open(init_path, 'w') as f:
        f.write("# This file makes src/ a Python package\n")
    print(f"  Created: {init_path}")
    
    print()
    print("=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("  NEXT STEPS:")
    print(f"  1. Copy your 4 Excel files into:")
    print(f"     {os.path.abspath(os.path.join(project_name, 'data', 'raw'))}/")
    print(f"     - Inflationdata.xlsx")
    print(f"     - Fiscalspacedata.xlsx")
    print(f"     - CMOHistoricalDataMonthly.xlsx")
    print(f"     - CMOHistoricalDataAnnual.xlsx")
    print()
    print(f"  2. Copy your 2 PDF papers into:")
    print(f"     {os.path.abspath(os.path.join(project_name, 'papers'))}/")
    print(f"     - machinelearningatcentralbanks.pdf")
    print(f"     - ssrn2679090.pdf")
    print()
    print(f"  3. Run the install script: python install_libraries.py")
    print(f"  4. Then run the verification: python verify_setup.py")
    print()

if __name__ == "__main__":
    create_project()
