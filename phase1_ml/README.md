# Phase 1 ML — Lipid Structure Prediction from MS/MS

Predict lipid class and acyl-chain descriptors (carbon count, double bonds, oxidation) from tandem mass spectra using classical ML models. This project compares multiple model families and selects the optimal per-target combination.

## Folder Structure

```
phase1_ml/
├── README.md
│
├── data/
│   ├── processed/                        # Feature-engineered data
│   │   ├── lipid_ms2_features.parquet    #   3,102-dim feature matrix + labels (106 K rows)
│   │   └── lipid_ms2_source_validated.parquet  # Source parquet with has_spectrum flag
│   └── splits/                           # Stratified 70/15/15 index arrays
│       ├── split_train.npy
│       ├── split_val.npy
│       └── split_test.npy
│
├── configs/                              # Per-model hyperparameter JSON files
│   ├── xgboost.json
│   ├── lightgbm.json
│   ├── random_forest.json
│   ├── decision_tree.json
│   └── random_baseline.json
│
├── src/                                  # Importable Python package
│   ├── __init__.py
│   ├── utils.py                          #   Constants, spectrum parser, binning, name builder
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaning.py                   #   Spectrum validation, class_to_numchain derivation
│   │   ├── features.py                   #   Binning, NL, normalisation, feature matrix build
│   │   └── splitting.py                  #   Stratified train/val/test split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                       #   BaseLipidModel ABC (fit/predict/save/load)
│   │   ├── xgboost.py                    #   XGBoost implementation (complete, migrated)
│   │   ├── lightgbm.py                   #   LightGBM (placeholder)
│   │   ├── random_forest.py              #   Random Forest (placeholder)
│   │   ├── decision_tree.py              #   Decision Tree (placeholder)
│   │   └── random_baseline.py            #   Random assignment (placeholder)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                    #   Hierarchical accuracy, MAE, confusion matrices
│   │   ├── reporting.py                  #   Cross-model comparison report generation
│   │   └── sum_comp_test.py              #   Sum-composition verification utility
│   └── pipeline/
│       ├── __init__.py
│       ├── inference.py                  #   End-to-end prediction (class→numchain→chains)
│       └── selection.py                  #   Best per-target model combination finder
│
├── scripts/                              # CLI entry-points (run from project root)
│   ├── 01_prepare_data.py                #   Clean + featurise + split (combines old 00–02)
│   ├── 02_train.py                       #   Train a single model: --model xgboost
│   ├── 03_train_all.py                   #   Train all 5 model families
│   ├── 04_evaluate.py                    #   Evaluate a single model
│   ├── 05_evaluate_all.py                #   Evaluate all + generate comparison table
│   └── 06_select_best.py                 #   Find optimal per-target model combination
│
├── outputs/                              # All generated artifacts (gitignored except shared/)
│   ├── shared/                           #   Encoders & metadata (model-independent)
│   │   ├── adduct_encoder.joblib
│   │   ├── class_encoder.joblib
│   │   ├── class_backbone_masses.json
│   │   ├── class_to_numchain.json
│   │   └── precursor_mz_stats.npy
│   ├── xgboost/                          #   ← XGBoost results (migrated, complete)
│   │   ├── models/                       #     xgb_adduct.joblib … xgb_nox4.joblib
│   │   ├── predictions/                  #     val_predictions.csv, test_predictions.csv
│   │   └── evaluation/                   #     metrics JSON, confusion CSV, report MD
│   ├── lightgbm/                         #   ← (empty, to be filled)
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   ├── random_forest/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   ├── decision_tree/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   ├── random_baseline/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   └── comparison/                       #   Cross-model comparison reports & best-combo
│
└── notebooks/                            #   Optional exploration / visualisation
```

## Pipeline (7-step cascaded prediction)

| Step | Target            | Features                                        |
|------|-------------------|-------------------------------------------------|
| 1    | Adduct            | base spectral (3,102 dim)                       |
| 2    | Class             | base + predicted adduct                         |
| 3    | Sum composition   | algebraic from backbone mass + precursor m/z    |
| 4    | Chain-1           | base + adduct + class + rule totals             |
| 5    | Chain-2           | above + predicted chain-1                       |
| 6    | Chain-3           | above + predicted chain-2                       |
| 7    | Chain-4           | above + predicted chain-3                       |

Class → num_chain is a strict 1-to-1 mapping (not predicted; looked up from `class_to_numchain.json`).

## Model Families

| Model            | Config file            | Key characteristics                    |
|------------------|------------------------|----------------------------------------|
| XGBoost          | `xgboost.json`         | Gradient boosting, strong baseline     |
| LightGBM         | `lightgbm.json`        | Faster training, native categorical    |
| Random Forest    | `random_forest.json`   | Bagging ensemble, less overfitting     |
| Decision Tree    | `decision_tree.json`   | Single tree, interpretability baseline |
| Random Baseline  | `random_baseline.json` | Class-frequency random draw, floor     |

## Evaluation Levels

| Level | Metric                        | Measures                                          |
|-------|-------------------------------|---------------------------------------------------|
| L0    | Class accuracy                | Lipid class correct                               |
| L1    | Sum composition accuracy      | Class + total_c + total_db + total_ox all correct |
| L2    | Full chain accuracy (multiset)| All individual chains correct (order-agnostic)    |
| L3    | Exact name match              | Reconstructed name string matches ground truth    |

## Quick Start

```bash
cd phase1_ml

# 1. Prepare data (already done — skip if data/processed/ exists)
python scripts/01_prepare_data.py

# 2. Train all models
python scripts/03_train_all.py

# 3. Evaluate all models + comparison
python scripts/05_evaluate_all.py

# 4. Find best per-target combination
python scripts/06_select_best.py
```
