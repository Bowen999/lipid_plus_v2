**project structure has been updated**

## Task

XGBoost baseline already complete, now implement, train, and evaluate four additional model families — **LightGBM**,
**Random Forest**, **Decision Tree**, and **Random Baseline** — using the exact
same data, splits, features, and evaluation pipeline as the existing XGBoost run.
Then find the **optimal per-target model combination**.

---

## Constraints

- Do NOT re-prepare the data; reuse `data/processed/` and `data/splits/` as-is.
- Do NOT modify any files in `outputs/xgboost/` — those results are frozen.
- All new code goes into `src/` (model implementations) and `scripts/` (entry-points).
- Every model must conform to the `BaseLipidModel` interface in `src/models/base.py`.
- Use the same 7-step cascaded pipeline that XGBoost uses (see Pipeline below).
- All random seeds fixed at 42.
- Save all outputs under `outputs/{model_name}/` following the same sub-structure
  as `outputs/xgboost/`.

---

## Pipeline (7-step cascaded prediction)

The prediction pipeline is a cascade where each step's output feeds the next.
During **training**, ground-truth values are used as inputs (teacher forcing).
During **inference**, predicted values are propagated.

```
Step 1 — Adduct:        features = base spectral (3,102 dim)
Step 2 — Class:         features = base + predicted adduct (3,103 dim)
Step 3 — Sum comp:      algebraically derived from class backbone mass + precursor_mz
                         (no model — rule-based; see class_backbone_masses.json)
Step 4 — Chain-1:       features = base + adduct + class + rule_totals (3,107 dim)
Step 5 — Chain-2:       features = above + predicted nc1/ndb1/nox1 (3,110 dim)
Step 6 — Chain-3:       features = above + predicted nc2/ndb2/nox2 (3,113 dim)
Step 7 — Chain-4:       features = above + predicted nc3/ndb3/nox3 (3,116 dim)
```
---

## Model Specifications
1. LightGBM (src/models/lightgbm.py | name="lightgbm")

Estimator: lgb.LGBMClassifier (params from configs/lightgbm.json)

Config: verbose=-1, eval_set + callbacks=[lgb.early_stopping(30)]

Weights: sample_weight=compute_sample_weight("balanced", y_train)

2. Random Forest (src/models/random_forest.py | name="random_forest")

Estimator: RandomForestClassifier (params from configs/random_forest.json)

Config: max_features="sqrt", full n_estimators (no early stopping), RF with 200 trees 

Weights: class_weight="balanced"

3. Decision Tree (src/models/decision_tree.py | name="decision_tree")

Estimator: DecisionTreeClassifier (params from configs/decision_tree.json)

Config: No early stopping/eval

Weights: class_weight="balanced"

4. Random Baseline (src/models/random_baseline.py | name="random_baseline")

Logic: Predict based on training class frequencies (performance floor).

fit(): Normalise np.bincount(y_train) to get probabilities.

predict(): np.random.choice(classes, size=len(X), p=probs)

predict_proba(): Return fixed probability vector tiled for all rows.

save()/load(): Serialize probability vector using joblib.

---

## Script Specifications

### `scripts/01_prepare_data.py` (already done — no-op)

If `data/processed/lipid_ms2_features.parquet` exists, print "Data already
prepared" and exit. Otherwise, chain the cleaning → feature engineering →
splitting steps from `src/data/`.

### `scripts/02_train.py`

```
Usage: python scripts/02_train.py --model {xgboost,lightgbm,random_forest,decision_tree,random_baseline}
```

1. Load config from `configs/{model_name}.json`.
2. Load features and splits.
3. For each of the 14 targets (adduct, class, nc1..4, ndb1..4, nox1..4):
   a. Construct the feature matrix for that step (base + cascade columns).
      During training, cascade columns come from ground truth (teacher forcing).
   b. Filter to rows where the target is valid (e.g., nc3 only for 3/4-chain).
   c. Instantiate the model class, call `fit()`.
   d. Save model artifact to `outputs/{model_name}/models/{target_name}.joblib`.
   e. Log: target, train size, val size, classes, best iteration, val loss.
4. Write all logs to `outputs/{model_name}/evaluation/training_log.txt`.

### `scripts/03_train_all.py`

Loop over all 5 model families and call `02_train.py` for each.
Print a summary table at the end with training times.

1. have quick run mode (--quick), only test the full workflow can be run

### `scripts/04_evaluate.py`

```
Usage: python scripts/04_evaluate.py --model {model_name}
```

1. Load all 14 model artifacts from `outputs/{model_name}/models/`.
2. Load shared encoders and class_to_numchain.
3. For val and test splits:
   a. Run the 7-step cascaded inference pipeline (predictions propagated forward).
   b. Build the sum composition algebraically at Step 3.
   c. Reconstruct the predicted lipid name from predicted components.
   d. Save predictions CSV to `outputs/{model_name}/predictions/{split}_predictions.csv`.
4. Compute all metrics (same as XGBoost evaluation):
   - L0 class accuracy, L1 sum composition, L2 full chain, L3 exact name
   - Per-class breakdown (L0, L1)
   - Chain descriptor MAE (nc, ndb, nox per chain position)
   - Confusion matrix (78×78 CSV)
5. Save to `outputs/{model_name}/evaluation/`:
   - `val_metrics.json`, `test_metrics.json`
   - `val_class_confusion.csv`, `test_class_confusion.csv`
   - `evaluation_report.md`

### `scripts/05_evaluate_all.py`

1. Run `04_evaluate.py` for each model family.
2. Load all `test_metrics.json` files.
3. Generate `outputs/comparison/comparison_report.md`:
4. Save `outputs/comparison/all_test_metrics.json` (dict of model → metrics).

### `scripts/06_select_best.py`

**Purpose:** Find the optimal per-target model combination and evaluate it.

The 14 prediction targets can each use a different model. For example, the best
combo might be: XGBoost for adduct, LightGBM for class, XGBoost for nc1, RF for ndb1, etc.

**Approach:**

1. For each of the 14 targets, load the per-target accuracy from each model:
   - `adduct`: adduct accuracy
   - `class`: L0 class accuracy
   - `nc1`: 1 − MAE_nc1 (or accuracy if available)
   - `ndb1`: 1 − MAE_ndb1
   - `nox1`: nox1 accuracy
   - Same for chain positions 2–4

2. Select the best model per target (highest accuracy / lowest MAE on **val set**).

3. Assemble the "best combo" by loading the corresponding model artifacts.

4. Re-run the 7-step cascaded inference on the **test set** using the mixed models:
   - Step 1: use best adduct model's predictions
   - Step 2: use best class model (with Step 1's adduct predictions as input)
   - Steps 4–7: use each target's best model with cascade from previous best predictions

   **IMPORTANT:** Because cascade predictions propagate, the best combo's end-to-end
   accuracy may differ from cherry-picking per-target accuracies. The re-evaluation
   on test is essential.

5. Output:
   - `outputs/comparison/best_combo.json`:
     ```json
     {
       "adduct": "xgboost",
       "class": "lightgbm",
       "nc1": "xgboost",
       "ndb1": "xgboost",
       "nox1": "random_forest",
       "nc2": "xgboost",
       ...
     }
     ```
   - `outputs/comparison/best_combo_test_metrics.json` (full hierarchical metrics)
   - `outputs/comparison/best_combo_test_predictions.csv`
   - `outputs/comparison/selection_report.md`:
     ```markdown

# Best Model Combination Report

     ## Selected Models

     | Target | Model     | Val Accuracy |
     |--------|-----------|-------------|
     | adduct | xgboost   | 0.9802      |
     | class  | lightgbm  | 0.9635      |
     | ...    |           |             |

     ## End-to-End Test Metrics (Best Combo)

     | Metric          | Best Combo | XGBoost Only | Delta  |
     |-----------------|-----------|--------------|--------|
     | L0 — Class      |           |              |        |
     | L1 — Sum comp   |           |              |        |
     | L2 — Full chain |           |              |        |
     | L3 — Exact name |           |              |        |

     ## Conclusion
     (Auto-generated prose: did the combo beat single-model XGBoost?)
     ```

---

## Code Quality Requirements

- All model implementations inherit from `BaseLipidModel` in `src/models/base.py`.
- Every function has a docstring.
- Use `pathlib.Path` for all file I/O.
- `tqdm` progress bars on any loop over >1,000 items.
- Print `[DONE] {script_name} — {timestamp}` on completion of each script.
- Random seeds fixed at 42 throughout.
- Models saved with `joblib.dump(model, path, compress=3)`.
- All parquet files use snappy compression.