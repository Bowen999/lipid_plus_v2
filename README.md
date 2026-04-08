**Goal:** Predict lipid class and acyl-chain descriptors (carbon count, double bonds, oxidation) from tandem mass spectra, then reconstruct the full lipid name.

**Data:** ~106K MS/MS spectra → 3,102-dim feature matrix (binned peaks + neutral losses), with stratified 70/15/15 train/val/test splits shared across phases.

**Prediction targets:** Adduct, class, sum composition (total C / DB / oxidation), and individual acyl chains (up to 4). Class → num_chains is a fixed lookup, not predicted.

**Evaluation levels:** L0 class acc · L1 sum-composition acc · L2 multiset chain acc · L3 exact name match.

**Phase 1 — Classical ML (`phase1_ml/`)**
Uses a **cascaded 7-step pipeline**: Adduct → Class → Sum composition (algebraic from backbone mass + precursor m/z) → Chain-1 → Chain-2 → Chain-3 → Chain-4, where each step's prediction is fed as input to the next. Compares five model families (XGBoost, LightGBM, Random Forest, Decision Tree, Random Baseline), each with its own config JSON, trained per target. `06_select_best.py` picks the optimal per-target model combination. Scripts: `01_prepare_data → 02/03_train → 04/05_evaluate → 06_select_best`.

**Phase 2 — Deep Learning (`phase2_dl/`)**
Uses an **end-to-end multi-task** approach instead of a cascade. A shared encoder (MLP, CNN, or Transformer) feeds 14 parallel classification heads that predict all targets simultaneously from the spectral features — no conditioning between targets, no teacher forcing. Same scripts pattern: `01_prepare_data → 02/03_train → 04_evaluate → 05_compare`. Reuses Phase 1's processed features and splits.

**Code conventions:** Each phase has `src/` (importable package: data, models, evaluation/training, pipeline), `configs/` (per-model JSON), `scripts/` (numbered CLI entry points run from phase root), `outputs/<model>/{models,predictions,evaluation}` plus `outputs/shared/` for encoders and `outputs/comparison/` for cross-model reports. Phase 1 models inherit a `BaseLipidModel` ABC (fit/predict/save/load).

Other folders present: `phase3/`, `phase3_hybrid/`, `dataset/`, `rules/` (not covered here).