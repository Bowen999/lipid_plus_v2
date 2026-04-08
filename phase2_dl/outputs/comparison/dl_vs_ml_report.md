# DL vs ML Comparison Report

Models compared: xgboost, lightgbm, random_forest, decision_tree, random_baseline, mlp, cnn, transformer
Splits: val, test

## Metrics Table

| Model | Type | Val-Adduct | Val-L0-Class | Val-L1-SumComp | Val-L2-Chain | Val-L3-Name | Test-Adduct | Test-L0-Class | Test-L1-SumComp | Test-L2-Chain | Test-L3-Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | ML | 98.02% | 96.29% | 91.33% | 80.84% | 80.84% | 98.20% | 96.70% | 91.71% | 81.07% | 81.07% |
| lightgbm | ML | 97.96% | 96.18% | 90.86% | 80.40% | 80.40% | 98.20% | 96.43% | 91.26% | 80.47% | 80.46% |
| random_forest | ML | 0.18% | 75.25% | 0.00% | 0.00% | 0.00% | 0.21% | 75.59% | 0.01% | 0.00% | 0.00% |
| decision_tree | ML | 0.36% | 56.76% | 0.03% | 0.00% | 0.00% | 0.35% | 57.31% | 0.01% | 0.00% | 0.00% |
| random_baseline | ML | 0.28% | 10.19% | 0.01% | 0.00% | 0.00% | 0.26% | 10.37% | 0.01% | 0.00% | 0.00% |
| mlp | DL | 97.52% | 96.89% | 92.09% | 78.84% | 78.84% | 97.76% | 96.89% | 92.25% | 79.41% | 79.41% |
| cnn | DL | 91.93% | 86.11% | 79.68% | 49.07% | 49.07% | 92.43% | 86.10% | 79.85% | 49.19% | 49.19% |
| transformer | DL | 71.20% | 68.59% | 51.88% | 20.45% | 20.45% | 71.33% | 68.90% | 52.29% | 20.90% | 20.90% |

## Best Model

**xgboost** achieves the highest val L3 accuracy: **80.84%**

## Per-Metric Rankings (val set)

**Adduct**: xgboost (98.02%), lightgbm (97.96%), mlp (97.52%), cnn (91.93%), transformer (71.20%), decision_tree (0.36%), random_baseline (0.28%), random_forest (0.18%)

**L0-Class**: mlp (96.89%), xgboost (96.29%), lightgbm (96.18%), cnn (86.11%), random_forest (75.25%), transformer (68.59%), decision_tree (56.76%), random_baseline (10.19%)

**L1-SumComp**: mlp (92.09%), xgboost (91.33%), lightgbm (90.86%), cnn (79.68%), transformer (51.88%), decision_tree (0.03%), random_baseline (0.01%), random_forest (0.00%)

**L2-Chain**: xgboost (80.84%), lightgbm (80.40%), mlp (78.84%), cnn (49.07%), transformer (20.45%), random_forest (0.00%), decision_tree (0.00%), random_baseline (0.00%)

**L3-Name**: xgboost (80.84%), lightgbm (80.40%), mlp (78.84%), cnn (49.07%), transformer (20.45%), random_forest (0.00%), decision_tree (0.00%), random_baseline (0.00%)

