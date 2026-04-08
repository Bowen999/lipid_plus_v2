# Phase 1 Multi-Model Comparison

*Generated: 2026-04-08 13:02:28*

## Validation Set Metrics

| Metric | decision_tree | lightgbm | random_baseline | random_forest | xgboost |
|--------|--------|--------|--------|--------|--------|
| Adduct accuracy | 0.0036 | 0.9796 | 0.0028 | 0.0018 | 0.9802 |
| L0 class accuracy | 0.5676 | 0.9618 | 0.1019 | 0.7525 | 0.9629 |
| L1 sum composition | 0.0003 | 0.9086 | 0.0001 | 0.0000 | 0.9136 |
| L2 full chain | 0.0000 | 0.8040 | 0.0000 | 0.0000 | 0.8084 |
| L3 exact name | 0.0000 | 0.8040 | 0.0000 | 0.0000 | 0.8084 |

## Test Set Metrics

| Metric | decision_tree | lightgbm | random_baseline | random_forest | xgboost |
|--------|--------|--------|--------|--------|--------|
| Adduct accuracy | 0.0035 | 0.9820 | 0.0026 | 0.0021 | 0.9820 |
| L0 class accuracy | 0.5731 | 0.9643 | 0.1037 | 0.7559 | 0.9670 |
| L1 sum composition | 0.0001 | 0.9126 | 0.0001 | 0.0001 | 0.9172 |
| L2 full chain | 0.0000 | 0.8047 | 0.0000 | 0.0000 | 0.8107 |
| L3 exact name | 0.0000 | 0.8046 | 0.0000 | 0.0000 | 0.8107 |

## Best Model Per Level (Val)

| Level | Best Model | Val Score |
|-------|-----------|-----------|
| Adduct accuracy | xgboost | 0.9802 |
| L0 class accuracy | xgboost | 0.9629 |
| L1 sum composition | xgboost | 0.9136 |
| L2 full chain | xgboost | 0.8084 |
| L3 exact name | xgboost | 0.8084 |

## Sum Composition Status (Val)

| Status | decision_tree | lightgbm | random_baseline | random_forest | xgboost |
|--------|--------|--------|--------|--------|--------|
| matched | 2,082 | 9,716 | 834 | 2,010 | 9,795 |
| multi | 1,843 | 5,769 | 1,336 | 1,632 | 5,739 |
| no_match | 11,996 | 436 | 13,751 | 12,279 | 387 |
