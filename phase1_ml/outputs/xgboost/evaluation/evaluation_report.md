# Phase 1 XGBoost Evaluation Report

## Dataset Summary

| Split | Rows |
|-------|------|
| Train | 74,300 |
| Val   | 15,921 |
| Test  | 15,918 |
| Total | 106,139 |

| Detail | Value |
|--------|-------|
| Base features    | 3,102 (1550 F + 1550 NL + precursor_mz_norm + ion_mode_enc) |
| Models           | 14 (adduct + class + 12 chain) |
| Lipid classes    | 78 |
| Sum comp PPM tol | 10.0 |

## Pipeline

1. **Adduct** predicted from base spectral features.
2. **Class** predicted from base + predicted adduct.
3. **Sum composition** (total_nc, total_db, total_ox) derived algebraically
   from backbone masses + precursor mass within the PPM tolerance.
   Rows without a mass match are labelled `no_match` / `no_adduct` /
   `no_backbone`; their chain predictions are zeroed out.
4. **Chain-1** predicted from base + predicted adduct + rule totals (total_c, total_db, total_ox).
5. **Chain-2** predicted from above + predicted chain-1 (chain conditioning).
6. **Chain-3** predicted from above + predicted chain-2.
7. **Chain-4** predicted from above + predicted chain-3.

## Top-Level Metrics

| Metric | Val | Test |
|--------|-----|------|
| Adduct accuracy | 0.9802 | 0.982 |
| Level 0 — Class | 0.9629 | 0.967 |
| Level 1 — Sum composition | 0.9136 | 0.9172 |
| Level 2 — Full chain | 0.8084 | 0.8107 |
| Level 3 — Exact name | 0.8084 | 0.8107 |

## Sum Composition Status

| Status | Val | Test |
|--------|-----|------|
| matched | 9,795 | 9,854 |
| multi | 5,739 | 5,702 |
| no_match | 387 | 362 |

## Per-Class Breakdown (top 20)

| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |
|-------|------|-------|--------|--------|---------|---------|
| FA | 3143 | 3142 | 0.9774 | 0.9240 | 0.9835 | 0.9332 |
| PC | 2839 | 2839 | 0.9581 | 0.9443 | 0.9641 | 0.9468 |
| TG | 1968 | 1969 | 0.9898 | 0.8806 | 0.9873 | 0.8700 |
| PE | 1244 | 1244 | 0.9646 | 0.9333 | 0.9751 | 0.9429 |
| Cer | 788 | 787 | 0.9822 | 0.9251 | 0.9809 | 0.9161 |
| SM | 593 | 594 | 0.9815 | 0.9713 | 0.9865 | 0.9764 |
| PG | 572 | 572 | 0.9353 | 0.9038 | 0.9545 | 0.9336 |
| PI | 547 | 547 | 0.9086 | 0.8775 | 0.9068 | 0.8793 |
| LPC | 506 | 506 | 0.9941 | 0.9901 | 0.9921 | 0.9783 |
| PS | 470 | 470 | 0.9681 | 0.9468 | 0.9702 | 0.9319 |
| DG | 465 | 465 | 0.9656 | 0.9441 | 0.9720 | 0.9355 |
| PE-O | 420 | 419 | 0.9500 | 0.8548 | 0.9427 | 0.8568 |
| PE-P | 396 | 397 | 0.9722 | 0.9672 | 0.9874 | 0.9874 |
| LPE | 360 | 361 | 0.9972 | 0.9833 | 0.9972 | 0.9917 |
| PC-O | 243 | 244 | 0.8971 | 0.8025 | 0.8975 | 0.8320 |
| HexCer | 146 | 146 | 0.8836 | 0.7603 | 0.8562 | 0.7740 |
| GalCer | 126 | 125 | 0.9048 | 0.8889 | 0.8480 | 0.8320 |
| CL | 91 | 92 | 1.0000 | 0.8352 | 1.0000 | 0.8152 |
| SE | 85 | 85 | 0.9294 | 0.8353 | 0.9529 | 0.9176 |
| LPI | 75 | 74 | 0.9867 | 0.9467 | 0.9730 | 0.9730 |

## Chain Descriptor MAE

| Chain | Target | Val MAE | Test MAE |
|-------|--------|---------|---------|
| Chain 1 | nc | 0.6554 | 0.6149 |
| Chain 1 | ndb | 0.2250 | 0.2021 |
| Chain 1 | nox | 0.0960 | 0.0988 |
| Chain 2 | nc | 0.5883 | 0.5659 |
| Chain 2 | ndb | 0.1826 | 0.1723 |
| Chain 2 | nox | 0.0494 | 0.0548 |
| Chain 3 | nc | 0.3837 | 0.4155 |
| Chain 3 | ndb | 0.1259 | 0.1488 |
| Chain 3 | nox | 0.0014 | 0.0000 |

*Report generated: 2026-04-08 13:01:19*
