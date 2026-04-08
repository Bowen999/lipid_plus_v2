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
| Adduct accuracy | 0.0028 | 0.0026 |
| Level 0 — Class | 0.1019 | 0.1037 |
| Level 1 — Sum composition | 0.0001 | 0.0001 |
| Level 2 — Full chain | 0.0 | 0.0 |
| Level 3 — Exact name | 0.0 | 0.0 |

## Sum Composition Status

| Status | Val | Test |
|--------|-----|------|
| matched | 834 | 775 |
| multi | 1,336 | 1,371 |
| no_match | 13,751 | 13,772 |

## Per-Class Breakdown (top 20)

| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |
|-------|------|-------|--------|--------|---------|---------|
| FA | 3143 | 3142 | 0.1941 | 0.0000 | 0.2021 | 0.0000 |
| PC | 2839 | 2839 | 0.1789 | 0.0000 | 0.1765 | 0.0000 |
| TG | 1968 | 1969 | 0.1098 | 0.0000 | 0.1178 | 0.0000 |
| PE | 1244 | 1244 | 0.0828 | 0.0008 | 0.0876 | 0.0000 |
| Cer | 788 | 787 | 0.0444 | 0.0000 | 0.0521 | 0.0000 |
| SM | 593 | 594 | 0.0354 | 0.0000 | 0.0303 | 0.0000 |
| PG | 572 | 572 | 0.0420 | 0.0000 | 0.0420 | 0.0000 |
| PI | 547 | 547 | 0.0311 | 0.0000 | 0.0347 | 0.0000 |
| LPC | 506 | 506 | 0.0336 | 0.0000 | 0.0415 | 0.0000 |
| PS | 470 | 470 | 0.0319 | 0.0000 | 0.0191 | 0.0000 |
| DG | 465 | 465 | 0.0258 | 0.0000 | 0.0194 | 0.0000 |
| PE-O | 420 | 419 | 0.0286 | 0.0000 | 0.0239 | 0.0000 |
| PE-P | 396 | 397 | 0.0253 | 0.0025 | 0.0453 | 0.0050 |
| LPE | 360 | 361 | 0.0417 | 0.0000 | 0.0083 | 0.0000 |
| PC-O | 243 | 244 | 0.0123 | 0.0000 | 0.0000 | 0.0000 |
| HexCer | 146 | 146 | 0.0137 | 0.0000 | 0.0068 | 0.0000 |
| GalCer | 126 | 125 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CL | 91 | 92 | 0.0000 | 0.0000 | 0.0109 | 0.0000 |
| SE | 85 | 85 | 0.0118 | 0.0000 | 0.0000 | 0.0000 |
| LPI | 75 | 74 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Chain Descriptor MAE

| Chain | Target | Val MAE | Test MAE |
|-------|--------|---------|---------|
| Chain 1 | nc | 16.3733 | 16.3857 |
| Chain 1 | ndb | 2.1001 | 2.0833 |
| Chain 1 | nox | 0.3365 | 0.3418 |
| Chain 2 | nc | 15.2604 | 15.2544 |
| Chain 2 | ndb | 0.7960 | 0.7907 |
| Chain 2 | nox | 0.1076 | 0.1154 |
| Chain 3 | nc | 16.0855 | 16.0488 |
| Chain 3 | ndb | 0.5453 | 0.5502 |
| Chain 3 | nox | 0.0005 | 0.0000 |

*Report generated: 2026-04-05 01:05:07*
