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
| Adduct accuracy | 0.0018 | 0.0021 |
| Level 0 — Class | 0.7525 | 0.7559 |
| Level 1 — Sum composition | 0.0 | 0.0001 |
| Level 2 — Full chain | 0.0 | 0.0 |
| Level 3 — Exact name | 0.0 | 0.0 |

## Sum Composition Status

| Status | Val | Test |
|--------|-----|------|
| matched | 2,010 | 1,982 |
| multi | 1,632 | 1,558 |
| no_match | 12,279 | 12,378 |

## Per-Class Breakdown (top 20)

| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |
|-------|------|-------|--------|--------|---------|---------|
| FA | 3143 | 3142 | 0.9583 | 0.0000 | 0.9586 | 0.0000 |
| PC | 2839 | 2839 | 0.8848 | 0.0000 | 0.8905 | 0.0000 |
| TG | 1968 | 1969 | 0.9634 | 0.0000 | 0.9680 | 0.0000 |
| PE | 1244 | 1244 | 0.2661 | 0.0000 | 0.2757 | 0.0000 |
| Cer | 788 | 787 | 0.9099 | 0.0000 | 0.9060 | 0.0000 |
| SM | 593 | 594 | 0.9444 | 0.0000 | 0.9411 | 0.0000 |
| PG | 572 | 572 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PI | 547 | 547 | 0.0000 | 0.0000 | 0.0018 | 0.0000 |
| LPC | 506 | 506 | 0.9763 | 0.0000 | 0.9862 | 0.0000 |
| PS | 470 | 470 | 0.3787 | 0.0000 | 0.3596 | 0.0000 |
| DG | 465 | 465 | 0.9333 | 0.0000 | 0.9419 | 0.0000 |
| PE-O | 420 | 419 | 0.3762 | 0.0000 | 0.3962 | 0.0000 |
| PE-P | 396 | 397 | 0.4874 | 0.0000 | 0.5416 | 0.0000 |
| LPE | 360 | 361 | 0.9472 | 0.0000 | 0.9612 | 0.0000 |
| PC-O | 243 | 244 | 0.8272 | 0.0000 | 0.7992 | 0.0000 |
| HexCer | 146 | 146 | 0.7877 | 0.0000 | 0.7466 | 0.0000 |
| GalCer | 126 | 125 | 0.4444 | 0.0000 | 0.4720 | 0.0000 |
| CL | 91 | 92 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| SE | 85 | 85 | 0.8353 | 0.0000 | 0.8471 | 0.0000 |
| LPI | 75 | 74 | 0.3067 | 0.0000 | 0.2973 | 0.0000 |

## Chain Descriptor MAE

| Chain | Target | Val MAE | Test MAE |
|-------|--------|---------|---------|
| Chain 1 | nc | 15.1794 | 15.2636 |
| Chain 1 | ndb | 1.9872 | 1.9737 |
| Chain 1 | nox | 0.5064 | 0.5017 |
| Chain 2 | nc | 14.2819 | 14.3498 |
| Chain 2 | ndb | 0.7284 | 0.7228 |
| Chain 2 | nox | 0.1324 | 0.1332 |
| Chain 3 | nc | 15.9709 | 15.9099 |
| Chain 3 | ndb | 0.4082 | 0.4052 |
| Chain 3 | nox | 0.0005 | 0.0000 |

*Report generated: 2026-04-05 01:04:48*
