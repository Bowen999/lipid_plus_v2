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
| Adduct accuracy | 0.9796 | 0.982 |
| Level 0 — Class | 0.9618 | 0.9643 |
| Level 1 — Sum composition | 0.9086 | 0.9126 |
| Level 2 — Full chain | 0.804 | 0.8047 |
| Level 3 — Exact name | 0.804 | 0.8046 |

## Sum Composition Status

| Status | Val | Test |
|--------|-----|------|
| matched | 9,716 | 9,804 |
| multi | 5,769 | 5,705 |
| no_match | 436 | 409 |

## Per-Class Breakdown (top 20)

| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |
|-------|------|-------|--------|--------|---------|---------|
| FA | 3143 | 3142 | 0.9704 | 0.9052 | 0.9761 | 0.9204 |
| PC | 2839 | 2839 | 0.9687 | 0.9510 | 0.9743 | 0.9574 |
| TG | 1968 | 1969 | 0.9842 | 0.8664 | 0.9843 | 0.8654 |
| PE | 1244 | 1244 | 0.9590 | 0.9268 | 0.9646 | 0.9317 |
| Cer | 788 | 787 | 0.9657 | 0.9264 | 0.9682 | 0.9187 |
| SM | 593 | 594 | 0.9798 | 0.9663 | 0.9747 | 0.9646 |
| PG | 572 | 572 | 0.9441 | 0.9126 | 0.9458 | 0.9231 |
| PI | 547 | 547 | 0.9397 | 0.9104 | 0.9452 | 0.9177 |
| LPC | 506 | 506 | 0.9960 | 0.9921 | 0.9921 | 0.9822 |
| PS | 470 | 470 | 0.9723 | 0.9532 | 0.9681 | 0.9319 |
| DG | 465 | 465 | 0.9677 | 0.9441 | 0.9591 | 0.9226 |
| PE-O | 420 | 419 | 0.9357 | 0.8548 | 0.9212 | 0.8329 |
| PE-P | 396 | 397 | 0.9747 | 0.9646 | 0.9798 | 0.9798 |
| LPE | 360 | 361 | 0.9944 | 0.9806 | 0.9945 | 0.9861 |
| PC-O | 243 | 244 | 0.9012 | 0.8066 | 0.9262 | 0.8607 |
| HexCer | 146 | 146 | 0.8836 | 0.7397 | 0.8562 | 0.7534 |
| GalCer | 126 | 125 | 0.9127 | 0.8889 | 0.8560 | 0.8480 |
| CL | 91 | 92 | 0.9670 | 0.7692 | 1.0000 | 0.7717 |
| SE | 85 | 85 | 0.9176 | 0.8235 | 0.8824 | 0.8471 |
| LPI | 75 | 74 | 0.9867 | 0.9600 | 0.9730 | 0.9730 |

## Chain Descriptor MAE

| Chain | Target | Val MAE | Test MAE |
|-------|--------|---------|---------|
| Chain 1 | nc | 0.7280 | 0.6782 |
| Chain 1 | ndb | 0.2179 | 0.2065 |
| Chain 1 | nox | 0.1024 | 0.1033 |
| Chain 2 | nc | 0.5903 | 0.6005 |
| Chain 2 | ndb | 0.1801 | 0.1733 |
| Chain 2 | nox | 0.0457 | 0.0509 |
| Chain 3 | nc | 0.4476 | 0.4549 |
| Chain 3 | ndb | 0.1264 | 0.1371 |
| Chain 3 | nox | 0.0023 | 0.0009 |

*Report generated: 2026-04-05 01:03:43*
