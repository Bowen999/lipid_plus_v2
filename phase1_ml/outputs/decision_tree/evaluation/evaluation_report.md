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
| Adduct accuracy | 0.0036 | 0.0035 |
| Level 0 — Class | 0.5676 | 0.5731 |
| Level 1 — Sum composition | 0.0003 | 0.0001 |
| Level 2 — Full chain | 0.0 | 0.0 |
| Level 3 — Exact name | 0.0 | 0.0 |

## Sum Composition Status

| Status | Val | Test |
|--------|-----|------|
| matched | 2,082 | 2,037 |
| multi | 1,843 | 1,822 |
| no_match | 11,996 | 12,059 |

## Per-Class Breakdown (top 20)

| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |
|-------|------|-------|--------|--------|---------|---------|
| FA | 3143 | 3142 | 0.7703 | 0.0000 | 0.7794 | 0.0000 |
| PC | 2839 | 2839 | 0.8031 | 0.0000 | 0.7880 | 0.0000 |
| TG | 1968 | 1969 | 0.7358 | 0.0000 | 0.7486 | 0.0000 |
| PE | 1244 | 1244 | 0.2355 | 0.0000 | 0.2395 | 0.0000 |
| Cer | 788 | 787 | 0.4112 | 0.0000 | 0.3875 | 0.0000 |
| SM | 593 | 594 | 0.8820 | 0.0000 | 0.8923 | 0.0000 |
| PG | 572 | 572 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PI | 547 | 547 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LPC | 506 | 506 | 0.1739 | 0.0000 | 0.1877 | 0.0000 |
| PS | 470 | 470 | 0.2851 | 0.0000 | 0.3213 | 0.0000 |
| DG | 465 | 465 | 0.6645 | 0.0000 | 0.6495 | 0.0000 |
| PE-O | 420 | 419 | 0.3571 | 0.0000 | 0.3795 | 0.0000 |
| PE-P | 396 | 397 | 0.3510 | 0.0000 | 0.3955 | 0.0000 |
| LPE | 360 | 361 | 0.4306 | 0.0000 | 0.4820 | 0.0000 |
| PC-O | 243 | 244 | 0.8025 | 0.0000 | 0.8320 | 0.0000 |
| HexCer | 146 | 146 | 0.6644 | 0.0000 | 0.6301 | 0.0000 |
| GalCer | 126 | 125 | 0.2143 | 0.0000 | 0.2640 | 0.0000 |
| CL | 91 | 92 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SE | 85 | 85 | 0.6118 | 0.0000 | 0.6471 | 0.0000 |
| LPI | 75 | 74 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Chain Descriptor MAE

| Chain | Target | Val MAE | Test MAE |
|-------|--------|---------|---------|
| Chain 1 | nc | 15.1214 | 15.1692 |
| Chain 1 | ndb | 2.0205 | 2.0023 |
| Chain 1 | nox | 0.5331 | 0.5288 |
| Chain 2 | nc | 14.1386 | 14.2216 |
| Chain 2 | ndb | 0.7904 | 0.7920 |
| Chain 2 | nox | 0.1447 | 0.1509 |
| Chain 3 | nc | 16.0009 | 15.9573 |
| Chain 3 | ndb | 0.5420 | 0.5338 |
| Chain 3 | nox | 0.0005 | 0.0000 |

*Report generated: 2026-04-05 01:04:58*
