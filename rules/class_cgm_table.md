# Class–CGM Reference Table

> **Dataset:** `lipid_ms2_cleaned.parquet` (119,108 rows, 78 classes)
> **Updated:** 2026-04-01 — chain-inference applied; names standardised to suffix notation
> **Cleaning:** 6 rules applied (NaN exact_mass dropped; prefix-d/t → suffix ;nO;
> PC 1-chain → LPC; _0:0 positional entries removed; FA artefacts removed)
> **Chain inference:** SM (380/381 → 2-chain); PC-O (50/50 → 2-chain);
> Cer 1-chain dropped (101 total: 100 Cer + 1 SM — exact_mass encodes LCB mass only)

## CGM Formula

```
CGM = exact_mass − Σ acyl_mass_i
acyl_mass(nc, ndb, nox) = nc·12 + (2nc − 2ndb)·H + (2 + nox)·O − H₂O
H = 1.00782503207  O = 15.99491461957  H₂O = 18.01056469
```

## Table

| Class | num_chain | CGM (Da) | N entries | Example |
|-------|-----------|----------|-----------|---------|
| ADGGA | 3 | 268.0732 | 9 | `ADGGA 16:1_16:1_16:0` |
| ADGGA-O | 3 | 268.0780 | 7 | `ADGGA-O 18:2_16:0_16:0` |
| AHexCer-O | 3 | 165.0964 | 25 | `AHexCer-O 22:0;O_18:1;2O_16:0` |
| BMP | 2 | 246.0495 | 177 | `BMP 22:6_22:6` |
| CAR | 1 | 162.1129 | 1,140 | `CAR 16:0` |
| CE | 1 | 386.3539 | 165 | `CE 18:2` |
| CL | 4 | 400.0528 | 646 | `CL 18:1_18:1_18:1_18:1` |
| Cer | 2 | 3.0464 | 6,983 | `Cer 18:1;2O_16:0` |
| CerP | 2 | 83.0154 | 1 | `CerP 18:1;2O_12:0` |
| CoQ | 1 | 168.0791 | 287 | `CoQ 50:10` |
| DG | 2 | 92.0468 | 3,100 | `DG 18:2_18:1` |
| DG-O | 2 | 78.0666 | 91 | `DG-O 22:6_16:1` |
| DGDG | 2 | 416.1528 | 355 | `DGDG 18:1_16:0` |
| DGDG-O | 2 | 402.1703 | 4 | `DGDG-O 16:0_16:0` |
| DGGA | 2 | 268.0786 | 19 | `DGGA 18:2_16:0` |
| DGTSA | 2 | 235.1425 | 126 | `DGTSA 18:1_18:1` |
| DLCL | 2 | 400.0506 | 24 | `DLCL 16:0_16:0` |
| FA | 1 | 18.0105 | 26,186 | `FA 18:2` |
| FAHFA | 2 | 34.0018 | 160 | `FAHFA 18:1_16:0` |
| GM3 | 2 | 618.2468 | 20 | `GM3 18:1;2O_16:0` |
| GalCer | 2 | 165.1002 | 838 | `GalCer 18:1;2O_16:0` |
| GlcCer | 2 | 165.0998 | 6 | `GlcCer 18:1;2O_14:0` |
| HBMP | 3 | 246.0458 | 19 | `HBMP 16:0_16:0_16:0` |
| Hex2Cer | 2 | 327.1520 | 45 | `Hex2Cer 24:0_18:1;2O` |
| Hex3Cer | 2 | 489.2006 | 2 | `Hex3Cer 18:1;2O_16:0` |
| HexCer | 2 | 165.0990 | 1,000 | `HexCer 18:1;2O_16:0` |
| LNAPE | 2 | 215.0548 | 56 | `LNAPE 22:6_18:2` |
| LNAPS | 2 | 259.0424 | 11 | `LNAPS 18:0_16:0` |
| LPA | 1 | 172.0110 | 16 | `LPA 16:0` |
| LPC | 1 | 257.1030 | 3,447 | `LPC 16:0` |
| LPC-O | 1 | 243.1229 | 128 | `LPC-O 16:0` |
| LPC-P | 1 | 241.1084 | 2 | `LPC-P 18:1` |
| LPE | 1 | 215.0554 | 2,402 | `LPE 18:1` |
| LPE-O | 1 | 201.0754 | 301 | `LPE-O 18:1` |
| LPE-P | 1 | 199.0603 | 2 | `LPE-P 18:0` |
| LPG | 1 | 246.0489 | 115 | `LPG 18:1` |
| LPG-O | 1 | 232.0694 | 20 | `LPG-O 18:2` |
| LPI | 1 | 334.0654 | 495 | `LPI 18:0` |
| LPS | 1 | 259.0444 | 255 | `LPS 22:6` |
| LacCer | 2 | 329.1694 | 66 | `LacCer 24:1;2O_18:1` |
| MG | 1 | 92.0466 | 28 | `MG 26:4` |
| MG-O | 3 | 78.0676 | 11 | `MG-O 18:0_0:0_0:0` |
| MGDG | 2 | 254.0986 | 162 | `MGDG 16:0_16:0` |
| MGDG-O | 2 | 240.1170 | 123 | `MGDG-O 16:0_16:0` |
| MLCL | 3 | 400.0528 | 2 | `MLCL 16:0_16:0_16:0` |
| NAE | 1 | 61.0528 | 432 | `NAE 12:0` |
| NAGly | 2 | 91.0266 | 11 | `NAGly 17:0_15:0` |
| NAGlySer | 2 | 178.0563 | 8 | `NAGlySer 17:0_15:0` |
| NAOrn | 2 | 148.0848 | 3 | `NAOrn 17:0_15:0` |
| PA | 2 | 172.0128 | 301 | `PA 18:1_16:0` |
| PC | 2 | 257.1030 | 21,367 | `PC 14:0_14:0` |
| PC-O | 2 | 243.1239 | 1,742 *(+50 inferred)* | `PC-O 18:1_16:0` |
| PC-P | 2 | 241.1073 | 409 | `PC-P 16:0_16:0` |
| PE | 2 | 215.0551 | 8,911 | `PE 18:1_16:0` |
| PE-Cer | 2 | 126.0565 | 9 | `PE-Cer 20:0;O_14:0;2O` |
| PE-O | 2 | 201.0746 | 2,795 | `PE-O 22:5_18:1` |
| PE-P | 2 | 199.0609 | 2,702 | `PE-P 20:4_16:0` |
| PEtOH | 2 | 200.0442 | 159 | `PEtOH 18:2_16:0` |
| PG | 2 | 246.0496 | 4,073 | `PG 18:1_16:0` |
| PG-O | 2 | 232.0675 | 46 | `PG-O 18:1_16:0` |
| PG-P | 2 | 230.0546 | 6 | `PG-P 16:0_15:0` |
| PI | 2 | 334.0657 | 3,784 | `PI 20:4_18:0` |
| PI-Cer | 2 | 261.0606 | 129 | `PI-Cer 20:0;2O_13:0;O` |
| PI-O | 2 | 320.0842 | 84 | `PI-O 20:4_16:0` |
| PMeOH | 2 | 186.0271 | 42 | `PMeOH 18:2_16:0` |
| PS | 2 | 259.0454 | 3,299 | `PS 22:6_18:0` |
| PS-O | 2 | 245.0638 | 30 | `PS-O 22:6_18:1` |
| SE | 2 | -4.0331 | 565 | `SE 27:1_18:1` |
| SHexCer | 2 | 245.0568 | 22 | `SHexCer 22:0;O_18:1;2O` |
| SL | 2 | 83.0025 | 37 | `SL 17:0;O_15:0` |
| SM | 2 | 168.1030 | 4,552 *(+380 inferred)* | `SM 18:1;2O_16:0` |
| SPB | 1 | 3.0476 | 39 | `SPB 18:1;2O` |
| SQDG | 2 | 318.0620 | 230 | `SQDG 18:3_16:0` |
| ST | 2 | -20.0287 | 4 | `ST 24:1;4O_18:1` |
| TG | 3 | 92.0470 | 13,280 | `TG 16:0_16:0_16:0` |
| TG-O | 3 | 78.0657 | 395 | `TG-O 18:1_16:0_16:0` |
| VAE | 1 | 286.2284 | 6 | `VAE 16:0` |
| WE | 2 | 4.0326 | 559 | `WE 16:1_1:0` |

## Notes

### Name Standardisation

All names are rebuilt from `class` + chain columns using the format:
```
{class} {nc1}:{ndb1}{ox1}_{nc2}:{ndb2}{ox2}...
```
where the oxidation suffix is:

| nox | Suffix | Example |
|-----|--------|---------|
| 0 | *(none)* | `18:0` |
| 1 | `;O` | `18:0;O` |
| 2 | `;2O` | `18:1;2O` |
| 3 | `;3O` | `18:0;3O` |
| 4 | `;4O` | `20:3;4O` |

Chain order: longest (most C) first; tie → most double bonds; tie → most oxidation.

### Chain-Inference Details

**SM (380/381 inferred):**
- `acyl_mass_chain2 = exact_mass − 168.1027 − acyl_mass(chain1)`
- Iterate ndb2 ∈ {0…13}; accept if nc2 rounds to integer within ±0.015
- 1 entry (SM 24:1;2O) had no integer solution → dropped

**PC-O (50/50 inferred):**
- `acyl_mass_chain2 = exact_mass − 243.1227 − acyl_mass(chain1)`
- All 50 solved; sn-1 ether chain longer in all cases → no chain reordering

**Cer (100 entries — dropped):**
- `exact_mass = acyl_mass(LCB) + 3.048` — only LCB backbone mass available,
  not the full ceramide mass. Inference impossible; rows removed.

### Key CGM Constants

| Class | CGM (Da) | Structural interpretation |
|-------|----------|--------------------------|
| MG / DG / TG | 92.047 | Glycerol backbone |
| PC / LPC | 257.103 | Glycerophosphocholine head |
| PE / LPE | 215.055 | Glycerophosphoethanolamine head |
| Cer | 3.047 | Amide linker (5H + N − O) |
| SM | 168.103 | Phosphocholine + amide linker |
| HexCer / GlcCer / GalCer | 165.100 | Hexose + amide linker |
| PC-O | 243.122 | GPC + ether linkage correction |
| SE | −4.033 | Sterol ester (non-linear formula) |

### Column Layout

| Group | Columns |
|-------|---------|
| Identity | `name`, `class`, `num_chain` |
| Chain 1 | `num_c_1`, `num_db_1`, `num_ox_1` |
| Chain 2 | `num_c_2`, `num_db_2`, `num_ox_2` |
| Chain 3 | `num_c_3`, `num_db_3`, `num_ox_3` |
| Chain 4 | `num_c_4`, `num_db_4`, `num_ox_4` |
| Mass | `exact_mass` |
| MS measurement | `adduct`, `precursor_mz`, `ion_mode` |
| Quality | `chain2_inferred` |
| Spectrum | `MS2` |
| Provenance | `source`, `source_id`, `instrument` |
| Structure | `SMILES`, `InChIKey`, `InChIKey_main`, `Formula` |
| Archive | `raw_name` |