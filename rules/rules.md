# Adduct Rules: Precursor m/z ↔ Exact Mass

## Mass Relationship

```
precursor_mz = (n × exact_mass + offset) / |charge|
exact_mass   = (precursor_mz × |charge| − offset) / n
```

| Symbol | Meaning |
|--------|---------|
| `n` | Molecule multiplier — 1 for `[M…]`, 2 for `[2M…]` |
| `offset` | Net mass (Da) contributed by adduct species (positive = mass added) |
| `\|charge\|` | Absolute charge magnitude (1 for singly charged, 2 for doubly charged) |
| `exact_mass` | Monoisotopic neutral mass of the molecule (Da) |
| `precursor_mz` | Observed m/z value in the mass spectrum |

---

## Adduct Rules Table

| Adduct | n | \|z\| | Offset (Da) | Ion Mode | Inverse: exact_mass = … |
|--------|---|-------|-------------|----------|--------------------------|
| `[M+H]+` | 1 | 1 | +1.007276 | Positive | `precursor_mz − 1.007276` |
| `[M+Na]+` | 1 | 1 | +22.989218 | Positive | `precursor_mz − 22.989218` |
| `[M+K]+` | 1 | 1 | +38.963158 | Positive | `precursor_mz − 38.963158` |
| `[M+NH4]+` | 1 | 1 | +18.034374 | Positive | `precursor_mz − 18.034374` |
| `[M+Na-H]+` | 1 | 1 | +21.981942 | Positive | `precursor_mz − 21.981942` |
| `[M-H+2Na]+` | 1 | 1 | +44.971160 | Positive | `precursor_mz − 44.971160` |
| `[M+2Na]+` | 1 | 1 | +44.971160 | Positive | `precursor_mz − 44.971160` |
| `[M+CH3CN+H]+` | 1 | 1 | +42.060040 | Positive | `precursor_mz − 42.060040` |
| `[M-H2O+H]+` | 1 | 1 | −17.003289 | Positive | `precursor_mz + 17.003289` |
| `[M+H-2H2O]+` | 1 | 1 | −35.013854 | Positive | `precursor_mz + 35.013854` |
| `[M-3H2O+H]+` | 1 | 1 | −53.024419 | Positive | `precursor_mz + 53.024419` |
| `[M+2H]2+` | 1 | 2 | +2.014552 | Positive | `precursor_mz × 2 − 2.014552` |
| `[M]+` | 1 | 1 | 0.000000 | Positive | `precursor_mz` |
| `[M-H]-` | 1 | 1 | −1.007276 | Negative | `precursor_mz + 1.007276` |
| `[M+Cl]-` | 1 | 1 | +34.968853 | Negative | `precursor_mz − 34.968853` |
| `[M+Cl-H]-` | 1 | 1 | +33.961577 | Negative | `precursor_mz − 33.961577` |
| `[M+CH3COO]-` | 1 | 1 | +59.013304 | Negative | `precursor_mz − 59.013304` |
| `[M+HCOO]-` | 1 | 1 | +44.997655 | Negative | `precursor_mz − 44.997655` |
| `[M-H2O-H]-` | 1 | 1 | −19.017841 | Negative | `precursor_mz + 19.017841` |
| `[M-CH3]-` | 1 | 1 | −15.023475 | Negative | `precursor_mz + 15.023475` |
| `[M-C3H7O2]-` | 1 | 1 | −75.044605 | Negative | `precursor_mz + 75.044605` |
| `[M-2H]2-` | 1 | 2 | −2.014552 | Negative | `precursor_mz × 2 + 2.014552` |
| `[2M+H]+` | 2 | 1 | +1.007276 | Positive | `(precursor_mz − 1.007276) / 2` |
| `[2M+Na]+` | 2 | 1 | +22.989218 | Positive | `(precursor_mz − 22.989218) / 2` |
| `[2M+K]+` | 2 | 1 | +38.963158 | Positive | `(precursor_mz − 38.963158) / 2` |
| `[2M+NH4]+` | 2 | 1 | +18.034374 | Positive | `(precursor_mz − 18.034374) / 2` |
| `[2M-H]-` | 2 | 1 | −1.007276 | Negative | `(precursor_mz + 1.007276) / 2` |
| `[2M-2H+Na]-` | 2 | 1 | +20.974666 | Negative | `(precursor_mz − 20.974666) / 2` |

### Offset Derivation for `[2M-2H+Na]-`
```
offset = −2 × H_proton + Na_atomic
       = −2 × 1.007276 + 22.989218
       = +20.974666 Da
```

---

## Complex Neutral Loss Adducts

For adducts of the form `[M − X + H]+` (loss of neutral fragment + proton addition):

```
offset = −mass(X)_monoisotopic + H_proton
```

where `mass(X)` uses monoisotopic atomic masses (H = 1.007825 Da, not proton mass).

| Adduct | Neutral Loss X | mass(X) Da | Offset Da |
|--------|---------------|-----------|-----------|
| `[M-CH4O+H]+` | methanol CH₄O | 32.026215 | −31.018939 |
| `[M-C2H6O+H]+` | ethanol C₂H₆O | 46.041865 | −45.034589 |
| `[M-C2H7NO+H]+` | ethanolamine C₂H₇NO | 61.052764 | −60.045488 |
| `[M-C3H8O3+H]+` | glycerol C₃H₈O₃ | 92.047235 | −91.039959 |
| `[M-C3H8NO6P+H]+` | glycerophosphate-like | 183.066044 | −182.058768 |
| `[M-C2H8NO4P+H]+` | phosphoethanolamine | 141.019095 | −140.011819 |
| `[M-C3H10O4+H]+` | 1,3-propanediol phosphate | 110.057910 | −109.050634 |
| `[M-C16H32O2+H]+` | palmitic acid C₁₆H₃₂O₂ | 256.240230 | −255.232954 |
| `[M-C18H34O2+H]+` | oleic acid C₁₈H₃₄O₂ | 282.255880 | −281.248604 |
| `[M-C5H6O3+H]+` | pentadiendioic acid | 114.031694 | −113.024418 |

For any novel `[M − Formula + H]+` adduct, compute:

```python
offset = -formula_to_mass(formula) + 1.007276
```

using monoisotopic atomic masses: H=1.007825, C=12.000000, N=14.003074, O=15.994915, P=30.973762, S=31.972071.

---

## Notes on H in Adduct Context

| Context | Mass value (Da) | Rationale |
|---------|----------------|-----------|
| Lone `+H` / `−H` in adduct | **1.007276** | Proton (H⁺), no electron |
| H in molecular formula (H₂O, C₃H₈O₃, …) | **1.007825** | Hydrogen atom (H), includes electron |
| Difference | 0.000549 | Electron mass — negligible for most lipids but correct for high-accuracy work |

---

## Adduct Normalization Map

Non-standard adduct strings encountered in source databases are normalized to canonical form:

| Raw string | Normalized adduct |
|-----------|------------------|
| `[M+Hac-H]-` | `[M+CH3COO]-` |
| `[M.Cl]-` | `[M+Cl]-` |
| `[M+ACN+H]+` | `[M+CH3CN+H]+` |
| `[M-2H2O+H]+` | `[M+H-2H2O]+` |
| `[M+2Na-H]+` | `[M-H+2Na]+` |
| `[M-H+CH3COOH]-` | `[M+CH3COO]-` |
| `[M+CH3COOH-H]-` | `[M+CH3COO]-` |
| `[M+CHOO]-` | `[M+HCOO]-` |
| `[M+FA]-` | `[M+HCOO]-` |
| `[M+CH3COO]-/[M-CH3]-` | `[M+CH3COO]-` (first adduct taken) |
| `[M+H]` (no trailing ±) | `[M+H]+` |
| `[M-H]` (no trailing ±) | `[M-H]-` |

---

## Ion Mode Determination

Ion mode is derived directly from the trailing character of the canonical adduct string:

| Condition | Ion mode |
|-----------|----------|
| Adduct ends with `+` | Positive |
| Adduct ends with `-` | Negative |
| No adduct available | Preserved from source (or null) |

Examples: `[M+H]+` → Positive; `[M+2H]2+` → Positive (ends with `+`); `[M-H]-` → Negative; `[2M-2H+Na]-` → Negative.
