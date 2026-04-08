"""
selection.py — Per-target model selection and cascade comparison.

Compares all trained model families on the validation set and identifies
the best performing model for each pipeline level.
"""
from __future__ import annotations

from datetime import datetime


# ── Metric keys used for ranking ──────────────────────────────────────────────
LEVEL_KEYS: list[tuple[str, str]] = [
    ("L0 (class)",            "level0_class_accuracy"),
    ("L1 (sum comp)",         "level1_sum_composition_accuracy"),
    ("L2 (full chain)",       "level2_full_chain_accuracy"),
    ("L3 (exact name)",       "level3_name_exact_match"),
    ("Adduct",                "adduct_accuracy"),
]


def find_best_combination(
    all_val_metrics: dict[str, dict],
) -> dict:
    """
    Compare model families on the validation set and return a structured
    report with per-level rankings and the best overall model.

    Parameters
    ----------
    all_val_metrics : {model_name: val_metrics_dict}
        Validation metrics for each model family.

    Returns
    -------
    dict with keys:
        "best_overall"      : model_name with highest L3 val accuracy
        "rankings"          : {level_label: [(model_name, score), ...]}  sorted desc
        "per_level_winner"  : {level_label: model_name}
        "summary_table"     : list of strings (for printing)
    """
    if not all_val_metrics:
        return {}

    rankings: dict[str, list[tuple[str, float]]] = {}
    per_level_winner: dict[str, str] = {}

    for label, key in LEVEL_KEYS:
        scores = [
            (name, float(m.get(key, 0.0)))
            for name, m in all_val_metrics.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        rankings[label]         = scores
        per_level_winner[label] = scores[0][0] if scores else ""

    # Best overall model: highest L3 val accuracy
    l3_label = "L3 (exact name)"
    best_overall = rankings[l3_label][0][0] if rankings.get(l3_label) else ""

    # Build a printable summary table
    models = list(all_val_metrics.keys())
    col_w  = max(len(m) for m in models) + 2

    header = f"{'Metric':<28}" + "".join(f"{m:>{col_w}}" for m in models)
    sep    = "─" * len(header)
    rows   = [header, sep]
    for label, key in LEVEL_KEYS:
        row = f"{label:<28}"
        for m in models:
            v = all_val_metrics[m].get(key, float("nan"))
            row += f"{v:>{col_w}.4f}"
        rows.append(row)

    rows.append(sep)
    rows.append(f"Best overall (L3): {best_overall}")

    return {
        "best_overall":     best_overall,
        "rankings":         rankings,
        "per_level_winner": per_level_winner,
        "summary_table":    rows,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
