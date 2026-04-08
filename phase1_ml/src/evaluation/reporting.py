"""
reporting.py — Cross-model comparison reporting.

Generates a markdown comparison table across all trained model families.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


METRIC_LABELS: list[tuple[str, str]] = [
    ("Adduct accuracy",       "adduct_accuracy"),
    ("L0 class accuracy",     "level0_class_accuracy"),
    ("L1 sum composition",    "level1_sum_composition_accuracy"),
    ("L2 full chain",         "level2_full_chain_accuracy"),
    ("L3 exact name",         "level3_name_exact_match"),
]


def generate_comparison_report(
    all_metrics: dict[str, dict[str, dict]],
    out_path: Path,
) -> None:
    """
    Write a markdown comparison report across all model families.

    Parameters
    ----------
    all_metrics : {model_name: {"val": metrics_dict, "test": metrics_dict}}
    out_path    : destination .md file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = sorted(all_metrics.keys())
    lines  = [
        "# Phase 1 Multi-Model Comparison",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Validation Set Metrics",
        "",
    ]

    # Validation table
    header = "| Metric |" + "".join(f" {m} |" for m in models)
    sep    = "|--------|" + "".join("--------|" for _ in models)
    lines += [header, sep]
    for label, key in METRIC_LABELS:
        row = f"| {label} |"
        for m in models:
            v = all_metrics[m].get("val", {}).get(key)
            row += f" {v:.4f} |" if v is not None else " N/A |"
        lines.append(row)

    lines += ["", "## Test Set Metrics", ""]
    lines += [header, sep]
    for label, key in METRIC_LABELS:
        row = f"| {label} |"
        for m in models:
            v = all_metrics[m].get("test", {}).get(key)
            row += f" {v:.4f} |" if v is not None else " N/A |"
        lines.append(row)

    # Per-level winner
    lines += ["", "## Best Model Per Level (Val)", ""]
    lines += ["| Level | Best Model | Val Score |", "|-------|-----------|-----------|"]
    for label, key in METRIC_LABELS:
        best_model = ""
        best_score = -1.0
        for m in models:
            v = all_metrics[m].get("val", {}).get(key, 0.0)
            if v is not None and v > best_score:
                best_score = v
                best_model = m
        lines.append(f"| {label} | {best_model} | {best_score:.4f} |")

    # Sum comp status breakdown per model
    lines += ["", "## Sum Composition Status (Val)", ""]
    all_statuses: set[str] = set()
    for m in models:
        counts = all_metrics[m].get("val", {}).get("sum_comp_status_counts", {})
        all_statuses.update(counts.keys())
    status_list = sorted(all_statuses)
    if status_list:
        hdr = "| Status |" + "".join(f" {m} |" for m in models)
        sp  = "|--------|" + "".join("--------|" for _ in models)
        lines += [hdr, sp]
        for s in status_list:
            row = f"| {s} |"
            for m in models:
                cnt = all_metrics[m].get("val", {}).get("sum_comp_status_counts", {}).get(s, 0)
                row += f" {cnt:,} |"
            lines.append(row)

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Comparison report saved → {out_path}")
