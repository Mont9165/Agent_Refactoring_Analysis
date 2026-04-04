#!/usr/bin/env python3
"""Compute inter-rater reliability for project filtering validation.

Calculates Cohen's Kappa and classification metrics for:
  1. Annotator1 vs Annotator2 (inter-rater agreement)
  2. Final Human Label vs GPT Label (LLM accuracy)

Expected input CSV columns:
  - annotator1_label, annotator2_label
  - final_human_label, gpt_label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from calculate_cohen_kappa import (
    _classification_metrics_from_series,
    _cohen_kappa_series,
    _confusion_matrix,
    _save_summary,
    _slugify,
)

DEFAULT_INPUT = Path("data/analysis/project_filtering/project_filtering_validation_full.csv")
DEFAULT_OUTPUT_DIR = Path("data/analysis/project_filtering/kappa_results")


def _filter_labels(
    left: pd.Series,
    right: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Drop rows where either label is missing, empty, or 'none'."""
    left_s = left.astype(str).replace({"nan": pd.NA, "": pd.NA, "none": pd.NA})
    right_s = right.astype(str).replace({"nan": pd.NA, "": pd.NA, "none": pd.NA})
    mask = left_s.notna() & right_s.notna()
    return left_s[mask], right_s[mask]


def _print_results(
    description: str,
    left_col: str,
    right_col: str,
    left_series: pd.Series,
    right_series: pd.Series,
) -> Tuple[List[Dict[str, object]], Optional[pd.DataFrame], List[Dict]]:
    """Print and return metrics for a pair of label columns."""
    observed, expected, kappa = _cohen_kappa_series(left_series, right_series)
    metrics, confusion, per_class = _classification_metrics_from_series(left_series, right_series)
    sample_count = int(metrics["sample_count"])

    print(f"\n{description}:")
    print(f"  Columns: '{left_col}' (reference) vs '{right_col}' (comparison)")
    print(f"  Samples: {sample_count}")
    print(f"  Observed agreement: {observed:.4f}")
    print(f"  Expected agreement: {expected:.4f}")
    print(f"  Cohen's kappa: {kappa:.4f}")

    if sample_count > 0 and not confusion.empty:
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print("  Confusion matrix:")
        print(confusion.to_string())
        if per_class:
            per_class_df = pd.DataFrame(per_class)
            print("  Per-class metrics:")
            print(per_class_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary_rows = [
        {"pair": description, "metric": "samples", "value": sample_count},
        {"pair": description, "metric": "observed_agreement", "value": observed},
        {"pair": description, "metric": "expected_agreement", "value": expected},
        {"pair": description, "metric": "kappa", "value": kappa},
        {"pair": description, "metric": "accuracy", "value": metrics["accuracy"]},
        {"pair": description, "metric": "macro_precision", "value": metrics["macro_precision"]},
        {"pair": description, "metric": "macro_recall", "value": metrics["macro_recall"]},
        {"pair": description, "metric": "macro_f1", "value": metrics["macro_f1"]},
    ]

    return summary_rows, confusion, per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to validation CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, dtype=str)

    # Migrate legacy 4-category labels to binary scheme
    _LEGACY_LABEL_MAP = {
        "production_grade": "non_toy",
        "specialized_project": "non_toy",
        "toy_or_example": "toy_or_uncertain",
        "uncertain": "toy_or_uncertain",
    }
    _label_cols = ["gpt_label", "annotator1_label", "annotator2_label", "final_human_label"]
    for col in _label_cols:
        if col in df.columns:
            migrated = df[col].map(_LEGACY_LABEL_MAP)
            changed = migrated.notna() & (migrated != df[col])
            if changed.any():
                df[col] = migrated.fillna(df[col])
                print(f"Migrated {changed.sum()} legacy labels in '{col}' to binary scheme (non_toy/toy_or_uncertain).")

    pairs = [
        ("annotator1_label", "annotator2_label", "human_vs_human",
         "Inter-rater reliability (Annotator1 vs Annotator2)"),
        ("final_human_label", "gpt_label", "human_vs_gpt",
         "Human consensus vs GPT classification"),
    ]

    all_summary_rows: List[Dict[str, object]] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for left_col, right_col, slug, description in pairs:
        if left_col not in df.columns or right_col not in df.columns:
            print(f"Skipping '{description}': columns '{left_col}' or '{right_col}' not found")
            continue

        left_series, right_series = _filter_labels(df[left_col], df[right_col])
        if left_series.empty:
            print(f"Skipping '{description}': no valid label pairs found (columns may be empty)")
            continue

        summary_rows, confusion, per_class = _print_results(
            description, left_col, right_col, left_series, right_series,
        )
        all_summary_rows.extend(summary_rows)

        if confusion is not None and not confusion.empty:
            confusion.to_csv(output_dir / f"{slug}_confusion.csv")
            if per_class:
                pd.DataFrame(per_class).to_csv(output_dir / f"{slug}_per_class.csv", index=False)

    if all_summary_rows:
        _save_summary(all_summary_rows, output_dir / "summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
