#!/usr/bin/env python3
"""Compute inter-rater reliability and classification metrics for motivation labels."""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_INPUT = Path("data/analysis/refactoring_instances/gpt_refactoring_motivation_labeling.csv")


def _filter_labels(
    left: Iterable[str],
    right: Iterable[str],
) -> Tuple[pd.Series, pd.Series]:
    """Drop rows with missing values or logical_mismatch."""
    left_series = pd.Series(tuple(left), copy=False)
    right_series = pd.Series(tuple(right), copy=False)
    mask = (
        left_series.notna()
        & right_series.notna()
        & (left_series != "logical_mismatch")
        & (right_series != "logical_mismatch")
    )
    return left_series[mask].astype(str), right_series[mask].astype(str)


def _confusion_matrix(
    left_series: pd.Series,
    right_series: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    categories = sorted(set(left_series.unique()) | set(right_series.unique()))
    confusion = (
        pd.crosstab(left_series, right_series)
        .reindex(index=categories, columns=categories, fill_value=0)
    )
    return confusion, categories


def _cohen_kappa_series(
    left_series: pd.Series,
    right_series: pd.Series,
) -> Tuple[float, float, float]:
    """Return observed agreement, expected agreement, and kappa."""
    if left_series.empty or right_series.empty:
        return 0.0, 0.0, float("nan")

    confusion, _ = _confusion_matrix(left_series, right_series)
    total = confusion.values.sum()
    if total == 0:
        return 0.0, 0.0, float("nan")

    observed = confusion.values.diagonal().sum() / total
    row_totals = confusion.sum(axis=1).values
    col_totals = confusion.sum(axis=0).values
    expected = (row_totals * col_totals).sum() / (total ** 2)
    if math.isclose(expected, 1.0):
        kappa = float("nan")
    else:
        kappa = (observed - expected) / (1 - expected)
    return observed, expected, kappa


def _cohen_kappa(left: Iterable[str], right: Iterable[str]) -> Tuple[float, float, float]:
    filtered_left, filtered_right = _filter_labels(left, right)
    return _cohen_kappa_series(filtered_left, filtered_right)


def _classification_metrics_from_series(
    left_series: pd.Series,
    right_series: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict[str, float]]]:
    """Compute accuracy, macro precision/recall/F1, and per-class metrics."""
    metrics: Dict[str, float] = {
        "sample_count": float(len(left_series)),
        "accuracy": float("nan"),
        "macro_precision": float("nan"),
        "macro_recall": float("nan"),
        "macro_f1": float("nan"),
    }

    confusion, categories = _confusion_matrix(left_series, right_series)
    if confusion.empty:
        return metrics, confusion, []

    total = confusion.values.sum()
    if total > 0:
        metrics["accuracy"] = confusion.values.diagonal().sum() / total

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    per_class: List[Dict[str, float]] = []

    for category in categories:
        tp = float(confusion.at[category, category])
        predicted = float(confusion[category].sum())
        actual = float(confusion.loc[category].sum())

        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / actual if actual > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class.append(
            {
                "label": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(actual),
                "predicted": int(predicted),
            }
        )

    if categories:
        metrics["macro_precision"] = sum(precisions) / len(categories)
        metrics["macro_recall"] = sum(recalls) / len(categories)
        metrics["macro_f1"] = sum(f1s) / len(categories)
    else:
        metrics["macro_precision"] = float("nan")
        metrics["macro_recall"] = float("nan")
        metrics["macro_f1"] = float("nan")

    return metrics, confusion, per_class


def _save_summary(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        print(f"  No metrics to save at {path}")
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        try:
            markdown = df.to_markdown(index=False)
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"  Could not render markdown; metrics saved to {csv_path}")
        else:
            path.write_text(markdown)
            print(f"  Saved metric summary to {path}")
    else:
        df.to_csv(path, index=False)
        print(f"  Saved metric summary to {path}")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = slug.strip("_")
    return slug or "pair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa for GPT motivation labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to gpt_ref_motivation_label.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a metrics summary (CSV or Markdown).",
    )
    parser.add_argument(
        "--confusion-dir",
        type=Path,
        default=None,
        help="Optional directory for saving confusion matrices (CSV).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.input.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)

    pairs = [
        ("label1", "label2", "human_vs_human", "Inter-rater reliability (label1 vs label2)"),
        ("Final human label", "ChatGPT label", "human_vs_chatgpt", "LLM vs human gold labels"),
    ]

    summary_rows: List[Dict[str, object]] = []
    confusion_dir: Optional[Path] = args.confusion_dir
    if confusion_dir is not None:
        confusion_dir.mkdir(parents=True, exist_ok=True)

    for left, right, slug, description in pairs:
        if left not in df.columns or right not in df.columns:
            raise ValueError(f"Missing expected columns: {left!r}, {right!r}")
        left_series, right_series = _filter_labels(df[left], df[right])
        observed, expected, kappa = _cohen_kappa_series(left_series, right_series)
        metrics, confusion, per_class = _classification_metrics_from_series(left_series, right_series)

        sample_count = int(metrics["sample_count"])
        print(f"\n{description}:")
        print(f"  Columns: '{left}' (reference) vs '{right}' (comparison)")
        print(f"  Samples after filtering: {sample_count}")
        print(f"  Observed agreement: {observed:.6f}")
        print(f"  Expected agreement: {expected:.6f}")
        print(f"  Cohen's kappa: {kappa:.6f}")

        if sample_count > 0 and not confusion.empty:
            accuracy = metrics["accuracy"]
            macro_precision = metrics["macro_precision"]
            macro_recall = metrics["macro_recall"]
            macro_f1 = metrics["macro_f1"]
            print("  Classification metrics (macro-averaged):")
            print(f"    Accuracy: {accuracy:.6f}")
            print(f"    Precision: {macro_precision:.6f}")
            print(f"    Recall: {macro_recall:.6f}")
            print(f"    F1-score: {macro_f1:.6f}")

            print("  Confusion matrix (rows = reference / columns = comparison):")
            print(confusion.to_string())

            if per_class:
                per_class_df = pd.DataFrame(per_class)
                per_class_df = per_class_df.rename(
                    columns={
                        "label": "Label",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "support": "Support",
                        "predicted": "Predicted",
                    }
                )
                print("  Per-class metrics:")
                print(per_class_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        else:
            print("  Not enough overlapping labels to compute classification metrics.")

        summary_rows.extend(
            [
                {"pair": description, "metric": "samples", "value": sample_count},
                {"pair": description, "metric": "observed_agreement", "value": observed},
                {"pair": description, "metric": "expected_agreement", "value": expected},
                {"pair": description, "metric": "kappa", "value": kappa},
                {"pair": description, "metric": "accuracy", "value": metrics["accuracy"]},
                {"pair": description, "metric": "macro_precision", "value": metrics["macro_precision"]},
                {"pair": description, "metric": "macro_recall", "value": metrics["macro_recall"]},
                {"pair": description, "metric": "macro_f1", "value": metrics["macro_f1"]},
            ]
        )

        if confusion_dir is not None and not confusion.empty:
            confusion_path = confusion_dir / f"{_slugify(slug)}_confusion.csv"
            confusion.to_csv(confusion_path)
            per_class_path = confusion_dir / f"{_slugify(slug)}_per_class.csv"
            per_class_df = pd.DataFrame(per_class)
            per_class_df.to_csv(per_class_path, index=False)
            print(f"  Saved confusion matrix to {confusion_path}")
            print(f"  Saved per-class metrics to {per_class_path}")

    if args.summary is not None:
        _save_summary(summary_rows, args.summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
