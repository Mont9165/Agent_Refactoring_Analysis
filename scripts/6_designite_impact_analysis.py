#!/usr/bin/env python3
"""Aggregate Designite metrics and compare refactoring vs non-refactoring commits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import sys

import pandas as pd

# Ensure project root on path when running as script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.phase3_code_quality.designite_impact import (  # noqa: E402
    ANALYSIS_DIR,
    build_commit_metrics,
    summarize_impact,
)

COMMITS_PATH_PARQUET = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
COMMITS_PATH_CSV = Path("data/analysis/refactoring_instances/commits_with_refactoring.csv")


def load_commits() -> pd.DataFrame:
    if COMMITS_PATH_PARQUET.exists():
        return pd.read_parquet(COMMITS_PATH_PARQUET)
    if COMMITS_PATH_CSV.exists():
        return pd.read_csv(COMMITS_PATH_CSV)
    raise FileNotFoundError("commits_with_refactoring parquet/csv not found under data/analysis/refactoring_instances/")


def main() -> None:
    commits = load_commits()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    commit_metrics = build_commit_metrics(commits)
    if commit_metrics.empty:
        print("No Designite outputs found for commits in dataset.")
        return

    metrics_path_parquet = ANALYSIS_DIR / "commit_designite_metrics.parquet"
    metrics_path_csv = ANALYSIS_DIR / "commit_designite_metrics.csv"
    commit_metrics.to_parquet(metrics_path_parquet, index=False)
    commit_metrics.to_csv(metrics_path_csv, index=False)
    print(f"Saved per-commit metrics to {metrics_path_parquet} and {metrics_path_csv}")

    summary = summarize_impact(commit_metrics, commits)
    summary_path = ANALYSIS_DIR / "designite_impact_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {summary_path}")

    boxplot_path = ANALYSIS_DIR / "designite_boxplot_data.json"
    with open(boxplot_path, "w") as fh:
        json.dump(summary.get("boxplot_data", {}), fh, indent=2)
    print(f"Saved box plot data to {boxplot_path}")

    metrics_summary = summary.get("metrics", {})
    if metrics_summary:
        print("\nMetric deltas (refactoring - non-refactoring):")
        for metric, stats in metrics_summary.items():
            diff = stats.get("difference", {})
            mean_diff = diff.get("mean")
            median_diff = diff.get("median")
            pct_mean = diff.get("pct_change_mean")
            pct_median = diff.get("pct_change_median")

            def _fmt(value: Optional[float]) -> str:
                return f"{value:.3f}" if isinstance(value, (int, float)) and value is not None else "n/a"

            def _fmt_pct(value: Optional[float]) -> str:
                return f"{value:.2f}%" if isinstance(value, (int, float)) and value is not None else "n/a"

            print(
                "  {metric}: mean_diff={mean}, median_diff={median}, pct_mean={pct_mean}, pct_median={pct_median}".format(
                    metric=metric,
                    mean=_fmt(mean_diff),
                    median=_fmt(median_diff),
                    pct_mean=_fmt_pct(pct_mean),
                    pct_median=_fmt_pct(pct_median),
                )
            )


if __name__ == "__main__":
    main()
