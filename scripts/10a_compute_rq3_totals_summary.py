#!/usr/bin/env python3
"""Generate RQ3 totals summary outputs from the distribution CSV."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DISTRIBUTION = Path("outputs/research_questions/rq3/rq3_refactoring_type_distribution.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research_questions/rq3")
DEFAULT_RELATED_WORK = Path("data/related_work/summary.csv")


def _load_distribution(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Distribution CSV not found at {csv_path}. "
            "Run scripts/10_research_questions.py first."
        )
    df = pd.read_csv(csv_path)
    expected_cols = {"commit_sha", "refactoring_type", "instance_count", "is_self_affirmed"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Distribution CSV missing required columns: {', '.join(sorted(missing))}")
    return df


def _load_related_work(csv_path: Path) -> pd.DataFrame:
    """Load related work summary and extract refactoring type totals."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Related work CSV not found at {csv_path}."
        )
    df = pd.read_csv(csv_path)
    if "Refactoring Type" not in df.columns:
        raise ValueError("Related work CSV missing 'Refactoring Type' column")
    
    # Pandas renames duplicate columns as "Total", "Total.1", "Total.2", etc.
    # We expect: Total (production) and Total.1 (test)
    total_cols = [col for col in df.columns if col.startswith("Total")]
    if len(total_cols) < 1:
        raise ValueError("Related work CSV missing 'Total' columns")
    
    total_prod_col = total_cols[0] if len(total_cols) > 0 else None
    total_test_col = total_cols[1] if len(total_cols) > 1 else None
    
    # Build dataframe with the columns we need
    cols_to_use = ["Refactoring Type"]
    if total_prod_col:
        cols_to_use.append(total_prod_col)
    if total_test_col:
        cols_to_use.append(total_test_col)
    
    human_df = df[cols_to_use].copy()
    human_df.columns = ["refactoring_type_underscore"] + human_df.columns[1:].tolist()
    
    # Convert UPPERCASE_WITH_UNDERSCORES to Title Case format
    human_df["refactoring_type"] = human_df["refactoring_type_underscore"].str.replace("_", " ").str.title()
    # Replace OPERATION with METHOD for consistency
    human_df["refactoring_type"] = human_df["refactoring_type"].str.replace("Operation", "Method")
    # Add "And" back for Move/Rename/Inline operations to match distribution names
    human_df["refactoring_type"] = human_df["refactoring_type"].str.replace("Move Rename", "Move And Rename")
    human_df["refactoring_type"] = human_df["refactoring_type"].str.replace("Move Inline", "Move And Inline")
    
    # Calculate total human count from both prod and test
    human_df["human_prod"] = pd.to_numeric(human_df.get(total_prod_col, 0), errors="coerce").fillna(0).astype(int)
    human_df["human_test"] = (
        pd.to_numeric(human_df.get(total_test_col, 0), errors="coerce").fillna(0).astype(int)
        if total_test_col
        else 0
    )
    human_df["human_count"] = human_df["human_prod"] + human_df["human_test"]

    return human_df[["refactoring_type", "human_count", "human_prod", "human_test"]]


def _compute_summary(
    distribution_df: pd.DataFrame, human_data: Optional[pd.DataFrame]
) -> tuple[pd.DataFrame, dict]:
    """Compute refactoring type summary from distribution and human data."""
    # Count all refactorings (overall) and self-affirmed refactorings (SAR)
    overall_counts = (
        distribution_df.groupby("refactoring_type")["instance_count"]
        .sum()
        .reset_index(name="overall_count")
    )
    sar_counts = (
        distribution_df[distribution_df["is_self_affirmed"] == True]
        .groupby("refactoring_type")["instance_count"]
        .sum()
        .reset_index(name="sar_count")
    )

    overall_counts["overall_count"] = overall_counts["overall_count"].fillna(0).astype(int)
    sar_counts["sar_count"] = sar_counts["sar_count"].fillna(0).astype(int)

    # Merge overall and SAR counts, keeping types that appear in either dataset
    summary = pd.merge(overall_counts, sar_counts, on="refactoring_type", how="outer")
    summary["overall_count"] = summary["overall_count"].fillna(0).astype(int)
    summary["sar_count"] = summary["sar_count"].fillna(0).astype(int)

    has_human = human_data is not None and not human_data.empty
    if has_human:
        summary = pd.merge(summary, human_data, on="refactoring_type", how="outer")
        summary["human_count"] = summary["human_count"].fillna(0).astype(int)
        summary["human_prod"] = summary["human_prod"].fillna(0).astype(int)
        summary["human_test"] = summary["human_test"].fillna(0).astype(int)

    overall_total = int(summary["overall_count"].sum())
    sar_total = int(summary["sar_count"].sum())
    summary["overall_pct"] = (
        (summary["overall_count"] / overall_total * 100).round(2) if overall_total > 0 else 0.0
    )
    summary["sar_pct"] = (summary["sar_count"] / sar_total * 100).round(2) if sar_total > 0 else 0.0

    human_total = human_prod_total = human_test_total = 0
    if has_human:
        human_total = int(summary["human_count"].sum())
        human_prod_total = int(summary["human_prod"].sum())
        human_test_total = int(summary["human_test"].sum())
        summary["human_pct"] = (
            (summary["human_count"] / human_total * 100).round(2) if human_total > 0 else 0.0
        )
        summary["human_prod_pct"] = (
            (summary["human_prod"] / human_prod_total * 100).round(2)
            if human_prod_total > 0
            else 0.0
        )
        summary["human_test_pct"] = (
            (summary["human_test"] / human_test_total * 100).round(2)
            if human_test_total > 0
            else 0.0
        )

    # Sort by SAR count foremost, falling back to overall count
    summary = summary.sort_values(["sar_count", "overall_count"], ascending=False).reset_index(drop=True)

    totals: dict[str, int] = {
        "overall_total": overall_total,
        "sar_total": sar_total,
    }
    if has_human:
        totals.update(
            {
                "human_total": human_total,
                "human_prod_total": human_prod_total,
                "human_test_total": human_test_total,
            }
        )

    return summary, totals


def _write_csv(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write summary to CSV."""
    output_path = output_dir / "rq3_refactoring_type_totals_summary.csv"
    output_cols = ["refactoring_type", "overall_count", "sar_count", "overall_pct", "sar_pct"]
    optional_cols = [
        "human_count",
        "human_prod",
        "human_test",
        "human_pct",
        "human_prod_pct",
        "human_test_pct",
    ]
    for col in optional_cols:
        if col in summary_df.columns:
            output_cols.append(col)
    csv_df = summary_df[output_cols].copy()
    csv_df.to_csv(output_path, index=False)
    logging.info(f"Wrote {len(csv_df)} rows to {output_path}")
    return output_path


def _write_tex(summary: pd.DataFrame, output_dir: Path) -> Path:
    tex_path = output_dir / "rq3_refactoring_type_totals_summary.tex"

    def _format_entry(count: int, pct: float) -> str:
        return f"{count:,} ({pct:.2f}\\%)"

    columns_to_include = ["refactoring_type", "sar_count", "sar_pct", "overall_count", "overall_pct"]
    
    # Add human columns if they exist
    if "human_count" in summary.columns:
        columns_to_include.extend(["human_count", "human_pct"])
    
    latex_df = summary[columns_to_include].copy()
    
    # Fill NaN values with 0 for count columns
    latex_df["sar_count"] = latex_df["sar_count"].fillna(0).astype(int)
    latex_df["overall_count"] = latex_df["overall_count"].fillna(0).astype(int)
    if "human_count" in latex_df.columns:
        latex_df["human_count"] = latex_df["human_count"].fillna(0).astype(int)
    
    # Fill NaN values with 0 for percentage columns
    latex_df["sar_pct"] = latex_df["sar_pct"].fillna(0)
    latex_df["overall_pct"] = latex_df["overall_pct"].fillna(0)
    if "human_pct" in latex_df.columns:
        latex_df["human_pct"] = latex_df["human_pct"].fillna(0)
    
    # Format columns with count and percentage
    if "sar_count" in latex_df.columns and "sar_pct" in latex_df.columns:
        latex_df["SAR"] = latex_df.apply(
            lambda row: _format_entry(int(row["sar_count"]), row["sar_pct"]), axis=1
        )
    
    if "overall_count" in latex_df.columns and "overall_pct" in latex_df.columns:
        latex_df["Overall"] = latex_df.apply(
            lambda row: _format_entry(int(row["overall_count"]), row["overall_pct"]), axis=1
        )
    
    if "human_count" in latex_df.columns and "human_pct" in latex_df.columns:
        latex_df["Human"] = latex_df.apply(
            lambda row: _format_entry(int(row["human_count"]), row["human_pct"]), axis=1
        )
    
    # Select final columns: SAR, Overall, and Human
    final_columns = ["refactoring_type", "SAR", "Overall"]
    if "Human" in latex_df.columns:
        final_columns.append("Human")
    
    latex_df = latex_df[final_columns]
    latex_df = latex_df.rename(columns={"refactoring_type": "Refactoring Type"})
    
    latex_df.to_latex(
        tex_path,
        index=False,
        caption="Refactoring type totals and percentages: SAR, Overall, and Human (related work) comparison",
        label="tab:rq3_refactoring_totals_comparison",
        escape=False,
    )
    return tex_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the RQ3 refactoring type totals and export CSV + LaTeX summaries."
    )
    parser.add_argument(
        "--distribution-csv",
        type=Path,
        default=DEFAULT_DISTRIBUTION,
        help=f"Path to rq3_refactoring_type_distribution.csv (default: {DEFAULT_DISTRIBUTION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--related-work-csv",
        type=Path,
        default=DEFAULT_RELATED_WORK,
        help=f"Path to related work summary.csv (default: {DEFAULT_RELATED_WORK})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv or sys.argv[1:])
    distribution = args.distribution_csv.resolve()
    output_dir = args.output_dir.resolve()
    related_work_csv = args.related_work_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_distribution(distribution)
    human_data = _load_related_work(related_work_csv) if related_work_csv.exists() else None
    summary, totals = _compute_summary(df, human_data)
    csv_path = _write_csv(summary, output_dir)
    tex_path = _write_tex(summary, output_dir)

    print(f"Wrote {len(summary)} rows to {csv_path}")
    print(f"Overall total: {totals['overall_total']}, SAR total: {totals['sar_total']}")
    if "human_total" in totals:
        print(f"Human (related work) total: {totals['human_total']}")
        print(f"  - Production: {totals['human_prod_total']}")
        print(f"  - Test: {totals['human_test_total']}")
    print(f"LaTeX table available at {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
