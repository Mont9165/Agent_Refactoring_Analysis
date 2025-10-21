#!/usr/bin/env python3
"""Generate motivation and refactoring heatmaps for significant metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS_DIR = Path("data/analysis/designite/metric_analysis")
SIGNIFICANT_TABLE = Path("outputs/designite/metric_distributions/significant_metrics_summary.csv")
OUTPUT_DIR = Path("outputs/designite/group_heatmaps")
CATEGORY_ORDER = ["implementation_smell", "design_smell", "type_metric", "method_metric"]
SIGNIFICANCE_THRESHOLD = 0.05
CATEGORY_LABELS = {
    "implementation_smell": "Implementation Smell",
    "design_smell": "Design Smell",
    "type_metric": "Class-Level Metric",
    "method_metric": "Method-Level Metric",
}
METRIC_LABELS = {
    "LOC": "Lines of Code",
    "CC": "Cyclomatic Complexity",
    "PC": "Parameter Count",
    "WMC": "Weighted Methods per Class",
    "NOM": "Number of Methods",
    "NOF": "Number of Fields",
    "NOPF": "Number of Public Fields",
    "NOPM": "Number of Public Methods",
    "NC": "Number of Children",
    "DIT": "Depth of Inheritance Tree",
    "LCOM": "Lack of Cohesion in Methods",
    "FANIN": "Fan-In",
    "FANOUT": "Fan-Out",
}


def pretty_category(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


def pretty_metric(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " ").replace("-", " ").title())


def load_significant_metrics(top_n: int) -> pd.DataFrame:
    if not SIGNIFICANT_TABLE.exists():
        raise FileNotFoundError(f"Significant metrics summary not found: {SIGNIFICANT_TABLE}")
    df = pd.read_csv(SIGNIFICANT_TABLE)
    df["metric_category"] = pd.Categorical(df["metric_category"], categories=CATEGORY_ORDER, ordered=True)
    df = df.sort_values(
        ["metric_category", "rank_biserial_effect"],
        ascending=[True, False],
        key=lambda s: np.abs(s) if s.name == "rank_biserial_effect" else s,
    )
    df["metric_key"] = list(zip(df["metric_category"].astype(str), df["metric_name"].astype(str)))
    df["display_label"] = df["metric_key"].apply(lambda key: f"{pretty_category(key[0])} – {pretty_metric(key[1])}")
    return df.head(top_n)


def load_summary(path: Path, group_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={group_col: "group"})
    df["metric_label"] = df["metric_category"] + " – " + df["metric_name"]
    return df


def load_tests(path: Path, p_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Tests file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={p_col: "adjusted_p"})
    return df


def format_heatmap(
    data: pd.DataFrame,
    metric_keys: List[tuple[str, str]],
    value_col: str,
    cmap,
    value_formatter: str,
    center: float | None,
    vmin: float | None,
    vmax: float | None,
    significant_keys: set[tuple[str, str]] | None,
    output_path: Path,
    title: str,
    group_limit: int | None = None,
    csv_path: Path | None = None,
) -> None:
    if not metric_keys:
        return
    subset = data.copy()
    subset["metric_key"] = list(zip(subset["metric_category"], subset["metric_name"]))
    subset = subset[subset["metric_key"].isin(metric_keys)]
    if subset.empty:
        return

    if value_col == "improvement_rate":
        subset[value_col] = subset[value_col] * 100.0

    pivot = (
        subset.pivot(index="metric_key", columns="group", values=value_col)
        .loc[metric_keys]
    )
    column_order = (
        pivot.apply(lambda col: np.nanmean(np.abs(col)), axis=0)
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    if group_limit is not None and len(column_order) > group_limit:
        column_order = column_order[:group_limit]
    pivot = pivot[column_order]

    labels = []
    for key in metric_keys:
        label = f"{pretty_category(key[0])} – {pretty_metric(key[1])}"
        if significant_keys and key in significant_keys:
            label += "*"
        labels.append(label)
    pivot.index = labels

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pivot.reset_index().rename(columns={"index": "metric"}).to_csv(csv_path, index=False)

    plt.figure(figsize=(max(6, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=value_formatter,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": value_col.replace("_", " ").title() + (" (%)" if value_col == "improvement_rate" else "")},
    )
    plt.title(title)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=900)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10, help="Number of significant metrics to plot (default: 10).")
    parser.add_argument(
        "--value",
        type=str,
        choices=["median_delta", "improvement_rate"],
        default="median_delta",
        help="Metric to display in the heatmaps.",
    )
    parser.add_argument("--refactoring-groups", type=int, default=6, help="Maximum refactoring types to show (default: 6).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = load_significant_metrics(args.top_n)
    metric_keys = list(metrics_df["metric_key"])
    key_to_label = dict(zip(metrics_df["metric_key"], metrics_df["display_label"]))

    motivation_summary = load_summary(ANALYSIS_DIR / "metric_by_motivation_summary.csv", "motivation_label")
    motivation_tests = load_tests(ANALYSIS_DIR / "metric_by_motivation_tests.csv", "kruskal_p_value_fdr")

    try:
        motivation_summary_sar = load_summary(ANALYSIS_DIR / "metric_by_motivation_sar_summary.csv", "motivation_label")
        motivation_tests_sar = load_tests(ANALYSIS_DIR / "metric_by_motivation_sar_tests.csv", "kruskal_p_value_fdr")
    except FileNotFoundError:
        motivation_summary_sar = None
        motivation_tests_sar = None

    refactoring_summary = load_summary(ANALYSIS_DIR / "metric_by_refactoring_summary.csv", "refactoring_type")
    refactoring_tests = load_tests(ANALYSIS_DIR / "metric_by_refactoring_tests.csv", "kruskal_p_value_fdr")

    try:
        refactoring_summary_sar = load_summary(ANALYSIS_DIR / "metric_by_refactoring_sar_summary.csv", "refactoring_type")
        refactoring_tests_sar = load_tests(ANALYSIS_DIR / "metric_by_refactoring_sar_tests.csv", "kruskal_p_value_fdr")
    except FileNotFoundError:
        refactoring_summary_sar = None
        refactoring_tests_sar = None

    sar_summary = load_summary(ANALYSIS_DIR / "metric_by_sar_summary.csv", "is_self_affirmed")
    sar_summary["group"] = sar_summary["group"].map({True: "Self-Affirmed", False: "Non-SAR"}).fillna(sar_summary["group"])
    sar_tests = load_tests(ANALYSIS_DIR / "metric_by_sar_tests.csv", "mannwhitney_p_value_fdr")
    sar_sig = {
        (row.metric_category, row.metric_name)
        for row in sar_tests.itertuples(index=False)
        if getattr(row, "adjusted_p", 1.0) < SIGNIFICANCE_THRESHOLD
    }

    if args.value == "median_delta":
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        value_formatter = ".2f"
        center = 0.0
        vmin = vmax = None
    else:
        cmap = sns.color_palette("light:b", as_cmap=True)
        value_formatter = ".1f"
        center = None
        vmin = 0.0
        vmax = 100.0

    motivation_sig = {
        (row.metric_category, row.metric_name)
        for row in motivation_tests.itertuples(index=False)
        if getattr(row, "adjusted_p", 1.0) < SIGNIFICANCE_THRESHOLD
    }
    motivation_metrics = [key for key in metric_keys if key in motivation_sig]
    if motivation_metrics:
        motivation_pdf = OUTPUT_DIR / f"motivation_heatmap_{args.value}.pdf"
        format_heatmap(
            motivation_summary,
            motivation_metrics,
            args.value,
            cmap=cmap,
            value_formatter=value_formatter,
            center=center,
            vmin=vmin,
            vmax=vmax,
            significant_keys=motivation_sig,
            output_path=motivation_pdf,
            title=f"Motivation impact for {len(motivation_metrics)} significant metrics ({args.value})",
            group_limit=None,
            csv_path=motivation_pdf.with_suffix(".csv"),
        )
    else:
        print("No motivation groups showed significant differences; skipping motivation heatmap.")

    if motivation_summary_sar is not None and motivation_tests_sar is not None:
        motivation_sar_sig = {
            (row.metric_category, row.metric_name)
            for row in motivation_tests_sar.itertuples(index=False)
            if getattr(row, "adjusted_p", 1.0) < SIGNIFICANCE_THRESHOLD
        }
        sar_motivation_metrics = [key for key in metric_keys if key in motivation_sar_sig]
        if sar_motivation_metrics:
            motivation_sar_pdf = OUTPUT_DIR / f"motivation_heatmap_sar_{args.value}.pdf"
            format_heatmap(
                motivation_summary_sar,
                sar_motivation_metrics,
                args.value,
                cmap=cmap,
                value_formatter=value_formatter,
                center=center,
                vmin=vmin,
                vmax=vmax,
                significant_keys=motivation_sar_sig,
                output_path=motivation_sar_pdf,
                title=f"Motivation impact for {len(sar_motivation_metrics)} SAR metrics ({args.value})",
                group_limit=None,
                csv_path=motivation_sar_pdf.with_suffix(".csv"),
            )
        else:
            print("No SAR motivation cohorts met the significance criteria; skipping SAR motivation heatmap.")
    else:
        print("SAR-specific motivation summaries not found; skipping SAR motivation heatmap.")

    top_ref_types = (
        refactoring_summary.groupby("group")["observation_count"]
        .sum()
        .sort_values(ascending=False)
        .head(args.refactoring_groups)
        .index
        .tolist()
    )
    refactoring_summary = refactoring_summary[refactoring_summary["group"].isin(top_ref_types)]
    refactoring_sig = {
        (row.metric_category, row.metric_name)
        for row in refactoring_tests.itertuples(index=False)
        if getattr(row, "adjusted_p", 1.0) < SIGNIFICANCE_THRESHOLD
    }
    refactoring_metrics = [key for key in metric_keys if key in refactoring_sig]
    if refactoring_metrics:
        refactoring_pdf = OUTPUT_DIR / f"refactoring_heatmap_{args.value}.pdf"
        format_heatmap(
            refactoring_summary,
            refactoring_metrics,
            args.value,
            cmap=cmap,
            value_formatter=value_formatter,
            center=center,
            vmin=vmin,
            vmax=vmax,
            significant_keys=refactoring_sig,
            output_path=refactoring_pdf,
            title=f"Refactoring impact for {len(refactoring_metrics)} significant metrics ({args.value})",
            group_limit=args.refactoring_groups,
            csv_path=refactoring_pdf.with_suffix(".csv"),
        )
    else:
        print("No refactoring types showed significant differences; skipping refactoring heatmap.")

    if refactoring_summary_sar is not None and refactoring_tests_sar is not None:
        refactoring_sar_sig = {
            (row.metric_category, row.metric_name)
            for row in refactoring_tests_sar.itertuples(index=False)
            if getattr(row, "adjusted_p", 1.0) < SIGNIFICANCE_THRESHOLD
        }
        sar_ref_metrics = [key for key in metric_keys if key in refactoring_sar_sig]
        if sar_ref_metrics:
            refactoring_sar_pdf = OUTPUT_DIR / f"refactoring_heatmap_sar_{args.value}.pdf"
            format_heatmap(
                refactoring_summary_sar,
                sar_ref_metrics,
                args.value,
                cmap=cmap,
                value_formatter=value_formatter,
                center=center,
                vmin=vmin,
                vmax=vmax,
                significant_keys=refactoring_sar_sig,
                output_path=refactoring_sar_pdf,
                title=f"Refactoring impact for {len(sar_ref_metrics)} SAR metrics ({args.value})",
                group_limit=args.refactoring_groups,
                csv_path=refactoring_sar_pdf.with_suffix(".csv"),
            )
        else:
            print("No SAR refactoring cohorts met the significance criteria; skipping SAR refactoring heatmap.")
    else:
        print("SAR-specific refactoring summaries not found; skipping SAR refactoring heatmap.")

    sar_metrics = [key for key in metric_keys if key in sar_sig]
    if sar_metrics:
        sar_pdf = OUTPUT_DIR / f"sar_heatmap_{args.value}.pdf"
        format_heatmap(
            sar_summary,
            sar_metrics,
            args.value,
            cmap=cmap,
            value_formatter=value_formatter,
            center=center,
            vmin=vmin,
            vmax=vmax,
            significant_keys=sar_sig,
            output_path=sar_pdf,
            title=f"SAR impact for {len(sar_metrics)} significant metrics ({args.value})",
            group_limit=None,
            csv_path=sar_pdf.with_suffix(".csv"),
        )
    else:
        print("No SAR differences detected; skipping SAR heatmap.")

    print(f"Saved heatmaps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
