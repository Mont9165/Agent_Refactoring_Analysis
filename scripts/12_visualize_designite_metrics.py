#!/usr/bin/env python3
"""Generate plots and LaTeX tables for Designite metric analysis results."""
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

ANALYSIS_DIR_DEFAULT = Path("data/analysis/designite/metric_analysis")
OUTPUT_DIR_DEFAULT = Path("outputs/designite/metric_analysis")
NON_SAR_COLOR = "#1f77b4"
DEGRADE_COLOR = "#d62728"
CATEGORY_ORDER = ["implementation_smell", "design_smell", "type_metric", "method_metric"]
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


def _pretty_category(name: str) -> str:
    return CATEGORY_LABELS.get(name, name.replace("_", " ").title())


def _pretty_metric(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " ").replace("-", " ").title())


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required analysis file not found: {path}")
    return pd.read_csv(path)


def _format_metric_label(row: pd.Series) -> str:
    return f"{_pretty_category(row['metric_category'])} – {_pretty_metric(row['metric_name'])}"


def create_diverging_baseline(
    baseline: pd.DataFrame,
    output_dir: Path,
    metric_label_order: List[str],
    output_filename: str,
    max_metrics: int = 25,
) -> None:
    data = baseline.copy()
    data["metric_label"] = data.apply(_format_metric_label, axis=1)
    selector = (
        (data["median_delta"].abs() > 1e-6)
        | (data["wilcoxon_p_value_fdr"] < 0.05)
        | (data["rank_biserial_effect"].abs() >= 0.1)
    )
    if selector.any():
        data = data[selector]
    if data.empty:
        data = baseline.copy()
    data = data.sort_values("median_delta")
    available_labels = [label for label in metric_label_order if label in data["metric_label"].values]
    data = data.set_index("metric_label").loc[available_labels].reset_index()
    if len(data) > max_metrics:
        half = max_metrics // 2
        remainder = max_metrics - half
        top = data.head(half)
        bottom = data.tail(remainder)
        data = pd.concat([top, bottom], ignore_index=True).sort_values("median_delta")

    values = data["median_delta"].to_numpy()
    use_mean = False
    if np.all(np.isclose(values, 0.0)):
        values = data["mean_delta"].to_numpy()
        use_mean = True

    colors = np.where(values < 0, NON_SAR_COLOR, DEGRADE_COLOR)

    plt.figure(figsize=(10, max(6, len(data) * 0.35)))
    bars = plt.barh(data["metric_label"], values, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    xlabel = "Median Δ (after − before)"
    if use_mean:
        xlabel += " (mean shown; medians were zero)"
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title("Overall median change per metric")

    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        max_abs = 1.0
    plt.xlim(-max_abs * 1.1, max_abs * 1.1)

    for rect, p_fdr in zip(bars, data["wilcoxon_p_value_fdr"]):
        if p_fdr < 0.05:
            x = rect.get_width()
            y = rect.get_y() + rect.get_height() / 2
            offset = 0.02 * np.sign(x) if x != 0 else 0.02
            plt.text(x + offset, y, "*", va="center", ha="left" if x >= 0 else "right", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=900)
    plt.close()


def create_baseline_plots(
    baseline: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    metric_label_order: List[str],
    top_filename: str,
    bottom_filename: str,
) -> None:
    baseline["metric_label"] = baseline.apply(_format_metric_label, axis=1)
    baseline["net_improvement"] = baseline["improvement_rate"] - baseline["worsening_rate"]

    top_positive = (
        baseline.sort_values("net_improvement", ascending=False)
        .head(top_n)
    )
    top_positive_labels = [label for label in metric_label_order if label in top_positive["metric_label"].values]
    top_positive = top_positive.set_index("metric_label").loc[top_positive_labels].reset_index()
    top_positive = top_positive.iloc[::-1]
    top_negative = (
        baseline.sort_values("net_improvement", ascending=True)
        .head(top_n)
    )
    top_negative_labels = [label for label in metric_label_order if label in top_negative["metric_label"].values]
    top_negative = top_negative.set_index("metric_label").loc[top_negative_labels].reset_index()
    top_negative = top_negative.iloc[::-1]

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    sns.barplot(
        data=top_positive,
        x="net_improvement",
        y="metric_label",
        hue="metric_label",
        palette="crest",
        ax=ax,
        legend=False,
    )
    ax.set_title(f"Top {top_n} Metrics by Net Improvement Rate")
    ax.set_xlabel("Improvement Rate − Worsening Rate")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / top_filename, dpi=900)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    sns.barplot(
        data=top_negative,
        x="net_improvement",
        y="metric_label",
        hue="metric_label",
        palette="rocket",
        ax=ax,
        legend=False,
    )
    ax.set_title(f"Worst {top_n} Metrics by Net Improvement Rate")
    ax.set_xlabel("Improvement Rate − Worsening Rate")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / bottom_filename, dpi=900)
    plt.close(fig)


def create_heatmap(
    summary: pd.DataFrame,
    category_col: str,
    values_col: str,
    output_path: Path,
    metric_keys: List[tuple[str, str]],
    order_labels: List[str],
    cmap: str = "viridis",
    title: str = "",
) -> None:
    if not metric_keys:
        return
    summary = summary.copy()
    summary["metric_key"] = list(zip(summary["metric_category"], summary["metric_name"]))
    subset = summary[summary["metric_key"].isin(metric_keys)].copy()
    if subset.empty:
        return
    if values_col == "improvement_rate":
        subset[values_col] = subset[values_col] * 100.0
    key_to_label = {key: label for key, label in zip(metric_keys, order_labels)}
    subset["metric_label"] = subset["metric_key"].map(key_to_label)
    pivot = subset.pivot_table(
        index="metric_label",
        columns=category_col,
        values=values_col,
    ).loc[order_labels]
    plt.figure(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f" if values_col == "improvement_rate" else ".2f",
        cmap=cmap,
        cbar_kws={"label": values_col.replace("_", " ").title() + (" (%)" if values_col == "improvement_rate" else "")},
        linewidths=0.5,
    )
    plt.title(title or f"{values_col} by {category_col}")
    plt.ylabel("")
    plt.xlabel(category_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=900)
    plt.close()


def save_latex_table(df: pd.DataFrame, columns: List[str], path: Path, caption: str, label: str) -> None:
    table_df = df[columns].copy()
    if "metric_category" in table_df.columns:
        table_df["metric_category"] = table_df["metric_category"].map(_pretty_category)
    if "metric_name" in table_df.columns:
        table_df["metric_name"] = table_df["metric_name"].map(_pretty_metric)
    for rate_col in ("improvement_rate", "worsening_rate"):
        if rate_col in table_df.columns:
            table_df[rate_col] = table_df[rate_col] * 100.0
    table_df = table_df.rename(
        columns={
            "metric_category": "Metric Category",
            "metric_name": "Metric Name",
            "observation_count": "Observations",
            "median_delta": "Median Δ",
            "improvement_rate": "Improvement Rate (%)",
            "worsening_rate": "Worsening Rate (%)",
            "rank_biserial_effect": "Rank-Biserial Effect",
            "wilcoxon_p_value_fdr": "Wilcoxon p (FDR)",
        }
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            table_df.to_latex(
                index=False,
                float_format="%.3f",
                caption=caption,
                label=label,
                escape=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_DIR_DEFAULT,
        help=f"Directory containing metric analysis CSVs (default: {ANALYSIS_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory to write plots and LaTeX tables (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top/bottom metrics to display in summaries (default: 10).",
    )
    parser.add_argument(
        "--top-heatmap",
        type=int,
        default=5,
        help="Number of metrics to include in motivation/SAR heatmaps (default: 5).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_csv(args.analysis_dir / "metric_baseline_summary.csv")
    baseline["metric_label"] = baseline.apply(_format_metric_label, axis=1)
    baseline["metric_key"] = list(zip(baseline["metric_category"], baseline["metric_name"]))
    baseline["category_order"] = baseline["metric_category"].map(lambda cat: CATEGORY_ORDER.index(cat))
    baseline = baseline.sort_values(
        ["category_order", "rank_biserial_effect"],
        ascending=[True, False],
        key=lambda s: np.abs(s) if s.name == "rank_biserial_effect" else s,
    ).reset_index(drop=True)
    metric_label_order = baseline["metric_label"].tolist()
    metric_key_order = baseline["metric_key"].tolist()
    key_to_label = dict(zip(metric_key_order, metric_label_order))

    create_diverging_baseline(
        baseline,
        args.output_dir,
        metric_label_order,
        output_filename="baseline_diverging_median_delta.pdf",
    )
    create_baseline_plots(
        baseline,
        args.output_dir,
        args.top_n,
        metric_label_order,
        top_filename="baseline_top_improvement.pdf",
        bottom_filename="baseline_worst_improvement.pdf",
    )

    baseline["net_improvement"] = baseline["improvement_rate"] - baseline["worsening_rate"]
    top_positive = baseline.sort_values("net_improvement", ascending=False).head(args.top_n)
    top_negative = baseline.sort_values("net_improvement", ascending=True).head(args.top_n)

    save_latex_table(
        top_positive,
        columns=[
            "metric_category",
            "metric_name",
            "observation_count",
            "median_delta",
            "improvement_rate",
            "worsening_rate",
            "rank_biserial_effect",
            "wilcoxon_p_value_fdr",
        ],
        path=args.output_dir / "baseline_top_improvement.tex",
        caption=f"Top {args.top_n} metrics by net improvement rate.",
        label="tab:designite_top_improvement",
    )

    save_latex_table(
        top_negative,
        columns=[
            "metric_category",
            "metric_name",
            "observation_count",
            "median_delta",
            "improvement_rate",
            "worsening_rate",
            "rank_biserial_effect",
            "wilcoxon_p_value_fdr",
        ],
        path=args.output_dir / "baseline_worst_improvement.tex",
        caption=f"Worst {args.top_n} metrics by net improvement rate.",
        label="tab:designite_worst_improvement",
    )

    motivation_summary = _load_csv(args.analysis_dir / "metric_by_motivation_summary.csv")
    sar_summary = _load_csv(args.analysis_dir / "metric_by_sar_summary.csv")
    sar_tests = _load_csv(args.analysis_dir / "metric_by_sar_tests.csv")
    sar_tests = sar_tests.rename(columns={"mannwhitney_p_value_fdr": "wilcoxon_p_value_fdr"})
    sar_summary = sar_summary.merge(
        sar_tests[["metric_category", "metric_name", "rank_biserial_effect", "wilcoxon_p_value_fdr"]],
        on=["metric_category", "metric_name"],
        how="left",
    )
    sar_summary["rank_biserial_effect"] = sar_summary["rank_biserial_effect"].fillna(0.0)
    sar_summary["wilcoxon_p_value_fdr"] = sar_summary["wilcoxon_p_value_fdr"].fillna(1.0)
    sar_summary["metric_key"] = list(zip(sar_summary["metric_category"], sar_summary["metric_name"]))
    sar_summary["metric_label"] = sar_summary["metric_key"].map(key_to_label)
    missing_labels = sar_summary["metric_label"].isna()
    if missing_labels.any():
        sar_summary.loc[missing_labels, "metric_label"] = sar_summary[missing_labels].apply(_format_metric_label, axis=1)
    sar_summary["net_improvement"] = sar_summary["improvement_rate"] - sar_summary["worsening_rate"]

    sar_group_labels = {
        True: ("sar_self_affirmed", "Self-affirmed (SAR)"),
        False: ("sar_non_self_affirmed", "Non-SAR"),
    }
    for flag, (prefix, human_label) in sar_group_labels.items():
        subset = sar_summary[sar_summary["is_self_affirmed"] == flag].dropna(subset=["metric_label"]).copy()
        if subset.empty:
            print(f"No data available for {human_label}; skipping SAR-specific outputs.")
            continue

        create_diverging_baseline(
            subset,
            args.output_dir,
            metric_label_order,
            output_filename=f"{prefix}_diverging_median_delta.pdf",
        )
        create_baseline_plots(
            subset,
            args.output_dir,
            args.top_n,
            metric_label_order,
            top_filename=f"{prefix}_top_improvement.pdf",
            bottom_filename=f"{prefix}_worst_improvement.pdf",
        )

        top_positive_subset = subset.sort_values("net_improvement", ascending=False).head(args.top_n)
        top_negative_subset = subset.sort_values("net_improvement", ascending=True).head(args.top_n)

        save_latex_table(
            top_positive_subset,
            columns=[
                "metric_category",
                "metric_name",
                "observation_count",
                "median_delta",
                "improvement_rate",
                "worsening_rate",
                "rank_biserial_effect",
                "wilcoxon_p_value_fdr",
            ],
            path=args.output_dir / f"{prefix}_top_improvement.tex",
            caption=f"Top {args.top_n} metrics by net improvement rate ({human_label}).",
            label=f"tab:designite_{prefix}_top_improvement",
        )

        save_latex_table(
            top_negative_subset,
            columns=[
                "metric_category",
                "metric_name",
                "observation_count",
                "median_delta",
                "improvement_rate",
                "worsening_rate",
                "rank_biserial_effect",
                "wilcoxon_p_value_fdr",
            ],
            path=args.output_dir / f"{prefix}_worst_improvement.tex",
            caption=f"Worst {args.top_n} metrics by net improvement rate ({human_label}).",
            label=f"tab:designite_{prefix}_worst_improvement",
        )

    heatmap_metric_keys = metric_key_order[:args.top_heatmap]
    heatmap_labels = [key_to_label[key] for key in heatmap_metric_keys]

    create_heatmap(
        motivation_summary,
        category_col="motivation_label",
        values_col="improvement_rate",
        metric_keys=heatmap_metric_keys,
        order_labels=heatmap_labels,
        output_path=args.output_dir / "motivation_improvement_heatmap.pdf",
        cmap="Blues",
        title="Improvement Rate by Motivation (Top Metrics)",
    )

    create_heatmap(
        motivation_summary,
        category_col="motivation_label",
        values_col="median_delta",
        metric_keys=heatmap_metric_keys,
        order_labels=heatmap_labels,
        output_path=args.output_dir / "motivation_median_delta_heatmap.pdf",
        cmap="PuRd",
        title="Median Delta by Motivation (Top Metrics)",
    )

    create_heatmap(
        sar_summary,
        category_col="is_self_affirmed",
        values_col="improvement_rate",
        metric_keys=heatmap_metric_keys,
        order_labels=heatmap_labels,
        output_path=args.output_dir / "sar_improvement_heatmap.pdf",
        cmap="Greens",
        title="Improvement Rate by SAR Flag (Top Metrics)",
    )

    motivation_summary["metric_key"] = list(zip(motivation_summary["metric_category"], motivation_summary["metric_name"]))
    motivation_table = motivation_summary[motivation_summary["metric_key"].isin(heatmap_metric_keys)].copy()
    motivation_table["metric_key"] = pd.Categorical(motivation_table["metric_key"], categories=heatmap_metric_keys, ordered=True)
    motivation_table = motivation_table.sort_values(["metric_key", "motivation_label"])
    motivation_table = motivation_table.drop(columns=["metric_key"])
    save_latex_table(
        motivation_table,
        columns=[
            "metric_category",
            "metric_name",
            "motivation_label",
            "observation_count",
            "median_delta",
            "improvement_rate",
        ],
        path=args.output_dir / "motivation_summary_top_metrics.tex",
        caption="Motivation-specific summary statistics for top metrics.",
        label="tab:designite_motivation_summary",
    )

    sar_summary["metric_key"] = list(zip(sar_summary["metric_category"], sar_summary["metric_name"]))
    sar_table = sar_summary[sar_summary["metric_key"].isin(heatmap_metric_keys)].copy()
    sar_table["metric_key"] = pd.Categorical(sar_table["metric_key"], categories=heatmap_metric_keys, ordered=True)
    sar_table = sar_table.sort_values(["metric_key", "is_self_affirmed"])
    sar_table = sar_table.drop(columns=["metric_key"])
    save_latex_table(
        sar_table,
        columns=[
            "metric_category",
            "metric_name",
            "is_self_affirmed",
            "observation_count",
            "median_delta",
            "improvement_rate",
        ],
        path=args.output_dir / "sar_summary_top_metrics.tex",
        caption="SAR vs. non-SAR summary statistics for top metrics.",
        label="tab:designite_sar_summary",
    )

    print(f"Plots and LaTeX tables saved under {args.output_dir}")


if __name__ == "__main__":
    main()
