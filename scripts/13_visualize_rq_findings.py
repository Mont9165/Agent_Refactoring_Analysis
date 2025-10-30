#!/usr/bin/env python3
"""Generate RQ1-RQ4 figures and LaTeX tables for the AI-assisted refactoring study."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_OUTPUT_DIR = Path("outputs/research_questions")
FIGURE_DIR = BASE_OUTPUT_DIR / "figures"
TABLE_DIR = BASE_OUTPUT_DIR / "tables"

NON_SAR_COLOR = "#1f77b4"
SAR_COLOR = "#ff7f0e"
IMPROVE_COLOR = "#1f77b4"
DEGRADE_COLOR = "#d62728"


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _load_commits() -> pd.DataFrame:
    df = pd.read_parquet("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
    df["is_self_affirmed"] = df["is_self_affirmed"].fillna(False)
    df["has_refactoring"] = df["has_refactoring"].fillna(False)
    return df


def _load_refminer() -> pd.DataFrame:
    return pd.read_parquet("data/analysis/refactoring_instances/refminer_refactorings.parquet")


def _load_motivations() -> pd.DataFrame:
    df = pd.read_csv("data/analysis/refactoring_instances/gpt_refactoring_motivation_update.csv")
    df = df.rename(columns={"sha": "commit_sha", "type": "motivation_label"})
    df["motivation_label"] = df["motivation_label"].fillna("unknown")
    return df


# ---------------------------------------------------------------------------
# Figure 1 – Frequency and role of SAR
# ---------------------------------------------------------------------------
def figure_rq1(commits: pd.DataFrame, refminer: pd.DataFrame) -> None:
    agentic = commits[commits["agent"].notna()]
    total_commits = agentic["sha"].nunique()

    refactor_commits = agentic[agentic["has_refactoring"]]
    refactor_count = refactor_commits["sha"].nunique()

    sar_commits = refactor_commits[refactor_commits["is_self_affirmed"]]
    sar_count = sar_commits["sha"].nunique()

    total_instances = len(refminer)
    sar_instance_count = refminer[refminer["commit_sha"].isin(sar_commits["sha"])].shape[0]

    pct_refactor = refactor_count / total_commits * 100 if total_commits else 0
    pct_sar_within_refactor = sar_count / refactor_count * 100 if refactor_count else 0
    pct_sar_instances = sar_instance_count / total_instances * 100 if total_instances else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Commit-level nested bar
    axes[0].barh(["Agentic Commits"], [total_commits], color="#d9d9d9")
    axes[0].barh(["Agentic Commits"], [refactor_count], color=NON_SAR_COLOR)
    axes[0].barh(["Agentic Commits"], [sar_count], color=SAR_COLOR)
    axes[0].set_title("Commit-Level Breakdown")
    axes[0].set_xlabel("Commit Count")
    axes[0].invert_yaxis()
    axes[0].text(refactor_count / 2, 0, f"{pct_refactor:.1f}% refactoring commits", color="white", ha="center", va="center")
    axes[0].text(
        sar_count / 2,
        0,
        f"{pct_sar_within_refactor:.1f}% SAR\n(refactoring commits)",
        color="white",
        ha="center",
        va="center",
    )

    # Instance-level nested bar
    axes[1].barh(["Refactoring Instances"], [total_instances], color="#d9d9d9")
    axes[1].barh(["Refactoring Instances"], [sar_instance_count], color=SAR_COLOR)
    axes[1].set_title("Instance-Level Breakdown")
    axes[1].set_xlabel("Instance Count")
    axes[1].invert_yaxis()
    axes[1].text(
        sar_instance_count / 2,
        0,
        f"{pct_sar_instances:.1f}% of instances\nfrom SAR commits",
        color="white",
        ha="center",
        va="center",
    )

    sns.despine()
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_rq1_refactoring_frequency.pdf", dpi=900)
    plt.close(fig)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total agentic commits",
                "Refactoring commits",
                "SAR refactoring commits",
                "Total refactoring instances",
                "SAR refactoring instances",
            ],
            "Count": [
                total_commits,
                refactor_count,
                sar_count,
                total_instances,
                sar_instance_count,
            ],
            "Percentage": [
                "",
                f"{pct_refactor:.2f}%",
                f"{pct_sar_within_refactor:.2f}% (within refactoring commits)",
                "",
                f"{pct_sar_instances:.2f}% (of instances)",
            ],
        }
    )
    summary_df.to_latex(TABLE_DIR / "table_rq1_summary.tex", index=False, caption="RQ1 refactoring frequency summary.", label="tab:rq1_frequency")


# ---------------------------------------------------------------------------
# Figure 2 – Top refactoring types by median
# ---------------------------------------------------------------------------
def compute_refactoring_medians(refminer: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    counts = (
        refminer.groupby(["commit_sha", "refactoring_type"])
        .size()
        .reset_index(name="count")
    )
    counts = counts.merge(
        commits[["sha", "is_self_affirmed"]],
        left_on="commit_sha",
        right_on="sha",
        how="left",
    )
    counts["group"] = np.where(counts["is_self_affirmed"], "SAR", "Non-SAR")
    medians = (
        counts.groupby(["group", "refactoring_type"])["count"]
        .median()
        .reset_index()
    )
    medians = medians.rename(columns={"count": "median_instances"})
    return medians


def figure_rq2(medians: pd.DataFrame, top_n: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, max(6, top_n * 0.35)), sharex=True)
    for ax, group, color in zip(axes, ["Non-SAR", "SAR"], [NON_SAR_COLOR, SAR_COLOR]):
        data = (
            medians[medians["group"] == group]
            .sort_values("median_instances", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        sns.barplot(
            data=data,
            x="median_instances",
            y="refactoring_type",
            color=color,
            ax=ax,
        )
        ax.set_title(f"{group}: Top {top_n} Refactoring Types (Median instances per commit)")
        ax.set_xlabel("Median instances per commit")
        ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_rq2_refactoring_types.pdf", dpi=900)
    plt.close(fig)

    top_non = medians[medians["group"] == "Non-SAR"].sort_values("median_instances", ascending=False).head(top_n)
    top_sar = medians[medians["group"] == "SAR"].sort_values("median_instances", ascending=False).head(top_n)

    table = top_non.merge(top_sar, on="refactoring_type", how="outer", suffixes=("_non_sar", "_sar"))
    table = table.rename(columns={"median_instances_non_sar": "Non-SAR median", "median_instances_sar": "SAR median"})
    table = table.fillna(0)
    table.to_latex(
        TABLE_DIR / "table_rq2_refactoring_types.tex",
        index=False,
        float_format="%.2f",
        caption=f"Top {top_n} refactoring types by median instances per commit (Non-SAR vs SAR).",
        label="tab:rq2_refactoring_types",
    )


# ---------------------------------------------------------------------------
# Figure 3 – Refactoring purposes
# ---------------------------------------------------------------------------
def figure_rq3(motivations: pd.DataFrame, commits: pd.DataFrame) -> None:
    data = motivations.merge(
        commits[["sha", "is_self_affirmed"]],
        left_on="commit_sha",
        right_on="sha",
        how="left",
    )
    data = data.dropna(subset=["motivation_label"])
    data = data[data["is_self_affirmed"]]

    counts = (
        data.groupby("motivation_label")["commit_sha"]
        .nunique()
        .reset_index(name="commit_count")
    )
    if counts.empty:
        print("  No SAR motivation labels available.")
        table_path = TABLE_DIR / "table_rq3_refactoring_purposes.tex"
        table_path.write_text(
            "\\begin{table}[h]\n\\centering\n\\caption{Distribution of refactoring purposes across SAR commits.}\n"
            "\\label{tab:rq3-purpose-comparison}\n\\begin{tabular}{lr}\n\\toprule\n"
            "\\textbf{Purpose Category} & \\textbf{SAR} \\\\\n\\midrule\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
            encoding="utf-8",
        )
        return

    counts = counts.sort_values("commit_count", ascending=False)
    total_commits = int(counts["commit_count"].sum())
    counts["percentage"] = counts["commit_count"] / total_commits * 100.0 if total_commits else 0.0
    labels_display = counts["motivation_label"].str.replace("_", " ").str.title().str.replace(" To ", " to ")

    plt.figure(figsize=(10, max(5, len(counts) * 0.45)))
    ax = plt.gca()
    ax.barh(labels_display.iloc[::-1], counts["percentage"].iloc[::-1], color="#E4572E")
    ax.set_xlabel("Share of SAR commits (%)")
    ax.set_ylabel("Motivation category")
    ax.set_title("Developer motivations for refactoring (SAR only)")
    ax.set_xlim(0, counts["percentage"].max() * 1.15 if not counts["percentage"].empty else 1)
    for y, (label, pct) in enumerate(zip(labels_display.iloc[::-1], counts["percentage"].iloc[::-1])):
        ax.text(pct + 0.5, y, f"{pct:.1f}%", va="center", ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure_rq3_refactoring_purposes.pdf", dpi=900)
    plt.close()

    def _fmt(count: int, pct: float) -> str:
        return f"{count:,} ({pct:.1f}\\%)"

    table_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Distribution of refactoring purposes across SAR commits.}",
        "\\label{tab:rq3-purpose-comparison}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "\\textbf{Purpose Category} & \\textbf{SAR} \\\\",
        "\\midrule",
    ]

    for label, row in zip(labels_display, counts.itertuples()):
        table_lines.append(f"{label} & {_fmt(row.commit_count, row.percentage)} \\\\")

    table_lines.extend(
        [
            "\\midrule",
            f"\\textbf{{Total Commits}} & \\textbf{{{total_commits:,}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table_path = TABLE_DIR / "table_rq3_refactoring_purposes.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 4 – Overall impact on metrics
# ---------------------------------------------------------------------------
def select_metrics_for_fig4(baseline: pd.DataFrame, max_metrics: int) -> pd.DataFrame:
    significant = baseline[baseline["wilcoxon_p_value_fdr"] < 0.05]
    top_counts = baseline.sort_values("observation_count", ascending=False).head(max_metrics)
    combined = pd.concat([significant, top_counts], ignore_index=True).drop_duplicates(subset=["metric_category", "metric_name"])
    return combined.sort_values("median_delta")


def figure_rq4(baseline: pd.DataFrame, max_metrics: int) -> List[Tuple[str, str]]:
    selected = select_metrics_for_fig4(baseline, max_metrics)
    selected["metric_label"] = selected.apply(lambda row: f"{row['metric_category']} – {row['metric_name']}", axis=1)
    selected = selected.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(selected) * 0.4)))
    colors = [IMPROVE_COLOR if val < 0 else DEGRADE_COLOR for val in selected["median_delta"]]
    ax.barh(selected["metric_label"], selected["median_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Median Δ (after − before)")
    ax.set_ylabel("")
    ax.set_title("Overall impact of refactoring on quality metrics")

    significant_metrics = []
    for y, (label, delta, p_fdr) in enumerate(zip(selected["metric_label"], selected["median_delta"], selected["wilcoxon_p_value_fdr"])):
        if p_fdr < 0.05:
            ax.text(
                delta,
                y,
                "*",
                ha="left" if delta >= 0 else "right",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )
            metric_parts = label.split(" – ", 1)
            significant_metrics.append((metric_parts[0], metric_parts[1]))

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_rq4_quality_impact.pdf", dpi=900)
    plt.close(fig)

    selected[
        [
            "metric_category",
            "metric_name",
            "observation_count",
            "median_delta",
            "improvement_rate",
            "worsening_rate",
            "wilcoxon_p_value_fdr",
        ]
    ].to_latex(
        TABLE_DIR / "table_rq4_quality_impact.tex",
        float_format="%.3f",
        index=False,
        caption="Overall quality impact metrics with median delta and significance.",
        label="tab:rq4_quality",
    )

    return significant_metrics


# ---------------------------------------------------------------------------
# Figure 5 – SAR vs Non-SAR comparison
# ---------------------------------------------------------------------------
def figure_rq5(
    sar_summary: pd.DataFrame,
    sar_tests: pd.DataFrame,
    metrics_to_show: List[Tuple[str, str]],
) -> None:
    if not metrics_to_show:
        return
    target = sar_summary.merge(
        pd.DataFrame(metrics_to_show, columns=["metric_category", "metric_name"]),
        on=["metric_category", "metric_name"],
        how="inner",
    )
    pivot = target.pivot_table(
        index=["metric_category", "metric_name"],
        columns="is_self_affirmed",
        values="median_delta",
    ).reset_index()
    pivot = pivot.rename(columns={False: "Non-SAR", True: "SAR"})
    pivot["metric_label"] = pivot.apply(lambda row: f"{row['metric_category']} – {row['metric_name']}", axis=1)

    signif = sar_tests[
        (sar_tests["mannwhitney_p_value_fdr"] < 0.05)
        & sar_tests["metric_category"].isin(pivot["metric_category"])
        & sar_tests["metric_name"].isin(pivot["metric_name"])
    ][["metric_category", "metric_name"]]
    signif_set = { (row.metric_category, row.metric_name) for row in signif.itertuples(index=False) }

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    y_positions = np.arange(len(pivot))
    ax.axvline(0, color="black", linewidth=1)
    for idx, row in pivot.iterrows():
        non_sar = row["Non-SAR"]
        sar = row["SAR"]
        y = y_positions[idx]
        ax.plot([non_sar, sar], [y, y], color="#7f7f7f", linewidth=2 if (row["metric_category"], row["metric_name"]) in signif_set else 1.2)
        ax.scatter(non_sar, y, color=NON_SAR_COLOR, s=60, label="Non-SAR" if idx == 0 else "")
        ax.scatter(sar, y, color=SAR_COLOR, s=60, label="SAR" if idx == 0 else "")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot["metric_label"])
    ax.set_xlabel("Median Δ (after − before)")
    ax.set_ylabel("")
    ax.set_title("Quality impact comparison: SAR vs Non-SAR")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_rq5_sar_vs_non_sar.pdf", dpi=900)
    plt.close(fig)

    pivot.to_latex(
        TABLE_DIR / "table_rq5_sar_vs_non_sar.tex",
        float_format="%.3f",
        index=False,
        caption="Median quality metric deltas for SAR and Non-SAR commits.",
        label="tab:rq5_sar_vs_non_sar",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=20, help="Top N items for refactoring type ranking (default: 20).")
    parser.add_argument("--fig4-metrics", type=int, default=15, help="Maximum number of metrics to display in Figure 4 (default: 15).")
    args = parser.parse_args()

    _ensure_dirs()
    sns.set_theme(style="whitegrid")

    commits = _load_commits()
    refminer = _load_refminer()
    motivations = _load_motivations()
    baseline = pd.read_csv("data/analysis/designite/metric_analysis/metric_baseline_summary.csv")
    sar_summary = pd.read_csv("data/analysis/designite/metric_analysis/metric_by_sar_summary.csv")
    sar_tests = pd.read_csv("data/analysis/designite/metric_analysis/metric_by_sar_tests.csv")

    figure_rq1(commits, refminer)

    medians = compute_refactoring_medians(refminer, commits)
    figure_rq2(medians, args.top_n)

    figure_rq3(motivations, commits)

    significant_metrics = figure_rq4(baseline, args.fig4_metrics)
    figure_rq5(sar_summary, sar_tests, significant_metrics)

    print(f"Saved figures to {FIGURE_DIR}")
    print(f"Saved LaTeX tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
