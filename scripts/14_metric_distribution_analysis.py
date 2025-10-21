#!/usr/bin/env python3
"""Filter significant metrics, export summary table, and plot violin distributions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASELINE_CSV = Path("data/analysis/designite/metric_analysis/metric_baseline_summary.csv")
DELTA_FILES = {
    "design_smell": Path("data/analysis/designite/deltas/design_smell_deltas.parquet"),
    "implementation_smell": Path("data/analysis/designite/deltas/implementation_smell_deltas.parquet"),
    "type_metric": Path("data/analysis/designite/deltas/type_metric_deltas.parquet"),
    "method_metric": Path("data/analysis/designite/deltas/method_metric_deltas.parquet"),
}
REFACTORING_COMMITS = Path("data/analysis/refactoring_instances/refactoring_commits.csv")
CATEGORY_ORDER = ["implementation_smell", "design_smell", "type_metric", "method_metric"]
OUTPUT_DIR = Path("outputs/designite/metric_distributions")
CATEGORY_LABELS = {
    "implementation_smell": "Implementation Smell",
    "design_smell": "Design Smell",
    "type_metric": "Class-Level Metric",
    "method_metric": "Method-Level Metric",
}
METRIC_LABELS = {
    "LOC": "Lines of Code",
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
    "CC": "Cyclomatic Complexity",
    "PC": "Parameter Count",
}


def pretty_category(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


def pretty_metric(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " ").replace("-", " ").title())


def load_baseline() -> pd.DataFrame:
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline summary not found: {BASELINE_CSV}")
    df = pd.read_csv(BASELINE_CSV)
    return df


def filter_significant_metrics(
    baseline: pd.DataFrame,
    alpha: float,
    metric_key_order: List[tuple[str, str]],
    key_to_label: Dict[tuple[str, str], str],
) -> pd.DataFrame:
    significant = baseline[baseline["wilcoxon_p_value_fdr"] < alpha].copy()
    significant = significant[significant["metric_key"].isin(metric_key_order)]
    significant["metric_key"] = pd.Categorical(significant["metric_key"], categories=metric_key_order, ordered=True)
    significant = significant.sort_values("metric_key")
    significant["metric_label"] = significant["metric_key"].map(key_to_label)
    significant["metric_category"] = pd.Categorical(significant["metric_category"], categories=CATEGORY_ORDER, ordered=True)
    return significant.reset_index(drop=True)


def save_summary_table(significant: pd.DataFrame, path: Path, top_n: int | None = None) -> pd.DataFrame:
    cols = [
        "metric_category",
        "metric_name",
        "median_delta",
        "improvement_rate",
        "unchanged_rate",
        "rank_biserial_effect",
        "wilcoxon_p_value_fdr",
    ]
    table = significant.loc[:, cols].copy()
    if top_n is not None:
        table = table.head(top_n)
    table["metric_category"] = pd.Categorical(table["metric_category"], categories=CATEGORY_ORDER, ordered=True)
    table = table.sort_values(
        ["metric_category", "rank_biserial_effect"],
        ascending=[True, False],
        key=lambda s: np.abs(s) if s.name == "rank_biserial_effect" else s,
    )
    table.to_csv(path, index=False)
    display_table = table.copy()
    display_table["metric_category"] = display_table["metric_category"].map(pretty_category)
    display_table["metric_name"] = display_table["metric_name"].map(pretty_metric)
    for col in ("improvement_rate", "worsening_rate", "unchanged_rate"):
        if col in display_table.columns:
            display_table[col] = display_table[col] * 100.0
    display_table = display_table.rename(
        columns={
            "metric_category": "Metric Category",
            "metric_name": "Metric Name",
            "median_delta": "Median Δ",
            "improvement_rate": "Improvement Rate (%)",
            "unchanged_rate": "Unchanged Rate (%)",
            "worsening_rate": "Worsening Rate (%)",
            "rank_biserial_effect": "Rank-Biserial Effect",
            "wilcoxon_p_value_fdr": "Wilcoxon p (FDR)",
        }
    )
    display_table.to_latex(
        path.with_suffix(".tex"),
        index=False,
        float_format="%.4f",
        caption="Top significant metrics ranked by absolute rank-biserial effect.",
        label="tab:significant_metrics_baseline",
    )
    return table


def load_deltas(metric_category: str) -> pd.DataFrame:
    path = DELTA_FILES.get(metric_category)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Delta file not found for category '{metric_category}': {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["metric_category"] = metric_category
    return df


def collect_metric_deltas(
    significant_metrics: Iterable[tuple[str, str]],
    metric_key_order: List[tuple[str, str]],
    commit_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cat_to_metrics: Dict[str, List[str]] = {}
    for category, metric in significant_metrics:
        cat_to_metrics.setdefault(category, []).append(metric)

    frames: List[pd.DataFrame] = []
    for category, metrics in cat_to_metrics.items():
        df = load_deltas(category)
        subset = df[df["metric"].isin(metrics)].copy()
        subset = subset.rename(columns={"metric": "metric_name"})
        frames.append(subset[["commit_sha", "metric_category", "metric_name", "delta"]])

    if not frames:
        columns = ["commit_sha", "metric_category", "metric_name", "delta", "metric_key"]
        if commit_metadata is not None:
            columns.extend(commit_metadata.columns.difference(["commit_sha"]).tolist())
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    result["metric_key"] = list(zip(result["metric_category"], result["metric_name"]))
    result = result[result["metric_key"].isin(metric_key_order)]
    result["metric_key"] = pd.Categorical(result["metric_key"], categories=metric_key_order, ordered=True)
    if commit_metadata is not None:
        result = result.merge(commit_metadata, on="commit_sha", how="left")
    return result


def plot_violins(
    deltas: pd.DataFrame,
    output_path: Path,
    metric_key_order: List[tuple[str, str]],
    key_to_label: Dict[tuple[str, str], str],
    title: str | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    if deltas.empty:
        return
    sns.set_theme(style="whitegrid")
    deltas = deltas.copy()
    deltas["metric_key"] = pd.Categorical(deltas["metric_key"], categories=metric_key_order, ordered=True)
    deltas = deltas.dropna(subset=["metric_key"])
    deltas = deltas.sort_values("metric_key")
    deltas["metric_label"] = deltas["metric_key"].map(key_to_label)
    order = [key_to_label[key] for key in metric_key_order if key in deltas["metric_key"].cat.categories and (deltas["metric_key"] == key).any()]
    plt.figure(figsize=(12, max(5, len(order) * 0.3)))

    ax = sns.violinplot(
        data=deltas,
        y="metric_label",
        x="delta",
        order=order,
        inner="quartile",
        cut=0,
        linewidth=1,
        density_norm="width",
        hue=None,
        color="#9ecae1",
    )

    plt.axvline(0, color="black", linewidth=1)
    ax.set_xscale("symlog", linthresh=3)
    if x_limits is not None:
        plt.xlim(x_limits)
    else:
        max_abs = np.nanmax(np.abs(deltas["delta"].to_numpy()))
        if np.isfinite(max_abs) and max_abs > 0:
            limit = max_abs * 1.05
            plt.xlim(-limit, limit)
    plt.xlabel("Δ (after − before)")
    plt.ylabel("")
    plt.title(title or "Delta distributions for significant metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=900)
    plt.close()


def plot_rate_breakdown(
    significant: pd.DataFrame,
    output_path: Path,
    metric_key_order: List[tuple[str, str]],
    key_to_label: Dict[tuple[str, str], str],
    top_n: int,
    title: str | None = None,
) -> None:
    if significant.empty:
        return
    chart = significant.copy()
    chart["metric_key"] = list(zip(chart["metric_category"], chart["metric_name"]))
    chart["abs_effect"] = chart["rank_biserial_effect"].abs()
    chart = chart.sort_values("abs_effect", ascending=False)
    if top_n is not None:
        selected_keys = chart.head(top_n)["metric_key"].tolist()
    else:
        selected_keys = chart["metric_key"].tolist()
    ordered_keys = [key for key in metric_key_order if key in selected_keys]
    if not ordered_keys:
        return
    chart = chart.set_index("metric_key").loc[ordered_keys].reset_index()
    chart["metric_label"] = chart["metric_key"].map(key_to_label)
    chart["improvement_rate"] = chart["improvement_rate"] * 100.0
    chart["unchanged_rate"] = chart["unchanged_rate"] * 100.0
    chart["worsening_rate"] = chart["worsening_rate"] * 100.0

    plt.figure(figsize=(12, max(3, len(chart) * 0.4)))
    y = np.arange(len(chart))
    width = 0.8
    plt.barh(y, chart["unchanged_rate"], width, label="Unchanged", color="#bfbfbf")
    plt.barh(y, chart["improvement_rate"], width, left=chart["unchanged_rate"], label="Improved", color="#1f77b4")
    plt.barh(
        y,
        chart["worsening_rate"],
        width,
        left=chart["unchanged_rate"] + chart["improvement_rate"],
        label="Worsened",
        color="#d62728",
    )
    totals = chart["unchanged_rate"] + chart["improvement_rate"] + chart["worsening_rate"]
    ax = plt.gca()
    max_total = float(np.nanmax(totals)) if not totals.empty else 100.0
    if not np.isfinite(max_total) or max_total <= 0:
        max_total = 100.0
    ax.set_xlim(0, max_total)
    ax.margins(x=0)
    plt.yticks(y, chart["metric_label"])
    plt.xlabel("Rate (%)")
    plt.ylabel("")
    plt.title(title or "Improvement / unchanged / worsening rates for significant metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=900)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR-adjusted p-value threshold (default: 0.05).")
    parser.add_argument("--top-n-table", type=int, default=None, help="Optional limit for summary table rows.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline()
    baseline["metric_key"] = list(zip(baseline["metric_category"], baseline["metric_name"]))
    baseline["metric_label"] = baseline["metric_key"].apply(lambda key: f"{pretty_category(key[0])} – {pretty_metric(key[1])}")
    baseline["category_order"] = baseline["metric_category"].map(lambda cat: CATEGORY_ORDER.index(cat))
    baseline = baseline.sort_values(
        ["category_order", "rank_biserial_effect"],
        ascending=[True, False],
        key=lambda s: np.abs(s) if s.name == "rank_biserial_effect" else s,
    ).reset_index(drop=True)

    metric_key_order = baseline["metric_key"].tolist()
    key_to_label = dict(zip(baseline["metric_key"], baseline["metric_label"]))

    significant = filter_significant_metrics(baseline, alpha=args.alpha, metric_key_order=metric_key_order, key_to_label=key_to_label)
    if significant.empty:
        print("No metrics passed the significance threshold.")
        return

    significant["unchanged_rate"] = 1.0 - (significant["improvement_rate"] + significant["worsening_rate"])

    summary_csv = OUTPUT_DIR / "significant_metrics_summary.csv"
    save_summary_table(significant, summary_csv, top_n=args.top_n_table)

    pairs = list(zip(significant["metric_category"], significant["metric_name"]))
    deltas = collect_metric_deltas(pairs, metric_key_order)
    default_top = args.top_n_table if args.top_n_table is not None else 15
    plot_rate_breakdown(
        significant,
        OUTPUT_DIR / "significant_metric_change_breakdown.pdf",
        metric_key_order,
        key_to_label,
        top_n=default_top,
    )

    sar_summary_path = BASELINE_CSV.parent / "metric_by_sar_summary.csv"
    sar_tests_path = BASELINE_CSV.parent / "metric_by_sar_tests.csv"
    sar_summary = pd.DataFrame()
    commit_labels = pd.DataFrame()
    sar_deltas = pd.DataFrame()
    if sar_summary_path.exists() and sar_tests_path.exists() and REFACTORING_COMMITS.exists():
        sar_summary = pd.read_csv(sar_summary_path)
        sar_tests = pd.read_csv(sar_tests_path)
        sar_tests = sar_tests.rename(columns={"mannwhitney_p_value_fdr": "wilcoxon_p_value_fdr"})
        sar_summary = sar_summary.merge(
            sar_tests[["metric_category", "metric_name", "rank_biserial_effect", "wilcoxon_p_value_fdr"]],
            on=["metric_category", "metric_name"],
            how="left",
        )
        sar_summary["wilcoxon_p_value_fdr"] = sar_summary["wilcoxon_p_value_fdr"].fillna(1.0)
        sar_summary["metric_key"] = list(zip(sar_summary["metric_category"], sar_summary["metric_name"]))
        commit_labels = pd.read_csv(REFACTORING_COMMITS, usecols=["sha", "is_self_affirmed"]).rename(columns={"sha": "commit_sha"})
    else:
        missing = []
        if not sar_summary_path.exists():
            missing.append(str(sar_summary_path))
        if not sar_tests_path.exists():
            missing.append(str(sar_tests_path))
        if not REFACTORING_COMMITS.exists():
            missing.append(str(REFACTORING_COMMITS))
        if missing:
            print(f"Skipping SAR-specific summaries; missing required files: {', '.join(missing)}")

    if not sar_summary.empty and not commit_labels.empty:
        selected_keys = set(pairs)
        sar_significant = sar_summary[
            (sar_summary["is_self_affirmed"] == True) & sar_summary["metric_key"].isin(selected_keys)
        ].copy()
        if sar_significant.empty:
            print("No SAR entries matched the baseline significant metrics; skipping SAR-specific graphs.")
        else:
            sar_significant["unchanged_rate"] = 1.0 - (sar_significant["improvement_rate"] + sar_significant["worsening_rate"])
            sar_significant["metric_key"] = list(zip(sar_significant["metric_category"], sar_significant["metric_name"]))
            sar_significant["metric_key"] = pd.Categorical(sar_significant["metric_key"], categories=metric_key_order, ordered=True)
            sar_significant = sar_significant.sort_values("metric_key")

            sar_summary_csv = OUTPUT_DIR / "sar_significant_metrics_summary.csv"
            save_summary_table(sar_significant, sar_summary_csv, top_n=args.top_n_table)

            sar_pairs = list(zip(sar_significant["metric_category"], sar_significant["metric_name"]))
            sar_deltas = collect_metric_deltas(sar_pairs, metric_key_order, commit_metadata=commit_labels)
            sar_deltas = sar_deltas[sar_deltas["is_self_affirmed"] == True].copy()
            plot_rate_breakdown(
                sar_significant,
                OUTPUT_DIR / "sar_significant_metric_change_breakdown.pdf",
                metric_key_order,
                key_to_label,
                top_n=default_top,
                title="Improvement / unchanged / worsening rates for significant metrics (SAR only)",
            )

    shared_xlim: tuple[float, float] | None = None
    overall_max = 0.0
    if not deltas.empty:
        overall_max = float(np.nanmax(np.abs(deltas["delta"].to_numpy())))
    sar_max = 0.0
    if not sar_deltas.empty:
        sar_max = float(np.nanmax(np.abs(sar_deltas["delta"].to_numpy())))
    max_abs = max(overall_max, sar_max)
    if np.isfinite(max_abs) and max_abs > 0:
        limit = max_abs * 1.05
        shared_xlim = (-limit, limit)

    plot_violins(
        deltas,
        OUTPUT_DIR / "significant_metric_violins.pdf",
        metric_key_order,
        key_to_label,
        x_limits=shared_xlim,
    )

    if not sar_deltas.empty:
        plot_violins(
            sar_deltas,
            OUTPUT_DIR / "sar_significant_metric_violins.pdf",
            metric_key_order,
            key_to_label,
            title="Delta distributions for significant metrics (SAR only)",
            x_limits=shared_xlim,
        )

    print(f"Wrote summary to {summary_csv} (and .tex)")
    print(f"Wrote violin plot to {OUTPUT_DIR/'significant_metric_violins.pdf'}")


if __name__ == "__main__":
    main()
