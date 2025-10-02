#!/usr/bin/env python3
"""Compute Research Questions RQ1–RQ4 using Phase 3 outputs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

# Add project root to sys.path when executed as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.research_questions.rq_common import OUTPUT_DIR, load_phase3_outputs
from src.research_questions.rq1_refactoring_instances import rq1_refactoring_instances_agentic
from src.research_questions.rq2_self_affirmed import rq2_self_affirmed_percentage
from src.research_questions.rq3_refactoring_types import rq3_top_refactoring_types
from src.research_questions.rq4_refactoring_purpose import rq4_refactoring_purpose


def _plot_rq3_distribution(
    csv_path: Path,
    top_types: Iterable[str],
    *,
    suffix: str = "",
    color: str = "#5DA5DA",
    subset_label: str = "Overall",
) -> Optional[Path]:
    """Create violin-with-box plots for per-commit refactoring counts."""
    top_types = list(top_types)
    if not csv_path.exists() or not top_types:
        return None

    try:
        import os
        import pandas as pd

        # Ensure matplotlib can build its cache inside the workspace sandbox
        mpl_cache = csv_path.parent / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")  # Safe in headless environments
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"  Skipping RQ3 plot (dependency error: {exc})")
        return None

    df = pd.read_csv(csv_path)
    filtered = df[df["refactoring_type"].isin(top_types)].copy()
    filtered = filtered[filtered["instance_count"] > 0]
    if filtered.empty:
        return None

    data_by_type = []
    labels: list[str] = []
    for ref_type in top_types:
        series = filtered.loc[filtered["refactoring_type"] == ref_type, "instance_count"]
        if series.empty:
            continue
        data_by_type.append(series.values)
        labels.append(ref_type)

    if not data_by_type:
        return None

    positions = list(range(1, len(labels) + 1))
    name_suffix = f"_{suffix}" if suffix else ""
    output_path = csv_path.with_name(f"rq3_refactoring_type_distribution{name_suffix}.png")

    width = max(12, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(width, 6))

    parts = ax.violinplot(
        data_by_type,
        positions=positions,
        showmeans=False,
        showmedians=False,
        widths=0.9,
    )
    for body in parts["bodies"]:
        rgba = mcolors.to_rgba(color, alpha=0.6)
        edge_rgba = mcolors.to_rgba(color, alpha=0.9)
        body.set_facecolor(rgba)
        body.set_edgecolor(edge_rgba)
        body.set_alpha(0.6)

    medianprops = dict(color=mcolors.to_rgba(color, alpha=0.9), linewidth=2.5)
    ax.boxplot(
        data_by_type,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        medianprops=medianprops,
        showcaps=True,
        whiskerprops={"linewidth": 1.5, "color": edge_rgba},
        capprops={"linewidth": 1.3, "color": edge_rgba},
        boxprops={"facecolor": "white", "alpha": 0.9, "linewidth": 1.2, "edgecolor": edge_rgba},
        showfliers=False,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Instances per commit (log)")
    ax.set_xlabel("Refactoring type")
    ax.set_title(f"Refactoring instances per commit ({subset_label})")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_rq3_totals(
    csv_path: Path,
    *,
    sar_only: bool = False,
    top_n: Optional[int] = None,
    suffix: str = "",
    color: str = "#4C72B0",
    title_label: str = "Overall",
) -> Optional[Path]:
    """Render total refactoring instances per type as a bar chart."""
    try:
        import os
        import pandas as pd

        mpl_cache = csv_path.parent / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping totals plot ({title_label}) due to dependency error: {exc}")
        return None

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if sar_only:
        df = df[df.get("is_self_affirmed") == True]
    aggregated = df.groupby("refactoring_type")["instance_count"].sum().sort_values(ascending=False)
    if aggregated.empty:
        return None

    if top_n is not None:
        aggregated = aggregated.head(top_n)

    labels = list(aggregated.index)
    values = aggregated.values

    suffix_parts = []
    if sar_only:
        suffix_parts.append("sar")
    if suffix:
        suffix_parts.append(suffix)
    suffix_str = "" if not suffix_parts else "_" + "_".join(suffix_parts)
    output_path = csv_path.with_name(f"rq3_refactoring_type_totals{suffix_str}.png")

    height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(12, height))
    bar_color = mcolors.to_rgba(color, alpha=0.85)
    edge_color = mcolors.to_rgba(color, alpha=1.0)
    ax.barh(labels[::-1], values[::-1], color=bar_color, edgecolor=edge_color)

    ax.set_xlabel("Total refactoring instances")
    ax.set_ylabel("Refactoring type")
    ax.set_title(f"Total refactoring instances per type ({title_label})")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_rq4_purposes(
    distribution: Dict[str, int],
    total_count: int,
    *,
    suffix: str = "",
    color: str = "#4C72B0",
    top_n: Optional[int] = None,
) -> Optional[Path]:
    """Render a bar chart for RQ4 purpose counts."""
    if not distribution or total_count <= 0:
        return None

    try:
        import os
        import matplotlib

        mpl_cache = OUTPUT_DIR / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping RQ4 plot due to dependency error: {exc}")
        return None

    items = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]
    if not items:
        return None

    labels, counts = zip(*items)
    percentages = [(c / total_count) * 100.0 for c in counts]
    height = max(4, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(9, height))
    bar_color = mcolors.to_rgba(color, alpha=0.85)
    edge_color = mcolors.to_rgba(color, alpha=1.0)
    ax.barh(labels[::-1], percentages[::-1], color=bar_color, edgecolor=edge_color)
    ax.set_xlabel("Share of analysed commits (%)")
    ax.set_ylabel("Purpose label")
    title_suffix = " (Top {})".format(top_n) if top_n else ""
    ax.set_title(f"GPT motivation categories{title_suffix}")
    max_pct = max(percentages)
    ax.set_xlim(0, (max_pct * 1.15) if max_pct > 0 else 1)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    output_path = OUTPUT_DIR / f"rq4_purpose_distribution{suffix_str}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    print("=" * 60)
    print("         RESEARCH QUESTIONS (RQ1–RQ4)")
    print("=" * 60)

    data = load_phase3_outputs()
    commits = data["commits"]
    refminer = data["refminer"]

    if commits is None:
        print("ERROR: Missing Phase 3 outputs. Run scripts/3_detect_refactoring.py first.")
        sys.exit(1)

    print("\nRQ1: Counting refactoring instances in agentic PRs…")
    rq1 = rq1_refactoring_instances_agentic(commits, refminer)
    print(f"  Agentic commits: {rq1['total_agentic_commits']}")
    print(f"  Agentic refactoring commits: {rq1['agentic_refactoring_commits']}")
    if rq1["refminer_available"]:
        print(f"  Agentic refactoring instances: {rq1['agentic_refactoring_instances']}")
    else:
        print("  RefactoringMiner instances unavailable (no RM results).")

    print("\nRQ2: Self-affirmed refactoring percentage in agentic commits…")
    rq2 = rq2_self_affirmed_percentage(commits)
    print(f"  Agentic refactoring commits: {rq2['agentic_refactoring_commits']}")
    print(f"  Self-affirmed commits: {rq2['self_affirmed_commits']}")
    print(f"  Self-affirmed percentage: {rq2['self_affirmed_percentage']:.2f}%")

    print("\nRQ3: Most common refactoring types (overall vs SAR)…")
    rq3 = rq3_top_refactoring_types(refminer, commits, top_n=None, min_count=2)
    if "overall_top_types" in rq3:
        order = rq3.get("overall_order_median_desc") or list(rq3["overall_top_types"].keys())
        distribution = rq3.get("overall_distribution", {})
        sample_counts = rq3.get("overall_sample_counts", {})
        print("  Types sorted by median instances/commit:")
        for ref_type in order:
            count = rq3["overall_top_types"].get(ref_type, 0)
            median = distribution.get(ref_type, {}).get("median")
            commit_count = sample_counts.get(ref_type)
            sample_str = f"n={commit_count}" if isinstance(commit_count, (int, float)) else "n=?"
            median_str = f"median={median:.2f}" if isinstance(median, (int, float)) else "median=n/a"
            print(f"    {ref_type}: total={count}, {median_str}, {sample_str}")
        sar_types = rq3.get("sar_top_types")
        if sar_types:
            sar_order = rq3.get("sar_order_median_desc") or list(sar_types.keys())
            sar_distribution = rq3.get("sar_distribution", {})
            sar_sample_counts = rq3.get("sar_sample_counts", {})
            print("  SAR types sorted by median instances/commit:")
            for ref_type in sar_order:
                count = sar_types.get(ref_type, 0)
                median = sar_distribution.get(ref_type, {}).get("median")
                commit_count = sar_sample_counts.get(ref_type)
                sample_str = f"n={commit_count}" if isinstance(commit_count, (int, float)) else "n=?"
                median_str = f"median={median:.2f}" if isinstance(median, (int, float)) else "median=n/a"
                print(f"    {ref_type}: total={count}, {median_str}, {sample_str}")
        else:
            print("  SAR-only list unavailable (no SAR flags or no RM results).")
        if rq3.get("distribution_csv"):
            print(f"  Distribution CSV for plotting: {rq3['distribution_csv']}")
            plot_path = _plot_rq3_distribution(
                Path(rq3["distribution_csv"]),
                order,
                color="#5DA5DA",
                subset_label="Overall",
            )
            if plot_path:
                print(f"  Saved violin/box plot: {plot_path}")
            if len(order) > 20:
                narrowed_path = _plot_rq3_distribution(
                    Path(rq3["distribution_csv"]),
                    order[:20],
                    suffix="top20",
                    color="#1F77B4",
                    subset_label="Overall (Top 20)",
                )
                if narrowed_path:
                    print(f"  Saved Top20 violin/box plot: {narrowed_path}")
            totals_path = _plot_rq3_totals(
                Path(rq3["distribution_csv"]),
                sar_only=False,
                color="#4C72B0",
                title_label="Overall",
            )
            if totals_path:
                print(f"  Saved totals bar chart: {totals_path}")
            totals_top20 = _plot_rq3_totals(
                Path(rq3["distribution_csv"]),
                sar_only=False,
                top_n=20,
                suffix="top20",
                color="#2451A4",
                title_label="Overall (Top 20)",
            )
            if totals_top20:
                print(f"  Saved totals Top20 bar chart: {totals_top20}")
            if sar_types:
                sar_plot_path = _plot_rq3_distribution(
                    Path(rq3["distribution_csv"]),
                    rq3.get("sar_order_median_desc") or sar_types.keys(),
                    suffix="sar",
                    color="#F15854",
                    subset_label="SAR",
                )
                if sar_plot_path:
                    print(f"  Saved SAR violin/box plot: {sar_plot_path}")
                sar_order_full = rq3.get("sar_order_median_desc") or list(sar_types.keys())
                if sar_plot_path and len(sar_order_full) > 20:
                    sar_top20_path = _plot_rq3_distribution(
                        Path(rq3["distribution_csv"]),
                        sar_order_full[:20],
                        suffix="sar_top20",
                        color="#C1392B",
                        subset_label="SAR (Top 20)",
                    )
                    if sar_top20_path:
                        print(f"  Saved SAR Top20 violin/box plot: {sar_top20_path}")
                sar_totals = _plot_rq3_totals(
                    Path(rq3["distribution_csv"]),
                    sar_only=True,
                    color="#F15854",
                    title_label="SAR",
                )
                if sar_totals:
                    print(f"  Saved SAR totals bar chart: {sar_totals}")
                sar_totals_top20 = _plot_rq3_totals(
                    Path(rq3["distribution_csv"]),
                    sar_only=True,
                    top_n=20,
                    suffix="top20",
                    color="#E4572E",
                    title_label="SAR (Top 20)",
                )
                if sar_totals_top20:
                    print(f"  Saved SAR totals Top20 bar chart: {sar_totals_top20}")
    else:
        print(f"  {rq3.get('error', 'No data')}")

    print("\nRQ4: Refactoring purposes in agentic commits (heuristic)…")
    rq4 = rq4_refactoring_purpose(commits, agentic_only=True)
    total_rq4 = rq4.get("total_refactoring_commits", 0)
    print(f"  Refactoring commits analysed: {total_rq4}")
    distribution = rq4.get("purpose_distribution", {})
    if distribution:
        sorted_purposes = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        limit = min(10, len(sorted_purposes))
        print("  Top purpose labels:")
        for label, count in sorted_purposes[:limit]:
            pct = (count / total_rq4 * 100.0) if total_rq4 else 0.0
            print(f"    {label}: {count} ({pct:.1f}%)")
    else:
        print("  No purpose labels identified.")
    examples_file = rq4.get("examples_file")
    if examples_file:
        print(f"  Examples CSV: {examples_file}")
    note = rq4.get("note")
    if note:
        print(f"  Note: {note}")
    full_chart = _plot_rq4_purposes(distribution, total_rq4, color="#4C72B0")
    if full_chart:
        print(f"  Saved RQ4 purpose chart: {full_chart}")
    top_chart = _plot_rq4_purposes(distribution, total_rq4, suffix="top10", color="#2C3E50", top_n=10)
    if top_chart:
        print(f"  Saved RQ4 Top10 purpose chart: {top_chart}")

    print("\nDone. See outputs in outputs/research_questions/.")


if __name__ == "__main__":
    main()
