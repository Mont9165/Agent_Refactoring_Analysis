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
from src.research_questions.rq5_quality_impact import rq5_quality_impact


def _plot_rq3_distribution(
    csv_path: Path,
    top_types: Iterable[str],
    *,
    suffix: str = "",
    color: str = "#5DA5DA",
    subset_label: str = "Overall",
    filter_non_sar: bool = False,
    subset_folder: str = "overall",
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
    filtered = df[df["instance_count"] > 0].copy()
    label_lower = subset_label.lower()
    if "sar" in label_lower and "non" not in label_lower:
        filtered = filtered[filtered.get("is_self_affirmed") == True]
    elif filter_non_sar or "non-sar" in label_lower:
        filtered = filtered[filtered.get("is_self_affirmed") == False]
    if filtered.empty:
        return None

    grouped = filtered.groupby("refactoring_type")["instance_count"]
    medians = grouped.median()
    means = grouped.mean()
    order = [t for t in medians.sort_values(ascending=False).index if t in top_types]
    if not order:
        return None

    data_by_type = []
    labels: list[str] = []
    for ref_type in order:
        series = filtered.loc[filtered["refactoring_type"] == ref_type, "instance_count"]
        if series.empty:
            continue
        data_by_type.append(series.values)
        labels.append(ref_type)

    if not data_by_type:
        return None

    positions = list(range(1, len(labels) + 1))
    suffix_parts = []
    if "sar" in label_lower and "non" not in label_lower:
        suffix_parts.append("sar")
    elif filter_non_sar or "non-sar" in label_lower:
        suffix_parts.append("non_sar")
    if suffix:
        suffix_parts.append(suffix)
    name_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    subset_dir = csv_path.parent / subset_folder
    subset_dir.mkdir(parents=True, exist_ok=True)
    output_path = subset_dir / f"rq3_refactoring_type_distribution{name_suffix}.pdf"

    width = max(12, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(width, 6))

    parts = ax.violinplot(
        data_by_type,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    edge_rgba = mcolors.to_rgba(color, alpha=0.85)
    for body in parts["bodies"]:
        body.set_facecolor(mcolors.to_rgba(color, alpha=0.35))
        body.set_edgecolor(edge_rgba)
        body.set_alpha(0.6)

    ax.boxplot(
        data_by_type,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor=edge_rgba, linewidth=1.4),
        whiskerprops=dict(color=edge_rgba, linewidth=1.2),
        capprops=dict(color=edge_rgba, linewidth=1.2),
        medianprops=dict(color="white", linewidth=2.3),
    )

    for pos, series in zip(positions, data_by_type):
        mean = pd.Series(series).mean()
        ax.scatter(
            pos,
            mean,
            marker="^",
            color=edge_rgba,
            s=40,
            zorder=3,
            edgecolors="white",
            linewidths=1.0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=14)
    ax.set_yscale("log")
    ax.set_ylabel("Instances per commit", fontsize=16)
    ax.set_xlabel("Refactoring type", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(f"Refactoring instances per commit ({subset_label})")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=900)
    plt.close(fig)
    return output_path


def _plot_rq3_totals(
    csv_path: Path,
    *,
    sar_only: bool = False,
    exclude_sar: bool = False,
    top_n: Optional[int] = None,
    suffix: str = "",
    color: str = "#4C72B0",
    title_label: str = "Overall",
    subset_folder: str = "overall",
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
    elif exclude_sar:
        df = df[df.get("is_self_affirmed") == False]
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
    elif exclude_sar:
        suffix_parts.append("non_sar")
    if suffix:
        suffix_parts.append(suffix)
    suffix_str = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    subset_dir = csv_path.parent / subset_folder
    subset_dir.mkdir(parents=True, exist_ok=True)
    output_path = subset_dir / f"rq3_refactoring_type_totals{suffix_str}.pdf"

    height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(12, height))
    bar_color = mcolors.to_rgba(color, alpha=0.85)
    edge_color = mcolors.to_rgba(color, alpha=1.0)
    ax.barh(labels[::-1], values[::-1], color=bar_color, edgecolor=edge_color)

    ax.set_xlabel("Total refactoring instances", fontsize=16)
    ax.set_ylabel("Refactoring type", fontsize=16)
    ax.set_title(f"Total refactoring instances per type ({title_label})")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
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
    subset_label: str = "overall",
    target_dir: Optional[Path] = None,
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
    target_dir = target_dir or (OUTPUT_DIR / "rq4" / (subset_label or "overall"))
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"rq4_purpose_distribution{suffix_str}.pdf"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    print("=" * 60)
    print("         RESEARCH QUESTIONS (RQ1–RQ5)")
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
            non_sar_plot_path = _plot_rq3_distribution(
                Path(rq3["distribution_csv"]),
                order,
                color="#2CA02C",
                subset_label="Non-SAR",
                filter_non_sar=True,
            )
            if non_sar_plot_path:
                print(f"  Saved non-SAR violin/box plot: {non_sar_plot_path}")
            non_sar_plot_top20 = None
            if len(order) > 20:
                non_sar_plot_top20 = _plot_rq3_distribution(
                    Path(rq3["distribution_csv"]),
                    order[:20],
                    suffix="top20",
                    color="#137B3E",
                    subset_label="Non-SAR (Top 20)",
                    filter_non_sar=True,
                )
                if non_sar_plot_top20:
                    print(f"  Saved non-SAR Top20 violin/box plot: {non_sar_plot_top20}")
            non_sar_totals = _plot_rq3_totals(
                Path(rq3["distribution_csv"]),
                sar_only=False,
                exclude_sar=True,
                color="#2CA02C",
                title_label="Non-SAR",
            )
            if non_sar_totals:
                print(f"  Saved non-SAR totals bar chart: {non_sar_totals}")
            non_sar_totals_top20 = _plot_rq3_totals(
                Path(rq3["distribution_csv"]),
                sar_only=False,
                exclude_sar=True,
                top_n=20,
                suffix="top20",
                color="#137B3E",
                title_label="Non-SAR (Top 20)",
            )
            if non_sar_totals_top20:
                print(f"  Saved non-SAR Top20 bar chart: {non_sar_totals_top20}")
            if sar_types:
                sar_plot_path = _plot_rq3_distribution(
                    Path(rq3["distribution_csv"]),
                    rq3.get("sar_order_median_desc") or sar_types.keys(),
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
                        suffix="top20",
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

    print("\nRQ4: Refactoring purposes in agentic commits")
    agentic_commits = commits
    if "agent" in commits.columns:
        agentic_commits = commits[commits["agent"].notna()]
    if "is_self_affirmed" in agentic_commits.columns:
        non_sar_commits = agentic_commits[agentic_commits["is_self_affirmed"] == False]
        sar_commits = agentic_commits[agentic_commits["is_self_affirmed"] == True]
    else:
        empty_subset = agentic_commits.iloc[0:0]
        non_sar_commits = empty_subset
        sar_commits = empty_subset

    rq4_groups = [
        ("Overall", "overall", agentic_commits, True, "#4C72B0", "#2C3E50"),
        ("Non-SAR", "non_sar", non_sar_commits, False, "#2CA02C", "#137B3E"),
        ("SAR", "sar", sar_commits, False, "#E4572E", "#C13C1E"),
    ]

    for title, subset_label, subset_df, agentic_flag, color_full, color_top in rq4_groups:
        rq4_result = rq4_refactoring_purpose(
            subset_df,
            agentic_only=agentic_flag,
            subset_label=subset_label,
        )
        total_rq4 = int(rq4_result.get("total_refactoring_commits", 0))
        print(f"  [{title}] Refactoring commits analysed: {total_rq4}")
        distribution = rq4_result.get("purpose_distribution", {})
        if distribution:
            sorted_purposes = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
            limit = min(10, len(sorted_purposes))
            print("    Top purpose labels:")
            for label, count in sorted_purposes[:limit]:
                pct = (count / total_rq4 * 100.0) if total_rq4 else 0.0
                print(f"      {label}: {count} ({pct:.1f}%)")
        else:
            print("    No purpose labels identified.")
        examples_file = rq4_result.get("examples_file")
        if examples_file:
            print(f"    Examples CSV: {examples_file}")
        note = rq4_result.get("note")
        if note and (not distribution or total_rq4 == 0):
            print(f"    Note: {note}")
        if distribution and total_rq4:
            target_dir = rq4_result.get("output_dir")
            base_dir = Path(target_dir) if target_dir else None
            full_chart = _plot_rq4_purposes(
                distribution,
                total_rq4,
                color=color_full,
                subset_label=subset_label,
                target_dir=base_dir,
            )
            if full_chart:
                print(f"    Saved RQ4 purpose chart: {full_chart}")
            top_chart = _plot_rq4_purposes(
                distribution,
                total_rq4,
                suffix="top10",
                color=color_top,
                top_n=10,
                subset_label=subset_label,
                target_dir=base_dir,
            )
            if top_chart:
                print(f"    Saved RQ4 Top10 purpose chart: {top_chart}")

    print("\nRQ5: Quality impact of refactorings (Designite & Readability)…")
    rq5 = rq5_quality_impact()
    quality_json = OUTPUT_DIR / "rq5_quality" / "rq5_quality_impact.json"
    print(f"  Summary JSON: {quality_json}")
    if rq5.get("wilcoxon_available"):
        print("  Wilcoxon signed-rank test available (scipy installed).")
    else:
        print("  Wilcoxon test unavailable (scipy not installed).")

    designite_summary = rq5.get("designite", {})
    if designite_summary.get("status") == "ok":
        metrics = designite_summary.get("metrics", {})
        print(f"  Designite metrics analysed: {len(metrics)}")
        sample_metrics = list(metrics.items())[:3]
        for metric_name, metric_data in sample_metrics:
            by_type = metric_data.get("by_refactoring_type", {})
            if not by_type:
                continue

            def _median(entry: Dict[str, object]) -> float:
                stats = entry.get("stats", {})
                return float(stats.get("median", 0.0)) if isinstance(stats, dict) else 0.0

            top_type, top_payload = max(
                by_type.items(),
                key=lambda item: abs(_median(item[1])),
            )
            stats = top_payload.get("stats", {})
            median = stats.get("median")
            count = stats.get("count")
            if isinstance(median, (int, float)) and isinstance(count, (int, float)):
                print(f"    {metric_name}: strongest median delta {median:.3f} from {top_type} (n={count})")
            else:
                print(f"    {metric_name}: analysed {len(by_type)} refactoring types")
    elif designite_summary.get("status") == "missing":
        print("  Designite delta files not found. Run scripts/6b_compute_designite_deltas.py first.")
    elif designite_summary.get("status") == "empty":
        print("  Designite delta files contained no usable rows (all NaN deltas).")

    readability_summary = rq5.get("readability", {})
    if readability_summary.get("status") == "ok":
        ref_types = readability_summary.get("by_refactoring_type", {})
        print(f"  Readability deltas analysed: {len(ref_types)} refactoring types")
        if ref_types:
            def _readability_median(entry: Dict[str, object]) -> float:
                stats = entry.get("stats", {})
                return float(stats.get("median", 0.0)) if isinstance(stats, dict) else 0.0

            top_type, payload = max(
                ref_types.items(),
                key=lambda item: abs(_readability_median(item[1])),
            )
            stats = payload.get("stats", {})
            median = stats.get("median")
            count = stats.get("count")
            if isinstance(median, (int, float)) and isinstance(count, (int, float)):
                print(f"    Largest median readability delta {median:.3f} from {top_type} (n={count})")
    elif readability_summary.get("status") == "missing":
        print("  Readability delta file not found. Run scripts/6c_readability_impact.py first.")
    elif readability_summary.get("status") == "empty":
        print("  Readability delta file contained no usable rows.")

    print("\nDone. Review generated CSV/JSON files under outputs/research_questions/ for details.")


if __name__ == "__main__":
    main()
