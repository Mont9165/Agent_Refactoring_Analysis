#!/usr/bin/env python3
"""Compute Research Questions RQ1–RQ4 using Phase 3 outputs."""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency used for statistical testing
    from scipy.stats import wilcoxon as wilcoxon_signed_rank
except Exception:  # pragma: no cover - scipy might be unavailable
    wilcoxon_signed_rank = None

_RAW_RQ4_HUMAN_PURPOSE_PERCENTAGES: Dict[str, float] = {
    "maintainability": 11.2,
    "readability": 24.6,
    "testability": 9.6,
    "repurpose_reuse": 12.3,
    "dependency": 4.2,
    "legacy_code": 10.3,
    "hard_to_debug": 2.3,
    "slow_performance": 8.1,
    "duplication": 13.1,
}
_HUMAN_TOTAL = sum(_RAW_RQ4_HUMAN_PURPOSE_PERCENTAGES.values()) or 1.0
RQ4_HUMAN_PURPOSE_PERCENTAGES: Dict[str, float] = {
    key: (value / _HUMAN_TOTAL) * 100.0 for key, value in _RAW_RQ4_HUMAN_PURPOSE_PERCENTAGES.items()
}

RQ4_PURPOSE_DISPLAY_LABELS: Dict[str, str] = {
    "maintainability": "Maintainability",
    "readability": "Readability",
    "logical_mismatch": "Logical Mismatch",
    "testability": "Testability",
    "repurpose_reuse": "Repurpose/Reuse",
    "dependency": "Dependency",
    "legacy_code": "Legacy Code",
    "hard_to_debug": "Hard to Debug",
    "slow_performance": "Performance",
    "duplication": "Duplication",
}
# Add project root to sys.path when executed as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.research_questions.rq_common import OUTPUT_DIR, load_phase3_outputs
from src.research_questions.rq1_refactoring_instances import rq1_refactoring_instances_agentic
from src.research_questions.rq2_self_affirmed import (
    rq2_refactoring_type_affirmed_split,
    rq2_self_affirmed_percentage,
)
from src.research_questions.rq3_refactoring_types import rq3_top_refactoring_types
from src.research_questions.refactoring_classification import (
    LEVEL_DISPLAY_ORDER,
    LEVEL_NAME_BY_KEY,
    classification_key,
)
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


def _plot_rq1_sar_summary(summary_csv: Path) -> Optional[Path]:
    """Render a SAR vs Non-SAR refactoring rate bar chart."""
    if not summary_csv.exists():
        return None

    try:
        import os
        import pandas as pd

        mpl_cache = OUTPUT_DIR / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping RQ1 SAR plot (dependency error: {exc})")
        return None

    df = pd.read_csv(summary_csv)
    if df.empty or "category" not in df.columns or "refactoring_rate_pct" not in df.columns:
        print("  Skipping RQ1 SAR plot (summary CSV missing required columns)")
        return None

    categories = df["category"].tolist()
    rates = df["refactoring_rate_pct"].tolist()
    totals = df.get("total_commits", pd.Series([None] * len(df))).tolist()
    refs = df.get("refactoring_commits", pd.Series([None] * len(df))).tolist()
    instances = df.get("refactoring_instances", pd.Series([None] * len(df))).tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#6baed6", "#fd8d3c"]
    bar_colors = [mcolors.to_rgba(colors[i % len(colors)], alpha=0.85) for i in range(len(categories))]
    edge_colors = [mcolors.to_rgba(colors[i % len(colors)], alpha=1.0) for i in range(len(categories))]

    bars = ax.bar(categories, rates, color=bar_colors, edgecolor=edge_colors)
    ax.set_ylabel("Refactoring commits per total commits (%)", fontsize=14)
    ax.set_xlabel("Commit category", fontsize=14)
    ax.set_title("SAR vs Non-SAR refactoring rate (all Java commits)", fontsize=15)
    ax.set_ylim(0, max(5, max(rates) * 1.25 if rates else 1))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    for bar, rate, total, ref_count, inst in zip(bars, rates, totals, refs, instances):
        height = bar.get_height()
        parts = []
        if ref_count is not None and total is not None:
            parts.append(f"{ref_count}/{total}")
        if inst is not None and inst != 0:
            parts.append(f"{int(inst)} instances")
        info = ", ".join(parts)
        label = f"{rate:.2f}%"
        if info:
            label += f"\n({info})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(0.3, height * 0.05),
            label,
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    output_dir = OUTPUT_DIR / "rq1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rq1_sar_vs_non_sar_refactoring_rate.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_rq1_refactoring_commit_boxplot(refactoring_commits_csv: Path) -> Optional[Path]:
    """Box plot comparing refactoring instances per commit for SAR vs Non-SAR agentic commits."""
    if not refactoring_commits_csv.exists():
        return None

    try:
        import os
        import pandas as pd

        mpl_cache = OUTPUT_DIR / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping RQ1 refactoring commit box plot (dependency error: {exc})")
        return None

    df = pd.read_csv(refactoring_commits_csv)
    if df.empty:
        return None

    agentic = df[df.get("agent").notna()].copy()
    if agentic.empty:
        print("  Skipping RQ1 box plot (no agentic refactoring commits)")
        return None

    if "is_self_affirmed" not in agentic.columns or "refactoring_instance_count" not in agentic.columns:
        print("  Skipping RQ1 box plot (required columns missing)")
        return None

    agentic["category"] = agentic["is_self_affirmed"].map({True: "SAR", False: "Non-SAR"})
    counts = agentic["category"].value_counts()
    print(f"  Refactoring commits by category: {counts.to_dict()}")

    plot_data = agentic[["category", "refactoring_instance_count"]].copy()
    plot_data["refactoring_instance_count"] = plot_data["refactoring_instance_count"].astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=plot_data,
        x="category",
        y="refactoring_instance_count",
        palette={"Non-SAR": "#6baed6", "SAR": "#fd8d3c"},
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="category",
        y="refactoring_instance_count",
        color="black",
        size=3,
        alpha=0.35,
        ax=ax,
    )
    ax.set_xlabel("Commit category")
    ax.set_ylabel("Refactoring instances per refactoring commit")
    ax.set_title("Refactoring instances (SAR vs Non-SAR)")
    ax.set_yscale("log")
    fig.tight_layout()

    output_dir = OUTPUT_DIR / "rq1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rq1_refactoring_commit_boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_rq2_refactoring_type_butterflies(
    csv_path: Path,
    *,
    top_n: Optional[int] = 15,
) -> Dict[str, Optional[Path]]:
    """Render butterfly charts comparing Non-SAR vs SAR percentages per refactoring type."""
    if not csv_path.exists():
        return {"group": None, "type": None}

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
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping RQ2 butterfly plots (dependency error: {exc})")
        return {"group": None, "type": None}

    df = pd.read_csv(csv_path)
    required_cols = {
        "refactoring_type",
        "sar_group_percentage",
        "non_sar_group_percentage",
        "sar_percentage_of_type",
        "non_sar_percentage_of_type",
        "total_instances",
    }
    if df.empty or not required_cols.issubset(df.columns):
        print("  Skipping RQ2 butterfly plots (CSV missing required columns)")
        return {"group": None, "type": None}

    df = df.sort_values(by="total_instances", ascending=False)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    if df.empty:
        return {"group": None, "type": None}

    level_markers = {"high": "H", "medium": "M", "low": "L", "unclassified": "U"}

    def _build_labels(types: Iterable[str]) -> list[str]:
        return [f"{ref_type} ({level_markers.get(classification_key(ref_type), '?')})" for ref_type in types]

    def _plot(values_sar, values_non_sar, suffix: str, title: str, ylabel: str) -> Optional[Path]:
        sar_vals = list(values_sar)
        non_sar_vals = list(values_non_sar)
        if not sar_vals and not non_sar_vals:
            return None
        max_val = max(sar_vals + non_sar_vals) if sar_vals or non_sar_vals else 0.0
        if max_val == 0:
            return None

        labels_with_levels = _build_labels(df["refactoring_type"].tolist())
        labels = labels_with_levels[::-1]
        sar_plot_vals = sar_vals[::-1]
        non_sar_plot_vals = non_sar_vals[::-1]
        neg_non_sar = [-val for val in non_sar_plot_vals]

        fig_height = max(6, len(labels) * 0.45)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(labels, neg_non_sar, color="#4C72B0", label="Non-SAR share")
        ax.barh(labels, sar_plot_vals, color="#DD8452", label="SAR share")
        ax.axvline(0, color="#222222", linewidth=1.0)
        limit = max_val * 1.15
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("Share (%)", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title, fontsize=16)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{abs(tick):.0f}" for tick in xticks])
        ax.legend(loc="upper right")
        ax.tick_params(axis="both", labelsize=12)

        for idx, (sar_val, non_val) in enumerate(zip(sar_plot_vals, non_sar_plot_vals)):
            label = labels[idx]
            ax.text(
                sar_val + limit * 0.01,
                label,
                f"{sar_val:.1f}%",
                va="center",
                ha="left",
                fontsize=14,
                color="#4A2C17",
            )
            ax.text(
                -(non_val + limit * 0.01),
                label,
                f"{non_val:.1f}%",
                va="center",
                ha="right",
                fontsize=14,
                color="#11395F",
            )

        fig.tight_layout()
        output_path = csv_path.with_name(f"rq2_refactoring_type_butterfly_{suffix}.pdf")
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return output_path

    group_plot = _plot(
        df["sar_group_percentage"].tolist(),
        df["non_sar_group_percentage"].tolist(),
        "group",
        "Refactoring type distribution by SAR status (group share)",
        "Refactoring type (H/M/L)",
    )

    type_plot = _plot(
        df["sar_percentage_of_type"].tolist(),
        df["non_sar_percentage_of_type"].tolist(),
        "type",
        "Refactoring type split between SAR and Non-SAR (within type)",
        "Refactoring type (H/M/L)",
    )

    return {"group": group_plot, "type": type_plot}


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


def _compute_rq3_level_counts(
    csv_path: Path,
    *,
    sar_only: bool = False,
    exclude_sar: bool = False,
) -> tuple[list[tuple[str, int]], int, int]:
    """Return ordered level counts, total instances, and unclassified count."""
    if not csv_path.exists():
        return [], 0, 0

    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return [], 0, 0

    df = pd.read_csv(csv_path)
    if df.empty:
        return [], 0, 0

    if sar_only:
        df = df[df.get("is_self_affirmed") == True]
    elif exclude_sar:
        df = df[df.get("is_self_affirmed") == False]
    if df.empty:
        return [], 0, 0

    df = df.copy()
    df["level_key"] = df["refactoring_type"].map(classification_key)
    grouped = df.groupby("level_key")["instance_count"].sum()
    if grouped.empty:
        return [], 0, 0

    total_instances = int(grouped.sum())
    ordered_counts: list[tuple[str, int]] = []
    unclassified = int(grouped.get("unclassified", 0))
    for level_key in LEVEL_DISPLAY_ORDER:
        if level_key == "unclassified":
            continue
        label = LEVEL_NAME_BY_KEY[level_key]
        value = int(grouped.get(level_key, 0))
        ordered_counts.append((label, value))

    return ordered_counts, total_instances, unclassified


def _plot_rq3_levels(
    csv_path: Path,
    level_counts: list[tuple[str, int]],
    *,
    total_instances: int,
    title_label: str = "Overall",
    subset_folder: str = "overall",
    suffix: str = "",
) -> Optional[Path]:
    """Render a bar chart aggregated by refactoring level."""
    try:
        import os

        mpl_cache = csv_path.parent / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping refactoring level plot ({title_label}) due to dependency error: {exc}")
        return None

    filtered_counts = [(label, value) for label, value in level_counts if value > 0]
    if not filtered_counts:
        return None

    colors = {
        LEVEL_NAME_BY_KEY["high"]: "#4C72B0",
        LEVEL_NAME_BY_KEY["medium"]: "#DD8452",
        LEVEL_NAME_BY_KEY["low"]: "#55A868",
    }

    labels = [label for label, _ in filtered_counts]
    values = [value for _, value in filtered_counts]
    bar_colors = [colors.get(label, "#7F7F7F") for label in labels]
    edge_colors = [mcolors.to_rgba(color, alpha=1.0) for color in bar_colors]

    height = max(4, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(7, height))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor=edge_colors, linewidth=1.2)

    ax.set_ylabel("Total refactoring instances", fontsize=15)
    ax.set_xlabel("Refactoring level", fontsize=15)
    ax.set_title(f"Refactoring instances by level ({title_label})", fontsize=16)
    ax.tick_params(axis="x", labelrotation=12, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    upper = max(values)
    ax.set_ylim(0, upper * 1.18 if upper > 0 else 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        pct = (value / total_instances * 100.0) if total_instances else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, value * 0.05),
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    subset_dir = csv_path.parent / subset_folder
    subset_dir.mkdir(parents=True, exist_ok=True)
    output_path = subset_dir / f"rq3_refactoring_level_totals{suffix_str}.pdf"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path

def _plot_rq3_sar_vs_human_butterfly(
    csv_path: Path,
    *,
    top_n: Optional[int] = 20,
) -> Optional[Path]:
    """Render a butterfly chart contrasting SAR vs Human refactoring shares."""
    if not csv_path.exists():
        print("  Skipping RQ3 SAR vs Human butterfly plot (CSV not found)")
        return None

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
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping RQ3 SAR vs Human butterfly plot (dependency error: {exc})")
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "refactoring_type" not in df.columns:
        print("  Skipping RQ3 SAR vs Human butterfly plot (CSV missing required columns)")
        return None

    # Ensure percentage columns exist – derive them from counts when necessary.
    for pct_col, count_col in [
        ("sar_pct", "sar_count"),
        ("human_pct", "human_count"),
    ]:
        if pct_col not in df.columns and count_col in df.columns:
            total = pd.to_numeric(df[count_col], errors="coerce").fillna(0).sum()
            df[pct_col] = (
                pd.to_numeric(df[count_col], errors="coerce").fillna(0) / total * 100 if total > 0 else 0.0
            )

    required_cols = {"sar_pct", "human_pct"}
    if not required_cols.issubset(df.columns):
        print("  Skipping RQ3 SAR vs Human butterfly plot (missing SAR/Human percentages)")
        return None

    if df["human_pct"].fillna(0).eq(0).all():
        print("  Skipping RQ3 SAR vs Human butterfly plot (Human percentages are zero)")
        return None

    # Normalise numeric data.
    for col in ["sar_pct", "human_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.sort_values(by="sar_pct", ascending=False)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    if df.empty:
        return None

    labels = df["refactoring_type"].tolist()[::-1]
    sar_pcts = df["sar_pct"].tolist()[::-1]
    human_pcts = df["human_pct"].tolist()[::-1]

    max_val_candidates = [abs(val) for val in sar_pcts + human_pcts]
    max_val = max(max_val_candidates) if max_val_candidates else 0.0
    if max_val == 0:
        return None

    fig_height = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    positions = list(range(len(labels)))

    bar_height = 0.5
    ax.barh(
        positions,
        [-val for val in human_pcts],
        height=bar_height,
        color="#4C72B0",
        label="Human (related work)",
        align="center",
    )
    ax.barh(
        positions,
        sar_pcts,
        height=bar_height,
        color="#DD8452",
        label="Agentic Refactoring",
        align="center",
    )

    ax.axvline(0, color="#222222", linewidth=1.2)
    limit = max_val * 1.2
    ax.set_xlim(-limit, limit)
    title = "Refactoring type distribution comparison (Agentic Refactoring vs Human)"

    ax.set_xlabel("Percentage within category (%)", fontsize=16)
    ax.set_ylabel("Refactoring type", fontsize=16)
    ax.set_title(title, fontsize=17, pad=15)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=13)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(tick):.1f}" for tick in xticks])

    ax.legend(loc="lower right", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)

    threshold = 0.3

    for sar_val, human_val, y in zip(sar_pcts, human_pcts, positions):
        if abs(sar_val) >= threshold:
            ax.text(
                sar_val + limit * 0.01,
                y,
                f"{abs(sar_val):.2f}%",
                va="center",
                ha="left",
                fontsize=13,
                color="#4A2C17",
            )
        if abs(human_val) >= threshold:
            ax.text(
                -(human_val + limit * 0.01),
                y,
                f"{abs(human_val):.2f}%",
                va="center",
                ha="right",
                fontsize=13,
                color="#11395F",
            )

    fig.tight_layout()

    output_dir = OUTPUT_DIR / "rq3"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = f"_top{top_n}" if top_n else ""
    output_path = output_dir / f"rq3_sar_vs_human_butterfly{suffix_str}.pdf"

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _round_percentages(values: list[float]) -> list[float]:
    rounded = [round(float(val), 2) for val in values]
    remainder = round(100.0 - sum(rounded), 2)
    if rounded and remainder:
        rounded[-1] = round(rounded[-1] + remainder, 2)
    return rounded


def _collect_rq3_level_shares(df: "pd.DataFrame") -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for level_key in ["low", "medium", "high"]:
        mask = df["refactoring_type"].map(classification_key) == level_key
        subset = df[mask]
        if subset.empty:
            continue
        sar_pct = subset["sar_pct"].sum()
        human_pct = subset["human_pct"].sum()
        overall_pct = subset["overall_pct"].sum() if "overall_pct" in subset.columns else 0.0
        rows.append((LEVEL_NAME_BY_KEY[level_key], sar_pct, human_pct, overall_pct))
    return rows


def _plot_rq3_level_butterfly(level_data: list[tuple[str, float, float, float]]) -> Optional[Path]:
    if not level_data:
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
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping refactoring level butterfly plot (dependency error: {exc})")
        return None

    labels = [row[0] for row in level_data]
    sar_vals = [row[1] for row in level_data]
    human_vals = [row[2] for row in level_data]
    max_val = max(abs(val) for val in sar_vals + human_vals) if sar_vals or human_vals else 0.0
    if max_val == 0:
        return None

    positions = list(range(len(labels)))
    fig_height = max(4, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bar_height = 0.45
    ax.barh(
        positions,
        [-val for val in human_vals],
        height=bar_height,
        color="#4C72B0",
        label="Human",
        align="center",
    )
    ax.barh(
        positions,
        sar_vals,
        height=bar_height,
        color="#DD8452",
        label="Agentic Refactoring",
        align="center",
    )

    ax.axvline(0, color="#222222", linewidth=1.2)
    limit = max_val * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Percentage within level (%)", fontsize=15)
    ax.set_ylabel("Refactoring level", fontsize=15)
    ax.set_title("Refactoring level distribution comparison (SAR vs Human)", fontsize=16, pad=12)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=13)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(tick):.1f}" for tick in xticks])

    ax.legend(loc="upper center", fontsize=11)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)

    threshold = 0.3
    for sar_val, human_val, pos in zip(sar_vals, human_vals, positions):
        if abs(sar_val) >= threshold:
            ax.text(
                sar_val + limit * 0.01,
                pos,
                f"{sar_val:.2f}%",
                va="center",
                ha="left",
                fontsize=11,
                color="#4A2C17",
            )
        if abs(human_val) >= threshold:
            ax.text(
                -(human_val + limit * 0.01),
                pos,
                f"{human_val:.2f}%",
                va="center",
                ha="right",
                fontsize=11,
                color="#11395F",
            )

    fig.tight_layout()
    output_dir = OUTPUT_DIR / "rq3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rq3_sar_vs_human_levels.pdf"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _report_rq3_level_comparison(csv_path: Path) -> Optional[list[tuple[str, float, float, float]]]:
    """Print SAR vs Human percentages by refactoring level (high/medium/low)."""
    if not csv_path.exists():
        print(f"  Level comparison skipped (CSV missing at {csv_path})")
        return None

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  Level comparison skipped (failed to load {csv_path}: {exc})")
        return None

    if "refactoring_type" not in df.columns:
        print("  Level comparison skipped (missing refactoring_type column)")
        return None

    for pct_col, count_col in [
        ("sar_pct", "sar_count"),
        ("human_pct", "human_count"),
        ("overall_pct", "overall_count"),
    ]:
        if pct_col not in df.columns and count_col in df.columns:
            total = pd.to_numeric(df[count_col], errors="coerce").fillna(0).sum()
            df[pct_col] = (
                pd.to_numeric(df[count_col], errors="coerce").fillna(0) / total * 100 if total > 0 else 0.0
            )

    level_rows = _collect_rq3_level_shares(df)
    if not level_rows:
        print("  Level comparison skipped (no classified refactoring types)")
        return None

    sar_source = [row[1] for row in level_rows]
    human_source = [row[2] for row in level_rows]
    overall_source = [row[3] for row in level_rows]

    if sum(sar_source) == 0 or sum(human_source) == 0:
        print("  Level comparison skipped (zero totals)")
        return None

    sar_vals = _round_percentages(sar_source)
    human_vals = _round_percentages(human_source)
    overall_vals = (
        _round_percentages(overall_source)
        if sum(overall_source) > 0
        else [0.0] * len(level_rows)
    )
    include_overall = sum(overall_source) > 0

    print("  Refactoring level share (SAR vs Human):")
    level_data: list[tuple[str, float, float, float]] = []
    for (label, _, _, _), sar_pct, human_pct, overall_pct in zip(level_rows, sar_vals, human_vals, overall_vals):
        parts = [
            f"SAR {sar_pct:.2f}%",
            f"Human {human_pct:.2f}%",
        ]
        if include_overall:
            parts.append(f"Overall {overall_pct:.2f}%")
        print(f"    {label}: " + " | ".join(parts))
        level_data.append((label, sar_pct, human_pct, overall_pct if include_overall else 0.0))

    return level_data


def _plot_rq5_smell_violin(
    parquet_path: Path,
    *,
    smell_label: str,
    output_name: str,
    metadata: Optional["pd.DataFrame"] = None,
    sar_only: bool = False,
    stats_accumulator: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Render split violin plots for smell counts before vs after refactoring."""
    if not parquet_path.exists():
        print(f"  Skipping {smell_label.lower()} smell violin plot (missing {parquet_path})")
        return None

    try:
        import os
        import pandas as pd
        import numpy as np

        mpl_cache = OUTPUT_DIR / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping {smell_label.lower()} smell violin plot (dependency error: {exc})")
        return None

    df = pd.read_parquet(parquet_path, columns=["commit_sha", "before_value", "after_value"])
    if df.empty:
        print(f"  Skipping {smell_label.lower()} smell violin plot (no data)")
        return None

    if metadata is not None:
        join_cols = ["sha"]
        if sar_only:
            join_cols.append("is_self_affirmed")
        missing_cols = [col for col in join_cols if col not in metadata.columns]
        if missing_cols:
            if sar_only:
                print(
                    f"  Skipping {smell_label.lower()} smell violin plot "
                    f"(metadata missing columns needed for SAR filter: {', '.join(missing_cols)})"
                )
                return None
            metadata = metadata[[col for col in metadata.columns if col in join_cols]]

        df = df.merge(metadata[join_cols], left_on="commit_sha", right_on="sha", how="left")
        if sar_only:
            df = df[df["is_self_affirmed"] == True]
        df.drop(columns=[col for col in ["sha", "is_self_affirmed"] if col in df.columns], inplace=True)
        if sar_only and df.empty:
            print(f"  Skipping {smell_label.lower()} smell violin plot (no rows after SAR filter)")
            return None
    elif sar_only:
        print(f"  Skipping {smell_label.lower()} smell violin plot (metadata required for SAR filter)")
        return None

    df[["before_value", "after_value"]] = df[["before_value", "after_value"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    aggregated = (
        df.groupby("commit_sha")[["before_value", "after_value"]]
        .sum()
        .reset_index()
    )  # smell counts per commit
    if aggregated.empty:
        print(f"  Skipping {smell_label.lower()} smell violin plot (no aggregated data)")
        return None

    before_median = float(aggregated["before_value"].median())
    after_median = float(aggregated["after_value"].median())
    diff_series = aggregated["after_value"] - aggregated["before_value"]
    mean_diff = float(diff_series.mean())
    median_diff = float(diff_series.median())
    std_diff = float(diff_series.std(ddof=1)) if len(diff_series) > 1 else 0.0
    effect_size = (
        float(mean_diff / std_diff)
        if std_diff not in (0.0, float("nan")) and std_diff != 0.0
        else float("nan")
    )
    improvement_pct = np.where(
        aggregated["before_value"] > 0,
        (aggregated["before_value"] - aggregated["after_value"]) / aggregated["before_value"] * 100.0,
        np.nan,
    )
    improvement_series = pd.Series(improvement_pct).dropna()
    improvement_mean = float(improvement_series.mean()) if not improvement_series.empty else float("nan")
    improvement_median = float(improvement_series.median()) if not improvement_series.empty else float("nan")
    wilcoxon_stat = float("nan")
    wilcoxon_pvalue = float("nan")
    wilcoxon_rbc = float("nan")
    wilcoxon_note: Optional[str] = None
    if wilcoxon_signed_rank is not None and len(aggregated) > 0:
        try:
            result = wilcoxon_signed_rank(
                aggregated["before_value"].to_numpy(),
                aggregated["after_value"].to_numpy(),
                zero_method="pratt",
                alternative="two-sided",
            )
            wilcoxon_stat = float(result.statistic)
            wilcoxon_pvalue = float(result.pvalue)
            diff = aggregated["after_value"].to_numpy() - aggregated["before_value"].to_numpy()
            non_zero = diff[~np.isclose(diff, 0.0)]
            n = len(non_zero)
            if n > 0:
                wilcoxon_rbc = 1.0 - (2.0 * wilcoxon_stat) / (n * (n + 1))
        except ValueError as exc:
            wilcoxon_note = f"Wilcoxon test skipped ({exc})"
    else:
        if wilcoxon_signed_rank is None:
            wilcoxon_note = "Wilcoxon test unavailable (scipy not installed)"
        else:
            wilcoxon_note = "Wilcoxon test skipped (not enough paired observations)"

    print(
        f"  {smell_label}: before median = {before_median:.2f}, after median = {after_median:.2f}"
    )
    print(
        f"  {smell_label}: mean Δ = {mean_diff:.2f}, median Δ = {median_diff:.2f}, effect size (Cohen's d) = {effect_size:.3f}"
    )
    if not math.isnan(improvement_mean):
        print(
            f"  {smell_label}: mean improvement rate = {improvement_mean:.2f}%, median improvement rate = {improvement_median:.2f}%"
        )
    if wilcoxon_note:
        print(f"  {smell_label}: {wilcoxon_note}")
    elif not np.isnan(wilcoxon_stat) and not np.isnan(wilcoxon_pvalue):
        print(
            f"  {smell_label}: Wilcoxon signed-rank = {wilcoxon_stat:.2f}, p-value = {wilcoxon_pvalue:.4f}"
        )

    stats_out = OUTPUT_DIR / "rq5"
    stats_out.mkdir(parents=True, exist_ok=True)
    stats_row: Dict[str, Any] = {
        "smell_label": smell_label,
        "commit_count": len(aggregated),
        "before_median": before_median,
        "after_median": after_median,
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "cohens_d": effect_size,
        "improvement_rate_mean_pct": improvement_mean,
        "improvement_rate_median_pct": improvement_median,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "wilcoxon_rank_biserial": wilcoxon_rbc,
        "wilcoxon_fdr_pvalue": float("nan"),
        "wilcoxon_null_hypothesis": "Median delta = 0 (Wilcoxon H0)",
    }
    stats_df = pd.DataFrame([stats_row])
    stats_path = stats_out / f"{output_name}_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    if stats_accumulator is not None:
        stats_accumulator.append(
            {
                "display_label": smell_label,
                "stats_csv": stats_path,
                "row": stats_row,
                "wilcoxon_note": wilcoxon_note,
            }
        )

    before_values_raw = aggregated["before_value"].to_numpy()
    after_values_raw = aggregated["after_value"].to_numpy()
    before_has_signal = np.any(before_values_raw > 0)
    after_has_signal = np.any(after_values_raw > 0)
    if not before_has_signal or not after_has_signal:
        print(
            f"  Skipping {smell_label.lower()} smell violin plot (split violin needs non-zero counts in both phases)"
        )
        return None

    melted = aggregated.melt(
        id_vars="commit_sha",
        value_vars=["before_value", "after_value"],
        var_name="phase",
        value_name="smell_count",
    )
    melted["phase"] = melted["phase"].map({"before_value": "Before", "after_value": "After"})
    melted["smell_count"] = melted["smell_count"].astype(float)
    melted = melted.dropna(subset=["smell_count"])

    if melted.empty or np.isclose(melted["smell_count"].sum(), 0.0):
        print(f"  Skipping {smell_label.lower()} smell violin plot (counts all zero)")
        return None

    melted["category"] = smell_label
    if (melted["smell_count"] < 0).any():
        melted["smell_count"] = melted["smell_count"].clip(lower=0.0)
    # ensure numerical stability for KDE
    melted["smell_count"] = melted["smell_count"].where(melted["smell_count"] > 0.0, 1e-6)
    high_quantile = melted["smell_count"].quantile(0.995)
    if np.isfinite(high_quantile) and high_quantile > 0:
        melted["smell_count"] = melted["smell_count"].clip(upper=float(high_quantile))

    before_values = melted.loc[melted["phase"] == "Before", "smell_count"]
    after_values = melted.loc[melted["phase"] == "After", "smell_count"]
    if before_values.empty or after_values.empty:
        print(
            f"  Skipping {smell_label.lower()} smell violin plot (insufficient data for split violin)"
        )
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"Before": "#4C72B0", "After": "#DD8452"}
    sns.violinplot(
        data=melted,
        x="category",
        y="smell_count",
        hue="phase",
        split=True,
        inner=None,
        palette=palette,
        ax=ax,
    )
    centers = ax.get_xticks()
    center = centers[0] if len(centers) == 1 else 0.0
    box_offset = 0.08
    for phase, offset in (("Before", -box_offset), ("After", box_offset)):
        values = melted.loc[melted["phase"] == phase, "smell_count"].to_numpy()
        if values.size == 0:
            continue
        base_rgb = mcolors.to_rgb(palette[phase])
        light_rgb = tuple(min(1.0, 0.55 * c + 0.45) for c in base_rgb)
        fill_color = (*light_rgb, 0.28)
        ax.boxplot(
            values,
            positions=[center + offset],
            widths=box_offset * 0.9,
            patch_artist=True,
            boxprops={
                "facecolor": fill_color,
                "edgecolor": palette[phase],
                "linewidth": 1.2,
            },
            whiskerprops={"color": palette[phase], "linewidth": 1.0},
            capprops={"color": palette[phase], "linewidth": 1.0},
            medianprops={"color": palette[phase], "linewidth": 1.4},
            vert=True,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Smell count per commit")
    ax.set_title(f"{smell_label} Smell Count Distribution (Before vs After)")
    ax.legend(title="Phase")
    # ax.set_yscale("symlog", linthresh=10)

    output_dir = OUTPUT_DIR / "rq5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    stats_out = OUTPUT_DIR / "rq5"
    stats_out.mkdir(parents=True, exist_ok=True)
    aggregated_with_rate = aggregated.copy()
    aggregated_with_rate["improvement_rate_pct"] = improvement_pct
    aggregated_with_rate.to_csv(stats_out / f"{output_name}_aggregated.csv", index=False)
    return output_path


def _apply_fdr_correction(wilcoxon_records: List[Dict[str, Any]]) -> None:
    """Apply Benjamini–Hochberg FDR correction to Wilcoxon p-values and rewrite stats."""
    if not wilcoxon_records:
        return

    try:
        import pandas as pd  # Reuse pandas for rewriting stats CSVs
    except Exception:
        pd = None  # type: ignore[assignment]

    valid: List[tuple[int, float]] = []
    for idx, record in enumerate(wilcoxon_records):
        row = record.get("row", {})
        pval = row.get("wilcoxon_pvalue")
        if pval is None or math.isnan(pval):
            continue
        valid.append((idx, float(pval)))

    if valid:
        sorted_valid = sorted(valid, key=lambda item: item[1])
        m = len(sorted_valid)
        adjusted = []
        for rank, (_, pval) in enumerate(sorted_valid, start=1):
            adjusted.append(min(pval * m / rank, 1.0))
        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        for (idx, _), adj in zip(sorted_valid, adjusted):
            wilcoxon_records[idx]["row"]["wilcoxon_fdr_pvalue"] = adj

    for record in wilcoxon_records:
        csv_path = record.get("stats_csv")
        row = record.get("row")
        if not csv_path or not row or pd is None:
            continue
        try:
            pd.DataFrame([row]).to_csv(csv_path, index=False)
        except Exception:
            # If rewriting fails, continue without halting the pipeline
            continue

    print("\nRQ5 Wilcoxon tests (H0: median delta = 0)")
    for record in wilcoxon_records:
        label = record.get("display_label") or record.get("row", {}).get("smell_label") or "Unknown"
        note = record.get("wilcoxon_note")
        row = record.get("row", {})
        stat = row.get("wilcoxon_statistic")
        pval = row.get("wilcoxon_pvalue")
        fdr_p = row.get("wilcoxon_fdr_pvalue")
        if note:
            print(f"  {label}: {note}")
            continue
        if pval is None or math.isnan(pval):
            print(f"  {label}: Wilcoxon test unavailable")
            continue
        summary = f"  {label}: statistic = {stat:.2f}, p = {pval:.4f}"
        if fdr_p is not None and not math.isnan(fdr_p):
            summary += f", FDR-adjusted p = {fdr_p:.4f}"
        print(summary)


def _plot_rq4_purpose_butterfly(
    distribution: Dict[str, int],
    total_count: int,
    *,
    subset_label: str = "overall",
    target_dir: Optional[Path] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Render a butterfly chart comparing agentic vs human RQ4 purpose shares."""
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
        from matplotlib import ticker
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"    Skipping RQ4 butterfly chart due to dependency error: {exc}")
        return None

    excluded = set(exclude_keys or [])
    ordered_categories = [cat for cat in RQ4_PURPOSE_DISPLAY_LABELS.keys() if cat not in excluded]
    extra_categories = [
        cat for cat in distribution.keys() if cat not in ordered_categories and cat not in excluded
    ]
    categories = ordered_categories + sorted(extra_categories)
    agentic_pct = {
        cat: (distribution.get(cat, 0) / total_count * 100.0) if total_count else 0.0
        for cat in categories
    }
    data_rows = []
    for cat in categories:
        agentic_share = agentic_pct.get(cat, 0.0)
        human_share = RQ4_HUMAN_PURPOSE_PERCENTAGES.get(cat)
        display = RQ4_PURPOSE_DISPLAY_LABELS.get(cat, cat.replace("_", " ").title())
        if math.isclose(agentic_share, 0.0, abs_tol=1e-9) and (
            human_share is None or math.isclose(human_share, 0.0, abs_tol=1e-9)
        ):
            continue
        data_rows.append(
            {
                "key": cat,
                "label": display,
                "agentic": agentic_share,
                "human": human_share if human_share is not None else float("nan"),
            }
        )
    if not data_rows:
        return None

    data_rows.sort(key=lambda row: row["agentic"], reverse=True)

    labels = [row["label"] for row in data_rows]
    agentic_values = np.array([-row["agentic"] for row in data_rows])
    human_values = np.array(
        [row["human"] if not math.isnan(row["human"]) else 0.0 for row in data_rows]
    )

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(labels) * 0.6)))
    y_pos = np.arange(len(labels))

    ax.barh(
        y_pos,
        agentic_values,
        color="#4C72B0",
        alpha=0.85,
        label="Agentic refactoring",
    )
    ax.barh(
        y_pos,
        human_values,
        color="#DD8452",
        alpha=0.85,
        label="Human (Kim et al. 2014)",
    )

    max_extent = max(
        np.max(np.abs(agentic_values)) if agentic_values.size else 0,
        np.max(human_values) if human_values.size else 0,
    )
    if max_extent <= 0:
        max_extent = 1.0
    ax.set_xlim(-max_extent * 1.15, max_extent * 1.15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of commits (%)")
    title_suffix = f" ({subset_label})" if subset_label else ""
    ax.set_title(f"Refactoring purpose comparison{title_suffix}")
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{abs(value):.0f}"))
    ax.legend(loc="upper right")

    for y, (agentic_val, human_val) in zip(y_pos, zip(agentic_values, human_values)):
        sar_pct = abs(agentic_val)
        human_pct = human_val
        if not math.isclose(agentic_val, 0.0, abs_tol=1e-3):
            ax.text(
                agentic_val - 0.02 * max_extent,
                y,
                f"{sar_pct:.1f}%",
                va="center",
                ha="right",
                color="#1B4F72",
                fontsize=9,
            )
        if not math.isclose(human_val, 0.0, abs_tol=1e-3):
            ax.text(
                human_val + 0.02 * max_extent,
                y,
                f"{human_pct:.1f}%",
                va="center",
                ha="left",
                color="#7F2704",
                fontsize=9,
            )

    fig.tight_layout()
    target_dir = target_dir or (OUTPUT_DIR / "rq4" / (subset_label or "overall"))
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(target_dir) / "rq4_purpose_butterfly.pdf"
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return output_path


def _report_rq3_top_types_by_level(summary_csv: Path) -> None:
    """Print the top three refactoring types per level for SAR and human data."""
    if not summary_csv.exists():
        print(f"  RQ3 top-by-level summary unavailable (missing {summary_csv})")
        return

    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        print(f"  Unable to report RQ3 level tops (dependency error: {exc})")
        return

    df = pd.read_csv(summary_csv)
    if df.empty or "refactoring_type" not in df.columns:
        print("  RQ3 level tops skip (summary CSV empty or missing columns)")
        return

    for col in ("sar_count", "human_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["level_key"] = df["refactoring_type"].apply(classification_key)
    print("\nRQ3: Top refactoring types by level (SAR vs Human)")

    for level_key in LEVEL_DISPLAY_ORDER:
        if level_key == "unclassified":
            continue
        level_label = LEVEL_NAME_BY_KEY.get(level_key, level_key.title())
        level_df = df[df["level_key"] == level_key]
        if level_df.empty:
            continue

        sar_top = level_df.sort_values("sar_count", ascending=False).head(5)
        human_top = level_df.sort_values("human_count", ascending=False).head(5)
        sar_level_total = level_df["sar_count"].sum() or 1.0
        human_level_total = level_df["human_count"].sum() or 1.0

        print(f"  {level_label} (SAR):")
        if sar_top.empty:
            print("    No SAR refactorings recorded.")
        else:
            for _, row in sar_top.iterrows():
                share = (row["sar_count"] / sar_level_total) * 100.0 if sar_level_total else 0.0
                print(
                    f"    {row['refactoring_type']}: "
                    f"{int(row['sar_count']):,} SAR instances "
                    f"({share:.1f}% of SAR level, {row['sar_pct']:.1f}% overall)"
                )

        print(f"  {level_label} (Human):")
        if human_top.empty:
            print("    No human refactorings recorded.")
        else:
            for _, row in human_top.iterrows():
                share = (row["human_count"] / human_level_total) * 100.0 if human_level_total else 0.0
                print(
                    f"    {row['refactoring_type']}: "
                    f"{int(row['human_count']):,} human instances "
                    f"({share:.1f}% of human level, {row['human_pct']:.1f}% overall)"
                )


def _summarize_designite_metric_delta(
    parquet_path: Path,
    *,
    label: str,
    output_name: str,
    metadata: Optional["pd.DataFrame"] = None,
    sar_only: bool = False,
) -> Optional[Path]:
    """Summarise designite metric deltas (type/method) and export per-metric stats."""
    if not parquet_path.exists():
        print(f"  Skipping {label.lower()} metrics (missing {parquet_path})")
        return None

    try:
        import os
        import pandas as pd
        import numpy as np

        mpl_cache = OUTPUT_DIR / ".matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping {label.lower()} metrics (dependency error: {exc})")
        return None

    df = pd.read_parquet(
        parquet_path,
        columns=[
            "commit_sha",
            "before_value",
            "after_value",
            "delta",
            "metric",
        ],
    )
    if df.empty:
        print(f"  Skipping {label.lower()} metrics (no data)")
        return None

    if metadata is not None:
        join_cols = ["sha"]
        if sar_only:
            join_cols.append("is_self_affirmed")
        missing = [col for col in join_cols if col not in metadata.columns]
        if missing:
            if sar_only:
                print(
                    f"  Skipping {label.lower()} metrics "
                    f"(metadata missing columns for SAR filter: {', '.join(missing)})"
                )
                return None
            metadata = metadata[[col for col in metadata.columns if col in join_cols]]
        df = df.merge(metadata[join_cols], left_on="commit_sha", right_on="sha", how="left")
        if sar_only:
            df = df[df["is_self_affirmed"] == True]
        df.drop(columns=[col for col in ["sha", "is_self_affirmed"] if col in df.columns], inplace=True)
        if sar_only and df.empty:
            print(f"  Skipping {label.lower()} metrics (no rows after SAR filter)")
            return None
    elif sar_only:
        print(f"  Skipping {label.lower()} metrics (metadata required for SAR filter)")
        return None

    df[["before_value", "after_value", "delta"]] = df[
        ["before_value", "after_value", "delta"]
    ].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["metric", "before_value", "after_value", "delta"])
    if df.empty:
        print(f"  Skipping {label.lower()} metrics (no numeric data)")
        return None

    aggregated = (
        df.groupby(["commit_sha", "metric"], as_index=False)
        .agg(
            before_total=("before_value", "sum"),
            after_total=("after_value", "sum"),
            delta_total=("delta", "sum"),
            observation_count=("delta", "size"),
        )
    )
    if aggregated.empty:
        print(f"  Skipping {label.lower()} metrics (no aggregated rows)")
        return None

    aggregated["improvement_rate_pct"] = np.where(
        aggregated["before_total"] > 0,
        (aggregated["before_total"] - aggregated["after_total"])
        / aggregated["before_total"]
        * 100.0,
        np.nan,
    )

    summary_rows: List[Dict[str, object]] = []
    for metric, group in aggregated.groupby("metric"):
        delta_series = group["delta_total"]
        improvement_series = group["improvement_rate_pct"].dropna()
        improved = int((group["improvement_rate_pct"] > 0).sum())
        regressed = int((group["improvement_rate_pct"] < 0).sum())
        unchanged = int(group["improvement_rate_pct"].isna().sum() + (group["improvement_rate_pct"] == 0).sum())
        total = len(group)
        summary_rows.append(
            {
                "metric": metric,
                "commit_count": total,
                "delta_mean": float(delta_series.mean()),
                "delta_median": float(delta_series.median()),
                "delta_std": float(delta_series.std(ddof=1)) if len(delta_series) > 1 else 0.0,
                "delta_min": float(delta_series.min()),
                "delta_max": float(delta_series.max()),
                "improvement_rate_mean_pct": float(improvement_series.mean())
                if not improvement_series.empty
                else float("nan"),
                "improvement_rate_median_pct": float(improvement_series.median())
                if not improvement_series.empty
                else float("nan"),
                "improvement_rate_std_pct": float(improvement_series.std(ddof=1))
                if len(improvement_series) > 1
                else 0.0,
                "improved_commits": improved,
                "regressed_commits": regressed,
                "unchanged_commits": unchanged,
                "improved_share_pct": (improved / total * 100.0) if total else float("nan"),
                "regressed_share_pct": (regressed / total * 100.0) if total else float("nan"),
                "unchanged_share_pct": (unchanged / total * 100.0) if total else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("metric")
    output_dir = OUTPUT_DIR / "rq5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"\n  {label} metrics summary ({'SAR' if sar_only else 'overall'}):")
    for _, row in summary_df.iterrows():
        metric_name = row["metric"]
        mean_delta = row["delta_mean"]
        median_delta = row["delta_median"]
        mean_rate = row["improvement_rate_mean_pct"]
        median_rate = row["improvement_rate_median_pct"]
        improved = int(row["improved_commits"])
        regressed = int(row["regressed_commits"])
        unchanged = int(row["unchanged_commits"])
        total = int(row["commit_count"])
        print(
            f"    {metric_name}: Δmean = {mean_delta:.2f}, Δmedian = {median_delta:.2f}, "
            f"improvement mean = {mean_rate:.2f}%, median = {median_rate:.2f}%, "
            f"improved/regressed/unchanged = {improved}/{regressed}/{unchanged} (n={total})"
        )

    aggregated_path = output_dir / f"{output_name}_aggregated.csv"
    aggregated.to_csv(aggregated_path, index=False)
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

    print("\nRQ1: Counting refactoring instances in agentic PRs...")
    rq1 = rq1_refactoring_instances_agentic(commits, refminer)
    print(f"  Agentic commits: {rq1['total_agentic_commits']}")
    print(f"  Agentic refactoring commits: {rq1['agentic_refactoring_commits']}")
    if rq1["refminer_available"]:
        print(f"  Agentic refactoring instances: {rq1['agentic_refactoring_instances']}")
    else:
        print("  RefactoringMiner instances unavailable (no RM results).")

    sar_summary_candidates = [
        Path("data/analysis/refactoring_instances/sar_commit_summary_filtered.csv"),
        Path("data/analysis/refactoring_instances/sar_commit_summary.csv"),
    ]
    sar_summary_path = next((p for p in sar_summary_candidates if p.exists()), None)
    if sar_summary_path:
        plot_path = _plot_rq1_sar_summary(sar_summary_path)
        if plot_path:
            print(f"  Saved SAR vs Non-SAR refactoring rate plot: {plot_path}")
        else:
            print(f"  Skipped SAR plot (could not render from {sar_summary_path})")
    else:
        print("  SAR summary CSV not found; skipping SAR vs Non-SAR plot.")

    refactoring_commits_candidates = [
        Path("data/analysis/refactoring_instances/refactoring_commits_filtered.csv"),
        Path("data/analysis/refactoring_instances/refactoring_commits.csv"),
    ]
    refactoring_commits_path = next((p for p in refactoring_commits_candidates if p.exists()), None)
    if refactoring_commits_path:
        boxplot_path = _plot_rq1_refactoring_commit_boxplot(refactoring_commits_path)
        if boxplot_path:
            print(f"  Saved refactoring commit box plot: {boxplot_path}")
        else:
            print(f"  Skipped refactoring commit box plot (could not render from {refactoring_commits_path})")
    else:
        print("  Refactoring commit CSV not found; skipping RQ1 box plot.")

    print("\nRQ2: Self-affirmed refactoring percentage in agentic commits...")
    rq2 = rq2_self_affirmed_percentage(commits)
    print(f"  Agentic refactoring commits: {rq2['agentic_refactoring_commits']}")
    print(f"  Self-affirmed commits: {rq2['self_affirmed_commits']}")
    print(f"  Self-affirmed percentage: {rq2['self_affirmed_percentage']:.2f}%")

    rq2_split = rq2_refactoring_type_affirmed_split(refminer, commits)
    csv_path_str = rq2_split.get("csv_path") if isinstance(rq2_split, dict) else None
    if csv_path_str:
        csv_path = Path(csv_path_str)
        print(f"  Refactoring type SAR split CSV: {csv_path}")
        butterfly_paths = _plot_rq2_refactoring_type_butterflies(csv_path, top_n=rq2_split.get("top_n"))
        group_path = butterfly_paths.get("group")
        type_path = butterfly_paths.get("type")
        if group_path:
            print(f"  Saved RQ2 butterfly chart (group share): {group_path}")
        if type_path:
            print(f"  Saved RQ2 butterfly chart (within type): {type_path}")
    else:
        note = rq2_split.get("note") if isinstance(rq2_split, dict) else None
        if note:
            print(f"  Skipping RQ2 butterfly chart: {note}")

    sar_commits = commits[commits.get("is_self_affirmed") == True] if "is_self_affirmed" in commits.columns else commits.iloc[0:0]
    refminer_sar = (
        refminer[refminer["commit_sha"].isin(sar_commits["sha"])]
        if not sar_commits.empty
        else refminer.iloc[0:0]
    )

    print("\nRQ3: Most common refactoring types (SAR only)...")
    if sar_commits.empty or refminer_sar.empty:
        print("  No SAR commits or refactoring instances available.")
    else:
        rq3 = rq3_top_refactoring_types(refminer_sar, sar_commits, top_n=None, min_count=2)
        sar_types = rq3.get("sar_top_types") or rq3.get("overall_top_types")
        if sar_types:
            sar_order = rq3.get("sar_order_median_desc") or list(sar_types.keys())
            sar_distribution = rq3.get("sar_distribution", {}) or rq3.get("overall_distribution", {})
            sar_sample_counts = rq3.get("sar_sample_counts", {}) or rq3.get("overall_sample_counts", {})
            print("  SAR types sorted by median instances/commit:")
            for ref_type in sar_order:
                count = sar_types.get(ref_type, 0)
                median = sar_distribution.get(ref_type, {}).get("median")
                commit_count = sar_sample_counts.get(ref_type)
                sample_str = f"n={commit_count}" if isinstance(commit_count, (int, float)) else "n=?"
                median_str = f"median={median:.2f}" if isinstance(median, (int, float)) else "median=n/a"
                print(f"    {ref_type}: total={count}, {median_str}, {sample_str}")

            distribution_csv = rq3.get("distribution_csv")
            if distribution_csv:
                distribution_path = Path(distribution_csv)
                print(f"  Distribution CSV for plotting: {distribution_path}")
                sar_plot_path = _plot_rq3_distribution(
                    distribution_path,
                    sar_order,
                    color="#F15854",
                    subset_label="SAR",
                )
                if sar_plot_path:
                    print(f"  Saved SAR violin/box plot: {sar_plot_path}")
                if len(sar_order) > 20:
                    sar_top20_path = _plot_rq3_distribution(
                        distribution_path,
                        sar_order[:20],
                        suffix="top20",
                        color="#C1392B",
                        subset_label="SAR (Top 20)",
                    )
                    if sar_top20_path:
                        print(f"  Saved SAR Top20 violin/box plot: {sar_top20_path}")
                sar_totals = _plot_rq3_totals(
                    distribution_path,
                    sar_only=True,
                    color="#F15854",
                    title_label="SAR",
                    subset_folder="sar",
                )
                if sar_totals:
                    print(f"  Saved SAR totals bar chart: {sar_totals}")
                sar_totals_top20 = _plot_rq3_totals(
                    distribution_path,
                    sar_only=True,
                    top_n=20,
                    suffix="top20",
                    color="#E4572E",
                    title_label="SAR (Top 20)",
                    subset_folder="sar",
                )
                if sar_totals_top20:
                    print(f"  Saved SAR totals Top20 bar chart: {sar_totals_top20}")

                level_counts_sar, level_total_sar, unclassified_sar = _compute_rq3_level_counts(
                    distribution_path,
                    sar_only=True,
                )
                if level_total_sar:
                    print("  Refactoring levels (SAR):")
                    for label, count in level_counts_sar:
                        if count <= 0:
                            continue
                        pct = (count / level_total_sar) * 100.0 if level_total_sar else 0.0
                        print(f"    {label}: {count} ({pct:.1f}%)")
                    if unclassified_sar:
                        pct = (unclassified_sar / level_total_sar) * 100.0 if level_total_sar else 0.0
                        print(f"    Unclassified: {unclassified_sar} ({pct:.1f}%)")
                    sar_level_chart = _plot_rq3_levels(
                        distribution_path,
                        level_counts_sar,
                        total_instances=level_total_sar,
                        title_label="SAR",
                        subset_folder="sar",
                    )
                    if sar_level_chart:
                        print(f"  Saved SAR level totals chart: {sar_level_chart}")
        else:
            print("  SAR type summary unavailable (no SAR refactoring instances).")

    # Compare SAR vs Human using the combined summary
    print("\nRQ3 (continued): Comparing SAR vs Human refactoring distributions...")
    totals_summary_path = Path("outputs/research_questions/rq3/rq3_refactoring_type_totals_summary.csv")
    if totals_summary_path.exists():
        butterfly_path = _plot_rq3_sar_vs_human_butterfly(totals_summary_path, top_n=20)
        if butterfly_path:
            print(f"  Saved SAR vs Human butterfly chart (Top 20): {butterfly_path}")
        butterfly_all_path = _plot_rq3_sar_vs_human_butterfly(totals_summary_path, top_n=None)
        if butterfly_all_path:
            print(f"  Saved SAR vs Human butterfly chart (all types): {butterfly_all_path}")
        level_data = _report_rq3_level_comparison(totals_summary_path)
        if level_data:
            level_plot_path = _plot_rq3_level_butterfly(level_data)
            if level_plot_path:
                print(f"  Saved SAR vs Human level butterfly chart: {level_plot_path}")
    else:
        print(f"  SAR vs Human summary not found at {totals_summary_path}")
        print("  Run scripts/10a_compute_rq3_totals_summary.py first to generate this comparison.")

    _report_rq3_top_types_by_level(totals_summary_path)

    print("\nRQ4: Refactoring purposes in SAR (agentic) commits")
    if sar_commits.empty:
        print("  No SAR commits available for RQ4.")
    else:
        rq4_sar = rq4_refactoring_purpose(
            sar_commits,
            agentic_only=False,
            subset_label="sar",
        )
        total_rq4_sar = int(rq4_sar.get("total_refactoring_commits", 0))
        print(f"  [SAR] Refactoring commits analysed: {total_rq4_sar}")
        distribution_sar = rq4_sar.get("purpose_distribution", {})
        filtered_distribution_sar = {
            label: count for label, count in (distribution_sar or {}).items() if label != "logical_mismatch"
        }
        filtered_total_sar = sum(filtered_distribution_sar.values())
        if filtered_distribution_sar and filtered_total_sar > 0:
            sar_rows = sorted(
                (
                    (
                        label,
                        count,
                        (count / filtered_total_sar * 100.0) if filtered_total_sar else 0.0,
                    )
                    for label, count in filtered_distribution_sar.items()
                ),
                key=lambda row: row[2],
                reverse=True,
            )
            limit_sar = min(10, len(sar_rows))
            print("    Top purpose labels:")
            for label, count, pct in sar_rows[:limit_sar]:
                print(f"      {label}: {count} ({pct:.1f}%)")
        else:
            print("    No purpose labels identified (logical mismatch excluded).")
        examples_sar = rq4_sar.get("examples_file")
        if examples_sar:
            print(f"    Examples CSV: {examples_sar}")
        note_sar = rq4_sar.get("note")
        if note_sar and (not filtered_distribution_sar or filtered_total_sar == 0):
            print(f"    Note: {note_sar}")
        if filtered_distribution_sar and filtered_total_sar > 0:
            base_dir_sar = Path(rq4_sar.get("output_dir") or (OUTPUT_DIR / "rq4" / "sar"))
            sar_chart = _plot_rq4_purposes(
                filtered_distribution_sar,
                filtered_total_sar,
                color="#E4572E",
                subset_label="sar",
                target_dir=base_dir_sar,
            )
            if sar_chart:
                print(f"    Saved SAR purpose chart: {sar_chart}")
            sar_top_chart = _plot_rq4_purposes(
                filtered_distribution_sar,
                filtered_total_sar,
                suffix="top10",
                color="#C13C1E",
                top_n=10,
                subset_label="sar",
                target_dir=base_dir_sar,
            )
            if sar_top_chart:
                print(f"    Saved SAR Top10 purpose chart: {sar_top_chart}")
            sar_butterfly = _plot_rq4_purpose_butterfly(
                filtered_distribution_sar,
                filtered_total_sar,
                subset_label="sar",
                target_dir=base_dir_sar,
            )
            if sar_butterfly:
                print(f"    Saved SAR vs human purpose butterfly: {sar_butterfly}")

    print("\nRQ5: Smell counts before vs after refactoring")
    commits_meta = None
    try:
        import pandas as pd
        commits_meta = pd.read_parquet(
            Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet"),
            columns=["sha", "is_self_affirmed"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  Note: could not load commit metadata for SAR filter ({exc})")
        commits_meta = None

    wilcoxon_records: List[Dict[str, Any]] = []

    design_violin = _plot_rq5_smell_violin(
        Path("data/analysis/designite/deltas/design_smell_deltas.parquet"),
        smell_label="Design Smell",
        output_name="rq5_design_smell_violin",
        metadata=commits_meta,
        sar_only=False,
        stats_accumulator=wilcoxon_records,
    )
    if design_violin:
        print(f"  Saved design smell before/after violin: {design_violin}")
    design_violin_sar = _plot_rq5_smell_violin(
        Path("data/analysis/designite/deltas/design_smell_deltas.parquet"),
        smell_label="Design Smell (SAR)",
        output_name="rq5_design_smell_violin_sar",
        metadata=commits_meta,
        sar_only=True,
        stats_accumulator=wilcoxon_records,
    )
    if design_violin_sar:
        print(f"  Saved design smell (SAR) before/after violin: {design_violin_sar}")

    implementation_violin = _plot_rq5_smell_violin(
        Path("data/analysis/designite/deltas/implementation_smell_deltas.parquet"),
        smell_label="Implementation Smell",
        output_name="rq5_implementation_smell_violin",
        metadata=commits_meta,
        sar_only=False,
        stats_accumulator=wilcoxon_records,
    )
    if implementation_violin:
        print(f"  Saved implementation smell before/after violin: {implementation_violin}")
    implementation_violin_sar = _plot_rq5_smell_violin(
        Path("data/analysis/designite/deltas/implementation_smell_deltas.parquet"),
        smell_label="Implementation Smell (SAR)",
        output_name="rq5_implementation_smell_violin_sar",
        metadata=commits_meta,
        sar_only=True,
        stats_accumulator=wilcoxon_records,
    )
    if implementation_violin_sar:
        print(f"  Saved implementation smell (SAR) before/after violin: {implementation_violin_sar}")

    _apply_fdr_correction(wilcoxon_records)

    print("\nRQ5: Designite metric deltas (type & method)")
    type_metrics_summary = _summarize_designite_metric_delta(
        Path("data/analysis/designite/deltas/type_metric_deltas.parquet"),
        label="Type",
        output_name="rq5_type_metrics",
        metadata=commits_meta,
        sar_only=False,
    )
    if type_metrics_summary:
        print(f"  Saved type metric summary: {type_metrics_summary}")
    type_metrics_summary_sar = _summarize_designite_metric_delta(
        Path("data/analysis/designite/deltas/type_metric_deltas.parquet"),
        label="Type",
        output_name="rq5_type_metrics_sar",
        metadata=commits_meta,
        sar_only=True,
    )
    if type_metrics_summary_sar:
        print(f"  Saved type metric SAR summary: {type_metrics_summary_sar}")

    method_metrics_summary = _summarize_designite_metric_delta(
        Path("data/analysis/designite/deltas/method_metric_deltas.parquet"),
        label="Method",
        output_name="rq5_method_metrics",
        metadata=commits_meta,
        sar_only=False,
    )
    if method_metrics_summary:
        print(f"  Saved method metric summary: {method_metrics_summary}")
    method_metrics_summary_sar = _summarize_designite_metric_delta(
        Path("data/analysis/designite/deltas/method_metric_deltas.parquet"),
        label="Method",
        output_name="rq5_method_metrics_sar",
        metadata=commits_meta,
        sar_only=True,
    )
    if method_metrics_summary_sar:
        print(f"  Saved method metric SAR summary: {method_metrics_summary_sar}")


if __name__ == "__main__":
    main()
