#!/usr/bin/env python3
"""Generate descriptive plots summarising the Java refactoring dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataset_summary import get_java_commit_dataframe

DEFAULT_OUTDIR = Path("outputs/dataset_summary")
FONT_SIZE = 16

sns.set(style="whitegrid")


def _violin_plot(
    series: pd.Series,
    ylabel: str,
    output: Path,
    log_scale: bool = False,
    color: str = "#4C72B0",
    clip_pct: float = 0.995,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    raw = series.dropna()
    if log_scale:
        raw = raw[raw > 0]
    if raw.empty:
        return

    if clip_pct and 0 < clip_pct < 1:
        upper = raw.quantile(clip_pct)
        data = raw.clip(upper=upper)
    else:
        data = raw

    if clip_pct and 0 < clip_pct < 1:
        display_mean = raw.mean()
        display_median = raw.median()
    else:
        display_mean = data.mean()
        display_median = data.median()

    plt.figure(figsize=(8, 6))
    sns.violinplot(y=data, color=color, inner="box", cut=0)
    if log_scale:
        plt.yscale("log")
    lower = y_min if y_min is not None else (1.0 if log_scale else 0.0)
    upper = y_max if y_max is not None else None
    if upper is not None or lower is not None:
        if log_scale:
            min_positive = data[data > 0].min() if not data.empty else 1.0
            lower = max(lower, min_positive / 10.0)
        plt.ylim(lower if lower is not None else plt.gca().get_ylim()[0],
                 upper if upper is not None else plt.gca().get_ylim()[1])

    if not pd.isna(display_mean) and (not log_scale or display_mean > 0):
        plt.axhline(display_mean, color="red", linestyle="--", label=f"Mean: {display_mean:.2f}")
    if not pd.isna(display_median) and (not log_scale or display_median > 0):
        plt.axhline(display_median, color="green", linestyle="-", label=f"Median: {display_median:.2f}")
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.tick_params(axis="y", labelsize=FONT_SIZE)
    plt.tick_params(axis="x", labelsize=FONT_SIZE - 2)
    plt.legend(fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=900)
    plt.close()


def _plot_file_distributions(commit_df: pd.DataFrame, outdir: Path) -> None:
    _violin_plot(
        commit_df["file_changes_total"],
        ylabel="# Modified Files",
        output=outdir / "files_per_commit_distribution.pdf",
        color="#76B7B2",
        log_scale=True,
        y_min=1,
        y_max=120,
    )
    _violin_plot(
        commit_df["file_changes_java"],
        ylabel="# Modified Java Files",
        output=outdir / "java_files_per_commit_distribution.pdf",
        color="#59A14F",
        log_scale=True,
        y_min=1,
        y_max=120,
    )


def _plot_line_distributions(commit_df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("commit_stats_additions", "Total Addition Lines", "total_addition_lines_distribution_log.pdf", "lightgreen", (1, 1e5)),
        ("commit_stats_deletions", "Total Deletion Lines", "total_deletions_lines_distribution_log.pdf", "salmon", (1, 1e5)),
        ("commit_stats_total", "Total Line Changes", "total_line_changes_distribution_log.pdf", "#9370DB", None),
    ]
    for col, ylabel, filename, color, limits in metrics:
        _violin_plot(
            commit_df[col],
            ylabel=ylabel,
            output=outdir / filename,
            color=color,
            log_scale=True,
            y_min=limits[0] if limits else None,
            y_max=limits[1] if limits else None,
        )


def _plot_pr_status(commit_df: pd.DataFrame, outdir: Path) -> None:
    pr_group = commit_df.groupby("pr_id")
    pr_state = pr_group["state"].first().fillna("unknown")
    pr_counts = pr_state.value_counts().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    pr_counts.plot(kind="bar", color="#C44E52", alpha=0.85)
    plt.title("PR state distribution")
    plt.ylabel("Number of PRs")
    plt.tight_layout()
    plt.savefig(outdir / "pr_state_distribution.png", dpi=160)
    plt.close()

    closed_mask = pr_state == "closed"
    merged = pr_group["is_merged"].first().fillna(False)
    merged_counts = pd.Series({
        "merged": int((merged & closed_mask).sum()),
        "closed_unmerged": int(((~merged) & closed_mask).sum()),
    })
    plt.figure(figsize=(6, 4))
    merged_counts.plot(kind="bar", color="#8172B3", alpha=0.85)
    plt.title("Closed PR outcomes")
    plt.ylabel("Number of PRs")
    plt.tight_layout()
    plt.savefig(outdir / "pr_closed_merge_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory to store generated plots (default: outputs/dataset_summary)",
    )
    args = parser.parse_args()

    commit_df = get_java_commit_dataframe()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_file_distributions(commit_df, outdir)
    _plot_line_distributions(commit_df, outdir)
    _plot_pr_status(commit_df, outdir)

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
