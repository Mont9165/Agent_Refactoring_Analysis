"""RQ3: Identify common refactoring types and expose per-commit distributions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .rq_common import OUTPUT_DIR, write_csv, write_json


def _run_mannwhitneyu_tests(
    per_commit_df: pd.DataFrame, top_types: list[str]
) -> Dict[str, Dict[str, float]]:
    """Perform Mann-Whitney U tests comparing SAR vs. Overall distributions."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return {"error": "scipy is not installed, skipping Mann-Whitney U tests."}

    results = {}
    sar_df = per_commit_df[per_commit_df["is_self_affirmed"]]
    if sar_df.empty:
        return {"note": "No SAR commits to compare."}

    for ref_type in top_types:
        overall_series = per_commit_df[
            per_commit_df["refactoring_type"] == ref_type
        ]["instance_count"]
        sar_series = sar_df[sar_df["refactoring_type"] == ref_type]["instance_count"]

        if sar_series.empty or len(overall_series) < 2 or len(sar_series) < 2:
            continue

        # Compare SAR distribution against the non-SAR distribution for a cleaner test
        non_sar_series = per_commit_df[
            (per_commit_df["refactoring_type"] == ref_type) & (~per_commit_df["is_self_affirmed"])
        ]["instance_count"]

        if non_sar_series.empty:
            continue

        # Test if SAR instances are significantly GREATER than non-SAR
        try:
            stat, p_value = mannwhitneyu(
                sar_series, non_sar_series, alternative="greater"
            )
            results[ref_type] = {"statistic": stat, "p_value": p_value}
        except ValueError:
            # Can occur if all values are identical
            continue

    return results


def rq3_top_refactoring_types(
    refminer: Optional[pd.DataFrame],
    commits: Optional[pd.DataFrame],
    top_n: Optional[int] = None,
    *,
    min_count: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return refactoring type frequency summaries and per-commit distributions.

    Besides global counts, this function writes a commit-level CSV so downstream
    consumers can build box/violin plots that are robust to outlier commits.
    """
    if refminer is None or refminer.empty:
        result = {"error": "No RefactoringMiner results available"}
        write_json(result, base_dir / "rq3_top_refactoring_types.json")
        return result

    base_dir = (output_dir or (OUTPUT_DIR / "rq3"))
    base_dir.mkdir(parents=True, exist_ok=True)

    per_commit = (
        refminer.groupby(["commit_sha", "refactoring_type"]).size().reset_index(name="instance_count")
    )
    dist_path = base_dir / "rq3_refactoring_type_distribution.csv"

    if commits is not None and "is_self_affirmed" in commits.columns:
        sar_flags = commits.groupby("sha")["is_self_affirmed"].max()
        per_commit["is_self_affirmed"] = (
            per_commit["commit_sha"].map(sar_flags).fillna(False).astype(bool)
        )
    else:
        per_commit["is_self_affirmed"] = False

    grouped = per_commit.groupby("refactoring_type")["instance_count"]
    count_series = grouped.count()
    valid_types = count_series[count_series >= min_count].index
    if valid_types.empty:
        write_csv(per_commit, dist_path)
        result = {
            "overall_top_types": {},
            "overall_order_median_desc": [],
            "sar_top_types": None,
            "sar_order_median_desc": None,
            "overall_distribution": {},
            "sar_distribution": None,
            "distribution_csv": str(dist_path),
            "overall_sample_counts": {},
            "sar_sample_counts": None,
            "mannwhitneyu_results": None, # ADDED
        }
        write_json(result, OUTPUT_DIR / "rq3_top_refactoring_types.json")
        return result

    median_series = grouped.median().loc[valid_types]
    median_order = median_series.sort_values(ascending=False)
    selected_types = median_order.index.tolist()
    if top_n is not None:
        selected_types = selected_types[:top_n]

    totals_series = grouped.sum().loc[selected_types]
    sample_counts_series = count_series.loc[selected_types]

    def _stats(series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {
                "min": 0.0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "count": 0,
            }
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        std_val = desc.get("std")
        std_float = float(std_val) if pd.notna(std_val) else 0.0
        return {
            "min": float(desc.get("min", 0.0)),
            "q1": float(desc.get("25%", 0.0)),
            "median": float(desc.get("50%", 0.0)),
            "q3": float(desc.get("75%", 0.0)),
            "max": float(desc.get("max", 0.0)),
            "mean": float(desc.get("mean", 0.0)),
            "std": std_float,
            "count": int(desc.get("count", 0.0)),
        }

    overall_distribution: Dict[str, Dict[str, float]] = {}
    for ref_type in selected_types:
        series = grouped.get_group(ref_type)
        overall_distribution[ref_type] = _stats(series)

    sar_subset = per_commit[per_commit["is_self_affirmed"]]
    sar_top_types: Optional[Dict[str, int]] = None
    sar_distribution: Optional[Dict[str, Dict[str, float]]] = None
    sar_order: Optional[list[str]] = None
    sar_sample_counts: Optional[Dict[str, int]] = None
    if not sar_subset.empty:
        sar_grouped = sar_subset.groupby("refactoring_type")["instance_count"]
        sar_counts = sar_grouped.count()
        sar_valid = sar_counts[sar_counts >= min_count].index
        if not sar_valid.empty:
            sar_medians = sar_grouped.median().loc[sar_valid]
            sar_median_order = sar_medians.sort_values(ascending=False)
            sar_order = sar_median_order.index.tolist()
            if top_n is not None:
                sar_order = sar_order[:top_n]
            sar_totals = sar_grouped.sum().reindex(sar_order)
            sar_top_types = sar_totals.to_dict()
            sar_distribution = {
                ref_type: _stats(sar_grouped.get_group(ref_type)) for ref_type in sar_order
            }
            sar_sample_counts = sar_counts.loc[sar_order].to_dict()
        else:
            sar_top_types = None
            sar_distribution = None
            sar_order = None

    write_csv(per_commit, dist_path)

    # ADDED: Run the statistical test
    mwu_results = _run_mannwhitneyu_tests(per_commit, selected_types)

    # Non-SAR (not self-affirmed) subset
    non_sar_subset = per_commit[per_commit["is_self_affirmed"] == False]
    non_sar_top_types: Optional[Dict[str, int]] = None
    non_sar_distribution: Optional[Dict[str, Dict[str, float]]] = None
    non_sar_order: Optional[list[str]] = None
    non_sar_sample_counts: Optional[Dict[str, int]] = None
    if not non_sar_subset.empty:
        non_sar_grouped = non_sar_subset.groupby("refactoring_type")["instance_count"]
        non_sar_counts = non_sar_grouped.count()
        non_sar_valid = non_sar_counts[non_sar_counts >= min_count].index
        if not non_sar_valid.empty:
            non_sar_medians = non_sar_grouped.median().loc[non_sar_valid]
            non_sar_median_order = non_sar_medians.sort_values(ascending=False)
            non_sar_order = non_sar_median_order.index.tolist()
            if top_n is not None:
                non_sar_order = non_sar_order[:top_n]
            non_sar_totals = non_sar_grouped.sum().reindex(non_sar_order)
            non_sar_top_types = non_sar_totals.to_dict()
            non_sar_distribution = {
                ref_type: _stats(non_sar_grouped.get_group(ref_type)) for ref_type in non_sar_order
            }
            non_sar_sample_counts = non_sar_counts.loc[non_sar_order].to_dict()

    result: Dict[str, object] = {
        "overall_top_types": totals_series.to_dict(),
        "overall_order_median_desc": selected_types,
        "overall_sample_counts": sample_counts_series.to_dict(),
        "sar_top_types": sar_top_types,
        "sar_order_median_desc": sar_order,
        "overall_distribution": overall_distribution,
        "sar_distribution": sar_distribution,
        "sar_sample_counts": sar_sample_counts,
        "non_sar_top_types": non_sar_top_types,
        "non_sar_order_median_desc": non_sar_order,
        "non_sar_distribution": non_sar_distribution,
        "non_sar_sample_counts": non_sar_sample_counts,
        "distribution_csv": str(dist_path),
        "mannwhitneyu_results": mwu_results,
    }
    write_json(result, base_dir / "rq3_top_refactoring_types.json")
    return result


__all__ = ["rq3_top_refactoring_types"]
