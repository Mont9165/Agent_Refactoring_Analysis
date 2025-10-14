#!/usr/bin/env python3
"""Analyze Designite metric deltas (smells and structural) overall and by motivation/SAR cohorts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DELTAS_DIR = Path("data/analysis/designite/deltas")
DEFAULT_OUTPUT_DIR = Path("data/analysis/designite/metric_analysis")
MOTIVATION_CSV = Path("data/analysis/refactoring_instances/gpt_refactoring_motivation_update.csv")
COMMITS_PARQUET = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")


def _load_metric_deltas(deltas_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    mapping = [
        ("design_smell", deltas_dir / "design_smell_deltas.parquet"),
        ("implementation_smell", deltas_dir / "implementation_smell_deltas.parquet"),
        ("type_metric", deltas_dir / "type_metric_deltas.parquet"),
        ("method_metric", deltas_dir / "method_metric_deltas.parquet"),
    ]
    for category, path in mapping:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["metric_category"] = category
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No metric delta parquet files found under "
            f"{deltas_dir}. Expected one or more of design_smell_deltas.parquet, "
            "implementation_smell_deltas.parquet, type_metric_deltas.parquet, method_metric_deltas.parquet."
        )
    out = pd.concat(frames, ignore_index=True)
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
    out = out.dropna(subset=["delta"])
    out = out.rename(columns={"metric": "metric_name"})
    return out


def _load_metadata() -> pd.DataFrame:
    if not MOTIVATION_CSV.exists():
        raise FileNotFoundError(f"Motivation CSV not found: {MOTIVATION_CSV}")
    if not COMMITS_PARQUET.exists():
        raise FileNotFoundError(f"Commit metadata parquet not found: {COMMITS_PARQUET}")

    motivations = pd.read_csv(MOTIVATION_CSV, usecols=["sha", "type", "confidence"])
    motivations = motivations.rename(columns={"sha": "commit_sha", "type": "motivation_label"})
    motivations = motivations.drop_duplicates(subset="commit_sha", keep="first")

    commits = pd.read_parquet(COMMITS_PARQUET, columns=["sha", "is_self_affirmed"])
    commits = commits.rename(columns={"sha": "commit_sha"})

    meta = motivations.merge(commits, on="commit_sha", how="left")
    return meta


def _wilcoxon_signed_rank(series: pd.Series) -> Tuple[float, float]:
    non_zero = series[series != 0.0]
    if len(non_zero) == 0:
        return np.nan, np.nan
    try:
        stat, p_value = stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p_value = np.nan, np.nan
    return float(stat) if stat is not None else np.nan, float(p_value) if p_value is not None else np.nan


def _rank_biserial(series: pd.Series) -> float:
    improvements = (series < 0).sum()
    degradations = (series > 0).sum()
    total = improvements + degradations
    if total == 0:
        return 0.0
    return (improvements - degradations) / total


def _benjamini_hochberg(p_values: Iterable[float]) -> List[Optional[float]]:
    p_values_array = np.array([p if pd.notna(p) else np.nan for p in p_values], dtype=float)
    n = np.count_nonzero(~np.isnan(p_values_array))
    if n == 0:
        return [np.nan for _ in p_values_array]
    order = np.argsort(p_values_array, kind="mergesort")
    ranked = p_values_array[order]
    adjusted = np.full_like(ranked, np.nan, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        if np.isnan(ranked[i]):
            continue
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        prev = val
        adjusted[i] = val
    result = np.full_like(p_values_array, np.nan, dtype=float)
    result[order] = adjusted
    return result.tolist()


def compute_baseline_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (metric_category, metric_name), group in metric_df.groupby(["metric_category", "metric_name"]):
        deltas = group["delta"].dropna()
        if deltas.empty:
            continue
        observation_count = int(len(deltas))
        commit_count = int(group["commit_sha"].nunique()) if "commit_sha" in group.columns else observation_count
        improved = int((deltas < 0).sum())
        worsened = int((deltas > 0).sum())
        unchanged = int(observation_count - improved - worsened)
        stat, p_value = _wilcoxon_signed_rank(deltas)
        record = {
            "metric_category": metric_category,
            "metric_name": metric_name,
            "observation_count": observation_count,
            "unique_commit_count": commit_count,
            "median_delta": float(deltas.median()),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=0)),
            "q1_delta": float(deltas.quantile(0.25)),
            "q3_delta": float(deltas.quantile(0.75)),
            "iqr_delta": float(deltas.quantile(0.75) - deltas.quantile(0.25)),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "improved_commits": improved,
            "worsened_commits": worsened,
            "unchanged_commits": unchanged,
            "improvement_rate": improved / observation_count if observation_count else np.nan,
            "worsening_rate": worsened / observation_count if observation_count else np.nan,
            "wilcoxon_stat": stat,
            "wilcoxon_p_value": p_value,
            "rank_biserial_effect": _rank_biserial(deltas),
        }
        records.append(record)

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    summary["wilcoxon_p_value_fdr"] = _benjamini_hochberg(summary["wilcoxon_p_value"])
    return summary.sort_values(["metric_category", "metric_name"]).reset_index(drop=True)


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (metric_category, metric_name, group_value), group in df.groupby(["metric_category", "metric_name", group_col]):
        deltas = group["delta"].dropna()
        if deltas.empty:
            continue
        observation_count = int(len(deltas))
        commit_count = int(group["commit_sha"].nunique()) if "commit_sha" in group.columns else observation_count
        records.append(
            {
                "metric_category": metric_category,
                "metric_name": metric_name,
                group_col: group_value,
                "observation_count": observation_count,
                "unique_commit_count": commit_count,
                "median_delta": float(deltas.median()),
                "mean_delta": float(deltas.mean()),
                "q1_delta": float(deltas.quantile(0.25)),
                "q3_delta": float(deltas.quantile(0.75)),
                "improvement_rate": float((deltas < 0).mean()),
                "worsening_rate": float((deltas > 0).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _kruskal_tests(df: pd.DataFrame, group_col: str, min_group_size: int = 10) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (metric_category, metric_name), group in df.groupby(["metric_category", "metric_name"]):
        grouped_values: List[np.ndarray] = []
        counts: List[int] = []
        for label, sub in group.groupby(group_col):
            deltas = sub["delta"].dropna().to_numpy()
            if len(deltas) >= min_group_size:
                grouped_values.append(deltas)
                counts.append(len(deltas))
        if len(grouped_values) < 2:
            records.append(
                {
                    "metric_category": metric_category,
                    "metric_name": metric_name,
                    "groups_tested": len(grouped_values),
                    "kruskal_stat": np.nan,
                    "kruskal_p_value": np.nan,
                    "eta_squared": np.nan,
                    "epsilon_squared": np.nan,
                }
            )
            continue
        stat, p_value = stats.kruskal(*grouped_values)
        total_n = sum(counts)
        eta_squared = np.nan
        epsilon_squared = np.nan
        if total_n > len(grouped_values):
            eta_squared = (stat - len(grouped_values) + 1) / (total_n - len(grouped_values))
            epsilon_squared = stat / (total_n - 1)
        records.append(
            {
                "metric_category": metric_category,
                "metric_name": metric_name,
                "groups_tested": len(grouped_values),
                "kruskal_stat": float(stat),
                "kruskal_p_value": float(p_value),
                "eta_squared": float(eta_squared) if not np.isnan(eta_squared) else np.nan,
                "epsilon_squared": float(epsilon_squared) if not np.isnan(epsilon_squared) else np.nan,
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["kruskal_p_value_fdr"] = _benjamini_hochberg(result["kruskal_p_value"])
    return result


def _mannwhitney_tests(df: pd.DataFrame, group_col: str, positive_label: bool = True, min_group_size: int = 10) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (metric_category, metric_name), group in df.groupby(["metric_category", "metric_name"]):
        positive = group[group[group_col] == positive_label]["delta"].dropna().to_numpy()
        negative = group[group[group_col] != positive_label]["delta"].dropna().to_numpy()
        if len(positive) < min_group_size or len(negative) < min_group_size:
            records.append(
                {
                    "metric_category": metric_category,
                    "metric_name": metric_name,
                    "positive_count": int(len(positive)),
                    "negative_count": int(len(negative)),
                    "mannwhitney_stat": np.nan,
                    "mannwhitney_p_value": np.nan,
                    "rank_biserial_effect": np.nan,
                }
            )
            continue
        res = stats.mannwhitneyu(positive, negative, alternative="two-sided", method="auto")
        n_pos = len(positive)
        n_neg = len(negative)
        effect = 2 * res.statistic / (n_pos * n_neg) - 1
        records.append(
            {
                "metric_category": metric_category,
                "metric_name": metric_name,
                "positive_count": int(n_pos),
                "negative_count": int(n_neg),
                "mannwhitney_stat": float(res.statistic),
                "mannwhitney_p_value": float(res.pvalue),
                "rank_biserial_effect": float(effect),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["mannwhitney_p_value_fdr"] = _benjamini_hochberg(result["mannwhitney_p_value"])
    return result


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df.to_csv(path, index=False)
    else:
        df.sort_values(df.columns.tolist(), axis=0).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deltas-dir",
        type=Path,
        default=DEFAULT_DELTAS_DIR,
        help=f"Directory containing smell delta parquet files (default: {DEFAULT_DELTAS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write summary outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=10,
        help="Minimum commits per cohort to include in statistical tests (default: 10).",
    )
    args = parser.parse_args()

    metric_df = _load_metric_deltas(args.deltas_dir)
    metadata = _load_metadata()
    joined = metric_df.merge(metadata, left_on="commit_sha", right_on="commit_sha", how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = compute_baseline_summary(metric_df)
    _write_dataframe(baseline, args.output_dir / "metric_baseline_summary.csv")

    refactoring_summary = _group_summary(joined, "refactoring_type")
    if not refactoring_summary.empty:
        refactoring_summary = refactoring_summary.sort_values(
            ["metric_category", "metric_name", "refactoring_type"]
        )
    refactoring_summary.to_csv(args.output_dir / "metric_by_refactoring_summary.csv", index=False)

    refactoring_tests = _kruskal_tests(joined, "refactoring_type", min_group_size=args.min_group_size)
    refactoring_tests.to_csv(args.output_dir / "metric_by_refactoring_tests.csv", index=False)

    motivation_summary = _group_summary(joined, "motivation_label")
    if not motivation_summary.empty:
        motivation_summary = motivation_summary.sort_values(["metric_category", "metric_name", "motivation_label"])
    motivation_summary.to_csv(args.output_dir / "metric_by_motivation_summary.csv", index=False)

    motivation_tests = _kruskal_tests(joined, "motivation_label", min_group_size=args.min_group_size)
    motivation_tests.to_csv(args.output_dir / "metric_by_motivation_tests.csv", index=False)

    sar_summary = _group_summary(joined, "is_self_affirmed")
    if not sar_summary.empty:
        sar_summary = sar_summary.sort_values(["metric_category", "metric_name", "is_self_affirmed"])
    sar_summary.to_csv(args.output_dir / "metric_by_sar_summary.csv", index=False)

    sar_tests = _mannwhitney_tests(joined, "is_self_affirmed", positive_label=True, min_group_size=args.min_group_size)
    sar_tests.to_csv(args.output_dir / "metric_by_sar_tests.csv", index=False)

    print(f"Wrote baseline summary ({len(baseline)} rows) to {args.output_dir/'metric_baseline_summary.csv'}")
    print(f"Wrote refactoring, motivation, and SAR summaries/tests to {args.output_dir}")


if __name__ == "__main__":
    main()
