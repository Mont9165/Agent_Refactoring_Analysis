"""RQ5: Quantify quality delta distributions for Designite and readability metrics."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .rq_common import OUTPUT_DIR, write_json

DESIGNITE_DELTA_DIR = Path("data/analysis/designite/deltas")
READABILITY_DIR = Path("data/analysis/readability")
QUALITY_OUTPUT_DIR = OUTPUT_DIR / "rq5"

try:  # Optional dependency for Wilcoxon signed-rank test
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover - scipy not guaranteed in all environments
    wilcoxon = None


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _numeric_series(values: Iterable[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def _wilcoxon_stats(series: pd.Series) -> Optional[Dict[str, float]]:
    if wilcoxon is None or len(series) < 2:
        return None
    if math.isclose(series.abs().sum(), 0.0):
        return None
    try:
        stat, pvalue = wilcoxon(series)
    except ValueError:
        return None
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def _describe(series: Iterable[object]) -> Optional[Dict[str, float]]:
    numeric = _numeric_series(series)
    if numeric.empty:
        return None
    stats = {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)) if numeric.count() > 1 else 0.0,
        "min": float(numeric.min()),
        "q1": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.5)),
        "q3": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }
    wilcoxon_result = _wilcoxon_stats(numeric)
    if wilcoxon_result:
        stats["wilcoxon"] = wilcoxon_result
    return stats


def _sample_values(series: Iterable[object], limit: int = 500) -> List[float]:
    numeric = _numeric_series(series)
    if numeric.empty:
        return []
    if len(numeric) <= limit:
        return [float(x) for x in numeric]
    sample = numeric.sample(n=limit, random_state=0)
    return [float(x) for x in sample]


def _summarize_designite(designite_df: pd.DataFrame) -> Dict[str, object]:
    if designite_df.empty:
        return {"status": "missing"}

    designite_df = designite_df.copy()
    designite_df["delta"] = pd.to_numeric(designite_df["delta"], errors="coerce")
    designite_df["before_value"] = pd.to_numeric(designite_df.get("before_value"), errors="coerce")
    designite_df["after_value"] = pd.to_numeric(designite_df.get("after_value"), errors="coerce")
    designite_df["improvement_rate_pct"] = np.where(
        designite_df["before_value"] > 0,
        (designite_df["before_value"] - designite_df["after_value"]) / designite_df["before_value"] * 100.0,
        np.nan,
    )
    designite_df = designite_df.dropna(subset=["delta"])
    if designite_df.empty:
        return {"status": "empty"}

    summary: Dict[str, object] = {
        "status": "ok",
        "metrics": {},
    }

    for metric, metric_df in designite_df.groupby("metric"):
        metric_entry: Dict[str, object] = {
            "by_refactoring_type": {},
            "by_commit": {},
        }

        # Refactoring-type statistics at entity granularity
        for ref_type, group in metric_df.groupby("refactoring_type"):
            stats = _describe(group["delta"])
            if not stats:
                continue
            improvement_stats = _describe(group["improvement_rate_pct"])
            metric_entry["by_refactoring_type"][ref_type] = {
                "stats": stats,
                "entity_kind_counts": group["entity_kind"].value_counts().to_dict(),
                "violin_sample": _sample_values(group["delta"]),
            }
            if improvement_stats:
                metric_entry["by_refactoring_type"][ref_type]["improvement_rate"] = {
                    "stats": improvement_stats,
                    "violin_sample": _sample_values(group["improvement_rate_pct"]),
                }

        # Commit-level aggregation (sum of deltas per commit & ref type)
        commit_df = (
            metric_df
            .groupby(["commit_sha", "refactoring_type"], as_index=False)
            .agg(
                delta=("delta", "sum"),
                before_total=("before_value", "sum"),
                after_total=("after_value", "sum"),
            )
        )
        commit_df["improvement_rate_pct"] = np.where(
            commit_df["before_total"] > 0,
            (commit_df["before_total"] - commit_df["after_total"]) / commit_df["before_total"] * 100.0,
            np.nan,
        )
        for ref_type, group in commit_df.groupby("refactoring_type"):
            delta_stats = _describe(group["delta"])
            improvement_stats = _describe(group["improvement_rate_pct"])
            if not delta_stats and not improvement_stats:
                continue
            entry: Dict[str, object] = {}
            if delta_stats:
                entry["stats"] = delta_stats
                entry["violin_sample"] = _sample_values(group["delta"])
            if improvement_stats:
                entry["improvement_rate"] = {
                    "stats": improvement_stats,
                    "violin_sample": _sample_values(group["improvement_rate_pct"]),
                }
            if entry:
                metric_entry["by_commit"][ref_type] = entry

        summary["metrics"][metric] = metric_entry

    return summary


def _summarize_readability(readability_df: pd.DataFrame) -> Dict[str, object]:
    if readability_df.empty:
        return {"status": "missing"}

    readability_df = readability_df.copy()
    readability_df["delta"] = pd.to_numeric(readability_df["delta"], errors="coerce")
    readability_df = readability_df.dropna(subset=["delta"])
    if readability_df.empty:
        return {"status": "empty"}

    summary: Dict[str, object] = {
        "status": "ok",
        "metric": "readability_delta",
        "by_refactoring_type": {},
        "by_commit": {},
    }

    for ref_type, group in readability_df.groupby("refactoring_type"):
        stats = _describe(group["delta"])
        if not stats:
            continue
        summary["by_refactoring_type"][ref_type] = {
            "stats": stats,
            "violin_sample": _sample_values(group["delta"]),
        }

    commit_df = (
        readability_df
        .groupby(["commit_sha", "refactoring_type"], as_index=False)["delta"]
        .mean()
    )
    for ref_type, group in commit_df.groupby("refactoring_type"):
        stats = _describe(group["delta"])
        if not stats:
            continue
        summary["by_commit"][ref_type] = {
            "stats": stats,
            "violin_sample": _sample_values(group["delta"]),
        }

    return summary


def rq5_quality_impact(
    commit_shas: Optional[Iterable[str]] = None,
    *,
    subset_label: str = "overall",
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Compute RQ5 quality impact summaries for Designite and readability metrics."""

    target_dir = (output_dir or QUALITY_OUTPUT_DIR) / (subset_label or "overall")
    target_dir.mkdir(parents=True, exist_ok=True)

    designite_frames = []
    for path in (
        DESIGNITE_DELTA_DIR / "type_metric_deltas.parquet",
        DESIGNITE_DELTA_DIR / "method_metric_deltas.parquet",
        DESIGNITE_DELTA_DIR / "design_smell_deltas.parquet",
        DESIGNITE_DELTA_DIR / "implementation_smell_deltas.parquet",
    ):
        frame = _load_parquet(path)
        if frame is not None:
            designite_frames.append(frame)
    designite_df = pd.concat(designite_frames, ignore_index=True) if designite_frames else pd.DataFrame()

    readability_df = _load_parquet(READABILITY_DIR / "readability_deltas.parquet") or pd.DataFrame()

    if commit_shas is not None:
        commit_sha_set = {str(sha) for sha in commit_shas}
        if not designite_df.empty and "commit_sha" in designite_df.columns:
            designite_df = designite_df[designite_df["commit_sha"].astype(str).isin(commit_sha_set)]
        if not readability_df.empty and "commit_sha" in readability_df.columns:
            readability_df = readability_df[readability_df["commit_sha"].astype(str).isin(commit_sha_set)]

    result = {
        "designite": _summarize_designite(designite_df),
        "readability": _summarize_readability(readability_df),
        "wilcoxon_available": wilcoxon is not None,
    }

    write_json(result, target_dir / "summary.json")

    # Persist human-friendly CSV snapshots for downstream plotting
    if not designite_df.empty:
        designite_df.to_csv(target_dir / "designite_metric_deltas.csv", index=False)
    if not readability_df.empty:
        readability_df.to_csv(target_dir / "readability_deltas.csv", index=False)

    return result


__all__ = ["rq5_quality_impact"]
