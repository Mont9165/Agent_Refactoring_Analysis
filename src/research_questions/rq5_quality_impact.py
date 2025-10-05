"""RQ5: Quantify quality delta distributions for Designite and readability metrics."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .rq_common import OUTPUT_DIR, write_json

DESIGNITE_DELTA_DIR = Path("data/analysis/designite/deltas")
READABILITY_DIR = Path("data/analysis/readability")
QUALITY_OUTPUT_DIR = OUTPUT_DIR / "rq5_quality"

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
            metric_entry["by_refactoring_type"][ref_type] = {
                "stats": stats,
                "entity_kind_counts": group["entity_kind"].value_counts().to_dict(),
                "violin_sample": _sample_values(group["delta"]),
            }

        # Commit-level aggregation (sum of deltas per commit & ref type)
        commit_df = (
            metric_df
            .groupby(["commit_sha", "refactoring_type"], as_index=False)["delta"]
            .sum()
        )
        for ref_type, group in commit_df.groupby("refactoring_type"):
            stats = _describe(group["delta"])
            if not stats:
                continue
            metric_entry["by_commit"][ref_type] = {
                "stats": stats,
                "violin_sample": _sample_values(group["delta"]),
            }

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


def rq5_quality_impact(*_: pd.DataFrame) -> Dict[str, object]:
    """Compute RQ5 quality impact summaries for Designite and readability metrics."""

    QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    designite_frames = []
    for path in (
        DESIGNITE_DELTA_DIR / "type_metric_deltas.parquet",
        DESIGNITE_DELTA_DIR / "method_metric_deltas.parquet",
    ):
        frame = _load_parquet(path)
        if frame is not None:
            designite_frames.append(frame)
    designite_df = pd.concat(designite_frames, ignore_index=True) if designite_frames else pd.DataFrame()

    readability_df = _load_parquet(READABILITY_DIR / "readability_deltas.parquet") or pd.DataFrame()

    result = {
        "designite": _summarize_designite(designite_df),
        "readability": _summarize_readability(readability_df),
        "wilcoxon_available": wilcoxon is not None,
    }

    write_json(result, QUALITY_OUTPUT_DIR / "rq5_quality_impact.json")

    # Persist human-friendly CSV snapshots for downstream plotting
    if not designite_df.empty:
        designite_df.to_csv(QUALITY_OUTPUT_DIR / "designite_metric_deltas.csv", index=False)
    if not readability_df.empty:
        readability_df.to_csv(QUALITY_OUTPUT_DIR / "readability_deltas.csv", index=False)

    return result


__all__ = ["rq5_quality_impact"]

