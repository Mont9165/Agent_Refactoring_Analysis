"""Utilities to aggregate Designite metrics and compare refactoring vs non-refactoring commits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _resolve_designite_root() -> Path:
    env_root = os.environ.get("DESIGNITE_OUTPUT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if env_path.exists():
            return env_path

    for candidate in (
        Path("tools/DesigniteRunner/outputs"),
        Path("data/designite/outputs"),
    ):
        if candidate.exists():
            return candidate
    return Path("tools/DesigniteRunner/outputs")


DESIGNITE_OUTPUT_ROOT = _resolve_designite_root()
ANALYSIS_DIR = Path("data/analysis/designite")


@dataclass
class CommitLocator:
    owner: str
    repo: str
    sha: str

    @classmethod
    def from_html_url(cls, sha: str, html_url: Optional[str]) -> Optional["CommitLocator"]:
        if not html_url or not isinstance(html_url, str):
            return None
        parts = html_url.split("/")
        if len(parts) < 5:
            return None
        owner, repo = parts[3], parts[4]
        return cls(owner=owner, repo=repo, sha=str(sha))

    @property
    def designite_dir(self) -> Path:
        return DESIGNITE_OUTPUT_ROOT / self.owner / self.repo / self.sha


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


def _numeric_stats(df: pd.DataFrame, columns: Iterable[str], prefix: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if df is None or df.empty:
        for col in columns:
            out[f"{prefix}_{col.lower()}_mean"] = None
        return out
    for col in columns:
        if col not in df.columns:
            out[f"{prefix}_{col.lower()}_mean"] = None
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out[f"{prefix}_{col.lower()}_mean"] = float(series.mean()) if series.notna().any() else None
    return out


def parse_designite_commit(dir_path: Path) -> Dict[str, Optional[float]]:
    """Aggregate metrics from a single Designite output directory."""
    metrics: Dict[str, Optional[float]] = {}

    type_df = _safe_read_csv(dir_path / "typeMetrics.csv")
    if type_df is not None:
        metrics["type_count"] = int(len(type_df))
        if "LOC" in type_df.columns:
            loc_series = pd.to_numeric(type_df["LOC"], errors="coerce")
            metrics["type_loc_sum"] = float(loc_series.sum(skipna=True)) if loc_series.notna().any() else None
        else:
            metrics["type_loc_sum"] = None
        metrics.update(_numeric_stats(type_df, ["LOC", "WMC", "NOM", "NOF", "DIT", "LCOM", "FANIN", "FANOUT"], "type"))
    else:
        metrics["type_count"] = 0
        metrics["type_loc_sum"] = None
        metrics.update({f"type_{col.lower()}_mean": None for col in ["LOC", "WMC", "NOM", "NOF", "DIT", "LCOM", "FANIN", "FANOUT"]})

    method_df = _safe_read_csv(dir_path / "methodMetrics.csv")
    if method_df is not None:
        metrics["method_count"] = int(len(method_df))
        metrics.update(_numeric_stats(method_df, ["LOC", "CC", "PC"], "method"))
    else:
        metrics["method_count"] = 0
        metrics.update({f"method_{col.lower()}_mean": None for col in ["LOC", "CC", "PC"]})

    design_df = _safe_read_csv(dir_path / "designCodeSmells.csv")
    if design_df is not None:
        metrics["design_smell_count"] = int(len(design_df))
        if not design_df.empty and "Code Smell" in design_df.columns:
            counts = design_df["Code Smell"].value_counts()
            metrics["design_smell_unique"] = int(counts.shape[0])
            metrics["design_smell_top"] = str(counts.idxmax()) if not counts.empty else None
        else:
            metrics["design_smell_unique"] = 0
            metrics["design_smell_top"] = None
    else:
        metrics["design_smell_count"] = 0
        metrics["design_smell_unique"] = 0
        metrics["design_smell_top"] = None

    impl_df = _safe_read_csv(dir_path / "implementationCodeSmells.csv")
    if impl_df is not None:
        metrics["implementation_smell_count"] = int(len(impl_df))
        if not impl_df.empty and "Code Smell" in impl_df.columns:
            counts = impl_df["Code Smell"].value_counts()
            metrics["implementation_smell_unique"] = int(counts.shape[0])
            metrics["implementation_smell_top"] = str(counts.idxmax()) if not counts.empty else None
        else:
            metrics["implementation_smell_unique"] = 0
            metrics["implementation_smell_top"] = None
    else:
        metrics["implementation_smell_count"] = 0
        metrics["implementation_smell_unique"] = 0
        metrics["implementation_smell_top"] = None

    # Derived ratios where possible
    loc = metrics.get("type_loc_sum")
    type_count = metrics.get("type_count", 0) or 0
    method_count = metrics.get("method_count", 0) or 0
    design_smells = metrics.get("design_smell_count", 0) or 0
    impl_smells = metrics.get("implementation_smell_count", 0) or 0
    metrics["design_smell_density_per_type"] = float(design_smells / type_count) if type_count else None
    metrics["implementation_smell_density_per_method"] = float(impl_smells / method_count) if method_count else None
    metrics["smell_density_per_kloc"] = float((design_smells + impl_smells) / (loc / 1000.0)) if loc and loc > 0 else None

    return metrics


def _series_stats(series: pd.Series) -> Optional[Dict[str, float]]:
    """Return basic statistics for a numeric series, or None if empty."""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None

    def _to_float(value: float) -> float:
        return float(value) if value is not None and not np.isnan(value) else float("nan")

    stats = {
        "count": int(valid.count()),
        "mean": _to_float(valid.mean()),
        "median": _to_float(valid.median()),
        "std": _to_float(valid.std(ddof=0)),
        "min": _to_float(valid.min()),
        "max": _to_float(valid.max()),
        "q1": _to_float(valid.quantile(0.25)),
        "q3": _to_float(valid.quantile(0.75)),
    }

    # Replace NaN placeholders with None for JSON serialization
    return {k: (None if np.isnan(v) else v) for k, v in stats.items()}


def build_commit_metrics(commits: pd.DataFrame) -> pd.DataFrame:
    """Return per-commit Designite aggregates for commits with available outputs."""
    rows = []
    for _, row in commits.iterrows():
        locator = CommitLocator.from_html_url(row.get("sha"), row.get("html_url"))
        if locator is None:
            continue
        dir_path = locator.designite_dir
        if not dir_path.exists():
            continue
        metrics = parse_designite_commit(dir_path)
        metrics.update({
            "sha": locator.sha,
            "owner": locator.owner,
            "repo": locator.repo,
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_impact(commit_metrics: pd.DataFrame, commits: pd.DataFrame) -> Dict[str, object]:
    if commit_metrics.empty:
        return {}

    merged = commit_metrics.merge(
        commits[[c for c in ["sha", "has_refactoring", "refactoring_instance_count", "agent"] if c in commits.columns]],
        on="sha",
        how="left",
    )

    metrics_cols = [
        col for col in commit_metrics.columns
        if col not in {"sha", "owner", "repo"}
        and commit_metrics[col].dtype != object
    ]

    summary: Dict[str, object] = {
        "total_commits_with_designite": int(len(merged)),
        "refactoring_commits_with_designite": int(merged[merged.get("has_refactoring") == True].shape[0]),
        "non_refactoring_commits_with_designite": int(merged[merged.get("has_refactoring") != True].shape[0]),
    }

    grouped = merged.groupby(merged.get("has_refactoring") == True)
    metrics_summary: Dict[str, Dict[str, object]] = {}
    mean_only: Dict[str, Dict[str, Optional[float]]] = {}
    if True in grouped.groups and False in grouped.groups:
        ref_df = grouped.get_group(True)
        non_df = grouped.get_group(False)
        for col in metrics_cols:
            ref_stats = _series_stats(ref_df[col])
            non_stats = _series_stats(non_df[col])
            if not ref_stats or not non_stats:
                continue

            ref_mean = ref_stats.get("mean")
            non_mean = non_stats.get("mean")
            ref_median = ref_stats.get("median")
            non_median = non_stats.get("median")

            if ref_mean is None or non_mean is None or ref_median is None or non_median is None:
                continue

            diff_mean = float(ref_mean - non_mean)
            diff_median = float(ref_median - non_median)
            pct_mean = float(diff_mean / non_mean * 100.0) if non_mean != 0 else None
            pct_median = float(diff_median / non_median * 100.0) if non_median != 0 else None

            metrics_summary[col] = {
                "refactoring": ref_stats,
                "non_refactoring": non_stats,
                "difference": {
                    "mean": diff_mean,
                    "median": diff_median,
                    "pct_change_mean": pct_mean,
                    "pct_change_median": pct_median,
                },
            }

            mean_only[col] = {
                "refactoring_mean": ref_mean,
                "non_refactoring_mean": non_mean,
                "difference": diff_mean,
                "pct_change_vs_non_ref": pct_mean,
            }

    summary["by_refactoring_flag"] = mean_only
    summary["metrics"] = metrics_summary

    if metrics_summary:
        boxplot_data: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        for metric, stats in metrics_summary.items():
            ref_stats = stats.get("refactoring", {})
            non_stats = stats.get("non_refactoring", {})
            boxplot_data[metric] = {
                "refactoring": {
                    key: ref_stats.get(key)
                    for key in ("min", "q1", "median", "q3", "max")
                },
                "non_refactoring": {
                    key: non_stats.get(key)
                    for key in ("min", "q1", "median", "q3", "max")
                },
            }
        summary["boxplot_data"] = boxplot_data

    if "agent" in merged.columns:
        agent_stats = {}
        for agent, agent_df in merged.groupby("agent"):
            if not agent or agent_df.empty:
                continue
            counts = agent_df.groupby(agent_df.get("has_refactoring") == True).size().to_dict()
            agent_stats[str(agent)] = {
                "total_commits": int(len(agent_df)),
                "refactoring_commits": int(counts.get(True, 0)),
                "non_refactoring_commits": int(counts.get(False, 0)),
            }
        summary["by_agent_counts"] = agent_stats

    if "refactoring_instance_count" in merged.columns:
        corr_data = merged[["refactoring_instance_count"] + metrics_cols].copy()
        corr = corr_data.corr(numeric_only=True)
        # Extract correlation row for refactoring count if available
        if "refactoring_instance_count" in corr.columns:
            correlations = corr["refactoring_instance_count"].drop("refactoring_instance_count", errors="ignore")
            summary["refactoring_instance_correlations"] = correlations.dropna().to_dict()

    return summary
