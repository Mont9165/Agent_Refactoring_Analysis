"""RQ2: Measure self-affirmed refactoring rates in agentic commits."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .rq_common import OUTPUT_DIR, write_json


def rq2_self_affirmed_percentage(commits: Optional[pd.DataFrame]) -> Dict[str, float | int]:
    """Compute percentage of agentic refactoring commits that are self-affirmed."""
    if commits is None:
        raise FileNotFoundError("Missing commits_with_refactoring.parquet")

    df = commits.copy()
    if "is_self_affirmed" not in df.columns:
        df["is_self_affirmed"] = False

    agentic = df[df.get("agent").notna()] if "agent" in df.columns else df.iloc[0:0]
    agentic_ref = agentic[agentic.get("has_refactoring", False)] if not agentic.empty else agentic

    total_commits = int(agentic_ref["sha"].nunique()) if not agentic_ref.empty else 0
    affirmed_commits = 0
    if total_commits:
        affirmed_commits = int(agentic_ref.groupby("sha")["is_self_affirmed"].max().sum())
    percentage = (affirmed_commits / total_commits * 100.0) if total_commits else 0.0

    result: Dict[str, float | int] = {
        "agentic_refactoring_commits": total_commits,
        "self_affirmed_commits": affirmed_commits,
        "self_affirmed_percentage": round(percentage, 2),
    }

    write_json(result, OUTPUT_DIR / "rq2_self_affirmed_percentage.json")
    return result


__all__ = ["rq2_self_affirmed_percentage"]
