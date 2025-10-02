"""RQ1: Count refactoring instances and commits in agentic pull requests."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .rq_common import OUTPUT_DIR, write_json


def rq1_refactoring_instances_agentic(
    commits: Optional[pd.DataFrame],
    refminer: Optional[pd.DataFrame],
) -> Dict[str, int | bool]:
    """Return summary statistics for agentic refactoring activity."""
    if commits is None:
        raise FileNotFoundError("Missing commits_with_refactoring.parquet")

    agentic = commits[commits.get("agent").notna()] if "agent" in commits.columns else commits.iloc[0:0]
    total_agentic_commits = int(agentic["sha"].nunique()) if not agentic.empty else 0
    agentic_refactor_commits = int(agentic[agentic.get("has_refactoring", False)]["sha"].nunique()) if not agentic.empty else 0

    instances = 0
    if refminer is not None and not refminer.empty and total_agentic_commits:
        agentic_shas = set(agentic["sha"].unique())
        instances = int(refminer[refminer["commit_sha"].isin(agentic_shas)].shape[0])

    result: Dict[str, int | bool] = {
        "total_agentic_file_changes": int(len(agentic)),
        "total_agentic_commits": total_agentic_commits,
        "agentic_refactoring_commits": agentic_refactor_commits,
        "agentic_refactoring_instances": instances,
        "refminer_available": bool(refminer is not None and not refminer.empty),
    }

    write_json(result, OUTPUT_DIR / "rq1_refactoring_instances.json")
    return result


__all__ = ["rq1_refactoring_instances_agentic"]
