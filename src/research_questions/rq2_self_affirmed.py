"""RQ2: Measure self-affirmed refactoring rates in agentic commits."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .rq_common import OUTPUT_DIR, write_json


def rq2_self_affirmed_percentage(
    commits: Optional[pd.DataFrame],
    *,
    subset_label: str = "overall",
    output_dir: Optional[Path] = None,
) -> Dict[str, float | int]:
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

    target_dir = (output_dir or OUTPUT_DIR) / "rq2" / (subset_label or "overall")
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, target_dir / "summary.json")
    return result


def rq2_refactoring_type_affirmed_split(
    refminer: Optional[pd.DataFrame],
    commits: Optional[pd.DataFrame],
    *,
    subset_label: str = "overall",
    top_n: Optional[int] = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Aggregate refactoring instances per type split by SAR vs Non-SAR."""
    if refminer is None or refminer.empty:
        return {"note": "RefactoringMiner results unavailable."}
    if commits is None or commits.empty:
        return {"note": "Commit metadata unavailable."}
    if "commit_sha" not in refminer.columns:
        return {"note": "RefactoringMiner data missing commit_sha column."}
    if "sha" not in commits.columns:
        return {"note": "Commit metadata missing sha column."}

    target_dir = (output_dir or OUTPUT_DIR) / "rq2" / (subset_label or "overall")
    target_dir.mkdir(parents=True, exist_ok=True)

    sar_flags = commits.groupby("sha")["is_self_affirmed"].max() if "is_self_affirmed" in commits.columns else None
    if sar_flags is None or sar_flags.empty:
        return {"note": "is_self_affirmed flag missing from commits metadata."}

    df = refminer[["commit_sha", "refactoring_type"]].copy()
    df["is_self_affirmed"] = df["commit_sha"].map(sar_flags).fillna(False).astype(bool)

    counts = df.groupby(["refactoring_type", "is_self_affirmed"]).size().unstack(fill_value=0)
    if counts.empty:
        return {"note": "No refactoring instances matched commit metadata."}

    counts = counts.rename(columns={True: "sar_instances", False: "non_sar_instances"})
    for column in ("sar_instances", "non_sar_instances"):
        if column not in counts.columns:
            counts[column] = 0
    counts["total_instances"] = counts["sar_instances"] + counts["non_sar_instances"]
    counts = counts[counts["total_instances"] > 0]
    if counts.empty:
        return {"note": "No refactoring instances with totals greater than zero."}

    counts["sar_percentage_of_type"] = counts["sar_instances"] / counts["total_instances"] * 100.0
    counts["non_sar_percentage_of_type"] = counts["non_sar_instances"] / counts["total_instances"] * 100.0

    total_sar_instances = counts["sar_instances"].sum()
    total_non_sar_instances = counts["non_sar_instances"].sum()
    if total_sar_instances > 0:
        counts["sar_group_percentage"] = counts["sar_instances"] / total_sar_instances * 100.0
    else:
        counts["sar_group_percentage"] = 0.0
    if total_non_sar_instances > 0:
        counts["non_sar_group_percentage"] = counts["non_sar_instances"] / total_non_sar_instances * 100.0
    else:
        counts["non_sar_group_percentage"] = 0.0

    counts = counts.sort_values(by="total_instances", ascending=False)
    csv_path = target_dir / "refactoring_type_affirmed_split.csv"
    counts.reset_index().to_csv(csv_path, index=False)

    top_rows: Sequence[Dict[str, object]] = []
    if top_n is not None and top_n > 0:
        top_rows = counts.reset_index().head(top_n).to_dict("records")

    return {
        "csv_path": str(csv_path),
        "total_types": int(len(counts)),
        "top_types": top_rows,
        "top_n": top_n,
        "sar_total_instances": int(total_sar_instances),
        "non_sar_total_instances": int(total_non_sar_instances),
    }


__all__ = ["rq2_self_affirmed_percentage", "rq2_refactoring_type_affirmed_split"]
