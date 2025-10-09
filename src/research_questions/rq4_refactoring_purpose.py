"""RQ4: Categorise refactoring purposes using GPT motivations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .rq_common import OUTPUT_DIR, ANALYSIS_DIR, write_csv, write_json


MOTIVATION_PATH = ANALYSIS_DIR / "gpt_refactoring_motivation.csv"
CONFIDENCE_THRESHOLD = 6

TYPE_REMAP: Dict[str, str] = {
    "maintainability": "maintainability",
    "readability": "readability",
    "testability": "testability",
    "legacy_code": "legacy_code",
    "performance": "performance",
    "dependency": "dependency",
    "duplication": "duplication",
    "reuse": "reuse",
}


def _load_gpt_motivations(path: Path, confidence_threshold: int) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = df.copy()
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce").fillna(0).astype(int)
    df = df[df["confidence"] >= confidence_threshold]
    if df.empty:
        return None
    df["normalized_type"] = df["type"].str.lower().map(TYPE_REMAP).fillna("unspecified")
    return df


def rq4_refactoring_purpose(
    commits: pd.DataFrame,
    agentic_only: bool = True,
    *,
    subset_label: str = "overall",
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return GPT-derived purpose labels for refactoring commits."""
    if commits is None:
        raise FileNotFoundError("Missing commits_with_refactoring.parquet")

    target_dir = (output_dir or OUTPUT_DIR) / "rq4" / (subset_label or "overall")
    target_dir.mkdir(parents=True, exist_ok=True)

    motivations = _load_gpt_motivations(MOTIVATION_PATH, CONFIDENCE_THRESHOLD)
    if motivations is None:
        result = {
            "total_refactoring_commits": 0,
            "purpose_distribution": {},
            "examples_file": None,
            "note": "No GPT motivation data available or all confidence scores below threshold.",
            "output_dir": str(target_dir),
        }
        write_json(result, target_dir / "summary.json")
        return result

    target_shas = set(commits["sha"].unique())
    if agentic_only and "agent" in commits.columns:
        target_shas = set(commits[commits["agent"].notna()]["sha"].unique())

    if target_shas:
        motivations = motivations[motivations["sha"].isin(target_shas)]

    if motivations.empty:
        result = {
            "total_refactoring_commits": 0,
            "purpose_distribution": {},
            "examples_file": None,
            "note": "No GPT motivation rows match the selected commits.",
            "output_dir": str(target_dir),
        }
        write_json(result, target_dir / "summary.json")
        return result

    counts = motivations["normalized_type"].value_counts().to_dict()
    samples = motivations[["sha", "normalized_type", "reason", "type", "confidence"]]
    examples_path = write_csv(samples.head(1000), target_dir / "examples.csv")

    result: Dict[str, object] = {
        "total_refactoring_commits": int(motivations["sha"].nunique()),
        "purpose_distribution": counts,
        "examples_file": str(examples_path),
        "note": "Based on GPT motivations with confidence >= {threshold}.".format(threshold=CONFIDENCE_THRESHOLD),
        "output_dir": str(target_dir),
    }
    write_json(result, target_dir / "summary.json")
    return result


__all__ = ["rq4_refactoring_purpose", "MOTIVATION_PATH"]
