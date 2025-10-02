"""Shared helpers for research question analyses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


OUTPUT_DIR = Path("outputs/research_questions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_DIR = Path("data/analysis/refactoring_instances")


def load_phase3_outputs(
    commits_path: Optional[Path] = None,
    refminer_path: Optional[Path] = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Load Phase 3 parquet outputs needed by RQ scripts."""
    commits_path = commits_path or ANALYSIS_DIR / "commits_with_refactoring.parquet"
    refminer_path = refminer_path or ANALYSIS_DIR / "refminer_refactorings.parquet"

    commits = pd.read_parquet(commits_path) if commits_path.exists() else None
    refminer = pd.read_parquet(refminer_path) if refminer_path.exists() else None

    return {
        "commits": commits,
        "refminer": refminer,
    }


def write_json(obj: Dict, path: Path) -> Path:
    """Persist a dictionary to JSON with repo-relative paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Persist a dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


__all__ = ["OUTPUT_DIR", "ANALYSIS_DIR", "load_phase3_outputs", "write_json", "write_csv"]
