"""Aggregate research question helpers (legacy module).

This module re-exports the dedicated RQ1–RQ3 functions from their new modules
and retains the RQ4/RQ5 scaffolds for backwards compatibility.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .rq_common import (
    OUTPUT_DIR,
    ANALYSIS_DIR,
    load_phase3_outputs,
    write_json,
)
from .rq1_refactoring_instances import rq1_refactoring_instances_agentic
from .rq2_self_affirmed import rq2_self_affirmed_percentage
from .rq3_refactoring_types import rq3_top_refactoring_types
from .rq4_refactoring_purpose import rq4_refactoring_purpose


def rq5_quality_impact(commits: pd.DataFrame) -> Dict:
    """RQ5 scaffold: record availability of quality tools."""
    designite_path = os.environ.get("DESIGNITE_JAVA_PATH")
    read_tool_path = os.environ.get("READABILITY_TOOL_PATH")

    result = {
        "designite": {"status": "not-run"},
        "readability": {"status": "not-run"},
        "note": "Provide DESIGNITE_JAVA_PATH and READABILITY_TOOL_PATH to enable.",
    }

    if designite_path and Path(designite_path).exists():
        result["designite"] = {
            "status": "available",
            "path": designite_path,
            "todo": "Implement before/after analysis on sampled commits.",
        }

    if read_tool_path and Path(read_tool_path).exists():
        result["readability"] = {
            "status": "available",
            "path": read_tool_path,
            "todo": "Implement readability scoring deltas on sampled files.",
        }

    write_json(result, OUTPUT_DIR / "rq5_quality_impact.json")
    return result


__all__ = [
    "ANALYSIS_DIR",
    "OUTPUT_DIR",
    "load_phase3_outputs",
    "rq1_refactoring_instances_agentic",
    "rq2_self_affirmed_percentage",
    "rq3_top_refactoring_types",
    "rq4_refactoring_purpose",
    "rq5_quality_impact",
]
