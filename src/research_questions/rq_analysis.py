"""Aggregate research question helpers (legacy module).

This module re-exports the dedicated RQ1–RQ3 functions from their new modules
and retains the RQ4/RQ5 scaffolds for backwards compatibility.
"""
from __future__ import annotations

from .rq_common import OUTPUT_DIR, ANALYSIS_DIR, load_phase3_outputs  # noqa: F401
from .rq1_refactoring_instances import rq1_refactoring_instances_agentic
from .rq2_self_affirmed import rq2_self_affirmed_percentage
from .rq3_refactoring_types import rq3_top_refactoring_types
from .rq4_refactoring_purpose import rq4_refactoring_purpose
from .rq5_quality_impact import rq5_quality_impact


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
