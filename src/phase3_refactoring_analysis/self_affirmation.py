"""Shared self-affirmation pattern utilities."""
from __future__ import annotations

import re
from typing import List

TABLE_TERMS: List[str] = [
    "Refactor*",
    "Mov*",
    "Split*",
    "Fix*",
    "Introduce*",
    "Decompos*",
    "Reorganiz*",
    "Extract*",
    "Merg*",
    "Renam*",
    "Chang*",
    "Restructur*",
    "Reformat*",
    "Extend*",
    "Remov*",
    "Replac*",
    "Rewrit*",
    "Simplifi*",
    "Creat*",
    "Improv*",
    "Add*",
    "Modif*",
    "Enhanc*",
    "Rework*",
    "Inlin*",
    "Redesign*",
    "Cleanup",
    "Reduc*",
    "Encapsulat*",
    "Removed poor coding practice",
    "Improve naming consistency",
    "Removing unused classes",
    "Pull some code up",
    "Use better name",
    "Replace it with",
    "Make maintenance easier",
    "Code cleanup",
    "Minor Simplification",
    "Reorganize project structures",
    "Code maintenance for refactoring",
    "Remove redundant code",
    "Moved and gave clearer names to",
    "Refactor bad designed code",
    "Getting code out of",
    "Deleting a lot of old stuff",
    "Code revision",
    "Fix technical debt",
    "Fix quality issue",
    "Antipattern bad for performances",
    "Major/Minor structural changes",
    "Clean up unnecessary code",
    "Code reformatting & reordering",
    "Nicer code / formatted / structure",
    "Simplify code redundancies",
    "Added more checks for quality factors",
    "Naming improvements",
    "Renamed for consistency",
    "Refactoring towards nicer name analysis",
    "Change design",
    "Modularize the code",
    "Code cosmetics",
    "Moved more code out of",
    "Remove dependency",
    "Enhanced code beauty",
    "Simplify internal design",
    "Change package structure",
    "Use a safer method",
    "Code improvements",
    "Minor enhancement",
    "Get rid of unused code",
    "Fixing naming convention",
    "Fix module structure",
    "Code optimization",
    "Fix a design flaw",
    "Nonfunctional code cleanup",
    "Improve code quality",
    "Fix code smell",
    "Use less code",
    "Avoid future confusion",
    "More easily extended",
    "Polishing code",
    "Move unused file away",
    "Many cosmetic changes",
    "Inlined unnecessary classes",
    "Code cleansing",
    "Fix quality flaws",
    "Simplify the code",
]


def _build_pattern() -> re.Pattern[str]:
    stem_parts = []
    phrase_parts = []
    for term in TABLE_TERMS:
        term = term.strip()
        if not term:
            continue
        if term.endswith("*"):
            prefix = re.escape(term[:-1].lower())
            stem_parts.append(rf"{prefix}[a-zA-Z]*")
        else:
            escaped = re.escape(term.lower())
            phrase_parts.append(escaped.replace(r"\ ", r"\s+"))

    pattern_components = []
    if stem_parts:
        pattern_components.append(r"\b(?:" + "|".join(stem_parts) + r")\b")
    if phrase_parts:
        pattern_components.append(r"\b(?:" + "|".join(phrase_parts) + r")\b")

    if not pattern_components:
        raise ValueError("No terms available to build self-affirmation pattern.")

    pattern_str = "(?:" + "|".join(pattern_components) + ")"
    return re.compile(pattern_str, re.IGNORECASE)


SELF_AFFIRMATION_PATTERN = _build_pattern()


__all__ = ["SELF_AFFIRMATION_PATTERN", "TABLE_TERMS"]
