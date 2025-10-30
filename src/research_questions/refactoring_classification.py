"""Refactoring classification levels inspired by Murphy-Hill et al."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping


HIGH_LEVEL_TYPES: frozenset[str] = frozenset(
    {
        "Add Attribute Annotation",
        "Add Attribute Modifier",
        "Add Class Annotation",
        "Add Class Modifier",
        "Add Method Annotation",
        "Add Method Modifier",
        "Add Parameter",
        "Add Parameter Annotation",
        "Add Thrown Exception Type",
        "Change Attribute Access Modifier",
        "Change Class Access Modifier",
        "Change Method Access Modifier",
        "Change Thrown Exception Type",
        "Change Type Declaration Kind",
        "Encapsulate Attribute",
        "Extract Interface",
        "Extract Superclass",
        "Merge Attribute",
        "Merge Class",
        "Merge Package",
        "Merge Parameter",
        "Modify Attribute Annotation",
        "Modify Class Annotation",
        "Modify Method Annotation",
        "Modify Parameter Annotation",
        "Move And Rename Attribute",
        "Move And Rename Class",
        "Move And Rename Method",
        "Move Attribute",
        "Move Class",
        "Move Method",
        "Move Package",
        "Move Source Folder",
        "Parameterize Test",
        "Parameterize Variable",
        "Pull Up Attribute",
        "Pull Up Method",
        "Push Down Attribute",
        "Push Down Method",
        "Remove Attribute Annotation",
        "Remove Attribute Modifier",
        "Remove Class Annotation",
        "Remove Class Modifier",
        "Remove Method Annotation",
        "Remove Method Modifier",
        "Remove Parameter",
        "Remove Parameter Annotation",
        "Remove Thrown Exception Type",
        "Rename Attribute",
        "Rename Class",
        "Rename Method",
        "Rename Package",
        "Reorder Parameter",
        "Split Attribute",
        "Split Package",
        "Split Parameter",
    }
)


MEDIUM_LEVEL_TYPES: frozenset[str] = frozenset(
    {
        "Change Attribute Type",
        "Change Parameter Type",
        "Change Return Type",
        "Extract And Move Method",
        "Extract Attribute",
        "Extract Class",
        "Extract Method",
        "Extract Subclass",
        "Inline Attribute",
        "Inline Method",
        "Localize Parameter",
        "Merge Method",
        "Move And Inline Method",
        "Parameterize Attribute",
        "Replace Anonymous With Class",
        "Replace Attribute With Variable",
        "Replace Variable With Attribute",
        "Split Class",
        "Split Method",
    }
)


LOW_LEVEL_TYPES: frozenset[str] = frozenset(
    {
        "Add Parameter Modifier",
        "Add Variable Annotation",
        "Add Variable Modifier",
        "Assert Throws",
        "Change Variable Type",
        "Extract Variable",
        "Inline Variable",
        "Invert Condition",
        "Merge Catch",
        "Merge Conditional",
        "Merge Variable",
        "Move Code",
        "Remove Parameter Modifier",
        "Remove Variable Annotation",
        "Remove Variable Modifier",
        "Rename Parameter",
        "Rename Variable",
        "Replace Anonymous With Lambda",
        "Replace Attribute",
        "Replace Conditional With Ternary",
        "Replace Generic With Diamond",
        "Replace Loop With Pipeline",
        "Replace Pipeline With Loop",
        "Split Conditional",
        "Split Variable",
        "Try With Resources",
    }
)


LEVEL_NAME_BY_KEY: Dict[str, str] = {
    "high": "High-level (signature)",
    "medium": "Medium-level (signature + block)",
    "low": "Low-level (code block)",
    "unclassified": "Unclassified",
}


REF_TYPE_TO_LEVEL_KEY: Dict[str, str] = {}


def _register(level_key: str, refactoring_types: Iterable[str]) -> None:
    for ref_type in refactoring_types:
        REF_TYPE_TO_LEVEL_KEY[ref_type] = level_key


_register("high", HIGH_LEVEL_TYPES)
_register("medium", MEDIUM_LEVEL_TYPES)
_register("low", LOW_LEVEL_TYPES)


LEVEL_DISPLAY_ORDER: tuple[str, ...] = ("high", "medium", "low", "unclassified")


def classify_refactoring_type(refactoring_type: str) -> str:
    """Return the display label for the refactoring level."""
    key = REF_TYPE_TO_LEVEL_KEY.get(refactoring_type, "unclassified")
    return LEVEL_NAME_BY_KEY[key]


def classification_key(refactoring_type: str) -> str:
    """Return the internal key for a refactoring type (high/medium/low/unclassified)."""
    return REF_TYPE_TO_LEVEL_KEY.get(refactoring_type, "unclassified")


def classified_types() -> Mapping[str, str]:
    """Expose the refactoring type -> level key mapping."""
    return dict(REF_TYPE_TO_LEVEL_KEY)
