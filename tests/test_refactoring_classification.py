from src.research_questions import refactoring_classification as rc


def test_classify_known_types():
    assert rc.classification_key("Extract Method") == "medium"
    assert rc.classify_refactoring_type("Extract Method") == rc.LEVEL_NAME_BY_KEY["medium"]
    assert rc.classification_key("Rename Variable") == "low"
    assert rc.classification_key("Move Class") == "high"


def test_unknown_type_is_unclassified():
    assert rc.classification_key("Nonexistent Refactoring") == "unclassified"
    assert rc.classify_refactoring_type("Nonexistent Refactoring") == rc.LEVEL_NAME_BY_KEY["unclassified"]


def test_move_source_folder_is_high_level():
    assert rc.classification_key("Move Source Folder") == "high"
