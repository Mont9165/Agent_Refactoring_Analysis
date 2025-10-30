from pathlib import Path

import pandas as pd

from src.research_questions.rq2_self_affirmed import rq2_refactoring_type_affirmed_split


def test_rq2_refactoring_type_affirmed_split(tmp_path: Path):
    commits = pd.DataFrame(
        [
            {"sha": "a", "agent": "bot", "has_refactoring": True, "is_self_affirmed": True},
            {"sha": "b", "agent": "bot", "has_refactoring": True, "is_self_affirmed": False},
            {"sha": "c", "agent": "bot", "has_refactoring": True, "is_self_affirmed": False},
        ]
    )
    refminer = pd.DataFrame(
        [
            {"commit_sha": "a", "refactoring_type": "Extract Method"},
            {"commit_sha": "a", "refactoring_type": "Extract Method"},
            {"commit_sha": "b", "refactoring_type": "Extract Method"},
            {"commit_sha": "b", "refactoring_type": "Rename Variable"},
            {"commit_sha": "c", "refactoring_type": "Rename Variable"},
        ]
    )

    result = rq2_refactoring_type_affirmed_split(
        refminer, commits, subset_label="overall", output_dir=tmp_path, top_n=10
    )
    assert "csv_path" in result
    csv_path = Path(result["csv_path"])
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert set(df["refactoring_type"]) == {"Extract Method", "Rename Variable"}
    em_row = df[df["refactoring_type"] == "Extract Method"].iloc[0]
    # Extract Method: 2 SAR instances (commit a) and 1 Non-SAR (commit b)
    assert em_row["sar_instances"] == 2
    assert em_row["non_sar_instances"] == 1
    assert round(em_row["sar_percentage_of_type"], 1) == 66.7
    assert round(em_row["non_sar_percentage_of_type"], 1) == 33.3
    # Within-group shares: all SAR instances are Extract Method, Non-SAR split across both types.
    assert round(em_row["sar_group_percentage"], 1) == 100.0
    assert round(em_row["non_sar_group_percentage"], 1) == 33.3

    rv_row = df[df["refactoring_type"] == "Rename Variable"].iloc[0]
    assert rv_row["sar_instances"] == 0
    assert rv_row["non_sar_instances"] == 2
    assert round(rv_row["sar_group_percentage"], 1) == 0.0
    assert round(rv_row["non_sar_group_percentage"], 1) == 66.7
