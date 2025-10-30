import pandas as pd

from scripts.sample_refactoring_motivations import sample_per_category


def test_sample_per_category_top_up_existing_samples():
    df = pd.DataFrame(
        {
            "sha": ["s1", "s2", "s3", "s4", "s5"],
            "type": ["A", "A", "B", "B", "B"],
            "extra": [1, 2, 3, 4, 5],
        }
    )
    existing_records = {"A": {"s1"}}

    sampled_df, warnings, summary, undersized = sample_per_category(
        df,
        category_column="type",
        min_per_category=2,
        max_per_category=2,
        seed=123,
        existing_records=existing_records,
        id_column="sha",
    )

    assert len(sampled_df) == 3  # One new for A, two for B.
    assert set(sampled_df["sha"]).issubset({"s2", "s3", "s4", "s5"})
    assert summary["A"]["existing"] == 1
    assert summary["A"]["added"] == 1
    assert summary["A"]["requested"] == 2
    assert "A" not in undersized
    assert not warnings


def test_sample_per_category_reports_undersized_existing_only():
    df = pd.DataFrame(
        {
            "sha": ["s10", "s11"],
            "type": ["A", "A"],
        }
    )
    existing_records = {"C": {"cx1"}}

    sampled_df, warnings, summary, undersized = sample_per_category(
        df,
        category_column="type",
        min_per_category=3,
        max_per_category=3,
        seed=999,
        existing_records=existing_records,
        id_column="sha",
    )

    assert summary["C"]["existing"] == 1
    assert summary["C"]["added"] == 0
    assert "C" in undersized
    assert any("Category 'C' has only 1 existing rows" in msg for msg in warnings)
    # No new samples should be drawn for category C since it is absent in the input.
    if not sampled_df.empty and "type" in sampled_df.columns:
        assert "C" not in sampled_df["type"].unique()
