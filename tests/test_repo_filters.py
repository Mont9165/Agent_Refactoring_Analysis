import pandas as pd

from src.utils import repo_filters


def test_load_repo_whitelist(tmp_path):
    whitelist_path = tmp_path / "repos.csv"
    df = pd.DataFrame(
        [
            {"repo_id": 1, "repo_name": "org/high", "repo_stars": 10},
            {"repo_id": 2, "repo_name": "org/low", "repo_stars": 3},
        ]
    )
    df.to_csv(whitelist_path, index=False)
    
    loaded_df, repo_ids, repo_names = repo_filters.load_repo_whitelist(whitelist_path)
    assert len(loaded_df) == 2
    assert repo_ids == {"1", "2"}
    assert repo_names == {"org/high", "org/low"}


def test_filter_dataframe_by_repo_list():
    data = pd.DataFrame(
        [
            {"repo_id": 1, "value": "keep"},
            {"repo_id": 2, "value": "drop"},
            {"repo_id": 3, "value": "keep"},
        ]
    )
    kept, stats = repo_filters.filter_dataframe_by_repo_list(data, repo_ids={"1", "3"})
    assert kept["value"].tolist() == ["keep", "keep"]
    assert stats["filtered_rows"] == 2


def test_extract_repo_identifiers_normalizes_names():
    df = pd.DataFrame(
        [
            {"repo_id": 10, "repo_name": "Org/High"},
            {"repo_id": 11, "repo_name": "org/low"},
            {"repository_id": 12, "full_name": "Example/Repo"},
        ]
    )
    repo_ids, repo_names = repo_filters.extract_repo_identifiers(df)
    assert repo_ids == {"10", "11", "12"}
    # Names should be case-folded to ensure consistent matching.
    assert repo_names == {"org/high", "org/low", "example/repo"}


def test_build_commit_repo_lookup_with_pr_metadata():
    commit_df = pd.DataFrame(
        [
            {"sha": "a1", "pr_id": 10},
            {"sha": "b2", "pr_id": 20},
        ]
    )
    pr_df = pd.DataFrame(
        [
            {"pr_id": 10, "repo_id": 101, "repo_name": "org/high"},
            {"pr_id": 20, "repo_id": 102, "repo_name": "org/low"},
        ]
    )
    lookup, commit_col = repo_filters.build_commit_repo_lookup(commit_df, pr_df)
    assert commit_col == "sha"
    assert set(lookup.columns) == {"sha", "repo_id", "repo_name"}
    assert lookup.set_index("sha")["repo_id"].to_dict() == {"a1": 101, "b2": 102}
