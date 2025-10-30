import pandas as pd

from scripts import filter_outputs_by_repo_list as repo_filter_script


def test_augment_with_inferred_repository_extracts_names():
    df = pd.DataFrame(
        {
            "html_url": [
                "https://github.com/org/project/commit/abcdef",
                "https://github.com/another/repo/pull/42",
                "not-a-github-url",
            ],
            "sha": ["abcdef", "123456", "abcdef123456"],
        }
    )
    augmented = repo_filter_script.augment_with_inferred_repository(df)
    assert augmented is not None
    assert "repo_name" in augmented.columns
    assert augmented.loc[0, "repo_name"] == "org/project"
    assert augmented.loc[1, "repo_name"] == "another/repo"
    assert pd.isna(augmented.loc[2, "repo_name"])
