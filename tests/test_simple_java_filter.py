import pandas as pd

from src.phase1_java_extraction import simple_java_filter


def test_simple_java_filter_applies_min_repo_stars(monkeypatch):
    commit_details = pd.DataFrame(
        [
            {
                "pull_request_id": 1,
                "filename": "src/Main.java",
                "commit_sha": "sha-1",
                "repo_id": 100,
            },
            {
                "pull_request_id": 2,
                "filename": "src/Helper.java",
                "commit_sha": "sha-2",
                "repo_id": 101,
            },
        ]
    )

    pull_requests = pd.DataFrame(
        [
            {
                "id": 1,
                "state": "open",
                "repo_id": 100,
                "html_url": "https://example.com/pr1",
                "title": "Add feature",
                "agent": None,
                "user": "dev1",
            },
            {
                "id": 2,
                "state": "open",
                "repo_id": 101,
                "html_url": "https://example.com/pr2",
                "title": "Add helper",
                "agent": None,
                "user": "dev2",
            },
        ]
    )

    repositories = pd.DataFrame(
        [
            {"id": 100, "stars": 10, "forks": 2, "full_name": "org/high-star", "language": "Java"},
            {"id": 101, "stars": 3, "forks": 1, "full_name": "org/toy-project", "language": "Java"},
        ]
    )

    data_tables = {
        "pr_commit_details": commit_details,
        "all_pull_request": pull_requests,
        "all_repository": repositories,
    }

    class DummyLoader:
        def __init__(self, config_path):
            self.config = {
                "java_detection": {
                    "java_file_extensions": [".java"],
                    "min_java_percentage": 60,
                    "check_build_files": True,
                    "build_file_patterns": ["pom.xml"],
                },
                "filtering": {"min_repo_stars": 5},
            }

        def load_parquet_table(self, table_name):
            return data_tables[table_name].copy()

    monkeypatch.setattr(simple_java_filter, "HFDatasetLoader", DummyLoader)

    java_filter = simple_java_filter.SimpleJavaFilter(config_path="unused")
    filtered_prs = java_filter.filter_java_prs()

    assert filtered_prs["repo_id"].tolist() == [100]
    assert java_filter._last_filter_stats["removed_by_stars"] == 1

    summary = java_filter.get_summary_stats(filtered_prs)
    assert summary["filtering_stats"]["min_repo_stars"] == 5
    assert summary["filtering_stats"]["final_prs"] == 1
