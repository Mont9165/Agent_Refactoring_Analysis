# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python packages that power each phase. Data ingestion lives in `src/data_loader`, Java filtering in `src/phase1_*`, refactoring analysis in `src/phase3_*`, and research outputs in `src/research_questions/`.
- `scripts/` provides numbered CLI entry points that form the pipeline. Example: `python scripts/0_download_dataset.py` pulls HuggingFace data, and `python scripts/4_analyze_refactoring_instance_and_type.py` assembles commit-level refactoring summaries.
- `data/` (gitignored) stores parquet/CSV artifacts such as `data/analysis/refactoring_instances/refminer_refactorings.parquet`. `outputs/` captures publishable plots and tables. Configuration is centralized in `config/dataset_config.yaml`. Shared tooling (Designite, RefactoringMiner) sits under `tools/`.

## Build, Test, and Development Commands
- Bootstrap locally:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Run the end-to-end dataset pipeline in order: `scripts/0_download_dataset.py` → `1_simple_java_extraction.py` → `2_extract_commits.py` → `3_apply_refactoringminer.py` → `4_analyze_refactoring_instance_and_type.py`.
- Quality deltas: `python scripts/6b_compute_designite_deltas.py --workers 4` (ensure `DESIGNITE_JAVA_PATH` and `REPOS_BASE` are set). Summaries and plots: `python scripts/10_research_questions.py`.
- Tests: execute `pytest -q` from the repo root; use `pytest tests/test_phase1_filters.py::test_handles_gradle_build` when iterating on a specific case.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, descriptive snake_case for variables/modules, CapWords for classes, and imperative verb module names in `scripts/`.
- Add type hints for public APIs (`def load_dataset(...) -> Dataset`) and lightweight docstrings explaining data expectations.
- Gate script entry points with `if __name__ == "__main__":` and keep logging informative but concise.

## Testing Guidelines
- Targeted tests live under `tests/`, named `test_<area>.py`. Mock large parquet inputs; prefer fixtures stored alongside the test file.
- Write assertions around both data shape and key numeric metrics (e.g., refactoring counts). Aim to keep runtime under one minute.
- Always run `pytest -q` before opening a PR; integrate new datasets behind flags so tests remain deterministic.

## Commit & Pull Request Guidelines
- Use imperative, scoped commit messages (e.g., `phase3: dedupe refactoring shas`, `rq5: cache designite summary`). Avoid bundling unrelated pipeline and doc edits.
- PRs should include: purpose summary, affected scripts/modules, reproduction commands, new artifacts or schema changes, and links to relevant issues/notebooks. Attach plot thumbnails when modifying visualization code.
- Confirm required environment variables (`HF_TOKEN`, `DESIGNITE_JAVA_PATH`, `REPOS_BASE`) in the PR description if reviewers must rerun the pipeline. Ensure CI/tests pass before requesting review.
