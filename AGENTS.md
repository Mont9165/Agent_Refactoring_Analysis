# Repository Guidelines

## Project Structure & Module Organization
- `src/` — core Python packages. Phase-specific logic lives under `src/phase1_*`, `src/phase3_*`, and `src/research_questions/`.
- `scripts/` — CLI entry points that orchestrate each phase (e.g., `scripts/0_download_dataset.py`, `scripts/10_research_questions.py`).
- `data/` — cached datasets and derived parquet/CSV artifacts (gitignored). Subfolders such as `data/analysis/designite/` and `data/analysis/readability/` hold quality metrics.
- `outputs/` — generated reports, plots, and research-question summaries.
- `config/` — repository-level configuration (`dataset_config.yaml`).

## Build, Test, and Development Commands
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Reproduce the data pipeline: `python scripts/0_download_dataset.py` → `python scripts/1_simple_java_extraction.py` → `python scripts/2_extract_commits.py` → `python scripts/3_apply_refactoringminer.py` → `python scripts/4_analyze_refactoring_instance_and_type.py`.
- Generate plots and summaries for the paper: `python scripts/10_research_questions.py`.
- Tests (when present): `pytest -q` from the repository root.

## Coding Style & Naming Conventions
- Primary language is Python; follow PEP 8 with 4-space indentation and CapWords for classes. Modules and scripts use snake_case (`rq3_refactoring_types.py`).
- Prefer type hints and short docstrings for public functions. When writing CLI scripts, keep top-level execution guarded with `if __name__ == "__main__"`.
- When contributing notebooks or figures, place them under `notebooks/` or `outputs/` rather than `src/`.

## Testing Guidelines
- Tests use `pytest`; place new suites under `tests/test_*.py` with descriptive method names (e.g., `test_rq3_handles_missing_refminer`).
- Focus on lightweight unit tests that stub filesystem dependencies; large parquet files should be mocked or truncated fixtures.
- Run `pytest -q` before pushing to confirm compatibility with the shared virtual environment.

## Commit & Pull Request Guidelines
- Commits should be imperative and scoped (e.g., `phase1: cache java filter output`, `rq3: add non-sar plots`). Group related changes instead of mixing pipeline and documentation tweaks.
- Pull requests must include:
  - A concise summary of the change and affected scripts.
  - Links to relevant issues or notebook experiments.
  - Notes on new artifacts (e.g., `outputs/research_questions/...`) or data schema changes.
  - Screenshots or sample plots when modifying visualization scripts.
- Ensure CI/testing steps listed above pass and reference any required environment variables (e.g., `DESIGNITE_JAVA_PATH`, `REPOS_BASE`) in the PR description if reviewers need to reproduce results.

