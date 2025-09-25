# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code
  - `data_loader/`: HuggingFace dataset access
  - `phase1_java_extraction/`: Java PR and commit filtering
  - `phase3_refactoring_analysis/`: Pattern and RefactoringMiner analysis
  - `utils/`, `research_questions/`: helpers and RQ scaffolding
- `scripts/`: CLI entry points (`0_…py` → `4_…py`)
- `config/dataset_config.yaml`: dataset/cache paths and heuristics
- `data/`: downloaded and derived parquet/csv outputs
- `tools/RefactoringMiner/`: vendored/tooling for RefactoringMiner
- `notebooks/`, `outputs/`, `tests/`: exploration, reports, tests

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Download data: `python scripts/0_download_dataset.py`
- Phase 1 (Java PRs): `python scripts/1_simple_java_extraction.py`
- Phase 1.5 (Commits): `python scripts/2_extract_commits.py`
- Phase 3 (Patterns): `python scripts/3_detect_refactoring.py`
- RefactoringMiner (optional): `REFMINER_MAX_COMMITS=50 python scripts/4_refminer_analysis.py`
  - If the tool is missing, follow on-screen setup or build in `tools/RefactoringMiner` with `./gradlew shadowJar`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indentation; prefer type hints and docstrings.
- Files: snake_case modules (`simple_java_filter.py`), CapWords classes.
- Keep I/O paths relative to repo; read/write under `data/…` and respect `config/dataset_config.yaml`.

## Testing Guidelines
- Framework: pytest (add to dev deps as needed).
- Location: `tests/test_*.py`; name tests after module under test.
- Run: `pytest -q` (fast, unit‑level; avoid large downloads—use small fixtures/temporary parquet files).

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped prefix when helpful.
  - Example: `phase1: add PR status merge`, `data_loader: cache HF tables`
- PRs: clear summary, motivation, linked issues, sample commands, and notes on data/schema changes. Include before/after snippets or paths (e.g., `data/filtered/java_repositories/*.parquet`).

## Security & Configuration Tips
- Never hard‑code tokens; configure via `config/dataset_config.yaml` and environment variables.
- Do not commit large data under `data/`; treat outputs as artifacts.
- Prefer deterministic writes; avoid destructive operations outside `data/`.

## Agent‑Specific Instructions
- Preserve phase ordering and file contracts consumed by later scripts.
- Avoid absolute paths; make code idempotent; guard for missing columns and empty datasets.
