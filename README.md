# Agent Refactoring Analysis

Research infrastructure for studying how AI-assisted commits refactor Java code. The pipeline starts from the AIDev dataset, filters Java pull requests, detects refactorings (pattern-based + RefactoringMiner), and quantifies quality impacts with Designite metrics.

---

## 1. Repository Tour

```
config/                        Dataset paths, caching, heuristics
scripts/                       CLI entry points (0_* → 6_*)
src/                           Core libraries
  data_loader/                 HuggingFace dataset accessors
  phase1_java_extraction/      PR + commit filtering
  phase3_refactoring_analysis/ RefactoringMiner orchestration + parsers
  research_questions/          Analysis + metric aggregation utilities
  utils/                       Shared helpers

data/                          Downloaded inputs and generated outputs
  analysis/                    Phase-specific reports (refactoring, designite, quality)
  filtered/                    Intermediate parquet/csv extracted from dataset
  designite/                   Raw Designite CSV dumps grouped by repo/sha

tools/RefactoringMiner/        Vendored RefactoringMiner source/build
```

Key output folders:
- `data/filtered/java_repositories/*` – filtered PRs and commit tables.
- `data/analysis/refactoring_instances/*` – refactoring commits, type counts, summaries.
- `data/analysis/designite/*` – commit-level metrics, box-plot stats, entity deltas.

---

## 2. Local Setup (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.11+ and Java 17+ are required. The repository uses relative paths only; run scripts from the repo root.

### Environment configuration

Create a `.env` (optional) or export variables in your shell:

| Variable | Purpose | Example |
|----------|---------|---------|
| `HF_TOKEN` | Needed if the HuggingFace dataset requires authentication | `hf_...` |
| `GITHUB_TOKEN` / `OAuthToken` | Used by RefactoringMiner for GitHub API access | `ghp_...` (write to `github-oauth.properties`) |
| `DESIGNITE_JAVA_PATH` | Absolute path to `DesigniteJava.jar` | `/path/to/DesigniteJava.jar` |
| `REPOS_BASE` | Root directory containing `<owner>/<repo>` clones | `/work/.../DesigniteRunner/cloned_repos` |
| `REFMINER_LOCAL_REPO` | (Optional) analyze a single local repo instead of using `REPOS_BASE` | `/path/to/repo` |
| `READABILITY_TOOL_CMD` | Shell command template to compute readability (`{input}`, `{output}` placeholders) | `python tools/CoRed/run.py {input} --verbose --output {output}` |
| `DESIGNITE_AUTO` | Toggle automatic Designite runs for missing snapshots | `0` to disable (default `1`) |
| `DESIGNITE_AUTO_CLONE` | Automatically clone repos missing under `REPOS_BASE` | `0` to disable (default `1`) |
| `DESIGNITE_GIT_REMOTE_TMPL` | Template for cloning repos when auto-clone is enabled | `https://github.com/{owner}/{repo}.git` |
| `DESIGNITE_MAX_RETRIES` | Retry count when DesigniteJava fails transiently | `3` |
| `DESIGNITE_WORKERS` | Repository workers for Designite delta generation | `4` |
| `READABILITY_WORKERS` | Repository workers for readability delta generation | `4` |

If you plan to re-run Designite or RefactoringMiner, make sure `java` is available on your `PATH`.

---

## 3. Data Pipeline Overview

Each numbered script builds on the previous outputs. Run them from the repository root.

| Step | Command | Description | Key Outputs |
|------|---------|-------------|-------------|
| 0 | `python scripts/0_download_dataset.py` | Download & cache the source dataset | `data/huggingface/...` |
| 1 | `python scripts/1_simple_java_extraction.py` | Filter Java PRs and keep metadata | `data/filtered/java_repositories/simple_java_prs.parquet` |
| 2 | `python scripts/2_extract_commits.py` | Expand PRs into commit-level rows; filter out merges | `data/filtered/java_repositories/java_file_commits_for_refactoring.parquet` |
| 3 | `python scripts/3_detect_refactoring.py` | Pattern-based hints + (optional) RefactoringMiner pass | `data/analysis/refactoring_instances/commits_with_refactoring.parquet` |
| 4 | `python scripts/4_refminer_analysis.py` | Batch run RefactoringMiner & aggregate results; saves raw JSON | `data/analysis/refactoring_instances/refminer_refactorings.parquet` + `refminer_raw/` |
| 5 | `python scripts/5_designite_impact_analysis.py` | Aggregate Designite commit-level metrics; compare refactoring vs non-refactoring commits | `data/analysis/designite/commit_designite_metrics.*`, `designite_impact_summary.json`, `designite_boxplot_data.json` |
| 6a | `python scripts/6a_prepare_designite_projects.py` | Generate `tools/DesigniteRunner/projects.txt` listing repos to analyse with Designite | `tools/DesigniteRunner/projects.txt` |
| 6 | `python scripts/6_quality_analysis.py` | High-level quality deltas (Designite+readability) for sampled commits | `data/analysis/quality/quality_deltas.*`, `quality_summary.json` |
| 6b | `python scripts/6b_compute_designite_deltas.py [--workers N]` | Auto-fetch repos/commits, run Designite for missing snapshots in parallel, and emit per-entity deltas | `data/analysis/designite/deltas/*.csv|parquet` |
| 6c | `python scripts/6c_readability_impact.py [--workers N]` | Compute readability deltas (per repo worker pool) for files touched by refactorings | `data/analysis/readability/readability_deltas.*`, `_summary.*` |

> **Tip:** Most scripts accept environment variables (e.g., `REFMINER_MAX_COMMITS`, `REFMINER_SAVE_JSON`) to control sample size and caching. Check the script source or `--help` for details.

---

## 4. RefactoringMiner Integration

1. Build or drop the RefactoringMiner jar inside `tools/RefactoringMiner`:
   ```bash
   cd tools/RefactoringMiner
   ./gradlew shadowJar   # produces build/libs/RM-fat.jar
   ```

2. Provide GitHub credentials by writing `github-oauth.properties` with an OAuth token (see GitHub instructions below).

3. Use either `scripts/3_detect_refactoring.py` (quick validation) or `scripts/4_refminer_analysis.py` (full batch with raw JSON). Useful environment knobs:
   - `REFMINER_MAX_COMMITS=<N>` – limit the number of commits analysed.
   - `REFMINER_SAVE_JSON=1` – persist raw responses under `data/analysis/refactoring_instances/refminer_raw/`.

Outputs include:
- `refminer_refactorings.parquet/csv` – flattened refactoring instances.
- `refactoring_analysis.json` – commit/file-level summary (instances per agent, percentages, etc.).

---

## 5. Designite Quality Metrics

### 5.1 Generating commit-level metrics

Prerequisites:
- Designite command-line jar (`DESIGNITE_JAVA_PATH`).
- Designite output directories populated under `data/designite/outputs/<owner>/<repo>/<sha>/`. You can populate them by running Designite manually or by invoking the helper in `scripts/tools/`.

Run:
```bash
python scripts/5_designite_impact_analysis.py
```

Outputs:
- `commit_designite_metrics.parquet|csv` – aggregated metrics per commit.
- `designite_impact_summary.json` – mean/median deltas, descriptive stats, agent breakdowns.
- `designite_boxplot_data.json` – five-number summaries for plotting box plots.

### 5.2 Before/after deltas for refactoring entities

To quantify deltas for specific classes/methods touched by refactorings:

```bash
export DESIGNITE_JAVA_PATH=/absolute/path/to/DesigniteJava.jar
export REPOS_BASE=/absolute/path/to/cloned_repos
# optional: export HF_TOKEN, REFMINER_LOCAL_REPO, etc.
python scripts/6b_compute_designite_deltas.py --max-commits 50  # remove flag for full run

# (Optional) Run readability deltas with CoRed
export READABILITY_TOOL_CMD="python tools/CoRed/run.py {input} --verbose --output {output}"
python scripts/6c_readability_impact.py --max-commits 50

> **Parallelism:** Add `--workers N` or export `DESIGNITE_WORKERS` / `READABILITY_WORKERS` to process repositories concurrently. On an M1 Pro (8 cores), `N=4`–`6` keeps the JVM busy without overwhelming the system.

### 5.3 Readability tooling (CoRed)

[CoRed](https://github.com/grosa1/CoRed) implements the readability model from Scalabrino et al. (JSEP 2018). This repository includes a Python wrapper (`tools/CoRed/run.py`) that batches large repositories and accepts an `--output` path so the pipeline can consume its CSV results directly.

```
python tools/CoRed/run.py path/to/project --verbose --output report.csv
```

To integrate with the analysis scripts, export `READABILITY_TOOL_CMD` so it points to the wrapper (placeholders `{input}` and `{output}` are substituted at runtime):

```
export READABILITY_TOOL_CMD="python tools/CoRed/run.py {input} --verbose --output {output}"
```

The wrapper always writes `file_name,score,level` rows, which `scripts/6c_readability_impact.py` uses to measure before/after changes in files touched by each refactoring.
```

Behaviour:
1. Looks up each refactoring commit in `commits_with_refactoring.parquet`.
2. Ensures a clone exists under `REPOS_BASE` (auto-clones from GitHub when `DESIGNITE_AUTO_CLONE=1`).
3. Fetches the child commit and its parent and checks them out in temporary worktrees.
4. Runs `DesigniteJava.jar` for both snapshots (respects `DESIGNITE_MAX_RETRIES`).
5. Matches entity keys (type or method) using file paths + qualified names from RefactoringMiner locations.
6. Emits per-metric deltas and an aggregated summary (`delta_summary.*`).

Troubleshooting:
- **Designite outputs still missing**: confirm `DESIGNITE_JAVA_PATH` points to a licensed jar and that `java -jar …` succeeds for the target commit. The script logs failures with the offending SHA.
- **Repo clone fails**: set `DESIGNITE_AUTO_CLONE=0` to disable auto-clone and provide the repo manually under `REPOS_BASE`.
- **Large runs**: use `--max-commits` to limit the batch size while validating your setup.

---

## 6. GitHub Token Setup (for RefactoringMiner)

1. Visit **Settings → Developer settings → Personal access tokens (classic)**.
2. Generate a token with `public_repo` scope (or broader if needed).
3. Store it in `github-oauth.properties` at the repo root:
   ```
   OAuthToken=ghp_your_token_here
   ```
4. The scripts automatically read this file when invoking RefactoringMiner.

---

## 7. Research Question Artifacts

| RQ | Output | Location |
|----|--------|----------|
| RQ1 / RQ2 | Counts of refactoring commits & self-affirmation | `data/analysis/refactoring_instances/refactoring_analysis.json`, `refactoring_commits.parquet` |
| RQ3 | Refactoring type distributions | `data/analysis/refactoring_instances/refactoring_type_counts_*.{csv,json}` |
| RQ4 | Qualitative summaries / prompts (if generated) | `outputs/` or `notebooks/` (see repo history) |
| RQ5 | Quality metrics before/after refactorings | `data/analysis/designite/*`, `data/analysis/quality/*`, `outputs/research_questions/rq5_quality/*` |

After running the numbered scripts you can generate the RQ summary bundle (including the new RQ5 quality impact report) with:

```bash
python scripts/10_research_questions.py
```

All JSON/CSV artifacts land in `outputs/research_questions/`. For custom analyses, load the parquet files into notebooks under `notebooks/` or create new scripts under `scripts/`.

---

## 8. Troubleshooting Checklist

- **Missing dataset files**: re-run `scripts/0_download_dataset.py`; ensure `HF_TOKEN` is set if the dataset is private.
- **RefactoringMiner fails**: double-check the jar path in `tools/RefactoringMiner`, ensure the GitHub token is valid, and confirm `java` is Java 17+.
- **Designite delta script prints "No deltas computed"**:
  - `DESIGNITE_JAVA_PATH` must point to the jar.
  - `REPOS_BASE` must contain `<owner>/<repo>` clones with the analysed SHAs (`git fetch origin <sha>` if missing).
  - Existing child Designite outputs live under `data/designite/outputs/...`; the script creates parent snapshots automatically when possible.
- **Large parquet reads fail**: install optional dependencies like `pyarrow==15.x` (already listed in `requirements.txt`).

---

## 9. Contributing & Workflow Tips

1. Keep outputs under `data/` (already gitignored). Do not commit large binaries.
2. Follow PEP 8 (4-space indent) and use type hints + short docstrings for new modules.
3. When adding scripts, prefer a numbered naming scheme and document expected inputs/outputs.
4. Always mention new environment variables in this README and in script docstrings.
5. Before pushing analysis results, regenerate summaries with the latest scripts to ensure reproducibility.

For substantial contributions, consider adding pytest cases under `tests/` (named `test_*.py`). Use temporary directories and fixtures to avoid large downloads during testing.

---

Happy analysing! If you uncover new refactoring insights, extend the pipeline with additional research questions under `src/research_questions/` and expose them via new scripts in `scripts/`.
