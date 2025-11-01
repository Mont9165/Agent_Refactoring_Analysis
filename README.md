# Agent Refactoring Analysis

Research infrastructure for studying how AI-assisted commits refactor Java code. The pipeline starts from the AIDev dataset, filters Java pull requests, detects refactorings (pattern-based + RefactoringMiner), and quantifies quality impacts with Designite metrics.

## Overview

- **Dataset ingestion** – Scripts `0`–`2` pull the HuggingFace snapshot, keep Java PRs, and expand them into commit-level tables.
- **Refactoring detection** – Scripts `3a`/`3b` combine heuristic signals with RefactoringMiner to build `refactoring_commits.parquet` and raw JSON traces.
- **Quality analysis** – Scripts `6a`–`6d` run DesigniteJava to emit commit/entity deltas for structural and readability metrics.
- **Research questions** – `scripts/10_research_questions.py` aggregates the outputs to answer RQ1–RQ5 (counts, Agentic vs human comparisons, smell deltas) with plots and CSVs under `outputs/research_questions/`.
- **Auxiliary tooling** – Extra scripts (10a, 11–15, `calculate_cohen_kappa.py`) generate level-based breakdowns, visualisations, and inter-rater metrics for GPT motivation labels.

All derived artefacts live in `data/analysis/` (parquet/CSV) and `outputs/` (publication-ready plots). Environment variables such as `DESIGNITE_JAVA_PATH`, `REPOS_BASE`, `READABILITY_TOOL_CMD`, and `REFMINER_MAX_COMMITS` control the tooling without modifying code.

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
| `DESIGNITE_TIMEOUT` | Seconds before a Designite run aborts | `600` |
| `DESIGNITE_WORKERS` | Repository workers for Designite delta generation | `4` |
| `READABILITY_WORKERS` | Repository workers for readability delta generation | `4` |
| `READABILITY_TIMEOUT` | Seconds before the readability command aborts | `600` |

If you plan to re-run Designite or RefactoringMiner, make sure `java` is available on your `PATH`.

---

### Docker-based workflow (optional)

The repository ships with a container image that bundles Python 3.11, Java 17, Git LFS, and the Python requirements. To build it:

```bash
docker build -t agent-refactoring-analysis .
```

Run an interactive shell with the data, tools, and outputs mounted so results persist on the host:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/tools:/app/tools" \
  -v "$(pwd)/github-oauth.properties:/app/github-oauth.properties:ro" \
  agent-refactoring-analysis
```

On first start the entrypoint clones and builds RefactoringMiner under `tools/RefactoringMiner`. The container exports the same environment variables as the local setup (`HF_HOME`, `PYTHONPATH`, and `MPLCONFIGDIR=/tmp/matplotlib`), so Matplotlib caches are written to a writable location.

Prefer `docker compose` if you want reusable service definitions (for example the bundled Jupyter profile):

```bash
docker compose up --build agent-refactoring-analysis
# or `docker compose --profile jupyter up` for notebooks
```

Remember to populate `github-oauth.properties` (or provide a different path via a bind mount) so RefactoringMiner can authenticate against the GitHub API.

---

## 3. Data Pipeline Overview

Each numbered script builds on the previous outputs. Run them from the repository root.

| Step | Command | Description | Key Outputs |
|------|---------|-------------|-------------|
| 0 | `python scripts/0_download_dataset.py` | Download & cache the source dataset snapshot from HuggingFace | `data/huggingface/...` |
| 1 | `python scripts/1_simple_java_extraction.py` | Filter to Java pull requests and persist metadata for downstream phases | `data/filtered/java_repositories/simple_java_prs.*` |
| 2 | `python scripts/2_extract_commits.py` | Expand PRs into commit-level rows and drop merge commits | `data/filtered/java_repositories/java_file_commits_for_refactoring.*` |
| 3a | `python scripts/3_detect_refactoring.py` | Heuristic refactoring signals based on PR metadata and file changes | `data/analysis/refactoring_instances/commits_with_refactoring.parquet` |
| 3b | `python scripts/3_apply_refactoringminer.py` | Run RefactoringMiner on candidate commits (saves raw JSON + parquet) | `data/analysis/refactoring_instances/refminer_refactorings.parquet`, `refminer_raw/` |
| 4 | `python scripts/4_analyze_refactoring_instance_and_type.py` | Merge heuristics + RefactoringMiner output into commit/type summaries | `data/analysis/refactoring_instances/refactoring_commits.parquet`, `refactoring_analysis.json` |
| 6 | `python scripts/6_designite_impact_analysis.py` | Compare Designite metrics for refactoring vs. control commits | `data/analysis/designite/commit_designite_metrics.*`, `designite_impact_summary.json` |
| 6a | `python scripts/6a_prepare_designite_projects.py` | Emit Designite project list for bulk analysis | `tools/DesigniteRunner/projects.txt` |
| 6b | `python scripts/6b_compute_designite_deltas.py [--workers N]` | Fetch parent/child snapshots, run Designite, emit per-entity deltas | `data/analysis/designite/deltas/*.parquet` |
<!-- | 6c | `python scripts/6c_readability_impact.py [--workers N]` | Compute CoRed readability deltas for files touched by refactorings | `data/analysis/readability/readability_deltas.*` | -->
| 6d | `python scripts/6_quality_analysis.py` | Combine Designite + readability metrics into a quality-impact bundle | `data/analysis/quality/*.parquet`, `quality_summary.json` |
| 7 | `python scripts/7b_label_repositories_by_chatgpt.py` | Tag repositories (production vs. toy) via GPT prompts using README excerpts | `data/filtered/java_repositories/gpt_repository_labels.csv` |
| 10 | `python scripts/10_research_questions.py` | Generate the RQ1–RQ5 tables, plots, and CSV artefacts | `outputs/research_questions/**` |

> **Tip:** Most scripts accept environment variables (e.g., `REFMINER_MAX_COMMITS`, `REFMINER_SAVE_JSON`, `DESIGNITE_WORKERS`) to control sample size and caching. Check the script source or run `--help` for details.

### Additional analysis & visualisation helpers

The pipeline ships with focused scripts that build on the core outputs:

- `python scripts/7_manual_inspection_by_chatgpt.py` – export structured prompts for manual GPT review of specific commits.
- `python scripts/10_research_questions.py` – end-to-end RQ1–RQ5 bundle (tables, stats, violin plots, CSVs under `outputs/research_questions/`).
- `python scripts/10a_compute_rq3_totals_summary.py` – derive Agentic vs. human refactoring totals for the RQ3 plots and tables.
- `python scripts/11_analyze_designite_smells.py` – deep dive into Designite smell deltas (per-type aggregations and CSV summaries).
- `python scripts/12_visualize_designite_metrics.py` – render per-metric before/after charts from the Designite delta outputs.
- `python scripts/13_visualize_rq_findings.py` – collect the headline RQ plots into a single PDF bundle.
- `python scripts/14_metric_distribution_analysis.py` – histogram/ECDF views for selected Designite metrics.
- `python scripts/15_visualize_metric_groups.py` – grouped radar/heatmap views for metric families.
- `python scripts/calculate_cohen_kappa.py` – compute inter-rater reliability and classification metrics for the GPT motivation labels (`--summary` and `--confusion-dir` flags control exports).

Utility scripts such as `dataset_summary.py`, `plot_dataset.py`, `render_table.py`, and `sample_refactoring_motivations.py` are living notebooks in script form—run them ad-hoc to explore subsets of the data without touching the main pipeline artefacts.

<!-- ### Filtered Java Dataset

- Adjust `config/dataset_config.yaml` if needed; `filtering.min_repo_stars` now defaults to `5`, so PRs from repositories with fewer than five stars are excluded from downstream phases.
- Re-run the pipeline scripts in order to refresh all derived artifacts:
  ```bash
  python scripts/0_download_dataset.py         # optional if parquet cache already up-to-date
  python scripts/1_simple_java_extraction.py   # regenerates simple_java_prs.* with star filter applied
  python scripts/2_extract_commits.py
  python scripts/3_detect_refactoring.py       # or scripts/3_apply_refactoringminer.py if you use the batch runner
  python scripts/4_refminer_analysis.py
  python scripts/7b_label_repositories_by_chatgpt.py --max-workers 3  # classify repositories (resume-safe)
  ```
- Existing outputs in `data/filtered/` and `data/analysis/` will be overwritten with the filtered results; archive or relocate historical snapshots beforehand if you need to keep them.
- Generate an explicit whitelist of qualifying repositories with `python scripts/export_high_star_projects.py`; this emits both CSV and parquet files under `data/filtered/java_repositories/`.
- To retrofit previously generated artifacts, run `python scripts/filter_outputs_by_repo_list.py <files...>` (accepts CSV or parquet) so only the whitelisted repositories remain. The script automatically merges repository metadata from `simple_java_prs.parquet` (override with `--pr-stats`) and, if needed, from the commit table (`--commit-stats`) so even commit-only outputs can be filtered in place.

> **Note:** Step 7 pulls README excerpts via the GitHub API—set `GITHUB_TOKEN` (or populate `github-oauth.properties`) before running to avoid stringent anonymous rate limits. Pass `--extra-context` if you want the prompt to include stars/forks and sample PR titles in addition to the README text. -->

---

## 4. RefactoringMiner Integration

1. Build or drop the RefactoringMiner jar inside `tools/RefactoringMiner`:
   ```bash
   mkdir -p tools
   cd tools
   git clone https://github.com/tsantalis/RefactoringMiner.git
   cd RefactoringMiner
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
1. Clone DesigniteRunner `tools/DesigniteRunner`:
   ```bash
   cd tools
   git clone https://github.com/Mont9165/DesigniteRunner.git
   cd DesigniteRunner
   ```

2. Prepare designite projects (come back project directory):
  ```bash
  python scripts/6a_prepare_designite_projects.py
  ```


### 5.1 Generating commit-level metrics
Prerequisites:
- Designite command-line jar (`DESIGNITE_JAVA_PATH`).
- Designite output directories populated under `data/designite/outputs/<owner>/<repo>/<sha>/`. You can populate them by running Designite manually or by invoking the helper in `scripts/tools/`.

Run:
```bash
python scripts/6_designite_impact_analysis.py
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
> **Timeouts:** Use `--timeout` (or export `DESIGNITE_TIMEOUT` / `READABILITY_TIMEOUT`) to relax the default 600s limit when running expensive snapshots.

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
| RQ1 / RQ2 (Frequency) | Counts of refactoring commits/instances & self-affirmation | `data/analysis/refactoring_instances/refactoring_analysis.json`, `refactoring_commits.parquet` |
| RQ3 (Type) | Refactoring type distributions | `data/analysis/refactoring_instances/refactoring_type_counts_*.{csv,json}` |
| RQ4 (Purpose) | Qualitative summaries / prompts (if generated) | `outputs/` or `notebooks/` (see repo history) |
| RQ5 (Impact) | Quality metrics before/after refactorings | `data/analysis/designite/*`, `data/analysis/quality/*`, `outputs/research_questions/rq5_quality/*` |

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

<!-- --- -->

<!-- ## 9. Contributing & Workflow Tips

1. Keep outputs under `data/` (already gitignored). Do not commit large binaries.
2. Follow PEP 8 (4-space indent) and use type hints + short docstrings for new modules.
3. When adding scripts, prefer a numbered naming scheme and document expected inputs/outputs.
4. Always mention new environment variables in this README and in script docstrings.
5. Before pushing analysis results, regenerate summaries with the latest scripts to ensure reproducibility.

For substantial contributions, consider adding pytest cases under `tests/` (named `test_*.py`). Use temporary directories and fixtures to avoid large downloads during testing.

---

Happy analysing! If you uncover new refactoring insights, extend the pipeline with additional research questions under `src/research_questions/` and expose them via new scripts in `scripts/`. -->
