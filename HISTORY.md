# Project Creation History

## 2025-08-26: Initial Project Setup

### Purpose
Created a research project structure for analyzing agentic refactoring patterns using the AIDev dataset from Hugging Face. This project aims to answer 5 research questions about AI-assisted code refactoring.

### What was implemented

1. **Directory Structure Created**
   - `data/` - For storing datasets and analysis results
     - `huggingface/download/` - Raw data from HF
     - `huggingface/cache/` - HF datasets cache
     - `filtered/` - Processed data (Java repos, agentic data)
     - `analysis/` - Analysis results
   - `src/` - Source code organized by analysis phases
   - `notebooks/` - Jupyter notebooks for exploration
   - `scripts/` - Executable scripts
   - `config/` - Configuration files
   - `tests/` - Test files
   - `outputs/` - Generated reports and visualizations

2. **Python Package Structure**
   - Added `__init__.py` files to make src/ and subdirectories proper Python packages

3. **Configuration File**
   - `config/dataset_config.yaml` - Contains HuggingFace dataset settings, Java detection parameters, and known agentic tools/bots list

4. **Dependencies**
   - `requirements.txt` - Essential Python packages: datasets, pandas, pyarrow, tqdm, pyyaml

5. **HuggingFace Data Retrieval**
   - `src/data_loader/hf_dataset_loader.py` - Class for loading AIDev dataset, downloading tables as parquet files, and caching
   - `scripts/download_dataset.py` - Executable script to download 8 core dataset tables
   - Updated to use correct AIDev table names and handle DatasetDict structure

6. **Java Project Filtering (Phase 1) - COMPLETED**
   - `src/phase1_java_extraction/language_filter.py` - Repository language detection using metadata
   - `src/phase1_java_extraction/file_extension_checker.py` - Analyzes commit files for Java/Kotlin extensions  
   - `src/phase1_java_extraction/maven_gradle_detector.py` - Detects Maven/Gradle build files in commits
   - `src/phase1_java_extraction/java_project_validator.py` - Combines all detection methods with confidence scoring
   - `src/phase1_java_extraction/data_merger.py` - Handles joining repository, PR, and commit data
   - `scripts/run_java_extraction.py` - Executable script to run complete Java project identification
   - **SIMPLE VERSION ADDED:**
     - `src/phase1_java_extraction/simple_java_filter.py` - Simple filtering for PRs containing .java files
     - `scripts/simple_java_extraction.py` - Fast executable to filter Java PRs (recommended)

### Next Steps (Not Yet Implemented)
- README.md with project overview  
- Makefile with build targets
- .gitignore file

### Research Questions to Address
- RQ1: How many refactoring instances are included in Agentic PRs?
- RQ2: What percent of refactoring commits by Agentic coding tools are self-affirmed?
- RQ3: What types of refactoring are made the most?
- RQ4: What purpose do developers refactor code with Agentic Code?
- RQ5: To what extent can Agentic Coding Tools improve quality?