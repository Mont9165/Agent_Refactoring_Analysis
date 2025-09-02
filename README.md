# Agent Refactoring Analysis

A comprehensive research project analyzing refactoring patterns in AI-assisted development using the AIDev dataset.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- GitHub Personal Access Token (for RefactoringMiner)

### Setup

1. **Clone and setup:**
```bash
git clone <your-repo-url>
cd Agent_Refactoring_Analysis
```

2. **Create GitHub OAuth file:**
```bash
# Create github-oauth.properties with your token
echo "OAuthToken=ghp_your_token_here" > github-oauth.properties
```

3. **Run with Docker:**
```bash
# Start the analysis environment
docker-compose up --build -d

# Enter the container
docker-compose exec agent-refactoring-analysis bash

# Or run analysis directly
docker-compose run agent-refactoring-analysis python scripts/1_simple_java_extraction.py
```

### Optional: Jupyter Notebook
```bash
# Start Jupyter Lab
docker-compose --profile jupyter up -d

# Access at http://localhost:8888
```

## 📊 Analysis Pipeline

### Step 1: Extract Java PRs
```bash
python scripts/1_simple_java_extraction.py
```
- Filters 932K PRs → 1,462 Java PRs (0.16%)
- Adds repository star/fork counts
- Output: `data/filtered/java_repositories/simple_java_prs.parquet`

### Step 2: Extract Commits
```bash
python scripts/2_extract_commits.py
```
- Extracts 1,703 unique commits from Java PRs
- Filters out merge commits
- Output: `data/filtered/java_repositories/java_file_commits_for_refactoring.parquet`

### Step 3: Detect Refactoring (Pattern-based)
```bash
python scripts/3_detect_refactoring.py
```
- Pattern-based detection: 6.3% refactoring rate
- 77.5% self-affirmation rate
- Output: `data/analysis/refactoring_instances/refactoring_analysis.json`

### Step 4: RefactoringMiner Analysis (AST-based)
```bash
# Analyze with RefactoringMiner for research-grade precision
python scripts/4_refminer_analysis.py

# Analyze more commits
REFMINER_MAX_COMMITS=100 python scripts/4_refminer_analysis.py
```

## 🔬 Research Questions

| RQ | Question | Status | Result |
|----|----------|--------|---------|
| RQ1 | How many refactoring instances in Agentic PRs? | ✅ Complete | 107 commits (6.3% pattern-based) |
| RQ2 | What % are self-affirmed? | ✅ Complete | 77.5% self-affirmation rate |
| RQ3 | Most common refactoring types? | ✅ Complete | General refactoring (35.2%), Cleanup (28.7%) |
| RQ4 | What purposes do developers refactor with AI? | 🔄 Pending | PR description analysis needed |
| RQ5 | Quality improvements from AI refactoring? | 🔄 Pending | Code metrics analysis needed |

## 🏗️ Key Findings

### **Pattern-based vs AST-based Detection:**
- **Pattern-based**: 6.3% of commits mention refactoring keywords
- **RefactoringMiner**: 0% actual formal refactorings detected
- **Insight**: AI tools are transparent about refactoring but may be doing general improvements

### **Agent Comparison:**
- **Cursor**: 18.2% refactoring rate (highest)
- **Claude_Code**: 12.8% refactoring rate  
- **Devin**: 10.6% refactoring rate
- **OpenAI_Codex**: 5.4% refactoring rate

### **Repository Context:**
- **67 unique repositories** analyzed
- **Average 1,086 stars** per repository
- **Top repository**: Stirling-Tools/Stirling-PDF (63,892 ⭐)

## 🛠️ Manual Setup (without Docker)

### Prerequisites
- Python 3.11+
- Java 17+
- Git

### Installation
```bash
pip install -r requirements.txt
python scripts/1_simple_java_extraction.py
```

## 📁 Project Structure

```
├── config/dataset_config.yaml          # Dataset configuration
├── src/
│   ├── data_loader/                     # HuggingFace data loading
│   ├── phase1_java_extraction/          # Java project filtering
│   └── phase3_refactoring_analysis/     # Refactoring detection
├── scripts/                             # Analysis scripts (1-4)
├── data/                               # Analysis results
└── tools/RefactoringMiner/             # RefactoringMiner integration
```

## 🔑 GitHub Token Setup

For RefactoringMiner to work with private repositories:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate token with `public_repo` scope
3. Add to `github-oauth.properties`:
```
OAuthToken=ghp_your_token_here
```

## 📈 Results

All analysis results are saved to `data/analysis/refactoring_instances/`:
- `refactoring_analysis.json`: Complete analysis summary
- `refactoring_commits.parquet`: Commits with detected refactoring
- `refminer_*.json`: RefactoringMiner results (when available)

## 🎯 Research Impact

This is the **first systematic study** of refactoring patterns in AI-assisted development, providing insights into:
- AI tool transparency (77.5% self-affirmation)
- Semantic vs syntactic refactoring gap
- Agent-specific refactoring behaviors

## 🤝 Contributing

This research infrastructure is designed for reproducibility. Use Docker for consistent environments across different machines and operating systems.