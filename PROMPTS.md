# LLM Prompts Used in This Study

This document describes the prompts used for automated classification tasks in this study. Both tasks use GPT-4.1-mini with structured outputs (JSON Schema) and temperature=0.0.

---

## 1. Project Filtering (Repository Classification)

**Script**: `scripts/7b_label_repositories_by_chatgpt.py`
**Purpose**: Classify repositories into four categories to filter out toy/example projects from the dataset.
**Model**: `gpt-4.1-mini` | **Temperature**: `0.0` | **max_tokens**: `1024`

### System Prompt

```
You are an expert software engineering researcher evaluating GitHub repositories.
Classify each repository into one of the predefined categories based on the provided summary.
Categories:
- production_grade: Actively developed or widely used software (applications, libraries, tooling) that appears suitable for real-world use.
- specialized_project: Niche, experimental, academic, or research prototype projects that still represent substantive software (not a toy).
- toy_or_example: Toy applications, tutorials, coursework, tests, evaluation harnesses, or otherwise trivial/example repositories.
- uncertain: Insufficient evidence to decide.
Focus on whether the project appears to be a toy/example versus substantive software.
Respond strictly in JSON following the requested schema.
```

### User Message (per repository)

Constructed dynamically by `build_summary()`. Contains:
- Repository name (`owner/repo`)
- Star and fork counts
- Keyword hints (if the repository name contains: toy, sample, example, demo, tutorial, practice, exercise, academy, university, homework, assignment, bootcamp, template, playground, training)
- Recent PR titles (up to 6, when `--extra-context` is enabled)
- README excerpt (up to 100 lines, fetched via GitHub API)
- Truncated to 1,500 tokens

Example:
```
Repository: apache/kafka
Stars: 29,000 | Forks: 13,500
README excerpt:
Apache Kafka is an open-source distributed event streaming platform used by thousands of companies...
```

### Output Schema (Structured Outputs)

```json
{
  "name": "repository_label",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "label": {
        "type": "string",
        "enum": ["production_grade", "specialized_project", "toy_or_example", "uncertain"]
      },
      "reason": {
        "type": "string",
        "description": "Brief explanation citing evidence from the summary."
      },
      "confidence": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "description": "Confidence score (1-10)."
      }
    },
    "required": ["label", "reason", "confidence"],
    "additionalProperties": false
  }
}
```

---

## 2. Refactoring Purpose Classification (RQ3)

**Script**: `scripts/7_manual_inspection_by_chatgpt.py`
**Purpose**: Classify the primary motivation behind each refactoring commit based on the taxonomy of code symptoms by Kim et al. (2014).
**Model**: `gpt-4.1-mini` | **Temperature**: `0.0` | **max_tokens**: `4096`

### System Prompt

```
You are an expert software engineering researcher. Your task is to classify the primary motivation for a commit based on the 'Code Symptoms' defined by Kim et al. (2014).

Analyze the provided commit information and assign **exactly one** label from the following categories:
readability: Poor readability or code that is hard to understand
duplication: Duplicated code that needs to be unified
repurpose_reuse: Difficulty of repurposing or reusing existing code
maintainability: Poor maintainability or fragile code
testability: Difficulty of testing code without refactoring
slow_performance: Slow performance that needs optimization
dependency: Unwanted dependencies to other modules
legacy_code: Working on old legacy code that needs modernization
logical_mismatch: Incorrect or inconsistent logic that leads to wrong behavior
hard_to_debug: Structure makes troubleshooting/diagnosis difficult (e.g., poor logging, tangled flows)

Respond in JSON with the following schema:
- reason: string explaining your choice
- output: the chosen label (must be one of the enum)
- confidence: <Your Confidence Score (1-10)>
  - 1-2: Very Low Confidence
  - 3-4: Low Confidence
  - 5-6: Moderate Confidence
  - 7-8: High Confidence
  - 9-10: Very High Confidence
Do not emit any other keys or text.
```

### User Message (per commit)

```
Commit Title:
{title}

Commit Message:
{message}

Refactoring Operations Performed (Summary):
{refactoring_types_summary}
```

Where `refactoring_types_summary` is a comma-separated list of refactoring types detected by RefactoringMiner with their counts (e.g., `Rename Method (3 times), Extract Method (2 times), Move Class (1 times)`).

### Output Schema (Structured Outputs)

```json
{
  "name": "classification",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "reason": {
        "type": "string",
        "description": "A brief explanation for why this commit type was chosen"
      },
      "output": {
        "type": "string",
        "enum": ["readability", "duplication", "repurpose_reuse", "maintainability", "testability", "slow_performance", "dependency", "legacy_code", "logical_mismatch", "hard_to_debug"]
      },
      "confidence": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "description": "Confidence score (1-10)"
      }
    },
    "required": ["reason", "output", "confidence"],
    "additionalProperties": false
  }
}
```

---

## Prompt Development Process

For both classification tasks, prompts were developed through the following iterative process:

1. **Pilot classification**: Two authors independently classified a small sample of items manually to establish ground-truth labels and refine category definitions.
2. **Initial prompt design**: A baseline prompt was drafted incorporating category definitions from prior literature (Kim et al., 2014 for RQ3; standard software engineering criteria for project filtering).
3. **Iterative refinement**: The prompt was tested on the pilot sample, misclassifications were analyzed, and the prompt was refined to address ambiguous cases. This cycle was repeated until stable performance was achieved.
4. **Validation**: The final prompt was applied to a statistically significant random sample (n=344 for project filtering, n=100 for RQ3), and inter-rater agreement was measured using Cohen's Kappa between human annotators and between human consensus and LLM output.
