#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parents[1]))

SCOPE_DIR = Path("data/analysis/refactoring_instances")

import backoff
import openai
import pandas as pd
import tiktoken

load_dotenv()

# -----------------------------------------------------------------------------
# 1. Configuration (same as before)
# -----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

# from Kim et al., "An Empirical Study of Refactoring Challenges and Benefits at Microsoft"
TYPES = {
    "readability": "Poor readability or code that is hard to understand",
    "duplication": "Duplicated code that needs to be unified",
    "reuse": "Difficulty of repurposing or reusing existing code",
    "maintainability": "Poor maintainability or fragile code",
    "testability": "Difficulty of testing code without refactoring",
    "performance": "Slow performance that needs optimization",
    "dependency": "Unwanted dependencies to other modules",
    "legacy_code": "Working on old legacy code that needs modernization"
}

MODEL = "gpt-4.1-mini"  # needs to support json_schema response_format

# -----------------------------------------------------------------------------
# 2. Token utilities (unchanged)
# -----------------------------------------------------------------------------
ENC = tiktoken.encoding_for_model("gpt-4")


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    toks = ENC.encode(text)
    return text if len(toks) <= max_tokens else ENC.decode(toks[:max_tokens])


# -----------------------------------------------------------------------------
# 3. JSON Schema for Structured Outputs
# -----------------------------------------------------------------------------
JSON_SCHEMA = {
    "name": "classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "A brief explanation for why this commit type was chosen"
            },
            "output": {
                "type": "string",
                "enum": list(TYPES.keys()),
                "description": "One of the allowed Conventional Commit types"
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence score (1-10)"
            }
        },
        "required": ["reason", "output", "confidence"],
        "additionalProperties": False
    }
}


# -----------------------------------------------------------------------------
# 4. classify_with_gpt using Structured Outputs
# -----------------------------------------------------------------------------
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError), max_tries=5)
def classify_with_gpt(title: str, message: str, refactoring_types_summary: str) -> tuple[str, str, int]:
    """
    Returns (reason, output, confidence) where output ∈ TYPES.
    Uses GPT-4o Structured Outputs to guarantee valid JSON.
    """
    types_str = ""
    for key, value in TYPES.items():
        types_str += f"{key}: {value}\n"

    system = {
        "role": "system",
        "content": (
                "You are an expert software engineering researcher. "
                "Your task is to classify the primary motivation for a commit based on the 'Code Symptoms' defined by Kim et al. (2014).\n\n"
                "Analyze the provided commit information and assign **exactly one** label from the following categories:\n"
                + types_str
                + "\n"
                  "Respond in JSON with the following schema:\n"
                  "- reason: string explaining your choice\n"
                  "- output: the chosen label (must be one of the enum)\n"
                  "- confidence: <Your Confidence Score (1-10)>\n"
                  "  - 1-2: Very Low Confidence\n"
                  "  - 3-4: Low Confidence\n"
                  "  - 5-6: Moderate Confidence\n"
                  "  - 7-8: High Confidence\n"
                  "  - 9-10: Very High Confidence\n"
                  "Do not emit any other keys or text."
        )
    }
    user = {
        "role": "user",
        "content": (
            f"Commit Title:\n{title}\n\n"
            f"Commit Message:\n{message}\n\n"
            f"Refactoring Operations Performed (Summary):\n{refactoring_types_summary}"
        )
    }

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[system, user],
        temperature=0.0,
        max_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": JSON_SCHEMA
        },
    )

    # `resp.choices[0].message.content` is guaranteed to be valid JSON per our schema
    data = json.loads(resp.choices[0].message.content)
    return data["reason"], data["output"], int(data["confidence"])


# -----------------------------------------------------------------------------
# 5. Driver (same as before, calling classify_with_gpt)
# -----------------------------------------------------------------------------
import concurrent.futures
from threading import Lock


def process_agent(max_workers: int = 10) -> None:
    for fp in SCOPE_DIR.rglob("refactoring_commits.csv"):
        name = fp.parent.name
        commits_df = pd.read_csv(fp)
        out_fp = fp.parent / f"gpt_refactoring_motivation.csv"

        if out_fp.exists():
            df = pd.read_csv(out_fp)
        else:
            df = pd.DataFrame(columns=["sha", "title", "reason", "type", "confidence"])

        existing_ids = set(df["sha"].values)
        rows_to_process = []
        for _, row in commits_df.iterrows():
            if (row["sha"] not in existing_ids):
                    # row["state"] == 'closed' and
                    # row["is_self_affirmed"] == True

                # refactoring_types を集計
                try:
                    # 文字列 '["type1", "type2"]' をPythonのリストに変換
                    types_list = ast.literal_eval(row["refactoring_types"])
                    # 種類ごとに出現回数をカウント
                    summary = Counter(types_list)
                    # 見やすい文字列に整形
                    summary_str = ", ".join([f"{k} ({v} times)" for k, v in summary.items()])
                except (ValueError, SyntaxError):
                    summary_str = "N/A"  # パース失敗時のデフォルト値

                rows_to_process.append(
                    (row["sha"], row["title"], row["message"], summary_str)
                )

        print(f"Found {len(rows_to_process)} commits to process for '{name}'")

        results = []

        print(f"Found {len(rows_to_process)} PRs to process for agent '{name}'")

        results = []
        lock = Lock()

        def classify_task(sha: str, title: str, message: str, refactoring_summary: str):
            try:
                reason, label, conf = classify_with_gpt(title, message, refactoring_summary)
                print(f"[{sha}] → {label}: {reason} (conf {conf})")
                row = {
                    "sha": sha,
                    "title": title,
                    "reason": reason,
                    "type": label,
                    "confidence": conf,
                }
                with lock:
                    results.append(row)
            except Exception as e:
                print(f"[{sha}] ERROR: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(classify_task, sha, title, message, summary)
                for sha, title, message, summary in rows_to_process
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()  # Will raise exception if classify_task failed

                # Save progress every 10 results
                if i % 10 == 0:
                    if results:
                        added_df = pd.DataFrame(results)
                        df = pd.concat([df, added_df], ignore_index=True)
                        df.to_csv(out_fp, index=False)
                        print(f"Wrote {len(df)} → {out_fp}")
                        results.clear()

        # Final save
        if results:
            added_df = pd.DataFrame(results)
            df = pd.concat([df, added_df], ignore_index=True)
            df.to_csv(out_fp, index=False)
            print(f"Wrote {len(df)} → {out_fp}")


if __name__ == "__main__":
    process_agent()
