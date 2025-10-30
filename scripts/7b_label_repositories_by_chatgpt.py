#!/usr/bin/env python3
"""Automatically classify repositories (toy vs. serious) using GPT."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import base64

import backoff
import openai
import pandas as pd
import tiktoken
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

DEFAULT_PRS_CSV = Path("data/filtered/java_repositories/simple_java_prs.csv")
DEFAULT_OUTPUT = Path("data/filtered/java_repositories/gpt_repository_labels.csv")

MODEL = "gpt-4.1-mini"
ENC = tiktoken.encoding_for_model("gpt-4")

REPO_LABELS: Dict[str, str] = {
    "production_grade": "Actively developed or widely used software (applications, libraries, tooling) that appears suitable for real-world use.",
    "specialized_project": "Niche, experimental, academic, or research prototype projects that still represent substantive software (not a toy).",
    "toy_or_example": "Toy applications, tutorials, coursework, tests, evaluation harnesses, or otherwise trivial/example repositories.",
    "uncertain": "Insufficient evidence to decide.",
}

KEYWORD_HINTS = [
    "toy",
    "sample",
    "example",
    "demo",
    "tutorial",
    "practice",
    "exercise",
    "academy",
    "university",
    "homework",
    "assignment",
    "bootcamp",
    "template",
    "playground",
    "training",
]

JSON_SCHEMA = {
    "name": "repository_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(REPO_LABELS.keys()),
                "description": "Chosen project category (toy_or_example, production_grade, etc.)",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation citing evidence from the summary.",
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence score (1-10).",
            },
        },
        "required": ["label", "reason", "confidence"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class RepoRecord:
    repo_name: str
    owner: str
    repo: str
    repo_stars: int
    repo_forks: int
    repo_keywords: List[str]
    summary_context: str


def truncate_to_tokens(text: str, max_tokens: int = 1500) -> str:
    tokens = ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENC.decode(tokens[:max_tokens])


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _sanitize_title(title: str | float | None) -> str:
    if not isinstance(title, str):
        return ""
    compressed = " ".join(title.strip().split())
    return compressed[:300]


def _format_agent_counts(agent_counts: Dict[str, int], limit: int = 4) -> str:
    if not agent_counts:
        return "None observed"
    items = sorted(agent_counts.items(), key=lambda kv: kv[1], reverse=True)
    display = [f"{name}: {count}" for name, count in items[:limit]]
    remainder = len(items) - limit
    if remainder > 0:
        display.append(f"+{remainder} more")
    return ", ".join(display)


def _load_github_token() -> str | None:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token.strip()
    props_path = Path("github-oauth.properties")
    if props_path.exists():
        for line in props_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OAuthToken="):
                return line.split("=", 1)[1].strip()
    return None


def _fetch_readme(owner: str, repo: str, headers: Dict[str, str], timeout: int = 15) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(url, headers=headers or None, timeout=timeout)
        if response.status_code == 404:
            print(f"[README] {owner}/{repo}: not found")
            return None
        response.raise_for_status()
        data = response.json()
        content = data.get("content")
        if not content:
            print(f"[README] {owner}/{repo}: empty content")
            return None
        decoded = base64.b64decode(content)
        return decoded.decode("utf-8", errors="ignore")
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        print(f"[README] {owner}/{repo}: HTTP {status} ({exc})")
        return None
    except requests.RequestException as exc:
        print(f"[README] {owner}/{repo}: request error ({exc})")
        return None


def build_repo_records(
    prs_df: pd.DataFrame,
    max_titles: int,
) -> List[RepoRecord]:
    prs_df = prs_df.copy()
    prs_df["created_at_dt"] = pd.to_datetime(prs_df.get("created_at"), errors="coerce")

    def _first_int(series: pd.Series, default: int = 0) -> int:
        series = series.dropna()
        if series.empty:
            return default
        try:
            return int(series.iloc[0])
        except (ValueError, TypeError):
            return default

    records: List[RepoRecord] = []
    for repo_name, group in prs_df.groupby("repo_name"):
        if not isinstance(repo_name, str) or not repo_name.strip():
            continue
        if "/" not in repo_name:
            continue
        owner, repository = repo_name.split("/", 1)

        repo_stars = _first_int(group.get("repo_stars"), 0)
        if repo_stars == 0:
            repo_stars = max(_first_int(group.get("stars"), 0), repo_stars)
        repo_forks = _first_int(group.get("repo_forks"), 0)
        if repo_forks == 0:
            repo_forks = max(_first_int(group.get("forks"), 0), repo_forks)

        keywords_hit = [kw for kw in KEYWORD_HINTS if kw in repo_name.lower()]
        sample_titles = [
            _sanitize_title(title)
            for title in group.sort_values("created_at_dt", ascending=False)["title"].dropna().astype(str).head(max_titles)
            if title
        ]

        context_lines: List[str] = [
            f"Repository: {repo_name}",
            f"Stars: {repo_stars:,} | Forks: {repo_forks:,}",
        ]
        if keywords_hit:
            context_lines.append(f"Repository name contains keywords: {', '.join(keywords_hit)}")
        if sample_titles:
            context_lines.append("Recent PR titles:")
            context_lines.extend([f"- {title}" for title in sample_titles])

        records.append(
            RepoRecord(
                repo_name=repo_name,
                owner=owner,
                repo=repository,
                repo_stars=repo_stars,
                repo_forks=repo_forks,
                repo_keywords=keywords_hit,
                summary_context="\n".join(context_lines),
            )
        )

    return records


def build_summary(
    record: RepoRecord,
    readme_excerpt: str | None = None,
    *,
    include_context: bool = False,
) -> str:
    chunks: List[str] = []
    if include_context and record.summary_context:
        chunks.append(record.summary_context)
    if readme_excerpt:
        chunks.append("README excerpt:")
        chunks.append(readme_excerpt)
    if not chunks:
        chunks.append(f"Repository: {record.repo_name}")
    return truncate_to_tokens("\n".join(chunks), max_tokens=1500)


def ensure_api_key() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError), max_tries=5)
def classify_repository(summary: str) -> Tuple[str, str, int]:
    descriptions = "\n".join(f"- {label}: {desc}" for label, desc in REPO_LABELS.items())
    system_message = {
        "role": "system",
        "content": (
            "You are an expert software engineering researcher evaluating GitHub repositories.\n"
            "Classify each repository into one of the predefined categories based on the provided summary.\n"
            "Categories:\n"
            f"{descriptions}\n"
            "Focus on whether the project appears to be a toy/example versus substantive software.\n"
            "Respond strictly in JSON following the requested schema."
        ),
    }
    user_message = {
        "role": "user",
        "content": summary,
    }

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[system_message, user_message],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
    )
    data = json.loads(resp.choices[0].message.content)
    return data["label"], data["reason"], int(data["confidence"])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label repositories (toy vs. production) using GPT with structured outputs."
    )
    parser.add_argument(
        "--prs-csv",
        type=Path,
        default=DEFAULT_PRS_CSV,
        help=f"Path to simple_java_prs.csv (default: {DEFAULT_PRS_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write repository labels (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of concurrent GPT requests (default: 5).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Optional limit on number of repositories to process.",
    )
    parser.add_argument(
        "--max-titles",
        type=int,
        default=6,
        help="Maximum sample PR titles to include when --extra-context is enabled (default: 6).",
    )
    parser.add_argument(
        "--extra-context",
        action="store_true",
        help="Include minimal repo metadata (stars, forks, recent PR titles) alongside the README.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Disable README retrieval from GitHub before labeling.",
    )
    parser.add_argument(
        "--readme-timeout",
        type=int,
        default=15,
        help="Timeout (seconds) for fetching README files (default: 15).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_api_key()

    prs_df = _load_dataframe(args.prs_csv)
    records = build_repo_records(prs_df, max_titles=args.max_titles)
    if args.max_repos is not None:
        records = records[: args.max_repos]

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        labeled_repos = set(existing_df.get("repo_name", pd.Series(dtype=str)).dropna().astype(str))
    else:
        existing_df = pd.DataFrame()
        labeled_repos = set()

    remaining = [record for record in records if record.repo_name not in labeled_repos]
    print(f"Total repositories discovered: {len(records)}")
    print(f"Already labeled repositories: {len(labeled_repos)}")
    print(f"Repositories left to process: {len(remaining)}")

    if not remaining:
        return 0

    include_readme = not args.skip_readme
    github_token = _load_github_token() if include_readme else None
    headers = {
        "Accept": "application/vnd.github+json",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    elif include_readme:
        print("Warning: GITHUB_TOKEN not set; README fetches will use unauthenticated requests (severely rate-limited).")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    results: List[Dict[str, object]] = []
    lock = Lock()

    def classify_task(record: RepoRecord) -> None:
        readme_excerpt: str | None = None
        if include_readme:
            readme = _fetch_readme(record.owner, record.repo, headers=headers, timeout=args.readme_timeout)
            if readme:
                # Keep the first ~4000 characters for reproducibility.
                readme_excerpt = readme.replace("\r\n", "\n").strip()
                readme_excerpt = "\n".join(readme_excerpt.splitlines()[:100])
        summary = build_summary(
            record,
            readme_excerpt=readme_excerpt,
            include_context=args.extra_context,
        )
        try:
            label, reason, confidence = classify_repository(summary)
            row = {
                "repo_name": record.repo_name,
                "owner": record.owner,
                "repo": record.repo,
                "repo_stars": record.repo_stars,
                "repo_forks": record.repo_forks,
                "summary_context": record.summary_context,
                "label": label,
                "reason": reason,
                "confidence": confidence,
            }
            if readme_excerpt:
                row["readme_excerpt"] = readme_excerpt
            with lock:
                results.append(row)
                print(f"[{record.repo_name}] → {label} (conf {confidence})")
        except Exception as exc:  # noqa: BLE001
            with lock:
                print(f"[{record.repo_name}] ERROR: {exc}")

    batch_size = 10
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(classify_task, record) for record in remaining]
        for idx, future in enumerate(as_completed(futures), 1):
            future.result()
            if idx % batch_size == 0 and results:
                new_df = pd.DataFrame(results)
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                existing_df.to_csv(output_path, index=False)
                results.clear()
                print(f"Progress saved for {idx} repositories.")

    if results:
        new_df = pd.DataFrame(results)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        existing_df.to_csv(output_path, index=False)
        print(f"Final progress saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
