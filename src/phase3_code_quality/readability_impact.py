"""Compute readability deltas for refactoring commits using CoRed or similar tools."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .quality_analysis import ToolConfig, load_tool_config, run_readability


READABILITY_DIR = Path("data/analysis/readability")
READABILITY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RepoLocator:
    owner: str
    repo: str
    sha: str
    parent_sha: str


@dataclass
class RefactoringFile:
    commit_sha: str
    refactoring_type: str
    before_path: Optional[str]
    after_path: Optional[str]


class ReadabilityImpactCalculator:
    def __init__(self, cfg: ToolConfig, max_commits: Optional[int] = None) -> None:
        self.cfg = cfg
        self.max_commits = max_commits
        self.tmp_root = READABILITY_DIR / "tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # git helpers
    # ------------------------------------------------------------------
    def _run(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 600) -> Tuple[int, str, str]:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr

    def _infer_repo_path(self, owner: str, repo: str) -> Optional[Path]:
        if self.cfg.local_repo and self.cfg.local_repo.exists():
            return self.cfg.local_repo
        if not self.cfg.repos_base:
            return None
        candidate = self.cfg.repos_base / owner / repo
        return candidate if candidate.exists() else None

    def _get_parent_sha(self, repo_path: Path, sha: str) -> Optional[str]:
        code, out, _ = self._run(["git", "rev-parse", f"{sha}^"], cwd=repo_path)
        return out.strip() if code == 0 else None

    def _checkout_worktree(self, repo_path: Path, sha: str) -> Optional[Path]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="readability_wt_"))
        worktree = tmp_dir / sha[:10]
        code, _, _ = self._run(["git", "worktree", "add", str(worktree), sha], cwd=repo_path)
        if code != 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
        return worktree

    def _cleanup_worktree(self, repo_path: Path, worktree: Optional[Path]) -> None:
        if not worktree:
            return
        self._run(["git", "worktree", "remove", str(worktree), "--force"], cwd=repo_path)
        shutil.rmtree(worktree.parent, ignore_errors=True)

    # ------------------------------------------------------------------
    # readability helpers
    # ------------------------------------------------------------------
    def _run_readability_for_snapshot(self, worktree: Path, sha: str, label: str) -> Optional[Path]:
        out_path = self.tmp_root / sha / f"{label}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_readability(worktree, out_path, self.cfg)
        return out_path if result else None

    @staticmethod
    def _parse_readability(csv_path: Path, root: Path) -> pd.DataFrame:
        if not csv_path or not csv_path.exists():
            return pd.DataFrame(columns=["relative_path", "score", "level"])
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=["relative_path", "score", "level"])
        root_resolved = root.resolve()

        def to_relative(path_str: str) -> str:
            try:
                rel = Path(path_str).resolve().relative_to(root_resolved)
                return rel.as_posix()
            except Exception:
                return Path(path_str).name

        df["relative_path"] = df["file_name"].astype(str).map(to_relative)
        df.rename(columns={"score": "readability_score"}, inplace=True)
        return df[["relative_path", "readability_score", "level"]]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_refminer_path(path: Optional[str]) -> Optional[str]:
        if not path or not isinstance(path, str):
            return None
        normalized = path.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _collect_refactoring_files(self, refminer_df: pd.DataFrame, sha: str) -> List[RefactoringFile]:
        rows: List[RefactoringFile] = []
        subset = refminer_df[refminer_df["commit_sha"] == sha]
        if subset.empty:
            return rows
        for _, row in subset.iterrows():
            left_locs = json.loads(row.get("left_side_locations", "[]") or "[]")
            right_locs = json.loads(row.get("right_side_locations", "[]") or "[]")
            pairs = list(zip_longest(left_locs, right_locs, fillvalue=None))
            if not pairs:
                pairs = [(None, None)]
            for left_loc, right_loc in pairs:
                before = self._normalize_refminer_path((left_loc or {}).get("filePath"))
                after = self._normalize_refminer_path((right_loc or {}).get("filePath"))
                rows.append(
                    RefactoringFile(
                        commit_sha=sha,
                        refactoring_type=row.get("refactoring_type", "unknown"),
                        before_path=before,
                        after_path=after,
                    )
                )
        return rows

    @staticmethod
    def _match_score(df: pd.DataFrame, path: Optional[str]) -> Optional[float]:
        if not path or df.empty:
            return None
        norm = path.replace("\\", "/")
        candidates = df[df["relative_path"] == norm]
        if candidates.empty:
            return None
        return float(candidates["readability_score"].iloc[0])

    def process(self, commits_df: pd.DataFrame, refminer_df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.readability_cmd and not self.cfg.readability_jar:
            raise RuntimeError("READABILITY_TOOL_CMD or READABILITY_JAR must be configured")

        target_commits = commits_df[commits_df.get("has_refactoring", False) == True]
        if self.max_commits is not None:
            target_commits = target_commits.head(self.max_commits)

        results: List[Dict[str, object]] = []

        for _, commit in target_commits.iterrows():
            html_url = commit.get("html_url")
            if not isinstance(html_url, str):
                continue
            parts = html_url.split("/")
            if len(parts) < 5:
                continue
            owner, repo = parts[3], parts[4]
            repo_path = self._infer_repo_path(owner, repo)
            if not repo_path:
                continue
            parent_sha = self._get_parent_sha(repo_path, commit["sha"])
            if not parent_sha:
                continue

            child_worktree = self._checkout_worktree(repo_path, commit["sha"])
            parent_worktree = self._checkout_worktree(repo_path, parent_sha)
            if not child_worktree or not parent_worktree:
                self._cleanup_worktree(repo_path, child_worktree)
                self._cleanup_worktree(repo_path, parent_worktree)
                continue

            try:
                before_csv = self._run_readability_for_snapshot(parent_worktree, commit["sha"], "before")
                after_csv = self._run_readability_for_snapshot(child_worktree, commit["sha"], "after")
                if not before_csv or not after_csv:
                    continue
                before_df = self._parse_readability(before_csv, parent_worktree)
                after_df = self._parse_readability(after_csv, child_worktree)
                if before_df.empty and after_df.empty:
                    continue

                ref_files = self._collect_refactoring_files(refminer_df, commit["sha"])
                if not ref_files:
                    continue

                for ref_file in ref_files:
                    before_path = ref_file.before_path
                    after_path = ref_file.after_path
                    before_score = self._match_score(before_df, before_path)
                    after_score = self._match_score(after_df, after_path)
                    if before_score is None or after_score is None:
                        continue
                    results.append(
                        {
                            "commit_sha": commit["sha"],
                            "refactoring_type": ref_file.refactoring_type,
                            "before_path": before_path,
                            "after_path": after_path,
                            "before_score": before_score,
                            "after_score": after_score,
                            "delta": after_score - before_score,
                        }
                    )
            finally:
                self._cleanup_worktree(repo_path, child_worktree)
                self._cleanup_worktree(repo_path, parent_worktree)

        df = pd.DataFrame(results)
        if not df.empty:
            out_file = READABILITY_DIR / "readability_deltas.parquet"
            df.to_parquet(out_file, index=False)
            df.to_csv(READABILITY_DIR / "readability_deltas.csv", index=False)
            summary = df.groupby("refactoring_type")["delta"].agg(["count", "mean", "median", "std", "min", "max"])
            summary.to_parquet(READABILITY_DIR / "readability_delta_summary.parquet")
            summary.to_csv(READABILITY_DIR / "readability_delta_summary.csv")
        return df


def run_readability_impact(
    max_commits: Optional[int] = None,
    skip_commits: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    commits_path = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
    refminer_path = Path("data/analysis/refactoring_instances/refminer_refactorings.parquet")
    if not commits_path.exists() or not refminer_path.exists():
        raise FileNotFoundError("Required inputs not found. Run refactoring analysis scripts first.")

    commits_df = pd.read_parquet(commits_path)
    if skip_commits:
        skip_set = {str(sha) for sha in skip_commits}
        commits_df = commits_df[~commits_df["sha"].astype(str).isin(skip_set)]
    if commits_df.empty:
        return pd.DataFrame()

    refminer_df = pd.read_parquet(refminer_path)
    cfg = load_tool_config()
    calculator = ReadabilityImpactCalculator(cfg, max_commits=max_commits)
    return calculator.process(commits_df, refminer_df)
