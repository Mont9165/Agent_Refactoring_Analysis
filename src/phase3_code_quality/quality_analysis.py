"""
Quality impact analysis for refactoring commits.

Computes deltas before/after a commit for:
- Code metrics / code smells via DesigniteJava (if DESIGNITE_JAVA_PATH configured)
- Readability via an external tool (if READABILITY_TOOL_CMD or READABILITY_JAR configured)

Assumptions:
- Local clones are available under REPOS_BASE/<owner>/<repo> parsed from commit html_url.
- Alternatively, set REFMINER_LOCAL_REPO to analyze a single local repo independent of html_url.

Outputs under data/analysis/quality/:
- quality_deltas.parquet/.csv: per-commit deltas
- quality_summary.json: aggregated summary across analyzed commits
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd


QUALITY_DIR = Path("data/analysis/quality")
QUALITY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ToolConfig:
    designite_path: Optional[str]
    readability_cmd: Optional[str]
    readability_jar: Optional[str]
    repos_base: Optional[Path]
    local_repo: Optional[Path]


def load_tool_config() -> ToolConfig:
    # Prefer explicit env var; fallback to vendored tools/DesigniteRunner/DesigniteJava.jar
    designite_default = Path("tools/DesigniteRunner/DesigniteJava.jar")
    designite_path = os.environ.get("DESIGNITE_JAVA_PATH")
    if not designite_path and designite_default.exists():
        designite_path = str(designite_default.resolve())
    return ToolConfig(
        designite_path=designite_path,
        readability_cmd=os.environ.get("READABILITY_TOOL_CMD"),
        readability_jar=os.environ.get("READABILITY_JAR"),
        repos_base=Path(os.environ["REPOS_BASE"]) if os.environ.get("REPOS_BASE") else None,
        local_repo=Path(os.environ["REFMINER_LOCAL_REPO"]) if os.environ.get("REFMINER_LOCAL_REPO") else None,
    )


def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 600) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _infer_repo_path(html_url: Optional[str], cfg: ToolConfig) -> Optional[Path]:
    if cfg.local_repo and cfg.local_repo.exists():
        return cfg.local_repo
    if not html_url or not cfg.repos_base:
        return None
    try:
        parts = html_url.split('/')
        owner, repo = parts[3], parts[4]
        return cfg.repos_base / owner / repo
    except Exception:
        return None


def _worktree_checkout(repo: Path, commit: str) -> Optional[Path]:
    if not repo.exists():
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="qa_wt_"))
    wt = tmpdir / commit[:10]
    code, out, err = _run(["git", "worktree", "add", str(wt), commit], cwd=repo)
    if code != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    return wt


def _get_parent_commit(repo: Path, commit: str) -> Optional[str]:
    code, out, err = _run(["git", "rev-parse", f"{commit}^"], cwd=repo)
    if code == 0:
        return out.strip()
    return None


def run_designite(input_dir: Path, out_dir: Path, cfg: ToolConfig) -> Optional[Path]:
    if not cfg.designite_path or not Path(cfg.designite_path).exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-jar", cfg.designite_path, "-i", str(input_dir), "-o", str(out_dir)]
    code, out, err = _run(cmd)
    return out_dir if code == 0 else None


def parse_designite(out_dir: Path) -> Dict[str, float]:
    res: Dict[str, float] = {}
    # Common outputs: DesigniteClassMetrics.csv, DesigniteDesignSmells.csv (names vary by version)
    metrics = list(out_dir.glob("**/*ClassMetrics*.csv"))
    smells = list(out_dir.glob("**/*DesignSmells*.csv"))
    if metrics:
        df = pd.read_csv(metrics[0])
        # Aggregate example metrics if present
        for col in ["LOC", "WMC", "NOM", "NOF"]:
            if col in df.columns:
                res[f"avg_{col}"] = float(df[col].mean())
    if smells:
        ds = pd.read_csv(smells[0])
        res["smell_instances"] = float(len(ds))
        if "Smell" in ds.columns:
            res["top_smell"] = str(ds["Smell"].value_counts().idxmax()) if len(ds) > 0 else ""
    return res


def run_readability(input_dir: Path, out_file: Path, cfg: ToolConfig) -> Optional[Path]:
    if cfg.readability_cmd:
        cmd = cfg.readability_cmd.format(input=str(input_dir), output=str(out_file))
        code, out, err = _run(["bash", "-lc", cmd])
        return out_file if code == 0 and out_file.exists() else None
    if cfg.readability_jar and Path(cfg.readability_jar).exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        code, out, err = _run(["java", "-jar", cfg.readability_jar, "-i", str(input_dir), "-o", str(out_file)])
        return out_file if code == 0 and out_file.exists() else None
    return None


def parse_readability(out_file: Path) -> Dict[str, float]:
    # Try JSON or CSV; expect at least an overall average field or per-file scores to average
    try:
        if out_file.suffix.lower() == ".json":
            data = json.loads(out_file.read_text())
            if isinstance(data, dict):
                for k in ("readability", "avg", "average"):
                    if k in data:
                        return {"avg_readability": float(data[k])}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                vals = [float(x.get("readability", x.get("score", 0))) for x in data]
                if vals:
                    return {"avg_readability": float(sum(vals) / len(vals))}
        else:
            df = pd.read_csv(out_file)
            for col in ["readability", "score", "avg", "average"]:
                if col in df.columns:
                    return {"avg_readability": float(pd.to_numeric(df[col], errors="coerce").mean())}
    except Exception:
        pass
    return {}


def analyze_commit_row(row: pd.Series, cfg: ToolConfig) -> Optional[Dict]:
    sha = str(row.get("sha"))
    html_url = row.get("html_url")
    repo = _infer_repo_path(html_url, cfg)
    if not repo or not repo.exists():
        return None

    parent = _get_parent_commit(repo, sha)
    if not parent:
        return None

    before_dir = _worktree_checkout(repo, parent)
    after_dir = _worktree_checkout(repo, sha)
    if not before_dir or not after_dir:
        # cleanup partial
        if before_dir:
            _run(["git", "worktree", "remove", str(before_dir), "--force"], cwd=repo)
            shutil.rmtree(before_dir.parent, ignore_errors=True)
        if after_dir:
            _run(["git", "worktree", "remove", str(after_dir), "--force"], cwd=repo)
            shutil.rmtree(after_dir.parent, ignore_errors=True)
        return None

    result: Dict[str, Optional[float]] = {"sha": sha}
    try:
        # Designite
        before_des = QUALITY_DIR / "tmp" / sha / "before" / "designite"
        after_des = QUALITY_DIR / "tmp" / sha / "after" / "designite"
        bdir = run_designite(before_dir, before_des, cfg)
        adir = run_designite(after_dir, after_des, cfg)
        if bdir and adir:
            b = parse_designite(bdir)
            a = parse_designite(adir)
            # simple deltas
            for k in set(b) | set(a):
                result[f"designite_delta_{k}"] = float(a.get(k, 0)) - float(b.get(k, 0))

        # Readability
        before_read = QUALITY_DIR / "tmp" / sha / "before" / "readability.csv"
        after_read = QUALITY_DIR / "tmp" / sha / "after" / "readability.csv"
        rb = run_readability(before_dir, before_read, cfg)
        ra = run_readability(after_dir, after_read, cfg)
        if rb and ra:
            b = parse_readability(before_read)
            a = parse_readability(after_read)
            if b and a and "avg_readability" in a and "avg_readability" in b:
                result["readability_delta"] = float(a["avg_readability"]) - float(b["avg_readability"]) 
    finally:
        # cleanup worktrees
        _run(["git", "worktree", "remove", str(before_dir), "--force"], cwd=repo)
        _run(["git", "worktree", "remove", str(after_dir), "--force"], cwd=repo)
        shutil.rmtree(before_dir.parent, ignore_errors=True)
        shutil.rmtree(after_dir.parent, ignore_errors=True)

    return result


def analyze_quality(commits_with_refactoring: Path, max_commits: int = 50) -> Tuple[pd.DataFrame, Dict]:
    cfg = load_tool_config()
    df = pd.read_parquet(commits_with_refactoring)
    # focus on refactoring commits
    df = df[df.get("has_refactoring", False) == True]
    # prefer agentic if available
    if "agent" in df.columns:
        agentic = df[df["agent"].notna()]
        if not agentic.empty:
            df = agentic
    sample = df.drop_duplicates("sha").head(max_commits)

    rows: List[Dict] = []
    for _, row in sample.iterrows():
        res = analyze_commit_row(row, cfg)
        if res:
            rows.append(res)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(QUALITY_DIR / "quality_deltas.parquet", index=False)
    out_df.to_csv(QUALITY_DIR / "quality_deltas.csv", index=False)

    summary: Dict[str, float] = {}
    if not out_df.empty:
        # Aggregate basic stats
        des_cols = [c for c in out_df.columns if c.startswith("designite_delta_")]
        for c in des_cols:
            summary[c + "_avg"] = float(out_df[c].mean())
        if "readability_delta" in out_df.columns:
            summary["readability_delta_avg"] = float(out_df["readability_delta"].mean())

    with open(QUALITY_DIR / "quality_summary.json", "w") as f:
        json.dump({"analyzed_commits": int(len(out_df)), "summary": summary}, f, indent=2)

    return out_df, summary
