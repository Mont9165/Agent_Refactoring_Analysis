#!/usr/bin/env python3
"""
Export a repository whitelist for projects meeting the minimum star threshold.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add project root to path for shared modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/filtered/java_repositories/simple_java_prs.parquet",
        help="Path to the PR statistics file (parquet or csv).",
    )
    parser.add_argument(
        "--output",
        default="data/filtered/java_repositories/high_star_repositories.csv",
        help="Output path for the whitelist (CSV). A parquet twin is saved alongside.",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=None,
        help="Minimum repository stars required. Defaults to filtering.min_repo_stars from config.",
    )
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _identify_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    repo_id_cols = ["repo_id", "repository_id", "repo", "id"]
    star_cols = ["repo_stars", "stars"]
    name_cols = ["repo_name", "full_name", "name", "repository_name"]
    
    repo_id_col = next((col for col in repo_id_cols if col in df.columns), None)
    star_col = next((col for col in star_cols if col in df.columns), None)
    name_col = next((col for col in name_cols if col in df.columns), None)
    
    if repo_id_col is None and name_col is None:
        raise ValueError("Input data must contain at least a repo identifier or name column")
    return repo_id_col, star_col, name_col


def _ensure_star_column(df: pd.DataFrame, star_col: str, loader: HFDatasetLoader, repo_id_col: str) -> pd.DataFrame:
    if star_col in df.columns:
        return df
    if repo_id_col is None:
        raise ValueError("Cannot enrich star counts without a repository identifier column")
    
    repo_df = loader.load_parquet_table("all_repository")
    if "stars" not in repo_df.columns and "watchers" not in repo_df.columns:
        raise ValueError("Repository metadata does not contain a stars column")
    
    star_source = "stars" if "stars" in repo_df.columns else "watchers"
    merged = df.merge(repo_df[["id", star_source]], left_on=repo_id_col, right_on="id", how="left")
    merged = merged.drop(columns=["id"])
    merged = merged.rename(columns={star_source: "repo_stars"})
    return merged


def main():
    args = parse_args()
    loader = HFDatasetLoader()
    min_stars = args.min_stars
    if min_stars is None:
        min_stars = int(loader.config.get("filtering", {}).get("min_repo_stars", 0))
        print(f"Using default min_repo_stars from config: {min_stars}")
    else:
        print(f"Using provided min_stars: {min_stars}")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = _load_dataframe(input_path)
    repo_id_col, star_col, name_col = _identify_columns(df)
    df = _ensure_star_column(df, star_col, loader, repo_id_col)
    
    star_column = "repo_stars" if "repo_stars" in df.columns else star_col
    numeric_stars = pd.to_numeric(df[star_column], errors="coerce").fillna(0)
    filtered = df[numeric_stars >= min_stars].copy()
    if filtered.empty:
        raise ValueError("No repositories meet the star threshold")
    
    dedup_cols = [col for col in [repo_id_col, name_col, star_column] if col]
    whitelist = filtered[dedup_cols].drop_duplicates(repo_id_col or name_col).reset_index(drop=True)
    whitelist = whitelist.sort_values(by=[repo_id_col or name_col])
    
    whitelist.to_csv(output_path, index=False)
    whitelist.to_parquet(output_path.with_suffix(".parquet"), index=False)
    
    print(f"Saved {len(whitelist)} repositories to {output_path}")
    print(f"Columns included: {list(whitelist.columns)}")


if __name__ == "__main__":
    main()
