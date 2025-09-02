"""
HuggingFace Dataset Loader for AIDev dataset
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm


class HFDatasetLoader:
    """Handles loading and caching of HuggingFace AIDev dataset"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.config = self._load_config(config_path)
        self.dataset_name = self.config["huggingface"]["dataset_name"]
        self.cache_dir = Path(self.config["huggingface"]["cache_dir"])
        self.download_dir = Path(self.config["huggingface"]["download_dir"])
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, config_name: str = None, split: str = None, streaming: bool = False) -> Dataset:
        """
        Load AIDev dataset from HuggingFace
        
        Args:
            config_name: Dataset config/table name (e.g., 'all_pull_request')
            split: Dataset split to load (e.g., 'train', 'validation', 'test')
            streaming: Whether to use streaming mode for large datasets
            
        Returns:
            Dataset object
        """
        print(f"Loading dataset: {self.dataset_name}, config: {config_name}")
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                config_name,
                split=split,
                cache_dir=str(self.cache_dir),
                streaming=streaming
            )
            print(f"Dataset loaded successfully. Keys: {list(dataset.keys()) if hasattr(dataset, 'keys') else 'N/A'}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def download_and_save_parquet(self, table_names: Optional[list] = None) -> Dict[str, str]:
        """
        Download specific tables and save as parquet files
        
        Args:
            table_names: List of table names to download. If None, downloads common tables.
            
        Returns:
            Dictionary mapping table names to saved file paths
        """
        if table_names is None:
            # Core tables for refactoring analysis
            table_names = [
                "all_repository",      # 116k rows - repository metadata
                "all_pull_request",    # 933k rows - all pull requests  
                "all_user",           # 72.2k rows - user information
                "pr_commits",         # 88.6k rows - commit information
                "pr_commit_details",  # 712k rows - detailed commit changes
                "pull_request",       # 33.6k rows - subset of PRs
                "pr_comments",        # 39.1k rows - PR comments
                "pr_reviews"          # 28.9k rows - PR reviews
            ]
        
        saved_files = {}
        
        for table_name in tqdm(table_names, desc="Downloading tables"):
            try:
                print(f"Loading table: {table_name}")
                dataset = self.load_dataset(config_name=table_name)
                
                # Handle DatasetDict - access the 'train' split
                if hasattr(dataset, 'keys') and 'train' in dataset:
                    dataset = dataset['train']
                
                # Convert to pandas DataFrame
                df = dataset.to_pandas()
                
                # Save as parquet
                output_path = self.download_dir / f"{table_name}.parquet"
                df.to_parquet(output_path, index=False)
                saved_files[table_name] = str(output_path)
                
                print(f"Saved {table_name}: {len(df)} rows -> {output_path}")
                
            except Exception as e:
                print(f"Error downloading {table_name}: {e}")
                continue
        
        return saved_files
    
    def load_parquet_table(self, table_name: str) -> pd.DataFrame:
        """Load a previously downloaded parquet table"""
        parquet_path = self.download_dir / f"{table_name}.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        return pd.read_parquet(parquet_path)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        info = {
            "dataset_name": self.dataset_name,
            "cache_dir": str(self.cache_dir),
            "download_dir": str(self.download_dir)
        }
        
        # Try to get info from a sample config
        try:
            sample_dataset = self.load_dataset(config_name="all_pull_request")
            if hasattr(sample_dataset, 'info'):
                info["description"] = sample_dataset.info.description
                info["features"] = str(sample_dataset.info.features)
            
            info["sample_config"] = "all_pull_request"
            
        except Exception as e:
            info["error"] = f"Could not load sample config: {e}"
        
        return info