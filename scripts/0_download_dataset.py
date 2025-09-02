#!/usr/bin/env python3
"""
Download AIDev dataset from HuggingFace
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


def main():
    """Download and save AIDev dataset tables as parquet files"""
    
    print("=== AIDev Dataset Downloader ===")
    
    # Initialize loader
    loader = HFDatasetLoader()
    
    # Get dataset info first
    print("\n1. Getting dataset information...")
    info = loader.get_dataset_info()
    print(f"Dataset: {info.get('dataset_name', 'Unknown')}")
    print(f"Cache dir: {info.get('cache_dir', 'Unknown')}")
    print(f"Download dir: {info.get('download_dir', 'Unknown')}")
    
    if 'error' in info:
        print(f"Error getting dataset info: {info['error']}")
        return
    
    # Download tables needed for refactoring analysis
    print("\n2. Downloading tables for refactoring analysis...")
    # Use default tables (will download 8 core tables including commit details)
    
    try:
        saved_files = loader.download_and_save_parquet()  # Uses default tables
        
        print(f"\n3. Download completed! Saved {len(saved_files)} tables:")
        for table, path in saved_files.items():
            print(f"  - {table}: {path}")
            
    except Exception as e:
        print(f"Error during download: {e}")
        return
    
    print("\n=== Download Complete ===")


if __name__ == "__main__":
    main()