#!/usr/bin/env python3
"""
Run Java project extraction (Phase 1) - Comprehensive Java project identification
"""
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_java_extraction.java_project_validator import JavaProjectValidator
from src.phase1_java_extraction.data_merger import DataMerger


def main():
    """Run comprehensive Java project extraction and validation"""
    
    print("="*60)
    print("         JAVA PROJECT EXTRACTION (PHASE 1)")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize components
        print("\n1. Initializing Java project validator...")
        validator = JavaProjectValidator()
        
        print("2. Initializing data merger...")
        merger = DataMerger()
        
        # Run comprehensive Java detection
        print("\n" + "="*50)
        print("STEP 1: COMPREHENSIVE JAVA PROJECT DETECTION")
        print("="*50)
        
        validation_results = validator.run_comprehensive_java_detection()
        
        # Combine validation results
        print("\n" + "="*50)
        print("STEP 2: COMBINING VALIDATION RESULTS")
        print("="*50)
        
        final_results = validator.combine_validation_results(validation_results)
        
        if final_results.empty:
            print("ERROR: No validation results generated")
            return
        
        # Generate validation report
        print("\n" + "="*50)
        print("STEP 3: GENERATING VALIDATION REPORT")
        print("="*50)
        
        report = validator.generate_validation_report(final_results)
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total PRs analyzed: {report['total_prs_analyzed']:,}")
        print(f"Java projects identified: {report['java_projects_identified']:,}")
        print(f"Java project rate: {report['java_project_percentage']:.1f}%")
        
        print(f"\nJava file statistics:")
        print(f"  Average Java percentage: {report['java_file_stats']['avg_java_percentage']:.1f}%")
        print(f"  Median Java percentage: {report['java_file_stats']['median_java_percentage']:.1f}%")
        
        print(f"\nBuild system statistics:")
        print(f"  Projects with Maven: {report['build_system_stats']['projects_with_maven']:,}")
        print(f"  Projects with Gradle: {report['build_system_stats']['projects_with_gradle']:,}")
        print(f"  Projects with both: {report['build_system_stats']['projects_with_both']:,}")
        
        print(f"\nConfidence statistics:")
        print(f"  High confidence (≥0.8): {report['confidence_stats']['high_confidence_projects']:,}")
        print(f"  Medium confidence (0.5-0.8): {report['confidence_stats']['medium_confidence_projects']:,}")
        print(f"  Low confidence (<0.5): {report['confidence_stats']['low_confidence_projects']:,}")
        
        # Save results
        print("\n" + "="*50)
        print("STEP 4: SAVING RESULTS")
        print("="*50)
        
        validator.save_validation_results(final_results, report)
        
        # Optional: Create comprehensive merged dataset
        create_merged = input("\nCreate comprehensive merged dataset? (y/N): ").lower().strip()
        if create_merged == 'y':
            print("\n" + "="*50)
            print("STEP 5: CREATING COMPREHENSIVE DATASET")
            print("="*50)
            
            # Load all tables
            tables = merger.load_all_tables()
            
            # Create comprehensive dataset
            merged_df = merger.create_comprehensive_dataset(tables)
            
            # Save merged dataset
            merger.save_merged_dataset(merged_df)
            
            # Get merge statistics
            merge_stats = merger.get_merge_statistics(tables, merged_df)
            print(f"\nMerge statistics:")
            print(f"  Final dataset size: {merge_stats['final_dataset_size']:,} records")
            print(f"  Final dataset columns: {merge_stats['final_dataset_columns']}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("JAVA EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Execution time: {execution_time:.1f} seconds")
        print(f"Results saved to: data/filtered/java_repositories/")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()