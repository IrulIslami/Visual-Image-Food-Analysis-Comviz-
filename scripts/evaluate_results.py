"""
Script to evaluate and compare results from different experiments.

Usage:
    python scripts/evaluate_results.py --results results/metrics/tes_2_raw_images_results.json
    python scripts/evaluate_results.py --raw results/metrics/tes_2_raw_images_results.json \
                                       --segmented results/metrics/tes_3_segmented_images_results.json
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.metrics import print_metrics_summary, compare_scenarios
from utils.visualization import Visualizer


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def evaluate_single_experiment(results_path: str):
    """Evaluate and display results from a single experiment."""
    results = load_results(results_path)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {results['experiment_name']}")
    print(f"Scenario: {results['scenario']}")
    print("="*80)
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Training samples: {results['n_train']}")
    print(f"  Test samples: {results['n_test']}")
    print(f"  Number of features: {results['n_features']}")
    print(f"  Using segmentation: {results['use_segmentation']}")
    
    # Print metrics
    print_metrics_summary(results)
    
    # Show top features
    if 'feature_importance' in results:
        print("Top 10 Most Important Features:")
        print("-" * 60)
        feature_df = pd.DataFrame(results['feature_importance'])
        for idx, row in feature_df.head(10).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<40s} {row['importance']:.6f}")
        print()
    
    # CV results summary
    if 'cv_results' in results:
        cv_df = pd.DataFrame(results['cv_results'])
        print("\nCross-Validation Results by Fold:")
        print("-" * 60)
        print(cv_df[['fold', 'mae', 'mse', 'r2']].to_string(index=False))
        print()


def compare_experiments(raw_path: str, segmented_path: str):
    """Compare results between raw and segmented scenarios."""
    results_raw = load_results(raw_path)
    results_segmented = load_results(segmented_path)
    
    print("\n" + "="*80)
    print("COMPARING EXPERIMENTS")
    print("="*80)
    
    print(f"\nExperiment 1 (Raw): {results_raw['experiment_name']}")
    print(f"  Samples: {results_raw['n_samples']} | Features: {results_raw['n_features']}")
    
    print(f"\nExperiment 2 (Segmented): {results_segmented['experiment_name']}")
    print(f"  Samples: {results_segmented['n_samples']} | Features: {results_segmented['n_features']}")
    
    # Compare metrics
    compare_scenarios(results_raw, results_segmented)
    
    # Compare CV results
    cv_raw = pd.DataFrame(results_raw['cv_results'])
    cv_seg = pd.DataFrame(results_segmented['cv_results'])
    
    print("Cross-Validation Comparison:")
    print("-" * 60)
    print(f"{'Metric':<10} {'Raw (mean±std)':<20} {'Segmented (mean±std)':<20}")
    print("-" * 60)
    
    for metric in ['mae', 'mse', 'r2']:
        raw_mean = cv_raw[metric].mean()
        raw_std = cv_raw[metric].std()
        seg_mean = cv_seg[metric].mean()
        seg_std = cv_seg[metric].std()
        
        print(f"{metric.upper():<10} {raw_mean:6.2f}±{raw_std:5.2f}{'':8s} "
              f"{seg_mean:6.2f}±{seg_std:5.2f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare experiment results"
    )
    
    # Single experiment evaluation
    parser.add_argument(
        "--results",
        type=str,
        help="Path to results JSON file for single experiment evaluation"
    )
    
    # Comparison mode
    parser.add_argument(
        "--raw",
        type=str,
        help="Path to raw images experiment results"
    )
    parser.add_argument(
        "--segmented",
        type=str,
        help="Path to segmented images experiment results"
    )
    
    args = parser.parse_args()
    
    if args.results:
        # Single experiment evaluation
        evaluate_single_experiment(args.results)
    
    elif args.raw and args.segmented:
        # Compare two experiments
        compare_experiments(args.raw, args.segmented)
    
    else:
        parser.print_help()
        print("\nError: Please provide either --results for single evaluation "
              "or both --raw and --segmented for comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()