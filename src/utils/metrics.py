import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


def save_results(results: Dict, output_path: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of results
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            results_serializable[key] = value.to_dict('records')
        else:
            results_serializable[key] = value
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_metrics_summary(results: Dict):
    """
    Print formatted summary of evaluation metrics.
    
    Args:
        results: Dictionary with metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {results['mae']:.2f}")
    print(f"  MSE:  {results['mse']:.2f}")
    print(f"  R²:   {results['r2']:.3f}")
    
    if 'mae_ci' in results:
        mae_ci = results['mae_ci']
        mse_ci = results['mse_ci']
        print(f"\n95% Confidence Intervals:")
        print(f"  MAE:  [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
        print(f"  MSE:  [{mse_ci[0]:.2f}, {mse_ci[1]:.2f}]")
    
    if 'cv_results' in results:
        cv_df = pd.DataFrame(results['cv_results'])
        print(f"\nNested Cross-Validation (mean ± std):")
        print(f"  MAE:  {cv_df['mae'].mean():.2f} ± {cv_df['mae'].std():.2f}")
        print(f"  MSE:  {cv_df['mse'].mean():.2f} ± {cv_df['mse'].std():.2f}")
        print(f"  R²:   {cv_df['r2'].mean():.3f} ± {cv_df['r2'].std():.3f}")
    
    print("\n" + "="*60 + "\n")


def compare_scenarios(results_raw: Dict, results_segmented: Dict):
    """
    Compare results between raw and segmented scenarios.
    
    Args:
        results_raw: Results from raw image scenario
        results_segmented: Results from segmented image scenario
    """
    print("\n" + "="*60)
    print("SCENARIO COMPARISON")
    print("="*60)
    
    metrics = ['mae', 'mse', 'r2']
    
    print(f"\n{'Metric':<10} {'Raw Images':<15} {'Segmented':<15} {'Difference':<15}")
    print("-" * 60)
    
    for metric in metrics:
        raw_val = results_raw.get(metric, 0)
        seg_val = results_segmented.get(metric, 0)
        diff = seg_val - raw_val
        
        # For MAE and MSE, lower is better (show as improvement)
        if metric in ['mae', 'mse']:
            improvement = "↓ Better" if diff < 0 else "↑ Worse"
        else:  # R² - higher is better
            improvement = "↑ Better" if diff > 0 else "↓ Worse"
        
        print(f"{metric.upper():<10} {raw_val:<15.3f} {seg_val:<15.3f} "
              f"{diff:+.3f} {improvement}")
    
    print("\n" + "="*60 + "\n")

