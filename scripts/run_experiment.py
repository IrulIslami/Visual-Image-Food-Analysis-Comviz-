
import argparse
import sys
from pathlib import Path
import joblib
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config as cfg_module
from data.loader import load_and_prepare_data
from features.extractor import FeatureExtractor
from models.trainer import ModelTrainer
from utils.visualization import Visualizer
from utils.metrics import save_results, print_metrics_summary


def main(config_path: str):
    """
    Run complete experiment pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    print("="*80)
    print("FOOD LEFTOVER PREDICTION EXPERIMENT")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = cfg_module.load_config(config_path)
    print(f"Experiment: {config.experiment_name}")
    print(f"Scenario: {config.get('experiment.scenario', 'unknown')}")
    print(f"Using segmentation: {config.use_segmentation}")
    
    # Setup random seeds for reproducibility
    cfg_module.setup_random_seeds(config.random_state)
    print(f"Random seed: {config.random_state}")
    
    # Create output directories
    output_base = Path(config.get('output.model_dir')).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    for dir_key in ['model_dir', 'features_dir', 'metrics_dir', 'figures_dir']:
        Path(config.get(f'output.{dir_key}')).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading and preparing data")
    print("-"*80)
    
    df_valid, before_index, after_index = load_and_prepare_data(config)
    print(f"✓ Loaded {len(df_valid)} valid samples")
    print(f"  Unique groups: {df_valid['group'].nunique()}")
    
    # ========================================================================
    # STEP 2: Extract features
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Extracting features")
    print("-"*80)
    
    feature_extractor = FeatureExtractor(config)
    
    # Check if cached features exist
    feature_cache_path = Path(config.get('output.features_dir')) / \
                        f"{config.experiment_name}_features.npz"
    
    if feature_cache_path.exists() and not config.get('force_recompute', False):
        print(f"Loading cached features from: {feature_cache_path}")
        cached = np.load(feature_cache_path, allow_pickle=True)  # ← Add allow_pickle=True
        X = cached['X']
        y = cached['y']
        groups = cached['groups']
        print(f"✓ Loaded features: {X.shape}")
    else:
        print("Extracting features from images...")
        X, y, groups, skipped = feature_extractor.extract_features_from_dataset(
            df_valid, show_progress=True
        )
        
        print(f"✓ Extracted features: {X.shape}")
        print(f"  Skipped samples: {len(skipped)}")
        
        # Save features for future use
        if config.get('output.save_features', True):
            np.savez(feature_cache_path, X=X, y=y, groups=groups)
            print(f"✓ Saved features to: {feature_cache_path}")
    
    # Get feature names and dimensions
    feature_names = feature_extractor.get_feature_names()
    feature_dims = feature_extractor.get_feature_dimensions()
    
    print(f"\nFeature dimensions:")
    for feat_type, dim in feature_dims.items():
        print(f"  {feat_type}: {dim}")
    
    # ========================================================================
    # STEP 3: Train and evaluate model
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Training and evaluating model")
    print("-"*80)
    
    trainer = ModelTrainer(config)
    
    # Perform nested cross-validation
    print("\nPerforming nested cross-validation...")
    cv_results_df = trainer.nested_cross_validation(X, y, groups, verbose=True)
    
    print("\nCross-validation summary (mean ± std):")
    summary = cv_results_df[['mae', 'mse', 'r2']].agg(['mean', 'std'])
    print(summary)
    
    # Train final model on single split
    print("\nTraining final model on train/test split...")
    model, X_train, X_test, y_train, y_test = trainer.train_final_model(
        X, y, groups
    )
    print(f"✓ Model trained")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    eval_results = trainer.evaluate_model(model, X_test, y_test, compute_ci=True)
    
    print_metrics_summary(eval_results)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance(feature_names, top_n=20)
    
    # ========================================================================
    # STEP 4: Save results and visualizations
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Saving results and visualizations")
    print("-"*80)
    
    # Save model
    if config.get('output.save_model', True):
        model_path = Path(config.get('output.model_dir')) / \
                    f"{config.experiment_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
    
    # Prepare complete results
    results = {
        'experiment_name': config.experiment_name,
        'scenario': config.get('experiment.scenario'),
        'use_segmentation': config.use_segmentation,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mae': float(eval_results['mae']),
        'mse': float(eval_results['mse']),
        'r2': float(eval_results['r2']),
        'mae_ci': eval_results.get('mae_ci'),
        'mse_ci': eval_results.get('mse_ci'),
        'cv_results': cv_results_df.to_dict('records'),
        'best_params': trainer.best_params,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    # Save results
    results_path = Path(config.get('output.metrics_dir')) / \
                  f"{config.experiment_name}_results.json"
    save_results(results, results_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    figures_dir = Path(config.get('output.figures_dir'))
    visualizer = Visualizer(save_dir=figures_dir)
    
    y_test_array = eval_results['y_true']
    y_pred_array = eval_results['y_pred']
    
    visualizer.plot_predictions(
        y_test_array, y_pred_array,
        title=f"Predictions vs Actual - {config.experiment_name}",
        filename=f"{config.experiment_name}_predictions.png"
    )
    
    visualizer.plot_residuals(
        y_test_array, y_pred_array,
        filename=f"{config.experiment_name}_residuals.png"
    )
    
    visualizer.plot_feature_importance(
        feature_importance,
        top_n=20,
        filename=f"{config.experiment_name}_feature_importance.png"
    )
    
    visualizer.plot_cv_results(
        cv_results_df,
        filename=f"{config.experiment_name}_cv_results.png"
    )
    
    print(f"✓ Visualizations saved to: {figures_dir}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_path}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run food leftover prediction experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

