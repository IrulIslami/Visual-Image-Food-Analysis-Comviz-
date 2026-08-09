import argparse
import sys
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# file diimport
import config as cfg_module
from data.loader import load_and_prepare_data
from features.extractor import FeatureExtractor
from models.trainer import ModelTrainer
import models.trainer as trainer_module
from utils.visualization import Visualizer
from utils.metrics import save_results, print_metrics_summary



def save_prediction_dataframe(eval_results, output_dir: Path, experiment_name: str):
    """
    Save test-set prediction results as CSV so they can be opened as a DataFrame.
    Columns:
    - y_true: actual leftover percentage
    - y_pred: predicted leftover percentage
    - residual: y_true - y_pred
    - abs_error: absolute prediction error
    """
    y_true = np.asarray(eval_results['y_true'])
    y_pred = np.asarray(eval_results['y_pred'])

    df_predictions = pd.DataFrame({
        'sample_no': np.arange(1, len(y_true) + 1),
        'y_true': y_true,
        'y_pred': y_pred,
        'residual': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred)
    })

    output_path = output_dir / f"{experiment_name}_predictions_dataframe.csv"
    df_predictions.to_csv(output_path, index=False)
    print(f"✓ Prediction DataFrame saved to: {output_path}")

    return df_predictions, output_path



def save_complete_feature_dataframe(
        df_valid,
        X,
        feature_names,
        train_idx,
        test_idx,
        eval_results,
        output_dir: Path,
        experiment_name: str
    ):
    
    """
    Save complete dataset containing metadata, prediction and all extracted features.
    """

    df = df_valid.reset_index(drop=True).copy()

    # nomor sampel
    df.insert(0, "sample_no", np.arange(1, len(df)+1))

    # ===== Metadata =====

    if "food_id" not in df.columns:
        df["food_id"] = np.nan

    if "Image Before Eaten" in df.columns:
        df.rename(columns={"Image Before Eaten": "id_before_image"}, inplace=True)

    if "Image After Eaten" in df.columns:
        df.rename(columns={"Image After Eaten": "id_after_image"}, inplace=True)

    if "y_percent_leftover" in df.columns:
        df.rename(columns={"y_percent_leftover": "actual_leftover"}, inplace=True)

    # ===== Dataset split =====

    df["dataset"] = "train"
    df.loc[test_idx, "dataset"] = "test"

    # ===== Prediction =====

    df["y_pred"] = np.nan
    df.loc[test_idx, "y_pred"] = eval_results["y_pred"]

    df["abs_error"] = np.nan
    df.loc[test_idx, "abs_error"] = np.abs(
        eval_results["y_true"] - eval_results["y_pred"]
    )

    # ===== Feature dataframe =====

    feature_df = pd.DataFrame(
        X,
        columns=feature_names
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            feature_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ===== Kolom utama di depan =====

    preferred_cols = [
        "sample_no",
        "food_id",
        "group",
        "id_before_image",
        "id_after_image",
        "actual_leftover",
        "dataset",
        "y_pred",
        "abs_error"
    ]

    feature_cols = [
        c for c in feature_df.columns
    ]

    remaining = [
        c for c in df.columns
        if c not in preferred_cols + feature_cols
    ]

    df = df[
        preferred_cols +
        remaining +
        feature_cols
    ]

    output_path = (
        output_dir /
        f"{experiment_name}_complete_features_dataframe.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✓ Complete feature DataFrame saved to: {output_path}")

    return df

def save_result_dataframes(cv_results_df, fi_mdi, fi_pfi, output_dir: Path, experiment_name: str):
    """Save cross-validation and feature-importance tables as CSV files."""
    cv_path = output_dir / f"{experiment_name}_cv_results_dataframe.csv"
    mdi_path = output_dir / f"{experiment_name}_feature_importance_mdi_dataframe.csv"
    pfi_path = output_dir / f"{experiment_name}_feature_importance_pfi_dataframe.csv"

    cv_results_df.to_csv(cv_path, index=False)
    fi_mdi.to_csv(mdi_path, index=False)
    fi_pfi.to_csv(pfi_path, index=False)

    print(f"✓ CV results DataFrame saved to: {cv_path}")
    print(f"✓ MDI feature importance DataFrame saved to: {mdi_path}")
    print(f"✓ PFI feature importance DataFrame saved to: {pfi_path}")

    return {
        'cv_results_dataframe': str(cv_path),
        'feature_importance_mdi_dataframe': str(mdi_path),
        'feature_importance_pfi_dataframe': str(pfi_path)
    }


def save_segmentation_example(feature_extractor, df_valid, figures_dir: Path, experiment_name: str):
    """
    Save one example visualization for the segmented experiment.
    The figure contains before/after image, binary mask, and masked result.
    """
    if not feature_extractor.config.get('segmentation.enabled', False):
        print("Segmentation example skipped because segmentation.enabled = False")
        return None

    if len(df_valid) == 0:
        print("Segmentation example skipped because no valid samples were found")
        return None

    row = df_valid.iloc[0]
    img_before, img_after, mask_before, mask_after = feature_extractor.preprocessor.process_image_pair(
        row['before_path'], row['after_path']
    )

    if img_before is None or img_after is None:
        print("Segmentation example skipped because the selected image pair could not be loaded")
        return None

    before_masked = feature_extractor.preprocessor.apply_mask_to_image(img_before, mask_before)
    after_masked = feature_extractor.preprocessor.apply_mask_to_image(img_after, mask_after)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_before)
    axes[0, 0].set_title('Before - Preprocessed')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_before, cmap='gray')
    axes[0, 1].set_title('Before - GrabCut Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(before_masked)
    axes[0, 2].set_title('Before - Segmented Result')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img_after)
    axes[1, 0].set_title('After - Preprocessed')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_after, cmap='gray')
    axes[1, 1].set_title('After - GrabCut Mask')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(after_masked)
    axes[1, 2].set_title('After - Segmented Result')
    axes[1, 2].axis('off')

    sample_title = f"Segmentation Example - {experiment_name}"
    if 'Image Before Eaten' in df_valid.columns and 'Image After Eaten' in df_valid.columns:
        sample_title += f"\nBefore: {row['Image Before Eaten']} | After: {row['Image After Eaten']}"
    fig.suptitle(sample_title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = figures_dir / f"{experiment_name}_segmentation_example.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Segmentation example saved to: {output_path}")
    return output_path

def extract_config_snapshot(config) -> dict:
    """
    Ambil snapshot konfigurasi ekstraksi fitur dan model
    yang relevan untuk dicatat bersama hasil eksperimen.
    Berguna untuk reprodusibilitas dan perbandingan antar run.
    """
    return {
        # ── Preprocessing ──────────────────────────────────
        "preprocessing": {
            "img_size": config.get("preprocessing.img_size"),
            "lighting_normalization": config.get(
                "preprocessing.lighting_normalization"
            ),
        },

        # ── Segmentasi ─────────────────────────────────────
        "segmentation": {
            "enabled": config.get("segmentation.enabled", False),
            "method": config.get("segmentation.method", None),
            "rect_scale": config.get("segmentation.rect_scale", None),
            "iterations": config.get("segmentation.iterations", None),
        },

        # ── Ekstraksi Fitur ────────────────────────────────
        "features": {
            "hsv_bins": config.get("features.hsv_bins"),
            "hsv_n_features": (
                config.get("features.hsv_bins", [8, 8, 4])[0]
                * config.get("features.hsv_bins", [8, 8, 4])[1]
                * config.get("features.hsv_bins", [8, 8, 4])[2]
                * 3  # before, after, delta
            ),
            "glcm_size": config.get("features.glcm_size"),
            "glcm_levels": config.get("features.glcm_levels"),
            "glcm_distances": config.get("features.glcm_distances"),
            "glcm_angles": config.get("features.glcm_angles"),
            "glcm_n_features": 12,  # 4 props × 3 (before, after, delta)
            "extract_area_features": config.get(
                "features.extract_area_features", False
            ),
        },

        # ── Cross-Validation ───────────────────────────────
        "cross_validation": {
            "outer_splits": config.get("cross_validation.outer_splits"),
            "inner_splits": config.get("cross_validation.inner_splits"),
            "test_size": config.get("cross_validation.test_size"),
            "stratify_by_group": config.get(
                "cross_validation.stratify_by_group"
            ),
        },

        # ── Hyperparameter Search ──────────────────────────
        "hyperparameter_search": {
            "enabled": config.get(
                "model.hyperparameter_search.enabled", False
            ),
            "method": config.get(
                "model.hyperparameter_search.method", None
            ),
            "param_grid": config.get(
                "model.hyperparameter_search.param_grid", None
            ),
            "scoring": config.get(
                "model.hyperparameter_search.scoring", None
            ),
        },

        # ── Random State ───────────────────────────────────
        "random_state": config.get("random_state", 42),
    }
    

#minta parameter config string
def main(config_path: str):
    """
    Run complete experiment pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    #header biar rapi di terminal
    print("="*80)
    print("FOOD LEFTOVER PREDICTION EXPERIMENT")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = cfg_module.load_config(config_path)
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Scenario: {config.get('experiment.scenario', 'unknown')}")
    print(f"Using segmentation: {config.get('segmentation.enabled', False)}")
    
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
    
    print(inspect.getfile(trainer_module))
    trainer = ModelTrainer(config)
    
    
    # Perform nested cross-validation
    print("\nPerforming nested cross-validation...")
    cv_results_df = trainer.nested_cross_validation(X, y, groups, verbose=True)
    
    print("\nCross-validation summary (mean ± std):")
    summary = cv_results_df[['mae', 'mse', 'r2']].agg(['mean', 'std'])
    print(summary)
    
    # Train final model on single split
    print("\nTraining final model on train/test split...")


    model, X_train, X_test, y_train, y_test, train_idx, test_idx = trainer.train_final_model( X, y, groups)

    print(f"✓ Model trained")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    eval_results = trainer.evaluate_model(model, X_test, y_test, compute_ci=True)
    
    print_metrics_summary(eval_results)
    
    # Get feature importance
    print("\nMenghitung Feature Importance...")

    # MDI — cepat, dari struktur pohon
    print("  [1/2] MDI (Mean Decrease Impurity)...")
    fi_mdi = trainer.get_feature_importance(feature_names, top_n=10)

    # PFI — lebih lambat, dari permutasi pada test set
    print("  [2/2] Permutation Feature Importance (PFI)...")
    fi_pfi = trainer.get_permutation_importance(
        X_test, y_test,
        feature_names,
        top_n=10,
        n_repeats=30
    )

    # Gabungan untuk perbandingan
    fi_combined = trainer.get_combined_importance(fi_mdi, fi_pfi)

    # Save DataFrame outputs to metrics directory
    metrics_dir = Path(config.get('output.metrics_dir'))
    df_predictions, prediction_df_path = save_prediction_dataframe(
        eval_results, metrics_dir, config.experiment_name
    )
    
    complete_df = save_complete_feature_dataframe(
        df_valid=df_valid,
        X=X,
        feature_names=feature_names,
        train_idx=train_idx,
        test_idx=test_idx,
        eval_results=eval_results,
        output_dir=metrics_dir,
        experiment_name=config.experiment_name
    )
    
    dataframe_paths = save_result_dataframes(
        cv_results_df, fi_mdi, fi_pfi, metrics_dir, config.experiment_name
    )
    dataframe_paths['predictions_dataframe'] = str(prediction_df_path)
    
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
        'use_segmentation': config.get('segmentation.enabled', False),
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
        'feature_importance_mdi': fi_mdi.to_dict('records'),
        'feature_importance_pfi': fi_pfi.to_dict('records'),
        'prediction_dataframe_preview': df_predictions.head(10).to_dict('records'),
        'dataframe_outputs': dataframe_paths,
        
        'config_snapshot': extract_config_snapshot(config)
    }
    
    # Save results
    results_path = Path(config.get('output.metrics_dir')) / \
                  f"{config.experiment_name}_results.json"
    save_results(results, results_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    figures_dir = Path(config.get('output.figures_dir'))
    visualizer = Visualizer(save_dir=figures_dir)

    # Save one segmentation visualization only for segmented scenario
    segmentation_example_path = save_segmentation_example(
        feature_extractor, df_valid, figures_dir, config.experiment_name
    )
    
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
    
    # Visualisasi MDI (sudah ada, ganti nama variabel)
    visualizer.plot_feature_importance(
        fi_mdi,
        top_n=10,
        filename=f"{config.experiment_name}_fi_mdi.png"
    )

    # Visualisasi PFI (baru)
    visualizer.plot_feature_importance(
        fi_pfi,
        top_n=10,
        filename=f"{config.experiment_name}_fi_pfi.png"
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
