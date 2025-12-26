"""
Debug script to test feature extraction on a single image pair.
This helps identify what's going wrong.

Usage:
    python scripts/debug_extraction.py --config config/config_segmented.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config as cfg_module
from data.loader import load_and_prepare_data
from features.extractor import FeatureExtractor


def debug_single_pair(config, df_valid, row_idx=0):
    """Debug feature extraction on a single image pair."""
    
    row = df_valid.iloc[row_idx]
    
    print("\n" + "="*80)
    print("DEBUGGING SINGLE IMAGE PAIR")
    print("="*80)
    
    print(f"\nRow index: {row_idx}")
    print(f"Before image: {row['Image Before Eaten']}")
    print(f"After image: {row['Image After Eaten']}")
    print(f"Before path: {row['before_path']}")
    print(f"After path: {row['after_path']}")
    print(f"Target value: {row['y_percent_leftover']:.2f}%")
    
    # Check if files exist
    print("\n" + "-"*80)
    print("FILE CHECK")
    print("-"*80)
    print(f"Before exists: {row['before_path'].exists()}")
    print(f"After exists: {row['after_path'].exists()}")
    
    # Try to load images manually
    print("\n" + "-"*80)
    print("MANUAL IMAGE LOAD TEST")
    print("-"*80)
    
    try:
        img_before = cv2.imread(str(row['before_path']))
        if img_before is not None:
            print(f"✓ Before image loaded: shape={img_before.shape}")
        else:
            print(f"✗ Before image failed to load")
            return
    except Exception as e:
        print(f"✗ Exception loading before image: {e}")
        return
    
    try:
        img_after = cv2.imread(str(row['after_path']))
        if img_after is not None:
            print(f"✓ After image loaded: shape={img_after.shape}")
        else:
            print(f"✗ After image failed to load")
            return
    except Exception as e:
        print(f"✗ Exception loading after image: {e}")
        return
    
    # Try feature extraction
    print("\n" + "-"*80)
    print("FEATURE EXTRACTION TEST")
    print("-"*80)
    
    feature_extractor = FeatureExtractor(config)
    
    try:
        print("Extracting features...")
        features = feature_extractor.extract_features_from_pair(
            row['before_path'],
            row['after_path']
        )
        
        if features is None:
            print("✗ Feature extraction returned None")
            
            # Debug preprocessing step by step
            print("\n" + "-"*80)
            print("STEP-BY-STEP PREPROCESSING DEBUG")
            print("-"*80)
            
            preprocessor = feature_extractor.preprocessor
            
            # Step 1: Load and preprocess
            print("\n1. Loading and preprocessing images...")
            img_b = preprocessor.load_and_preprocess(row['before_path'])
            img_a = preprocessor.load_and_preprocess(row['after_path'])
            
            if img_b is None:
                print("   ✗ Before image preprocessing failed")
                return
            else:
                print(f"   ✓ Before image: shape={img_b.shape}, dtype={img_b.dtype}")
            
            if img_a is None:
                print("   ✗ After image preprocessing failed")
                return
            else:
                print(f"   ✓ After image: shape={img_a.shape}, dtype={img_a.dtype}")
            
            # Step 2: Segmentation
            print("\n2. Applying segmentation...")
            try:
                mask_b = preprocessor.segment_food(img_b)
                mask_a = preprocessor.segment_food(img_a)
                print(f"   ✓ Before mask: shape={mask_b.shape}, sum={mask_b.sum()}, dtype={mask_b.dtype}")
                print(f"   ✓ After mask: shape={mask_a.shape}, sum={mask_a.sum()}, dtype={mask_a.dtype}")
                
                if mask_b.sum() == 0:
                    print("   ⚠ WARNING: Before mask is empty (all zeros)!")
                if mask_a.sum() == 0:
                    print("   ⚠ WARNING: After mask is empty (all zeros)!")
                    
            except Exception as e:
                print(f"   ✗ Segmentation failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Step 3: Apply mask
            print("\n3. Applying masks to images...")
            img_bm = preprocessor.apply_mask_to_image(img_b, mask_b)
            img_am = preprocessor.apply_mask_to_image(img_a, mask_a)
            print(f"   ✓ Before masked: shape={img_bm.shape}")
            print(f"   ✓ After masked: shape={img_am.shape}")
            
            # Step 4: Extract color features
            print("\n4. Extracting color features...")
            try:
                color_feats = feature_extractor.color_extractor.extract_pair_features(img_bm, img_am)
                print(f"   ✓ Color features: shape={color_feats.shape}, has_nan={np.any(np.isnan(color_feats))}")
            except Exception as e:
                print(f"   ✗ Color extraction failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 5: Extract texture features
            print("\n5. Extracting texture features...")
            try:
                texture_feats = feature_extractor.texture_extractor.extract_pair_features(img_bm, img_am)
                print(f"   ✓ Texture features: shape={texture_feats.shape}, has_nan={np.any(np.isnan(texture_feats))}")
            except Exception as e:
                print(f"   ✗ Texture extraction failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 6: Extract shape features
            print("\n6. Extracting shape features...")
            try:
                shape_feats = feature_extractor.shape_extractor.extract_pair_features(
                    img_bm, img_am, mask_b, mask_a
                )
                print(f"   ✓ Shape features: shape={shape_feats.shape}, has_nan={np.any(np.isnan(shape_feats))}")
            except Exception as e:
                print(f"   ✗ Shape extraction failed: {e}")
                import traceback
                traceback.print_exc()
                
        elif not np.all(np.isfinite(features)):
            print(f"✗ Features contain NaN or Inf values")
            print(f"   Shape: {features.shape}")
            print(f"   NaN count: {np.sum(np.isnan(features))}")
            print(f"   Inf count: {np.sum(np.isinf(features))}")
        else:
            print(f"✓ Features extracted successfully!")
            print(f"   Shape: {features.shape}")
            print(f"   Min: {features.min():.6f}")
            print(f"   Max: {features.max():.6f}")
            print(f"   Mean: {features.mean():.6f}")
            
    except Exception as e:
        print(f"✗ Exception during feature extraction: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Debug feature extraction")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--index", type=int, default=0, help="Row index to debug")
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = cfg_module.load_config(args.config)
    print(f"✓ Config loaded: {config.experiment_name}")
    print(f"  Segmentation enabled: {config.use_segmentation}")
    
    # Load data
    print("\nLoading data...")
    df_valid, _, _ = load_and_prepare_data(config)
    print(f"✓ Loaded {len(df_valid)} samples")
    
    # Debug single pair
    debug_single_pair(config, df_valid, args.index)
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()