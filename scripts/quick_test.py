"""
Quick test script to verify installation and basic functionality.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config import load_config, setup_random_seeds
        print("✓ config module")
        
        from data.loader import DataLoader
        print("✓ data.loader module")
        
        from data.preprocessor import ImagePreprocessor
        print("✓ data.preprocessor module")
        
        from features.extractor import FeatureExtractor
        print("✓ features.extractor module")
        
        from models.trainer import ModelTrainer
        print("✓ models.trainer module")
        
        from utils.visualization import Visualizer
        print("✓ utils.visualization module")
        
        from utils.metrics import save_results
        print("✓ utils.metrics module")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import load_config
        
        # Try to load example config (you'll need to create this)
        config_path = Path("config/config_segmented.yaml")
        
        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            print("   (This is expected if you haven't created it yet)")
            return True
        
        config = load_config(config_path)
        print(f"✓ Loaded config: {config.experiment_name}")
        print(f"✓ Random state: {config.random_state}")
        print(f"✓ Use segmentation: {config.use_segmentation}")
        
        print("\n✅ Configuration loading works!")
        return True
        
    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        return False


def main():
    print("="*60)
    print("FOOD LEFTOVER PREDICTION - QUICK TEST")
    print("="*60 + "\n")
    
    success = True
    
    success &= test_imports()
    success &= test_config()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're ready to run experiments!")
        print("\nNext steps:")
        print("1. Prepare your data in the correct directory structure")
        print("2. Create/verify configuration files in config/")
        print("3. Run: python scripts/run_experiment.py --config config/config_segmented.yaml")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the errors above before proceeding.")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())