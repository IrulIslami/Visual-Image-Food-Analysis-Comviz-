import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import re


class DataLoader:
    """Handles loading and preprocessing of meal image metadata."""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config.get('data.base_dir'))
        self.excel_path = Path(config.get('data.excel_file'))
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate metadata from Excel file.
        
        Returns:
            DataFrame with meal image metadata
        """
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        df = pd.read_excel(self.excel_path)
        
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        
        # Validate required columns
        required_cols = {
            "Image Before Eaten", 
            "Image After Eaten", 
            "Weight Before Eaten (g)", 
            "Weight After Eaten (g)", 
            "Name of the food"
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate target variable (percentage leftover)
        wb = df["Weight Before Eaten (g)"].astype(float)
        wa = df["Weight After Eaten (g)"].astype(float)
        
        # Calculate percentage: 100 * (after / before)
        y = 100.0 * np.divide(wa, wb, out=np.zeros_like(wa, dtype=float), where=wb!=0)
        y = np.clip(y, 0, 100)
        df["y_percent_leftover"] = y
        
        # Mark outliers (after > before or before < 1g)
        df["outlier_weight"] = (wa > wb) | (wb < 1)
        
        return df
    
    def build_image_index(self, folder: Path) -> Dict[str, Path]:
        """
        Build index mapping filenames to full paths.
        
        Args:
            folder: Root folder containing images
            
        Returns:
            Dictionary mapping lowercase filename to Path object
        """
        index = {}
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                index[p.name.lower()] = p
        
        return index
    
    def prepare_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Path], Dict[str, Path]]:
        """
        Prepare complete dataset with image paths.
        
        Returns:
            Tuple of (dataframe, before_index, after_index)
        """
        # Load metadata
        df = self.load_metadata()
        
        # Build image indices
        before_dir = self.base_dir / self.config.get('data.before_dirname', 'before')
        after_dir = self.base_dir / self.config.get('data.after_dirname', 'after')
        
        before_index = self.build_image_index(before_dir)
        after_index = self.build_image_index(after_dir)
        
        print(f"Indexed {len(before_index)} 'before' images")
        print(f"Indexed {len(after_index)} 'after' images")
        
        # Map image names to paths
        df["before_key"] = df["Image Before Eaten"].apply(self._to_key)
        df["after_key"] = df["Image After Eaten"].apply(self._to_key)
        df["before_path"] = df["before_key"].map(before_index)
        df["after_path"] = df["after_key"].map(after_index)
        
        # Filter valid rows (both images found)
        df_valid = df[
            df["before_path"].notna() & df["after_path"].notna()
        ].reset_index(drop=True)
        
        # Create grouping variable for cross-validation
        df_valid["group"] = df_valid.apply(
            lambda row: self._infer_group(
                Path(row["Image Before Eaten"]).name, 
                row["Name of the food"]
            ), 
            axis=1
        )
        
        print(f"Valid samples: {len(df_valid)} / {len(df)}")
        
        return df_valid, before_index, after_index
    
    @staticmethod
    def _to_key(filename: str) -> str:
        """Convert filename to lowercase key."""
        return Path(str(filename).strip()).name.lower()
    
    @staticmethod
    def _infer_group(filename: str, food_name: str) -> str:
        """
        Infer group ID from filename pattern (e.g., '001_002').
        Falls back to food name if pattern not found.
        
        Args:
            filename: Image filename
            food_name: Food type name
            
        Returns:
            Group identifier string
        """
        # Try to extract pattern like '001_002' from filename
        match = re.match(r"^(\d{3}_\d{3})", filename)
        if match:
            return match.group(1)
        
        # Fallback to food name
        return f"FOOD::{food_name.strip().lower()}"


def load_and_prepare_data(config) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Convenience function to load and prepare data.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (valid_dataframe, before_index, after_index)
    """
    loader = DataLoader(config)
    return loader.prepare_dataset()

