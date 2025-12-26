import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple


class ShapeFeatureExtractor:
    """Extract shape-based features and pair-wise similarity metrics."""
    
    def __init__(self, extract_area_features: bool = True):
        """
        Initialize shape feature extractor.
        
        Args:
            extract_area_features: Whether to extract area-based features
        """
        self.extract_area_features = extract_area_features
    
    def extract_pair_metrics(
        self, 
        img_before: np.ndarray, 
        img_after: np.ndarray
    ) -> np.ndarray:
        """
        Extract similarity metrics between image pair.
        
        Features:
        - SSIM (Structural Similarity Index)
        - MSE (Mean Squared Error)
        
        Args:
            img_before: Before image (RGB)
            img_after: After image (RGB)
            
        Returns:
            Feature vector [ssim, mse] (2,)
        """
        # Convert to grayscale
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        data_range = max(1e-6, float(gray_after.max() - gray_after.min()))
        ssim_value = ssim(gray_before, gray_after, data_range=data_range)
        
        # Calculate MSE
        mse_value = np.mean(
            (gray_before.astype(np.float32) - gray_after.astype(np.float32)) ** 2
        )
        
        return np.array([ssim_value, mse_value], dtype=np.float32)
    
    def extract_area_features(
        self, 
        mask_before: np.ndarray, 
        mask_after: np.ndarray
    ) -> np.ndarray:
        """
        Extract area-based features from segmentation masks.
        
        Features:
        - Normalized area before (relative to image size)
        - Normalized area after
        - Delta area (after - before)
        - Ratio area (after / before)
        
        Args:
            mask_before: Binary mask for before image (0 or 1)
            mask_after: Binary mask for after image (0 or 1)
            
        Returns:
            Feature vector [area_before, area_after, delta_area, ratio_area] (4,)
        """
        if not self.extract_area_features:
            return np.array([], dtype=np.float32)
        
        # Calculate total pixels
        total_pixels = mask_before.shape[0] * mask_before.shape[1]
        
        # Calculate normalized areas
        area_before = float(mask_before.sum()) / total_pixels
        area_after = float(mask_after.sum()) / total_pixels
        
        # Calculate derived features
        delta_area = area_after - area_before
        ratio_area = area_after / (area_before + 1e-8)  # Avoid division by zero
        
        return np.array(
            [area_before, area_after, delta_area, ratio_area], 
            dtype=np.float32
        )
    
    def extract_pair_features(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        mask_before: np.ndarray = None,
        mask_after: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract all shape-based features from image pair.
        
        Args:
            img_before: Before image (RGB)
            img_after: After image (RGB)
            mask_before: Binary mask for before image (optional)
            mask_after: Binary mask for after image (optional)
            
        Returns:
            Concatenated feature vector
        """
        # Extract pair metrics (SSIM, MSE)
        pair_metrics = self.extract_pair_metrics(img_before, img_after)
        
        # Extract area features if masks provided and enabled
        if (self.extract_area_features and 
            mask_before is not None and 
            mask_after is not None):
            area_features = self.extract_area_features(mask_before, mask_after)
            return np.concatenate([pair_metrics, area_features], axis=0)
        else:
            return pair_metrics
    
    def get_feature_names(self) -> list:
        """
        Get names for all shape features.
        
        Returns:
            List of feature names
        """
        names = ["pair_ssim", "pair_mse"]
        
        if self.extract_area_features:
            names.extend([
                "area_before",
                "area_after", 
                "delta_area",
                "ratio_area"
            ])
        
        return names
