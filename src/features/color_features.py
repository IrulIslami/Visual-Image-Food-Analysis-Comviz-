import cv2
import numpy as np
from typing import Tuple


class ColorFeatureExtractor:
    """Extract color features using HSV color space."""
    
    def __init__(self, bins: Tuple[int, int, int] = (8, 8, 4)):
        """
        Initialize color feature extractor.
        
        Args:
            bins: Number of bins for (H, S, V) channels
        """
        self.bins = bins
        self.feature_dim = bins[0] * bins[1] * bins[2]
    
    def extract_hsv_histogram(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Extract normalized HSV histogram from RGB image.
        
        Args:
            img_rgb: Input RGB image (H x W x 3)
            
        Returns:
            Flattened normalized histogram (bins[0] * bins[1] * bins[2],)
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Calculate 3D histogram
        # HSV ranges: H=[0,180], S=[0,256], V=[0,256]
        hist = cv2.calcHist(
            [hsv], 
            [0, 1, 2],  # All three channels
            None,  # No mask
            self.bins, 
            [0, 180, 0, 256, 0, 256]
        )
        
        # Flatten and normalize
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)  # Normalize to sum to 1
        
        return hist.astype(np.float32)
    
    def extract_pair_features(
        self, 
        img_before: np.ndarray, 
        img_after: np.ndarray
    ) -> np.ndarray:
        """
        Extract color features from image pair.
        
        Features include:
        - HSV histogram of before image
        - HSV histogram of after image  
        - Delta (difference) histogram
        
        Args:
            img_before: Before meal image (RGB)
            img_after: After meal image (RGB)
            
        Returns:
            Concatenated feature vector (3 * feature_dim,)
        """
        # Extract individual histograms
        hist_before = self.extract_hsv_histogram(img_before)
        hist_after = self.extract_hsv_histogram(img_after)
        
        # Calculate delta
        hist_delta = hist_after - hist_before
        
        # Concatenate all features
        features = np.concatenate([hist_before, hist_after, hist_delta], axis=0)
        
        return features
    
    def get_feature_names(self) -> list:
        """
        Get names for all color features.
        
        Returns:
            List of feature names
        """
        H, S, V = self.bins
        names = []
        
        # Helper function to generate names for one set
        def add_names(prefix: str):
            for h in range(H):
                for s in range(S):
                    for v in range(V):
                        names.append(f"{prefix}_hsv[{h},{s},{v}]")
        
        # Add names for before, after, and delta
        add_names("before")
        add_names("after")
        add_names("delta")
        
        return names

