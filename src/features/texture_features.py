import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List, Tuple


class TextureFeatureExtractor:
    """Extract texture features using GLCM."""
    
    def __init__(
        self, 
        glcm_size: Tuple[int, int] = (128, 128),
        levels: int = 8,
        distances: List[int] = [1],
        angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    ):
        """
        Initialize texture feature extractor.
        
        Args:
            glcm_size: Size to resize image for GLCM computation
            levels: Number of gray levels for quantization
            distances: List of pixel pair distances
            angles: List of angles in radians
        """
        self.glcm_size = glcm_size
        self.levels = levels
        self.distances = distances
        self.angles = angles
        self.properties = ["contrast", "homogeneity", "energy", "correlation"]
    
    def extract_glcm_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Extract GLCM texture features from RGB image.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            GLCM feature vector (4,) containing mean values of:
            [contrast, homogeneity, energy, correlation]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Resize for efficiency
        gray = cv2.resize(gray, self.glcm_size, interpolation=cv2.INTER_AREA)
        
        # Quantize to reduce levels (improves GLCM computation)
        quantized = np.floor(
            (gray.astype(np.float32) / 256.0) * self.levels
        ).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(
            quantized,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )
        
        # Extract properties and average across distances and angles
        features = []
        for prop in self.properties:
            prop_values = graycoprops(glcm, prop)
            # Average across all distances and angles
            features.append(np.mean(prop_values))
        
        return np.array(features, dtype=np.float32)
    
    def extract_pair_features(
        self, 
        img_before: np.ndarray, 
        img_after: np.ndarray
    ) -> np.ndarray:
        """
        Extract texture features from image pair.
        
        Features include:
        - GLCM features of before image (4)
        - GLCM features of after image (4)
        - Delta features (4)
        
        Args:
            img_before: Before meal image (RGB)
            img_after: After meal image (RGB)
            
        Returns:
            Concatenated feature vector (12,)
        """
        # Extract GLCM features
        glcm_before = self.extract_glcm_features(img_before)
        glcm_after = self.extract_glcm_features(img_after)
        
        # Calculate delta
        glcm_delta = glcm_after - glcm_before
        
        # Concatenate
        features = np.concatenate([glcm_before, glcm_after, glcm_delta], axis=0)
        
        return features
    
    def get_feature_names(self) -> list:
        """
        Get names for all texture features.
        
        Returns:
            List of feature names
        """
        names = []
        
        # Before features
        for prop in self.properties:
            names.append(f"before_glcm_{prop}")
        
        # After features
        for prop in self.properties:
            names.append(f"after_glcm_{prop}")
        
        # Delta features
        for prop in self.properties:
            names.append(f"delta_glcm_{prop}")
        
        return names

