import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from data.preprocessor import ImagePreprocessor
from features.color_features import ColorFeatureExtractor
from features.texture_features import TextureFeatureExtractor
from features.shape_features import ShapeFeatureExtractor


class FeatureExtractor:
    """Orchestrates extraction of all feature types."""
    
    def __init__(self, config):
        """
        Initialize feature extractor with all sub-extractors.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(config)
        
        # Initialize feature extractors
        self.color_extractor = ColorFeatureExtractor(
            bins=tuple(config.get('features.hsv_bins', [8, 8, 4]))
        )
        
        self.texture_extractor = TextureFeatureExtractor(
            glcm_size=tuple(config.get('features.glcm_size', [128, 128])),
            levels=config.get('features.glcm_levels', 8),
            distances=config.get('features.glcm_distances', [1]),
            angles=config.get('features.glcm_angles', [0, np.pi/4, np.pi/2, 3*np.pi/4])
        )
        
        self.shape_extractor = ShapeFeatureExtractor(
            extract_area_features=config.get('features.extract_area_features', True)
        )
    
    def extract_features_from_pair(
        self, 
        before_path: Path, 
        after_path: Path
    ) -> Optional[np.ndarray]:
        """
        Extract all features from a single image pair.
        
        Args:
            before_path: Path to before meal image
            after_path: Path to after meal image
            
        Returns:
            Feature vector or None if extraction fails
        """
        # Process image pair (load, normalize, segment)
        img_before, img_after, mask_before, mask_after = \
            self.preprocessor.process_image_pair(before_path, after_path)
        
        if img_before is None or img_after is None:
            return None
        
        # Apply masks to images if segmentation is enabled
        if self.config.get('segmentation.enabled', False):
            img_before_masked = self.preprocessor.apply_mask_to_image(
                img_before, mask_before
            )
            img_after_masked = self.preprocessor.apply_mask_to_image(
                img_after, mask_after
            )
        else:
            img_before_masked = img_before
            img_after_masked = img_after
            mask_before = None
            mask_after = None
        
        # Extract color features (HSV histograms)
        color_features = self.color_extractor.extract_pair_features(
            img_before_masked, img_after_masked
        )
        
        # Extract texture features (GLCM)
        texture_features = self.texture_extractor.extract_pair_features(
            img_before_masked, img_after_masked
        )
        
        # Extract shape features (SSIM, MSE, area)
        shape_features = self.shape_extractor.extract_pair_features(
            img_before_masked, img_after_masked,
            mask_before, mask_after
        )
        
        # Concatenate all features
        features = np.concatenate([
            color_features,
            texture_features,
            shape_features
        ], axis=0)
        
        return features
    
    def extract_features_from_dataset(
        self, 
        dataframe,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Extract features from entire dataset.
        
        Args:
            dataframe: DataFrame with 'before_path', 'after_path', 
                      'y_percent_leftover', and 'group' columns
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (X, y, groups, skipped_indices)
            - X: Feature matrix (n_samples, n_features)
            - y: Target values (n_samples,)
            - groups: Group labels for cross-validation (n_samples,)
            - skipped_indices: List of (index, error_message) for failed extractions
        """
        features_list = []
        valid_indices = []
        skipped = []
        
        iterator = tqdm(dataframe.iterrows(), total=len(dataframe)) \
                   if show_progress else dataframe.iterrows()
        
        for idx, row in iterator:
            try:
                features = self.extract_features_from_pair(
                    row['before_path'],
                    row['after_path']
                )
                
                if features is not None and np.all(np.isfinite(features)):
                    features_list.append(features)
                    valid_indices.append(idx)
                else:
                    skipped.append((idx, "Invalid features (NaN or Inf)"))
                    
            except Exception as e:
                skipped.append((idx, str(e)))
        
        # Convert to arrays
        X = np.vstack(features_list)
        y = dataframe.loc[valid_indices, 'y_percent_leftover'].to_numpy().astype(np.float32)
        groups = dataframe.loc[valid_indices, 'group'].to_numpy()
        
        return X, y, groups, skipped
    
    def get_feature_names(self) -> list:
        """
        Get names of all features in order.
        
        Returns:
            List of feature names
        """
        names = []
        names.extend(self.color_extractor.get_feature_names())
        names.extend(self.texture_extractor.get_feature_names())
        names.extend(self.shape_extractor.get_feature_names())
        return names
    
    def get_feature_dimensions(self) -> dict:
        """
        Get dimensions of each feature type.
        
        Returns:
            Dictionary with feature type dimensions
        """
        color_dim = self.color_extractor.feature_dim * 3  # before, after, delta
        texture_dim = len(self.texture_extractor.properties) * 3
        shape_dim = len(self.shape_extractor.get_feature_names())
        
        return {
            'color': color_dim,
            'texture': texture_dim,
            'shape': shape_dim,
            'total': color_dim + texture_dim + shape_dim
        }