import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ImagePreprocessor:
    """Handles image loading, normalization, and segmentation."""
    
    def __init__(self, config):
        self.config = config
        self.img_size = tuple(config.get('preprocessing.img_size', [256, 256]))
        self.lighting_mode = config.get('preprocessing.lighting_normalization', 'clahe')
        self.use_segmentation = config.get('segmentation.enabled', False)
        
        # Segmentation parameters
        if self.use_segmentation:
            self.seg_method = config.get('segmentation.method', 'grabcut')
            self.rect_scale = config.get('segmentation.rect_scale', 0.75)
            self.seg_iterations = config.get('segmentation.iterations', 3)
            self.morph_open_iters = config.get('segmentation.morphology.open_iterations', 1)
            self.morph_close_iters = config.get('segmentation.morphology.close_iterations', 1)
            self.kernel_size = tuple(config.get('segmentation.morphology.kernel_size', [3, 3]))
    
    def load_and_preprocess(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed RGB image or None if loading fails
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Apply lighting normalization
        img = self._normalize_lighting(img)
        
        return img
    
    def _normalize_lighting(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Normalize lighting using CLAHE or Gray World algorithm.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            Lighting-normalized RGB image
        """
        if self.lighting_mode == "grayworld":
            return self._gray_world_normalization(img_rgb)
        elif self.lighting_mode == "clahe":
            return self._clahe_normalization(img_rgb)
        else:
            return img_rgb  # No normalization
    
    @staticmethod
    def _gray_world_normalization(img_rgb: np.ndarray) -> np.ndarray:
        """Apply Gray World color constancy algorithm."""
        mean_rgb = np.mean(img_rgb, axis=(0, 1))
        gray_mean = np.mean(mean_rgb)
        scale = gray_mean / (mean_rgb + 1e-8)
        img_normalized = np.clip(img_rgb * scale, 0, 255).astype(np.uint8)
        return img_normalized
    
    @staticmethod
    def _clahe_normalization(img_rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back to RGB
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return img_normalized
    
    def segment_food(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Segment food region from image using GrabCut.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            Binary mask (0 or 1) of food region
        """
        if not self.use_segmentation:
            # Return full mask (all ones)
            h, w = img_rgb.shape[:2]
            return np.ones((h, w), dtype=np.uint8)
        
        if self.seg_method == "grabcut":
            return self._grabcut_segmentation(img_rgb)
        else:
            raise ValueError(f"Unknown segmentation method: {self.seg_method}")
    
    def _grabcut_segmentation(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply GrabCut segmentation with center rectangle initialization.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            Binary mask (0 or 1)
        """
        h, w = img_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        rw, rh = int(w * self.rect_scale), int(h * self.rect_scale)
        
        # Define rectangle around center
        x = max(0, cx - rw // 2)
        y = max(0, cy - rh // 2)
        rect = (x, y, min(rw, w - x), min(rh, h - y))
        
        # Initialize GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        mask = np.zeros((h, w), np.uint8)
        
        # Run GrabCut
        cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 
                   self.seg_iterations, cv2.GC_INIT_WITH_RECT)
        
        # Convert to binary mask (foreground = 1, background = 0)
        mask_binary = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 
            1, 0
        ).astype(np.uint8)
        
        # Apply morphological operations to clean up mask
        mask_binary = self._clean_mask(mask_binary)
        
        return mask_binary
    
    def _clean_mask(self, mask_binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean mask."""
        kernel = np.ones(self.kernel_size, np.uint8)
        
        # Opening: remove small noise
        mask_clean = cv2.morphologyEx(
            mask_binary, cv2.MORPH_OPEN, kernel, 
            iterations=self.morph_open_iters
        )
        
        # Closing: fill small holes
        mask_clean = cv2.morphologyEx(
            mask_clean, cv2.MORPH_CLOSE, kernel, 
            iterations=self.morph_close_iters
        )
        
        return mask_clean
    
    def apply_mask_to_image(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to image (zero out background).
        
        Args:
            img_rgb: Input RGB image
            mask: Binary mask (0 or 1)
            
        Returns:
            Masked image
        """
        if mask.ndim == 2:
            # Expand mask to 3 channels
            mask_3d = np.repeat(mask[..., None], 3, axis=2)
        else:
            mask_3d = mask
        
        # Apply mask
        img_masked = img_rgb.copy()
        img_masked[mask_3d == 0] = 0
        
        return img_masked
    
    def process_image_pair(
        self, 
        before_path: Path, 
        after_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a pair of before/after images.
        
        Args:
            before_path: Path to before image
            after_path: Path to after image
            
        Returns:
            Tuple of (before_img, after_img, before_mask, after_mask)
            Returns (None, None, None, None) if processing fails
        """
        # Load and preprocess both images
        img_before = self.load_and_preprocess(before_path)
        img_after = self.load_and_preprocess(after_path)
        
        if img_before is None or img_after is None:
            return None, None, None, None
        
        # Segment if enabled
        mask_before = self.segment_food(img_before)
        mask_after = self.segment_food(img_after)
        
        return img_before, img_after, mask_before, mask_after

