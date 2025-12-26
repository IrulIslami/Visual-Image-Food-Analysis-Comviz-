"""
Configuration loader and validator.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np


class Config:
    """Configuration container with validation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._validate()
        self._process_numpy_values()
    
    def _validate(self):
        """Validate required configuration keys."""
        required_sections = ['experiment', 'data', 'preprocessing', 
                           'features', 'model', 'output']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
    
    def _process_numpy_values(self):
        """Convert angle values to numpy arrays if needed."""
        if 'features' in self._config and 'glcm_angles' in self._config['features']:
            angles = self._config['features']['glcm_angles']
            self._config['features']['glcm_angles'] = [
                float(a) if isinstance(a, (int, float)) else a 
                for a in angles
            ]
    
    def get(self, key: str, default=None):
        """Get config value using dot notation (e.g., 'data.base_dir')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()
    
    @property
    def experiment_name(self) -> str:
        return self.get('experiment.name', 'unnamed_experiment')
    
    @property
    def use_segmentation(self) -> bool:
        return self.get('data.use_segmentation', False)
    
    @property
    def random_state(self) -> int:
        return self.get('random_state', 42)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def setup_random_seeds(seed: int):
    """Setup random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not available