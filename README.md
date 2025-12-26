# Visual Image Analysis for Predicting Hospital Food Leftovers

## Project Overview
This project predicts the percentage of leftover food from visual features extracted from "before" and "after" meal images using a Random Forest Regressor.

## Project Structure
```
food-leftover-prediction/
├── config/
│   ├── config_raw.yaml          # Configuration for raw images (Scenario 1)
│   └── config_segmented.yaml    # Configuration for segmented images (Scenario 2)
├── data/
│   ├── raw/                     # Raw meal images
│   │   ├── before/
│   │   └── after/
│   ├── segmented/               # Pre-segmented images
│   │   ├── before/
│   │   └── after/
│   └── data_original_edit.xlsx  # Metadata file
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading and indexing
│   │   └── preprocessor.py      # Image preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py         # Main feature extraction orchestrator
│   │   ├── color_features.py    # Color feature extraction
│   │   ├── texture_features.py  # Texture (GLCM) features
│   │   └── shape_features.py    # Shape/area features
│   ├── models/
│   │   ├── __init__.py
│   │   └── trainer.py           # Model training and evaluation
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py     # Plotting utilities
│       └── metrics.py           # Evaluation metrics and CI
├── scripts/
│   ├── run_experiment.py        # Main execution script
│   └── evaluate_results.py      # Post-analysis script
├── notebooks/
│   └── exploratory_analysis.ipynb
└── results/
    ├── models/                  # Saved models
    ├── features/                # Extracted features (cached)
    ├── metrics/                 # Evaluation results
    └── figures/                 # Plots and visualizations
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Place your data according to the structure:
- Excel file: `data/data_original_edit.xlsx`
- Images in respective folders based on scenario

### 3. Run Experiments

#### Scenario 1: Raw Images
```bash
python scripts/run_experiment.py --config config/config_raw.yaml
```

#### Scenario 2: Segmented Images
```bash
python scripts/run_experiment.py --config config/config_segmented.yaml
```

### 4. Analyze Results
```bash
python scripts/evaluate_results.py --results results/metrics/segmented_results.json
```

## Configuration

Each scenario has its own configuration file. Key parameters:
- `use_segmentation`: Whether to apply segmentation
- `img_size`: Image resize dimensions
- `hsv_bins`: HSV histogram bins
- `glcm_levels`: GLCM quantization levels
- Model hyperparameters

## Features Extracted

### Color Features (HSV Histogram)
- Before, After, and Delta (256 bins each = 768 features)

### Texture Features (GLCM)
- Contrast, Homogeneity, Energy, Correlation
- Before, After, and Delta (12 features)

### Pair Metrics
- SSIM and MSE between before/after (2 features)

### Shape Features (for segmented scenario)
- Area before, Area after, Delta area, Ratio area (4 features)

**Total: 782 features (raw) or 786 features (segmented)**

## Reproducibility

All experiments use fixed random seeds (default: 42) for:
- Data splitting
- Model training
- Bootstrap confidence intervals