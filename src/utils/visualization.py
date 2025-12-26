import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures (None = don't save)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        filename: str = "predictions.png"
    ):
        """Plot predicted vs actual values."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Leftover %', fontsize=12)
        ax.set_ylabel('Predicted Leftover %', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "residuals.png"
    ):
        """Plot residual distribution and residuals vs predicted."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual histogram
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residuals', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Leftover %', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "feature_importance.png"
    ):
        """Plot feature importance."""
        df_plot = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        ax.barh(range(len(df_plot)), df_plot['importance'], color='steelblue')
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cv_results(
        self,
        cv_results_df: pd.DataFrame,
        filename: str = "cv_results.png"
    ):
        """Plot cross-validation results."""
        metrics = ['mae', 'mse', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = cv_results_df[metric]
            
            ax.plot(cv_results_df['fold'], values, marker='o', linewidth=2, markersize=8)
            ax.axhline(values.mean(), color='r', linestyle='--', label=f'Mean: {values.mean():.3f}')
            ax.fill_between(
                cv_results_df['fold'],
                values.mean() - values.std(),
                values.mean() + values.std(),
                alpha=0.2, color='red'
            )
            
            ax.set_xlabel('Fold', fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(f'{metric.upper()} Across Folds', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
