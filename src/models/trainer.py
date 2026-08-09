"""
Model training, hyperparameter tuning, and evaluation.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List
import pandas as pd
import inspect
import models.trainer as trainer_module

class ModelTrainer:
    """Handles model training, tuning, and evaluation."""
    
    def __init__(self, config):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.best_params = None
        
    def create_model(self, params: Dict = None) -> RandomForestRegressor:
        """
        Create Random Forest model with specified parameters.
        
        Args:
            params: Model parameters (uses config defaults if None)
            
        Returns:
            RandomForestRegressor instance
        """
        if params is None:
            params = self.config.get('model.params', {})
        
        return RandomForestRegressor(**params)
    
    def nested_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Perform nested cross-validation for unbiased performance estimation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            groups: Group labels for GroupKFold (n_samples,)
            verbose: Whether to print progress
            
        Returns:
            DataFrame with results for each outer fold
        """
        outer_splits = self.config.get('cross_validation.outer_splits', 5)
        inner_splits = self.config.get('cross_validation.inner_splits', 3)
        
        outer_cv = GroupKFold(n_splits=outer_splits)
        inner_cv = GroupKFold(n_splits=inner_splits)
        
        # Get hyperparameter search configuration
        param_grid = self._get_param_grid()
        scoring = self.config.get('model.hyperparameter_search.scoring', 
                                 'neg_mean_absolute_error')
        
        results = []
        fold_num = 1
        
        for train_idx, test_idx in outer_cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            
            
            cv_splits = list(inner_cv.split(X_train, y_train, groups_train))
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                RandomForestRegressor(),
                param_grid,
                scoring=scoring,
                cv=cv_splits,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate on outer test set
            y_pred = best_model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'fold': fold_num,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'best_params': grid_search.best_params_
            })
            
            if verbose:
                print(f"Fold {fold_num}: MAE={mae:.2f} | MSE={mse:.2f} | "
                      f"R²={r2:.3f} | best_params={grid_search.best_params_}")
            
            fold_num += 1
        
        return pd.DataFrame(results)
    
    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        params: Dict = None
    ) -> Tuple[...]:
    
        test_size = self.config.get('cross_validation.test_size', 0.2)
        random_state = self.config.get('random_state', 42)
    
        gss = GroupShuffleSplit(
            test_size=test_size, 
            random_state=random_state, 
            n_splits=1
        )
        train_idx, test_idx = next(gss.split(X, y, groups))
    
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── TAMBAHKAN BLOK INI ──────────────────────────────────────
        # Jika tuning aktif, cari best params dulu pada train split
        if self.config.get('model.hyperparameter_search.enabled', False):
            print("  Tuning final model pada train split...")
            inner_cv = GroupKFold(
                n_splits=self.config.get('cross_validation.inner_splits', 3)
            )
            param_grid = self._get_param_grid()
            scoring = self.config.get(
                'model.hyperparameter_search.scoring',
                'neg_mean_absolute_error'
            )
            groups_train = groups[train_idx]
            cv_splits = list(inner_cv.split(X_train, y_train, groups_train))

            grid_search = GridSearchCV(
                RandomForestRegressor(),
                param_grid,
                scoring=scoring,
                cv=cv_splits,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            params = grid_search.best_params_
            self.best_params = params
            print(f"  Best params final model: {params}")
        # ── AKHIR BLOK TAMBAHAN ─────────────────────────────────────

        if params is None:
            params = self.config.get('model.params', {})

        self.model = self.create_model(params)
        self.model.fit(X_train, y_train)
        self.best_params = params

        return (
            self.model,
            X_train,
            X_test,
            y_train,
            y_test,
            train_idx,
            test_idx
        )
    
    def evaluate_model(
        self,
        model: RandomForestRegressor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        compute_ci: bool = True
    ) -> Dict:
        """
        Evaluate trained model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            compute_ci: Whether to compute confidence intervals
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        # Compute confidence intervals if requested
        if compute_ci and self.config.get('evaluation.confidence_interval.enabled', True):
            n_iterations = self.config.get('evaluation.confidence_interval.n_iterations', 1000)
            random_state = self.config.get('random_state', 42)
            
            mae_ci = self._bootstrap_ci(
                y_test, y_pred, mean_absolute_error, 
                n_iterations, random_state
            )
            mse_ci = self._bootstrap_ci(
                y_test, y_pred, mean_squared_error, 
                n_iterations, random_state
            )
            
            results['mae_ci'] = mae_ci
            results['mse_ci'] = mse_ci
        
        return results
    
    def _get_param_grid(self) -> Dict:
        """Get hyperparameter grid from config."""
        if not self.config.get('model.hyperparameter_search.enabled', True):
            return {}
        
        param_grid = self.config.get('model.hyperparameter_search.param_grid', {})
        
        # Add random_state and n_jobs if not in grid
        if 'random_state' not in param_grid:
            param_grid['random_state'] = [self.config.get('random_state', 42)]
        if 'n_jobs' not in param_grid:
            param_grid['n_jobs'] = [-1]
        
        return param_grid
    
    @staticmethod
    def _bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        n_iterations: int = 1000,
        random_state: int = 42
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_fn: Metric function (e.g., mean_absolute_error)
            n_iterations: Number of bootstrap iterations
            random_state: Random seed
            
        Returns:
            Tuple of (lower_bound, upper_bound) for 95% CI
        """
        rng = np.random.default_rng(random_state)
        n_samples = len(y_true)
        metric_values = []
        
        for _ in range(n_iterations):
            # Resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            metric_value = metric_fn(y_true[indices], y_pred[indices])
            metric_values.append(metric_value)
        
        # Compute 95% CI
        lower = np.percentile(metric_values, 2.5)
        upper = np.percentile(metric_values, 97.5)
        
        return (lower, upper)
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get MDI (Mean Decrease Impurity) feature importance.
        Dihitung dari struktur pohon saat training.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_

        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'method': 'MDI'
        })

        df_importance = df_importance.sort_values('importance', ascending=False)

        if top_n is not None:
            df_importance = df_importance.head(top_n)

        return df_importance


    def get_permutation_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
        n_repeats: int = 30
    ) -> pd.DataFrame:
        """
        Get Permutation Feature Importance (PFI).
    
        Mengacak nilai satu fitur pada data test, lalu mengukur 
        seberapa besar penurunan performa model. Fitur yang 
        penting akan menyebabkan penurunan besar saat diacak.

        Args:
            X_test: Test feature matrix
            y_test: Test target values
            feature_names: List of feature names
            top_n: Number of top features to return
            n_repeats: Number of times to permute each feature

        Returns:
            DataFrame with feature names, mean importance, 
            std importance, and method label
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        random_state = self.config.get('random_state', 42)

        print(f"  Computing PFI dengan {n_repeats} repeats "
            f"pada {X_test.shape[0]} sampel test...")

        pfi_result = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        df_pfi = pd.DataFrame({
            'feature': feature_names,
            'importance': pfi_result.importances_mean,
            'importance_std': pfi_result.importances_std,
            'method': 'PFI'
        })

        # Fitur dengan importance negatif atau nol berarti
        # mengacaknya tidak menurunkan performa (tidak penting)
        df_pfi = df_pfi.sort_values('importance', ascending=False)

        if top_n is not None:
            df_pfi = df_pfi.head(top_n)

        return df_pfi


    def get_combined_importance(
        self,
        df_mdi: pd.DataFrame,
        df_pfi: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Gabungkan hasil MDI dan PFI yang sudah dihitung 
        menjadi satu DataFrame untuk perbandingan.

        Args:
            df_mdi: Output dari get_feature_importance()
            df_pfi: Output dari get_permutation_importance()

        Returns:
            DataFrame gabungan dengan kolom: 
            feature, importance, importance_std, method
        """
        df_mdi = df_mdi.copy()
        df_mdi['importance_std'] = np.nan  # MDI tidak punya std

        return pd.concat([df_mdi, df_pfi], ignore_index=True)