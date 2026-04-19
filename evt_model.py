"""
Extreme Value Theory (Peaks-Over-Threshold) model for tail risk estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple

class EVTAnalyzer:
    """
    Fits Generalized Pareto Distribution (GPD) to extreme losses using POT method.
    """
    
    def __init__(self, threshold_quantile: float = 0.90, 
                 min_obs: int = 100,
                 ewma_halflife: int = 21):
        self.threshold_quantile = threshold_quantile
        self.min_obs = min_obs
        self.ewma_halflife = ewma_halflife
        
    def fit_gpd(self, losses: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Fit GPD to exceedances above threshold.
        Returns: (shape ξ, scale σ, threshold u) or None if insufficient data.
        """
        if len(losses) < self.min_obs:
            return None
            
        # Determine threshold as specified quantile of losses
        u = np.quantile(losses, self.threshold_quantile)
        exceedances = losses[losses > u] - u
        
        if len(exceedances) < 10:
            return None
            
        # Fit GPD using MLE
        try:
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            return shape, scale, u
        except:
            return None
    
    def calculate_var_es(self, losses: np.ndarray, 
                         shape: float, scale: float, u: float,
                         confidence: float = 0.99) -> Tuple[float, float]:
        """
        Compute Value-at-Risk and Expected Shortfall from fitted GPD.
        Based on McNeil & Frey (2000) conditional EVT.
        """
        n = len(losses)
        n_u = np.sum(losses > u)
        
        # Unconditional VaR
        if abs(shape) < 1e-8:
            var = u + scale * np.log((n / n_u) * (1 - confidence))
        else:
            var = u + (scale / shape) * (((n / n_u) * (1 - confidence)) ** (-shape) - 1)
        
        # Expected Shortfall
        if shape < 1:
            es = (var / (1 - shape)) + (scale - shape * u) / (1 - shape)
        else:
            es = np.inf
            
        return var, es
    
    def analyze_series(self, returns: pd.Series) -> pd.DataFrame:
        """
        Rolling window EVT analysis on a return series.
        Returns DataFrame with daily metrics: tail_shape, var_99, es_99, tail_warning.
        """
        # Work with negative returns (losses)
        losses = -returns.values
        dates = returns.index
        
        results = []
        for i in range(len(dates)):
            if i < self.min_obs:
                results.append({
                    'Date': dates[i],
                    'tail_shape': np.nan,
                    'var_99': np.nan,
                    'es_99': np.nan,
                    'tail_warning': 0
                })
                continue
                
            # Rolling window of losses up to current date
            window_losses = losses[max(0, i - 252 + 1):i+1]
            gpd_fit = self.fit_gpd(window_losses)
            
            if gpd_fit is None:
                results.append({
                    'Date': dates[i],
                    'tail_shape': np.nan,
                    'var_99': np.nan,
                    'es_99': np.nan,
                    'tail_warning': 0
                })
                continue
                
            shape, scale, u = gpd_fit
            var_99, es_99 = self.calculate_var_es(window_losses, shape, scale, u)
            
            results.append({
                'Date': dates[i],
                'tail_shape': shape,
                'var_99': var_99,
                'es_99': es_99,
                'tail_warning': 0
            })
        
        df = pd.DataFrame(results)
        
        # Apply EWMA smoothing to tail_shape
        df['tail_shape_smooth'] = df['tail_shape'].ewm(halflife=self.ewma_halflife, min_periods=1).mean()
        
        # Generate warning flag based on smoothed tail shape
        df['tail_warning'] = (df['tail_shape_smooth'] > 0.3).astype(int)
        
        return df[['Date', 'tail_shape', 'tail_shape_smooth', 'var_99', 'es_99', 'tail_warning']]
