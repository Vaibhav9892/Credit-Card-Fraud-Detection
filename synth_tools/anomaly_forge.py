# synth_tools/anomaly_forge.py
'''
This module contains functions for forging completely synthetic anomalies using Kernel Density Estimation (KDE).
The idea is to mathematically map the "mountain" of existing fraud in the feature space
and then sample new points from that mountain to create novel, never-before-seen synthetic frauds.
'''
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def forge_kde_anomalies(df: pd.DataFrame, features: list, target_col: str = 'Class', 
                        n_samples: int = 1000) -> pd.DataFrame:
    """
    Uses Kernel Density Estimation (KDE) to map the continuous probability space of 
    existing fraud, and generates completely synthetic, novel anomalies.

    Args:
        df (pd.DataFrame): The original dataset.
        features (list): The features to base the KDE on (e.g., ['V12', 'V14', 'V17']).
        target_col (str): The target column to filter for anomalies.
        n_samples (int): How many novel synthetic frauds to generate.

    Returns:
        pd.DataFrame: A dataset containing ONLY the newly forged synthetic anomalies.
    """
    print(f"Forging {n_samples} novel synthetic anomalies using KDE on {features}...")
    
    # Isolate only the actual fraud data
    fraud_data = df[df[target_col] == 1][features].dropna()
    
    # Fit the Gaussian KDE (This builds the mathematical "mountain")
    # Note: gaussian_kde expects data in shape (features, samples)
    kde = gaussian_kde(fraud_data.T)
    
    # Sample new points from the mountain
    synthetic_points = kde.resample(n_samples).T
    
    # Format into a clean DataFrame
    synthetic_df = pd.DataFrame(synthetic_points, columns=features)
    synthetic_df[target_col] = 1  # Label them all as fraud
    
    print("Synthetic anomaly generation complete!")
    return synthetic_df