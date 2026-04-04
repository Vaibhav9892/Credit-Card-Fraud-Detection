# synth_tools/bootstrap.py

import pandas as pd
from sklearn.utils import resample

def bootstrap_minority_class(df: pd.DataFrame, target_col: str = 'Class', 
                             target_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Upsamples the minority class (Fraud) using traditional bootstrapping (sampling with replacement)
    to create a larger, robust dataset for stress-testing models.

    Args:
        df (pd.DataFrame): The original dataset.
        target_col (str): The column denoting the class.
        target_samples (int): The number of minority samples you want to end up with.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with the upsampled minority class merged with the majority class.
    """
    print(f"Bootstrapping minority class to {target_samples} samples...")
    
    # Separate the classes
    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]
    
    # Resample the minority class with replacement
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=target_samples, 
                                     random_state=random_state)
    
    # Combine back together
    df_bootstrapped = pd.concat([df_majority, df_minority_upsampled])
    
    print(f"New dataset shape: {df_bootstrapped.shape}. "
          f"Fraud ratio: {df_bootstrapped[target_col].mean()*100:.2f}%")
    
    return df_bootstrapped