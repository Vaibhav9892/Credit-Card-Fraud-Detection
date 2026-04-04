# synth_tools/forward_sim.py

import pandas as pd
import numpy as np

def simulate_dag_environment(n_transactions: int = 10000, 
                             base_fraud_rate: float = 0.01) -> pd.DataFrame:
    """
    Forward-simulates an artificial banking environment based on our Causal DAG.
    Useful for testing if the Bayesian model can recover the true causal weights.

    Args:
        n_transactions (int): The total number of transactions to simulate.
        base_fraud_rate (float): The probability any given transaction is fraud.

    Returns:
        pd.DataFrame: A fully simulated dataset with known mathematical truths.
    """
    print(f"Simulating a causal DAG environment with {n_transactions} transactions...")
    np.random.seed(42)
    
    # 1. Simulate the unobserved cause (Fraud) using a binomial distribution
    is_fraud = np.random.binomial(n=1, p=base_fraud_rate, size=n_transactions)
    
    # 2. Forward simulate the effects based on arbitrary true weights we decide
    # Let's pretend: V14 drops by 5 if fraud, V12 drops by 3 if fraud
    
    # Normal transactions center around 0. Frauds center around -5.
    true_weight_v14 = -5.0
    v14 = np.random.normal(loc=is_fraud * true_weight_v14, scale=1.0, size=n_transactions)
    
    # Normal transactions center around 0. Frauds center around -3.
    true_weight_v12 = -3.0
    v12 = np.random.normal(loc=is_fraud * true_weight_v12, scale=1.5, size=n_transactions)
    
    # Compile into a DataFrame
    sim_df = pd.DataFrame({
        'Class': is_fraud,
        'V14': v14,
        'V12': v12
    })
    
    print(f"Simulation complete. True V14 causal weight: {true_weight_v14}")
    print(f"Simulation complete. True V12 causal weight: {true_weight_v12}")
    
    return sim_df