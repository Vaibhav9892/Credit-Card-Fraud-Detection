# bayes_tools/inference.py

import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run_posterior_predictive(model: pm.Model, trace: az.InferenceData) -> az.InferenceData:
    """
    Executes a Posterior Predictive Check (PPC). The model uses its learned 
    distributions to simulate/generate "fake" data to compare against reality.

    Args:
        model (pm.Model): The compiled PyMC model.
        trace (az.InferenceData): The fitted posterior trace.

    Returns:
        az.InferenceData: The trace object updated with 'posterior_predictive' samples.
    """
    print("Generating Synthetic Data via Posterior Predictive Check (PPC)...")
    with model:
        ppc_trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True, progressbar=True)
    return ppc_trace


def calculate_optimal_threshold(probabilities: np.ndarray, y_true: np.ndarray, 
                                cost_fp: float = 1.0, cost_fn: float = 10.0) -> float:
    """
    Applies Bayesian Decision Theory to find the optimal decision threshold.
    Instead of assuming 0.5 is the cutoff, it weighs the financial cost of a 
    False Positive (annoying a customer) vs a False Negative (letting a thief escape).

    Args:
        probabilities (np.ndarray): The continuous probability outputs from the model.
        y_true (np.ndarray): The actual ground truth binary labels.
        cost_fp (float): Financial or arbitrary cost of a False Positive.
        cost_fn (float): Financial or arbitrary cost of a False Negative.

    Returns:
        float: The mathematically optimal probability threshold (0.0 to 1.0).
    """
    print(f"Running Decision Theory Analysis (Cost FP: {cost_fp}, Cost FN: {cost_fn})...")
    
    thresholds = np.linspace(0.01, 0.99, 100)
    expected_costs = []
    
    for thresh in thresholds:
        # Convert continuous probabilities to binary predictions based on the threshold
        predictions = (probabilities >= thresh).astype(int)
        
        # Calculate rates
        false_positives = np.sum((predictions == 1) & (y_true == 0))
        false_negatives = np.sum((predictions == 0) & (y_true == 1))
        
        # Apply cost function
        total_cost = (false_positives * cost_fp) + (false_negatives * cost_fn)
        expected_costs.append(total_cost)
    
    optimal_idx = np.argmin(expected_costs)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plotting the Cost Curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, expected_costs, color='purple', lw=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Cutoff: {optimal_threshold:.2f}')
    plt.title("Expected Cost Curve")
    plt.xlabel("Decision Threshold (Probability)")
    plt.ylabel("Total Expected Cost")
    plt.legend()
    plt.show()
    
    print(f"Optimal Threshold Found: {optimal_threshold:.3f}")
    return optimal_threshold