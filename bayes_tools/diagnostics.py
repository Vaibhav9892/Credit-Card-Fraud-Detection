# bayes_tools/diagnostics.py
'''
This module provides diagnostic tools for Bayesian models, including functions to
summarize MCMC convergence statistics, plot trace and posterior densities, and visualize
forest plots of parameter intervals. These tools help assess the health and reliability of the sampled posterior distributions.
'''
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

def print_summary(trace: az.InferenceData, var_names: list = None) -> pd.DataFrame:
    """
    Generates a statistical summary of the MCMC trace, including R-hat and ESS.

    Args:
        trace (az.InferenceData): The sampled posterior trace.
        var_names (list): Specific variables to summarize (default is all).

    Returns:
        pd.DataFrame: A table containing mean, sd, HDI intervals, ESS, and R-hat.
    """
    print("Generating MCMC Convergence Summary...")
    print("Health Check: 'r_hat' should be strictly < 1.01. 'ess_bulk' should be > 400.")
    summary_df = az.summary(trace, var_names=var_names)
    return summary_df


def plot_mcmc_health(trace: az.InferenceData, var_names: list = None) -> None:
    """
    Plots the trace (caterpillar plots) and posterior density to visually assess 
    chain mixing and convergence.

    Args:
        trace (az.InferenceData): The sampled posterior trace.
        var_names (list): Specific variables to plot (default is all).
    """
    print("Plotting MCMC Trace and Posterior Densities...")
    # Traceplot: Left side shows density, right side shows the sampling chain history
    az.plot_trace(trace, var_names=var_names, figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def plot_forest_intervals(trace: az.InferenceData, var_names: list = None) -> None:
    """
    Generates a forest plot showing the 94% Highest Density Interval (HDI) 
    for the model parameters. Useful for seeing which features cross zero.

    Args:
        trace (az.InferenceData): The sampled posterior trace.
        var_names (list): Specific variables to plot.
    """
    print("Plotting Forest HDI Intervals...")
    az.plot_forest(trace, var_names=var_names, combined=True, hdi_prob=0.94, figsize=(10, 6))
    # Add a vertical line at 0. If a parameter's interval crosses 0, its effect is highly uncertain.
    plt.axvline(0, color='red', linestyle='--', alpha=0.6)
    plt.title("94% Highest Density Intervals (HDI)")
    plt.show()