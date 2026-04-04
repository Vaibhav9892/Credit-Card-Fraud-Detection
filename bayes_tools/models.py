# bayes_tools/models.py
'''
This module defines the core functions for building and sampling Bayesian Logistic Regression models using PyMC.
The `build_logistic_model` function constructs the model architecture based on the input features and specified priors,
while the `sample_model` function executes MCMC sampling to obtain the posterior distributions of
the model parameters. These functions are designed to be flexible and reusable for various datasets and modeling scenarios.
'''
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

def build_logistic_model(X: pd.DataFrame, y: pd.Series, 
                         prior_mu: float = 0.0, prior_sigma: float = 1.0) -> pm.Model:
    """
    Constructs a Bayesian Logistic Regression model dynamically based on input features.

    Args:
        X (pd.DataFrame): The predictor variables (must be scaled/standardized).
        y (pd.Series): The binary target variable (0 or 1).
        prior_mu (float): The mean for the Normal prior on coefficients. Default is 0.0.
        prior_sigma (float): The standard deviation for the Normal prior. Default is 1.0 
                             (represents strong skepticism of large effects).

    Returns:
        pm.Model: An un-sampled PyMC model object.
    """
    print(f"Building PyMC Logistic Model with {X.shape[1]} features...")
    
    feature_names = X.columns.tolist()
    
    with pm.Model() as bayes_model:
        # 1. Define Priors (The Baseline Beliefs)
        # Alpha is the intercept
        alpha = pm.Normal('alpha', mu=prior_mu, sigma=prior_sigma)
        
        # Betas are the weights for each feature
        betas = pm.Normal('betas', mu=prior_mu, sigma=prior_sigma, shape=len(feature_names))
        
        # 2. The Linear Deterministic Function (Matrix Multiplication)
        # Equivalent to: alpha + (beta_1 * X_1) + (beta_2 * X_2) ...
        mu = alpha + pm.math.dot(X.values, betas)
        
        # 3. The Link Function (Logit to Probability)
        # Converts the linear output into a strict 0.0 to 1.0 probability
        p = pm.math.invlogit(mu)
        
        # 4. The Likelihood (The Data)
        # Bernoulli distribution because the target is binary (Fraud vs Normal)
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y.values)
        
    print("Model architecture compiled successfully.")
    return bayes_model


def sample_model(model: pm.Model, draws: int = 1000, tune: int = 1000, 
                 chains: int = 4, target_accept: float = 0.9) -> az.InferenceData:
    """
    Executes the Markov Chain Monte Carlo (MCMC) sampling using the NUTS sampler.

    Args:
        model (pm.Model): The compiled PyMC model.
        draws (int): Number of posterior samples to draw per chain.
        tune (int): Number of tuning/burn-in steps per chain.
        chains (int): Number of independent Markov chains to run.
        target_accept (float): Step size tuning parameter for the NUTS sampler.

    Returns:
        az.InferenceData: The sampled trace containing the posterior distributions.
    """
    print(f"Starting MCMC Sampling: {chains} chains, {tune} tune, {draws} draws...")
    with model:
        # We use a higher target_accept (0.9) to prevent divergences in complex geometries
        trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                          target_accept=target_accept, return_inferencedata=True, 
                          progressbar=True)
    return trace