# bayes_tools/dags.py
'''
This module provides tools for constructing and visualizing Directed Acyclic Graphs (DAGs)
for causal inference, as well as assessing multicollinearity among features using Variance Inflation Factor (VIF).
It also includes a function to generate pairwise scatter plots with KDE diagonals for exploratory data analysis.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

def visualize_causal_graph(cause_node: str, effect_nodes: list) -> None:
    """
    Standard Modern Method: Generates a DAG using Graphviz directly.
    Bypasses the broken 'causalgraphicalmodels' package.
    """
    if not HAS_GRAPHVIZ:
        print("Graphviz not found. Install with: brew install graphviz && pip install graphviz")
        return None

    print(f"Generating Modern DAG: '{cause_node}' -> {effect_nodes}")
    
    # Create the Digraph object
    dot = graphviz.Digraph(comment='Causal Model')
    dot.attr(rankdir='LR') # Left to Right layout
    
    # Add the nodes
    dot.node('C', cause_node, shape='ellipse', color='tomato', style='bold')
    for i, effect in enumerate(effect_nodes):
        node_id = f'E{i}'
        dot.node(node_id, effect, shape='box')
        dot.edge('C', node_id) # Draw the arrow
        
    return dot

# ... keep assess_multicollinearity and plot_pairwise_distributions as they were ...


def assess_multicollinearity(df: pd.DataFrame, features: list, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) to detect multicollinearity
    among a set of independent variables.

    Args:
        df (pd.DataFrame): The dataset containing the features.
        features (list): A list of column names to evaluate.
        threshold (float): The VIF score above which a feature is considered 
                           highly collinear. Default is 5.0.

    Returns:
        pd.DataFrame: A DataFrame containing the features and their respective VIF scores,
                      sorted in descending order.
    """
    print(f"Calculating Variance Inflation Factor (VIF) for {len(features)} features...")
    
    # Isolate features and add the required constant for statsmodels
    X = df[features].dropna()
    X_with_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF_Score"] = [variance_inflation_factor(X_with_const.values, i) 
                             for i in range(X_with_const.shape[1])]
    
    # Remove the constant from the output and sort
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values(by="VIF_Score", ascending=False).reset_index(drop=True)
    
    # Format the diagnostic output
    for _, row in vif_data.iterrows():
        feature = row['Feature']
        score = row['VIF_Score']
        
        if score > (threshold * 2):
            status = "CRITICAL: Severe collinearity detected."
        elif score > threshold:
            status = "WARNING: Moderate collinearity detected."
        else:
            status = "PASS: Independent feature."
            
        print(f"{feature:<10} | VIF Score: {score:>6.2f} | {status}")
        
    return vif_data


def plot_pairwise_distributions(df: pd.DataFrame, features: list, target_col: str = 'Class', sample_size: int = 3000) -> None:
    """
    Generates a pairwise scatter plot matrix with KDE diagonals to visualize 
    feature separation and interaction with respect to a target variable.

    Args:
        df (pd.DataFrame): The dataset.
        features (list): The list of continuous features to plot.
        target_col (str): The categorical target column used for hue separation.
        sample_size (int): The number of majority-class samples to draw to prevent 
                           memory overflow. All minority-class samples are retained.
    """
    print(f"Generating pairwise distributions. Subsampling majority class to N={sample_size}...")
    
    # Stratified sampling for visualization performance
    majority_class = df[df[target_col] == 0].sample(n=sample_size, random_state=42)
    minority_class = df[df[target_col] == 1]
    plot_df = pd.concat([majority_class, minority_class])
    
    cols_to_plot = features + [target_col]
    
    sns.pairplot(plot_df[cols_to_plot], hue=target_col, 
                 palette={0: 'steelblue', 1: 'tomato'}, 
                 plot_kws={'alpha': 0.5},
                 diag_kind='kde')
    plt.show()