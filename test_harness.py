# test_harness.py
"""
INTEGRATION TEST HARNESS
Run this script from your terminal: python test_harness.py
This tests both the synth_tools and bayes_tools modules end-to-end.
"""
import os
# Tell PyTensor to bypass the Apple C++ compiler
os.environ["PYTENSOR_FLAGS"] = "cxx=" 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from synth_tools import forward_sim, bootstrap, anomaly_forge
from bayes_tools import dags, models, diagnostics, inference

def run_tests():
    print("==================================================")
    print("🚀 INITIALIZING INTEGRATION TEST HARNESS")
    print("==================================================\n")

    # ---------------------------------------------------------
    # PHASE 1: GENERATE TOY DATA (Testing synth_tools)
    # ---------------------------------------------------------
    print("--- 1. DATA GENERATION (synth_tools) ---")
    
    # 1A: Forward Simulation
    df_toy = forward_sim.simulate_dag_environment(n_transactions=1000, base_fraud_rate=0.05)
    features = ['V12', 'V14']
    
    # 1B: Bootstrapping
    df_boot = bootstrap.bootstrap_minority_class(df_toy, target_samples=200)
    
    # 1C: KDE Anomaly Forging
    df_forged = anomaly_forge.forge_kde_anomalies(df_toy, features=features, n_samples=50)


    # ---------------------------------------------------------
    # PHASE 2: CAUSAL EDA (Testing dags.py)
    # ---------------------------------------------------------
    print("\n--- 2. CAUSAL DAGs & VIF (bayes_tools/dags.py) ---")
    
    try:
        # Note: Depending on your Graphviz installation, this might render a PDF or just build the object
        dags.visualize_causal_graph(cause_node="Fraud", effect_nodes=features)
        print("DAG generation passed.")
    except Exception as e:
        print(f"DAG visualization skipped or failed (Graphviz issue?): {e}")
        
    vif_df = dags.assess_multicollinearity(df_toy, features=features)


    # ---------------------------------------------------------
    # PHASE 3: PYMC MODELING (Testing models.py)
    # ---------------------------------------------------------
    print("\n--- 3. BAYESIAN MODELING (bayes_tools/models.py) ---")
    
    # Standardize inputs (toy data is roughly standardized already)
    X_toy = df_toy[features]
    y_toy = df_toy['Class']

    # Build the architecture
    toy_model = models.build_logistic_model(X_toy, y_toy)

    # Sample (using very low numbers just to test if the engine runs without crashing)
    # This should take ~10-20 seconds max.
    toy_trace = models.sample_model(toy_model, draws=200, tune=200, chains=2)


    # ---------------------------------------------------------
    # PHASE 4: DIAGNOSTICS & INFERENCE (Testing diagnostics.py & inference.py)
    # ---------------------------------------------------------
    print("\n--- 4. DIAGNOSTICS (bayes_tools/diagnostics.py) ---")
    summary = diagnostics.print_summary(toy_trace)
    print("\n[MCMC Summary Report]")
    print(summary.to_string()) 

    print("\n--- 5. INFERENCE & DECISION THEORY (bayes_tools/inference.py) ---")
    
    # Test Posterior Predictive Check
    ppc_trace = inference.run_posterior_predictive(toy_model, toy_trace)

    # Test Decision Theory logic (Mocking probabilities to isolate the math test)
    mock_probabilities = np.random.uniform(0, 1, size=len(y_toy))
    
    print("\n(A plot will appear. Close the plot window to finish the test script!)")
    optimal_thresh = inference.calculate_optimal_threshold(
        probabilities=mock_probabilities, 
        y_true=y_toy.values, 
        cost_fp=1, 
        cost_fn=10
    )

    print("\n==================================================")
    print("🎉 ALL MODULES EXECUTED SUCCESSFULLY! 🎉")
    print("==================================================")


if __name__ == "__main__":
    # Ensure matplotlib plots dynamically but safely
    plt.ion() 
    run_tests()
    plt.ioff()
    plt.show() # Keeps the final decision theory plot open until you close it