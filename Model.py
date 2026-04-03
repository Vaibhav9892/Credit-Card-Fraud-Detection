import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

N = 284807
F_obs = 492

def main():
    with pm.Model() as fraud_model:
        theta = pm.Beta("theta", alpha=1, beta=500)
        F = pm.Binomial("F", n=N, p=theta, observed=F_obs)

        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=2,              # use 1 if you still get Windows multiprocessing issues
            target_accept=0.95,
            random_seed=42
        )

    print(az.summary(trace, var_names=["theta"]))
    az.plot_posterior(trace, var_names=["theta"])
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    main()