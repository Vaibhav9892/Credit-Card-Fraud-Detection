import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import multiprocessing as mp
from causalgraphicalmodels import CausalGraphicalModel

N = 284807
F_obs = 492

def show_dag():
    fraud_dag = CausalGraphicalModel(
        nodes=["alpha", "beta", "theta", "N", "F"],
        edges=[
            ("alpha", "theta"),
            ("beta", "theta"),
            ("theta", "F"),
            ("N", "F")
        ]
    )

    g = fraud_dag.draw()
    g.render(filename="fraud_dag", format="png", view=True)
    print("DAG saved as fraud_dag.png and opened.")

def main():
    show_dag()

    with pm.Model() as fraud_model:
        theta = pm.Beta("theta", alpha=1, beta=500)
        F = pm.Binomial("F", n=N, p=theta, observed=F_obs)

        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=2,
            target_accept=0.95,
            random_seed=42
        )

    print("\nPosterior summary for theta:")
    print(az.summary(trace, var_names=["theta"]))

    az.plot_posterior(trace, var_names=["theta"])
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    main()