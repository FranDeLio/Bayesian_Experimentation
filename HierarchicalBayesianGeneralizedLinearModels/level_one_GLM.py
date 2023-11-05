import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# import theano
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
np.set_printoptions(2)

import os, sys

# sys.stderr = open(os.devnull, "w")

# PyMC 4.0 imports
import pymc as pm

# import aesara.tensor as at
# import aesara

data = pd.read_csv(pm.get_data("radon.csv"))
county_names = data.county.unique()

data["log_radon"] = data["log_radon"]  # .astype(theano.config.floatX)
county_idx, counties = pd.factorize(data.county)
coords = {"county": counties, "obs_id": np.arange(len(county_idx))}



def build_model(pm):
    with pm.Model(coords=coords) as hierarchical_model:
        # Intercepts, non-centered
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", dims="county") * sigma_a + mu_a

        # Slopes, non-centered
        mu_b = pm.Normal("mu_b", mu=0.0, sigma=2.0)
        sigma_b = pm.HalfNormal("sigma_b", 1.0)
        b = pm.Normal("b", dims="county") * sigma_b + mu_b

        eps = pm.HalfNormal("eps", 1.5)

        radon_est = a[county_idx] + b[county_idx] * data.floor.values

        radon_like = pm.Normal(
            "radon_like",
            mu=radon_est,
            sigma=eps,
            observed=data.log_radon,
            dims="obs_id",
        )

    return hierarchical_model


model_pymc4 = build_model(pm)

with model_pymc4:
    idata_pymc4 = pm.sample(
        target_accept=0.9, draws=2000, tune=1000, chains=4, cores=4, progressbar=True
    )

