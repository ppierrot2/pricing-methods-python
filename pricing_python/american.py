import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from .diffusion import BlackScholesProcess, BlackScholesProcessMulti


def lsmc_price(model: (BlackScholesProcess, BlackScholesProcessMulti),
               regressor: RegressorMixin, init_val, K, payoff_fun,
               riskfree, maturity, n_path=1000, n_step=100,
               poly_features=True, R=6, **fit_params):
    """
    Pricing using least square Monte-Carlo
    """
    paths, _ = model.simulate(n_path, n_step, init_val, riskfree, maturity, random_seed=None)
    V = payoff_fun(paths[:, n_step, :], K)
    delta_t = maturity / n_step
    discount_factor = np.exp(-riskfree * delta_t)

    if poly_features:
        regressor = make_pipeline(PolynomialFeatures(degree=R), regressor)

    for t in range(n_step, -1, -1):
        regressor.fit(paths[:, t, :], V, **fit_params)
        cont_val = regressor.predict(paths[:, t, :])
        cont_val = cont_val.ravel()
        V = np.where(cont_val < payoff_fun(paths[:, t, :], K),
                     payoff_fun(paths[:, t, :], K),
                     discount_factor * V)

    mean = np.sum(V) / n_path
    var = np.sum((V - mean) ** 2) / (n_path - 1)
    stdev = np.sqrt(var / n_path)

    return mean, stdev


def fd_price():
    # ToDo
    pass
