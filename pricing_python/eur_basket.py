import numpy as np
from .diffusion import BlackScholesProcessMulti


def eur_basket_mc(model: BlackScholesProcessMulti,
                  n_path, init_val, K, riskfree,
                  maturity, payoff_fun, random_seed=None):

    S_T, S_T_anti = model.simulate_maturity(n_path, init_val, riskfree, maturity, random_seed=random_seed)

    payoff = np.exp(-riskfree * maturity) * payoff_fun(S_T, K)
    payoff_anti = np.exp(-riskfree * maturity) * payoff_fun(S_T_anti, K)
    payoff_avg = (payoff + payoff_anti) / 2
    est_price = np.mean(payoff_avg)
    stdev = np.std(payoff_avg) / np.sqrt(n_path)
    err = 1.96 * stdev
    return est_price, stdev, err
