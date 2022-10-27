import numpy as np
from scipy.stats import random_correlation
from pricing_python.diffusion import BlackScholesProcessMulti
from pricing_python.eur_basket import eur_basket_mc


def payoff_call_basket(S, K):
    G = np.maximum(np.mean(S, axis=1) - K, 0)
    return G


def test_eur_basket_mc():

    # Option parameters
    r = 0.05           # Interest rate
    K = 50             # Strike
    T = 1              # Terminal time
    S0 = 0.5           # Initial price

    sigmas = np.array([0.1, 0.25, 0.2, 0.05])   # Volatility vector
    correlations = random_correlation.rvs((.5, .8, 1.2, 1.5))
    model = BlackScholesProcessMulti(sigmas=sigmas, correlations=correlations)

    mc_price = eur_basket_mc(model, n_path=100, init_val=S0, K=K, riskfree=r,
                             maturity=T, payoff_fun=payoff_call_basket)