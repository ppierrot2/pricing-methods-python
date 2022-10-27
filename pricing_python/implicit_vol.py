import numpy as np
from scipy.stats import norm


def bs_vol_imp_call(S, K, T, V, r, sigma_init, tol=1e-7):

    d1 = (np.log(S / K) + (r - 0.5 * sigma_init ** 2) * T) / (sigma_init * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma_init ** 2) * T) / (sigma_init * np.sqrt(T))
    fx = S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0) - V
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    x0 = sigma_init
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tol:
        xold = xnew
        xnew = (xnew - fx - V) / vega

    return abs(xnew)
