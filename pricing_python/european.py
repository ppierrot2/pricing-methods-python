import numpy as np
from numpy.random import standard_normal, seed, uniform, randint
from scipy.stats import norm
from .diffusion import HestonProcess, BlackScholesProcess, MertonJumpProcess
from scipy.linalg import solve_triangular, lu


def eur_bs_analytical_price(S0, K, riskfree, sigma, maturity, call=True):
    """BS Closed formula for European call option"""
    d1 = (np.log(S0 / K) + (riskfree + 1 / 2 * sigma ** 2) * maturity) / sigma / np.sqrt(maturity)
    d2 = (np.log(S0 / K) + (riskfree - 1 / 2 * sigma ** 2) * maturity) / sigma / np.sqrt(maturity)
    if call:
        return S0 * norm.cdf(d1) - K * np.exp(-riskfree * maturity) * norm.cdf(d2)
    else:
        return K * np.exp(-riskfree * maturity) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def eur_merton_analytical_price(model: MertonJumpProcess, S0, K, riskfree, maturity, call=True):
    npv = 0
    k = np.exp(model.m_jump + model.v_jump ** 2 / 2) - 1
    for i in range(50):
        coef = np.exp(-model.lambda_jump) * model.lambda_jump ** i * maturity ** i / np.math.factorial(i)
        npv += coef * eur_bs_analytical_price(
            S0=S0 * np.exp(i * model.m_jump + i * model.v_jump ** 2 / 2 - model.lambda_jump * k * maturity),
            K=K,
            riskfree=riskfree,
            sigma=np.sqrt(model.sigma ** 2 + i * model.v_jump ** 2 / maturity),
            maturity=maturity,
            call=call)
    return npv


def eur_mc_price(model: (BlackScholesProcess, HestonProcess, MertonJumpProcess),
                        S0: float, K: float,
                        riskfree: float,
                        maturity: float, call=True,
                        n_path=10000, n_step=5000):

    if isinstance(model, BlackScholesProcess):
        S, S_anti = model.simulate(n_path=n_path, n_step=n_step, init_val=S0, riskfree=riskfree,
                                   maturity=maturity)
    elif isinstance(model, HestonProcess):
        S, S_anti = model.simulate(n_path=n_path, n_step=n_step, init_val=S0, riskfree=riskfree,
                                   maturity=maturity, proba='risk-neutral')
    elif isinstance(model, MertonJumpProcess):
        S = model.simulate(n_path=n_path, n_step=n_step, init_val=S0, riskfree=riskfree, maturity=maturity)
        S_anti = None
    else:
        raise NotImplementedError('MC not implemented for this model')

    payoffs = S[:, -1] - K if call else K - S[:, -1]
    payoffs[payoffs < 0] = 0
    if S_anti is not None:
        payoffs_anti = S_anti[:, -1] - K if call else K - S_anti[:, -1]
        payoffs_anti[payoffs_anti < 0] = 0
        payoffs_avg = (payoffs + payoffs_anti) / 2
    else:
        payoffs_avg = payoffs
    npv = np.exp(-riskfree * maturity) * np.mean(payoffs_avg)
    stdev = np.exp(-riskfree * maturity) * np.std(payoffs_avg) / np.sqrt(n_path)
    return npv, stdev


def eur_heston_price(model: HestonProcess, S0, K, riskfree,
                     maturity, call=True, n_path=10000, n_step=5000):
    S_rn, _ = model.simulate(n_path=n_path, n_step=n_step, init_val=S0, riskfree=riskfree,
                             maturity=maturity, proba='risk-neutral')
    S_fn, _ = model.simulate(n_path=n_path, n_step=n_step, init_val=S0, riskfree=riskfree,
                             maturity=maturity, proba='forward-neutral')
    I_rn = S_rn > K if call else S_rn < K
    I_fn = S_fn > K if call else S_rn < K
    npv = np.exp(-riskfree*maturity)*(-1)**call*(K*I_rn - S0*I_fn)
    return npv


def eur_fourier_price(model: (BlackScholesProcess, HestonProcess),
                      S0, K, riskfree, maturity, call=True, n_coef=10):

    # integration bonds
    a, b = model.density_integration_bounds(S0, riskfree, maturity)

    # coefficient factors
    coef = []
    if call:
        coef_0 = K * (np.log(K) - b - 1) + np.exp(b)
        coef.append(coef_0)
        for i in range(1, n_coef):
            coef.append(
                (np.exp(b) - K * (b - a) * np.sin(i * np.pi / (b - a) * (a - np.log(K))) / (i * np.pi)  - K * np.cos(
                    i * np.pi / (b - a) * (a - np.log(K))) ) / (1 + i ** 2 * np.pi ** 2 / (b - a) ** 2)

            )
    else:
        coef_0 = K * (np.log(K) - a - 1) + np.exp(a)
        coef.append(coef_0)
        for i in range(1, n_coef):
            coef.append(
                (np.exp(a) - K * (b - a) * np.sin(i * np.pi / (b - a) * (a - np.log(K))) / (i * np.pi) - K * np.cos(
                    i * np.pi / (b - a) * (a - np.log(K)))) \
                / (1 + i ** 2 * np.pi ** 2 / (b - a) ** 2)
            )

    discount_factor = np.exp(-riskfree * maturity)
    dens_decompo = coef[0]
    for i in range(1, n_coef):
        dens_decompo += 2 * model.characteristic_fun(i * np.pi / (b - a), init_val=S0,
                                                     riskfree=riskfree, maturity=maturity) * \
                        np.exp(-1j * i * np.pi * a / (b - a)) * coef[i]

    dens_decompo = np.real(dens_decompo) / (b-a)
    npv = discount_factor * dens_decompo

    return npv


def eur_fd_price(model: BlackScholesProcess, S0, K, riskfree,
                 maturity, call=True, method='implicit',
                 x_step=1000, time_step=1000, S_max=None,
                 theta=0.5):

    # discretize
    dt = maturity / (time_step - 1)
    if S_max is None:
        S_max = 4 * K
    # if method == 'explicit':
    #     dx = model.sigma*np.sqrt(3*dt)  # enforce stability conditions
    #     x_step = int(S_max / dx) + 1
    #     S = np.array([i * dx for i in range(x_step)])
    #     grid = np.zeros((x_step, time_step))
    # else:
    S, dx = np.linspace(start=0, stop=S_max, num=x_step, retstep=True)
    grid = np.zeros((x_step, time_step))

    # setup boundary conditions
    if call:
        grid[:, -1] = np.maximum(S - K, 0)
        grid[-1, :] = [(x_step - 1) * dx - K * np.exp(- (maturity - dt * n) * riskfree)
                       for n in range(time_step)]
    else:
        grid[:, -1] = np.maximum(K - S, 0)
        grid[-1, :] = 0

    # construct tridiagonal matrix coefficients
    a = (model.sigma * S[1:-1] / dx) ** 2 / 2 + riskfree * S[1:-1] / dx / 2
    b = - riskfree - (model.sigma * S[1:-1] / dx) ** 2
    c = (model.sigma * S[1:-1] / dx) ** 2 / 2 - riskfree * S[1:-1] / dx / 2

    if method == 'explicit':
        M = np.eye(x_step - 2) + dt * (np.diag(a[:-1], 1) + np.diag(b, 0) + np.diag(c[1:], -1))
        # solve the system backward-in-time
        for n in range(time_step - 2, -1, -1):
            grid[1:-1, n] = np.dot(M, grid[1:-1, n + 1])
            grid[-2, n] += a[-1] * dt * grid[-1, n + 1]  # complete the last differential with boundary val

    elif method == 'implicit':
        M = np.eye(x_step - 2) - dt * (np.diag(a[:-1], 1) + np.diag(b, 0) + np.diag(c[1:], -1))
        _, L, U = lu(M)
        # solve the system backward-in-time
        for n in range(time_step - 2, -1, -1):
            temp = grid[1:-1, n + 1].copy()
            temp[-1] += a[-1] * dt * grid[-1, n]
            x = solve_triangular(L, temp, lower=True)
            grid[1:-1, n] = solve_triangular(U, x)

    elif method == 'cn':
        M_1 = np.eye(x_step - 2) + (1 - theta) * dt * (np.diag(a[:-1], 1) + np.diag(b, 0) + np.diag(c[1:], -1))
        M_2 = np.eye(x_step - 2) - theta * dt * (np.diag(a[:-1], 1) + np.diag(b, 0) + np.diag(c[1:], -1))
        _, L, U = lu(M_2)
        for n in range(time_step - 2, -1, -1):
            temp = np.dot(M_1, grid[1:-1, n + 1])
            temp[-1] += theta * a[-1] * dt * grid[-1, n] + (1 - theta) * a[-1] * dt * grid[-1, n + 1]
            x = solve_triangular(L, temp, lower=True)
            grid[1:-1, n] = solve_triangular(U, x)

    else:
        raise ValueError('invalid method')

    npv = np.interp([S0], S, grid[:, 0])

    return npv, S, grid
