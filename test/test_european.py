from pricing_python.european import *
from pricing_python.diffusion import BlackScholesProcess


def test_european_fd_price():
    sigma = 0.2
    riskfree = 0.05
    maturity = 1
    S0 = 100
    K = 110
    model = BlackScholesProcess(sigma=sigma)
    explicit_pde_price = eur_fd_price(model, S0, K, riskfree,
                                      maturity, call=True, method='explicit',
                                      x_step=1000, time_step=100)
    implicit_pde_price = eur_fd_price(model, S0, K, riskfree,
                                      maturity, call=True, method='implicit',
                                      x_step=1000, time_step=1000)
    cn_pde_price = eur_fd_price(model, S0, K, riskfree,
                                maturity, call=True, method='cn',
                                x_step=1000, time_step=1000)

    bs_analytical_price = eur_bs_analytical_price(S0, K, riskfree, sigma, maturity, call=True)

