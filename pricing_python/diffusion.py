import numpy as np


class BlackScholesProcess:

    def __init__(self, sigma):
        self.sigma = sigma

    def simulate(self, n_path, n_step, init_val, riskfree, maturity):
        delta_t = maturity / n_step
        paths = np.ones((n_path, n_step))
        paths_anti = np.ones((n_path, n_step))  # antithetic
        dW = np.sqrt(delta_t) * np.random.randn(n_path, n_step-1)
        paths[:, 1:] = np.cumprod(np.exp(delta_t * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * dW), axis=1)
        paths_anti[:, 1:] = np.cumprod(np.exp(delta_t * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * -dW), axis=1)
        paths = paths * init_val
        paths_anti = paths_anti * init_val
        return paths, paths_anti

    def simulate_maturity(self, n_path, init_val, riskfree, maturity):
        DW = np.sqrt(maturity)*np.random.randn(n_path)
        s_T = init_val * np.exp(maturity * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * DW)
        s_T_anti = init_val * np.exp(maturity * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * -DW)
        return s_T, s_T_anti

    def characteristic_fun(self, x, init_val, riskfree, maturity):
        m = np.log(init_val) + (riskfree - 0.5 * self.sigma ** 2) * maturity
        return np.exp(1j * x * m - 0.5 * x ** 2 * self.sigma ** 2 ** maturity)

    def density_integration_bounds(self, init_val, riskfree, maturity):
        # integration bonds (mean(log(S_T))+-10*std dev)
        a = np.log(init_val) + riskfree * maturity - 10 * self.sigma * np.sqrt(maturity)
        b = np.log(init_val) + riskfree * maturity + 10 * self.sigma * np.sqrt(maturity)
        return a, b


class BlackScholesProcessMulti:

    def __init__(self, sigmas, correlations):
        assert correlations.shape == 2*(sigmas.shape[0],)
        self.dim = sigmas.shape[0]
        self.sigmas = sigmas
        self.correlations = correlations
        self.covariances = np.dot(np.dot(np.diag(sigmas), correlations), np.diag(sigmas))
        self.M = np.linalg.cholesky(correlations)

    def simulate(self, n_path, n_step, init_val, riskfree, maturity, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        delta_t = maturity / n_step
        paths = np.zeros((n_path, n_step+1, self.dim))
        paths_anti = np.zeros((n_path, n_step+1, self.dim)) # antithetic
        self.wiener_diffs = np.zeros((n_path, n_step, self.dim))

        if not isinstance(init_val, (list, np.ndarray)):
            init_val = self.dim * [init_val]
        paths[:, 0, :] = np.tile(init_val, (n_path, 1))
        paths_anti[:, 0, :] = np.tile(init_val, (n_path, 1))

        for m in range(1, n_step+1):
            dW = np.sqrt(delta_t) * np.dot(np.random.randn(n_path, self.dim), self.M.T)
            self.wiener_diffs[:, m - 1, :] = dW
            paths[:, m, :] = paths[:, m - 1, :] * np.exp(
                delta_t * np.ones((n_path, self.dim)) * (riskfree - np.square(self.sigmas) / 2)
                + self.sigmas * dW
            )
            paths_anti[:, m, :] = paths_anti[:, m - 1, :] * np.exp(
                delta_t * np.ones((n_path, self.dim)) * (riskfree - np.square(self.sigmas) / 2)
                - self.sigmas * dW
            )
        return paths, paths_anti

    def simulate_maturity(self, n_path, init_val, riskfree, maturity, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        DW = np.sqrt(maturity) * np.dot(np.random.randn(n_path, self.dim), self.M.T)
        s_T = np.exp(maturity * np.ones((n_path, self.dim)) * (riskfree - np.square(self.sigmas) / 2) + self.sigmas * DW)
        s_T_anti = np.exp(maturity * np.ones((n_path, self.dim)) * (riskfree - np.square(self.sigmas) / 2) - self.sigmas * DW)
        s_T = s_T * init_val
        s_T_anti = s_T_anti * init_val
        return s_T, s_T_anti

    def simulate_euler(self, n_path, n_step, init_val, riskfree, maturity):
        delta_t = maturity / n_step
        paths = np.zeros((n_path, n_step+1, self.dim))
        paths_anti = np.zeros((n_path, n_step+1, self.dim)) # antithetic
        self.wiener_diffs = np.zeros((n_path, n_step, self.dim))

        if not isinstance(init_val, (list, np.ndarray)):
            init_val = self.dim * [init_val]
        paths[:, 0, :] = np.tile(init_val, (n_path, 1))
        paths_anti[:, 0, :] = np.tile(init_val, (n_path, 1))

        for m in range(1, n_step+1):
            dW = np.sqrt(delta_t) * np.dot(np.random.randn(n_path, self.dim), self.M.T)
            self.wiener_diffs[:, m-1, :] = dW
            paths[:, m, :] = paths[:, m - 1, :] * (1 + riskfree * delta_t) + \
                             paths[:, m - 1, :] * self.sigmas * dW
            paths_anti[:, m, :] = paths_anti[:, m - 1, :] * (1 + riskfree * delta_t) - \
                                  paths_anti[:, m - 1, :] * self.sigmas * dW

        return paths, paths_anti


class HestonProcess:

    def __init__(self, v_init, v, kappa, gamma, rho):
        self.v_init = v_init
        self.v = v
        self.kappa = kappa
        self.gamma = gamma
        self.rho = rho

    def simulate(self, n_path, n_step, init_val, riskfree, maturity, proba='risk-neutral'):

        delta_t = maturity / n_step
        S = np.zeros((n_path, n_step+1))
        V = np.zeros((n_path, n_step+1))
        V_anti = np.zeros((n_path, n_step))  # antithetic variable
        S[:, 0] = init_val
        V[:, 0] = self.v_init
        V_anti[:, 0] = self.v_init

        S_anti = None
        if proba == 'risk-neutral':  # anthitetic variable only applicable for risk-neutral case
            S_anti = np.zeros((n_path, n_step))
            S_anti[:, 0] = init_val

        dW_S = np.random.randn(n_path, n_step)*np.sqrt(delta_t)
        dW_V = dW_S + np.sqrt(1 - self.rho**2)*np.random.randn(n_path, n_step)*np.sqrt(delta_t)

        for i in range(1, n_step):
            if proba == 'risk-neutral':
                S[:, i] = S[:, i - 1] + riskfree * S[:, i - 1] * delta_t + \
                          np.sqrt(V[:, i - 1]) * S[:, i - 1] * dW_S[:, i - 1]
                S_anti[:, i] = S_anti[:, i - 1] + riskfree * S_anti[:, i - 1] * delta_t + np.sqrt(
                    V_anti[:, i - 1]) * S_anti[:, i - 1] * -dW_S[:, i - 1]
            elif proba == 'forward-neutral':
                S[:, i] = S[:, i - 1] + (riskfree + V[:, i - 1]) * S[:, i - 1] * delta_t + \
                          np.sqrt(V[:, i - 1]) * S[:, i - 1] * dW_S[:, i - 1]
            V[:, i] = V[:, i - 1] + self.kappa * (self.v - V[:, i - 1]) * delta_t + self.gamma * np.sqrt(
                V[:, i - 1]) * dW_V[:, i - 1]
            V_anti[:, i] = V_anti[:, i - 1] + self.kappa * (self.v - V_anti[:, i - 1]) * delta_t + \
                           self.gamma * np.sqrt(V_anti[:, i - 1]) * (-dW_V[:, i - 1])

        return S, S_anti

    def characteristic_fun(self, x, init_val, riskfree, maturity):

        d = np.sqrt((self.kappa - 1j * self.rho * self.gamma * x) ** 2 + (x ** 2 + 1j * x) * self.gamma ** 2)
        g = (self.kappa - 1j * self.rho * self.gamma * x - d) / (self.kappa - 1j * self.rho * self.gamma * x + d)

        phi = np.exp(1j * x * riskfree * maturity + self.v_init / self.gamma ** 2 \
                     * (1 - np.exp(-d * maturity)) \
                     / (1 - g * np.exp(-d * maturity)) \
                     * (self.kappa - 1j * self.rho * self.gamma * x - d))

        phi = phi * np.exp(self.kappa * self.v / self.gamma ** 2 * \
                           (maturity * (self.kappa - 1j * self.rho * self.gamma * x - d) \
                            - 2 * np.log((1 - g * np.exp(-d * maturity)) / (1 - g))))

        return phi

    def density_integration_bounds(self, init_val, riskfree, maturity):

        m = np.log(init_val) + riskfree * maturity + (1 - np.exp(self.kappa * maturity)) * (self.v - self.v_init) / (
                2 * self.kappa) - 0.5 * self.v * maturity
        v = self.v / (8 * self.kappa ** 3) * (
                -self.gamma ** 2 * np.exp(-2 * self.kappa * maturity) + 4 * self.gamma * np.exp(
            - self.kappa * maturity) * (self.gamma - 2 * self.kappa * self.rho)
                + 2 * self.kappa * maturity * (
            4 * self.kappa ** 2 + self.gamma ** 2 - 4 * self.kappa * self.rho * self.gamma) + self.gamma * (
                        8 * self.kappa * self.rho - 3 * self.gamma))

        a = m - 20 * np.sqrt(np.abs(v))
        b = m + 20 * np.sqrt(np.abs(v))

        return a, b


class MertonJumpProcess:

    def __init__(self, sigma, lambda_jump, m_jump, v_jump):
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.m_jump = m_jump
        self.v_jump = v_jump

    def simulate(self, n_path, n_step, init_val, riskfree, maturity):
        delta_t = maturity / n_step
        paths = np.ones((n_path, n_step))
        dW = np.sqrt(delta_t) * np.random.randn(n_path, n_step-1)
        delta_X = np.random.normal(self.m_jump, self.v_jump, size=(n_path, n_step-1))
        poisson = np.random.poisson(self.lambda_jump * delta_t, size=(n_path, n_step-1))
        jump_part = np.exp(np.cumsum(delta_X * poisson, axis=1))
        paths[:, 1:] = np.cumprod(np.exp(delta_t * (riskfree - 0.5 * self.sigma ** 2 - self.lambda_jump * (
                    self.m_jump + self.v_jump ** 2 / 2)) + self.sigma * dW), axis=1)
        paths[:, 1:] *= jump_part

        paths *= init_val

        return paths

    def characteristic_fun(self, n_path, init_val, riskfree, maturity):
        pass
