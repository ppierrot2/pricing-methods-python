import numpy as np


class BlackScholesProcess:

    def __init__(self, sigma):
        self.sigma = sigma

    def simulate(self, n_path, n_step, init_val, riskfree, maturity):
        delta_t = maturity / n_step
        paths = np.zeros((n_path, n_step))
        paths_anti = np.zeros((n_path, n_step))  # antithetic
        paths[:, 0] = init_val
        paths_anti[:, 0] = init_val
        for m in range(1, n_step):
            dW = np.sqrt(delta_t) * np.random.randn(n_path)
            paths[:, m] = paths[:, m - 1] * np.exp(delta_t * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * dW)
            paths_anti[:, m] = paths_anti[:, m - 1] * np.exp(
                delta_t * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * -dW)

        return paths, paths_anti

    def simulate_maturity(self, n_path, init_val, riskfree, maturity):
        DW = np.sqrt(maturity)*np.random.randn(n_path)
        s_T = init_val*np.exp(maturity*(riskfree - 0.5*self.sigma**2) + self.sigma*DW)
        s_T_anti = init_val * np.exp(maturity * (riskfree - 0.5 * self.sigma ** 2) + self.sigma * -DW)
        return s_T, s_T_anti

    def characteristic_fun(self):
        # ToDo
        pass


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

        for i in range(1, n_step+1):
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

    def characteristic_fun(self):
        # ToDo
        pass


class HestonProcessMulti:
    # ToDo
    pass



