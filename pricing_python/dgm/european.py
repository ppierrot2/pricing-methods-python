from pricing_python.diffusion import BlackScholesProcess
from .dgm import DGMNet
import tensorflow as tf
import numpy as np


class EurDGM:
    t_min = 0 + 1e-10  # time lower bound
    S_min = 0.0 + 1e-10  # spot price lower bound
    S_multiplier = 1.5  # multiplier for oversampling : draw S from [S_min, S_max * S_multiplier]

    def __init__(self, model: BlackScholesProcess, K,
                 maturity, riskfree, dgm_net: DGMNet,
                 call=True, S_max=None,n_sim_interior=1000,
                 n_sim_terminal=100, n_sim_boundary=None):
        self.model = model
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.call = call
        self.dgm = dgm_net
        self.n_sim_interior = n_sim_interior
        self.n_sim_terminal = n_sim_terminal
        self.n_sim_boundary = n_sim_boundary
        self.S_max = 2 * K if S_max is None else S_max

    def _sampler(self):

        t_interior = np.random.uniform(low=self.t_min, high=self.maturity,
                                       size=[self.n_sim_interior, 1]).astype('float32')
        S_interior = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_interior, 1]).astype('float32')
        t_terminal = self.maturity * np.ones((self.n_sim_terminal, 1)).astype('float32')
        S_terminal = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_terminal, 1]).astype('float32')

        return t_interior, S_interior, t_terminal, S_terminal

    def _sampler_boundary(self):

        t_boundary_inf = np.random.uniform(low=self.t_min, high=self.maturity,
                                           size=[self.n_sim_boundary, 1]).astype('float32')
        t_boundary_sup = np.random.uniform(low=self.t_min, high=self.maturity,
                                           size=[self.n_sim_boundary, 1]).astype('float32')
        S_boundary_inf = self.S_min * np.ones((self.n_sim_boundary, 1)).astype('float32')
        S_boundary_sup = self.S_max * np.ones((self.n_sim_boundary, 1)).astype('float32')

        return t_boundary_inf, S_boundary_inf, t_boundary_sup, S_boundary_sup

    def _loss(self, t_interior, S_interior, t_terminal, S_terminal):
        """ Compute total loss for training"""

        # L1: PDE
        # compute function value and derivatives at current sampled points
        V = self.dgm(t_interior, S_interior)
        V_t = tf.gradients(V, t_interior)[0]
        V_s = tf.gradients(V, S_interior)[0]
        V_ss = tf.gradients(V_s, S_interior)[0]
        # V_ss = tf.hessians(V, S_interior)[0] # cause memory overflow

        diff_V = V_t + 0.5 * self.model.sigma ** 2 * S_interior ** 2 * V_ss + \
                 self.riskfree * S_interior * V_s - self.riskfree * V
        L1 = tf.reduce_mean(tf.square(diff_V))

        # L3: initial/terminal condition
        if self.call:
            target_payoff = tf.maximum(tf.subtract(S_terminal, self.K * tf.ones_like(S_terminal)), 0)
        else:
            target_payoff = tf.maximum(tf.subtract(self.K * tf.ones_like(S_terminal), S_terminal), 0)

        fitted_payoff = self.dgm(t_terminal, S_terminal)

        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L3

    def _loss_boundary(self, t_boundary_inf, S_boundary_inf, t_boundary_sup, S_boundary_sup):

        V_boundary_sup = S_boundary_sup - self.K * tf.math.exp(- (self.maturity - t_boundary_sup)
                                                               * self.riskfree)
        V_boundary_inf = S_boundary_inf
        V_boundary_sup_fitted = self.dgm(t_boundary_sup, S_boundary_sup)
        V_boundary_inf_fitted = self.dgm(t_boundary_inf, S_boundary_inf)
        L2_sup =  tf.reduce_sum(tf.square(V_boundary_sup - V_boundary_sup_fitted))
        L2_inf = tf.reduce_sum(tf.square(V_boundary_inf - V_boundary_inf_fitted))
        return L2_sup + L2_inf

    def solve(self, sampling_iter=100, steps_per_sample=10,
              learning_rate=0.001, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        self.train_losses = []

        @tf.function
        def train_one_iter(t_interior, S_interior, t_terminal, S_terminal,
                           t_boundary_inf, S_boundary_inf, t_boundary_sup,
                           S_boundary_sup):
            with tf.GradientTape() as tape:
                L1, L3 = self._loss(t_interior, S_interior, t_terminal, S_terminal)
                loss = L1 + L3
                L2 = 0.
                if self.n_sim_boundary is not None:
                    L2 = self._loss_boundary(t_boundary_inf, S_boundary_inf, t_boundary_sup,
                                             S_boundary_sup)
                    loss += L2
            gradients = tape.gradient(loss, self.dgm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.dgm.trainable_variables))
            return loss, L1, L2, L3

        for iter in range(sampling_iter):
            iter_loss_avg = tf.keras.metrics.Mean()
            iter_L1_avg = tf.keras.metrics.Mean()
            iter_L2_avg = tf.keras.metrics.Mean()
            iter_L3_avg = tf.keras.metrics.Mean()

            t_int, S_int, t_term, S_term = self._sampler()
            t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup = None, None, None, None
            if self.n_sim_boundary is not None:
                t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup = self._sampler_boundary()

            for _ in range(steps_per_sample):
                loss, L1, L2, L3 = train_one_iter(t_int, S_int, t_term, S_term,
                                                  t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup)
                iter_loss_avg.update_state(loss)
                iter_L1_avg.update_state(L1)
                iter_L2_avg.update_state(L2)
                iter_L3_avg.update_state(L3)

            self.train_losses.append([iter_loss_avg.result(),
                                      iter_L1_avg.result(),
                                      iter_L2_avg.result(),
                                      iter_L3_avg.result()])
            if verbose:
                print("iteration {:03d}: Loss: {:.3f}"
                      " L1: {:.3f} L2: {:.3f} L3: {:.3f}".format(iter,
                                                                 iter_loss_avg.result(),
                                                                 iter_L1_avg.result(),
                                                                 iter_L2_avg.result(),
                                                                 iter_L3_avg.result()))

    def eval(self, S_val, t_val):
        fitted_option_val = self.dgm(t_val, S_val)
        return fitted_option_val


class EurDGMV2:
    t_min = 0 + 1e-10  # time lower bound
    S_min = 0.0 + 1e-10  # spot price lower bound
    S_multiplier = 1.5  # multiplier for oversampling : draw S from [S_min, S_max * S_multiplier]

    def __init__(self, model: BlackScholesProcess, K,
                 maturity, riskfree, dgm_net: DGMNet,
                 call=True, S_max=None):
        self.model = model
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.call = call
        self.dgm = dgm_net
        self.S_max = 2 * K if S_max is None else S_max

    def _sampler(self, n_sim_interior, n_sim_terminal):

        t_interior = np.random.uniform(low=self.t_min, high=self.maturity,
                                       size=[n_sim_interior, 1]).astype('float32')
        S_interior = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[n_sim_interior, 1]).astype('float32')
        t_terminal = self.maturity * np.ones((n_sim_terminal, 1)).astype('float32')
        S_terminal = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[n_sim_terminal, 1]).astype('float32')

        return t_interior, S_interior, t_terminal, S_terminal

    def _sampler_boundary(self, n_sim):

        t_boundary_inf = np.random.uniform(low=self.t_min, high=self.maturity,
                                           size=[n_sim, 1]).astype('float32')
        t_boundary_sup = np.random.uniform(low=self.t_min, high=self.maturity,
                                           size=[n_sim, 1]).astype('float32')
        S_boundary_inf = self.S_min * np.ones((n_sim, 1)).astype('float32')
        S_boundary_sup = self.S_max * np.ones((n_sim, 1)).astype('float32')

        return t_boundary_inf, S_boundary_inf, t_boundary_sup, S_boundary_sup

    def _loss(self, t_interior, S_interior, t_terminal, S_terminal):
        """ Compute total loss for training"""

        # L1: PDE
        # compute function value and derivatives at current sampled points
        V = self.dgm(t_interior, S_interior)
        V_t = tf.gradients(V, t_interior)[0]
        V_s = tf.gradients(V, S_interior)[0]
        V_ss = tf.gradients(V_s, S_interior)[0]
        # V_ss = tf.hessians(V, S_interior)[0] # cause memory overflow

        diff_V = V_t + 0.5 * self.model.sigma ** 2 * S_interior ** 2 * V_ss + \
                 self.riskfree * S_interior * V_s - self.riskfree * V
        L1 = tf.reduce_mean(tf.square(diff_V))

        # L3: initial/terminal condition
        if self.call:
            target_payoff = tf.maximum(tf.subtract(S_terminal, self.K * tf.ones_like(S_terminal)), 0)
        else:
            target_payoff = tf.maximum(tf.subtract(self.K * tf.ones_like(S_terminal), S_terminal), 0)

        fitted_payoff = self.dgm(t_terminal, S_terminal)

        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L3

    def _loss_boundary(self, t_boundary_inf, S_boundary_inf, t_boundary_sup, S_boundary_sup):

        V_boundary_sup = S_boundary_sup - self.K * tf.math.exp(- (self.maturity - t_boundary_sup)
                                                               * self.riskfree)
        V_boundary_inf = S_boundary_inf
        V_boundary_sup_fitted = self.dgm(t_boundary_sup, S_boundary_sup)
        V_boundary_inf_fitted = self.dgm(t_boundary_inf, S_boundary_inf)
        L2_sup =  tf.reduce_sum(tf.square(V_boundary_sup - V_boundary_sup_fitted))
        L2_inf = tf.reduce_sum(tf.square(V_boundary_inf - V_boundary_inf_fitted))
        return L2_sup + L2_inf

    def solve(self, sampling_iter=100, n_sim_interior=1000, n_sim_terminal=100,
              n_sim_boundary=None, learning_rate=0.001, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        self.train_losses = []

        @tf.function
        def train_one_iter(t_interior, S_interior, t_terminal, S_terminal,
                           t_boundary_inf, S_boundary_inf, t_boundary_sup,
                           S_boundary_sup):
            with tf.GradientTape() as tape:
                L1, L3 = self._loss(t_interior, S_interior, t_terminal, S_terminal)
                loss = L1 + L3
                L2 = 0.
                if n_sim_boundary is not None:
                    L2 = self._loss_boundary(t_boundary_inf, S_boundary_inf, t_boundary_sup,
                                             S_boundary_sup)
                    loss += L2
            gradients = tape.gradient(loss, self.dgm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.dgm.trainable_variables))
            return loss, L1, L2, L3

        iter_loss_avg = tf.keras.metrics.Mean()
        iter_L1_avg = tf.keras.metrics.Mean()
        iter_L2_avg = tf.keras.metrics.Mean()
        iter_L3_avg = tf.keras.metrics.Mean()

        for iter in range(sampling_iter):

            t_int, S_int, t_term, S_term = self._sampler(n_sim_interior, n_sim_terminal)
            t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup = None, None, None, None
            if n_sim_boundary is not None:
                t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup = self._sampler_boundary(n_sim_boundary)

            loss, L1, L2, L3 = train_one_iter(t_int, S_int, t_term, S_term,
                                              t_bd_inf, S_bd_inf, t_bd_sup, S_bd_sup)
            iter_loss_avg.update_state(loss)
            iter_L1_avg.update_state(L1)
            iter_L2_avg.update_state(L2)
            iter_L3_avg.update_state(L3)

            self.train_losses.append([iter_loss_avg.result(),
                                      iter_L1_avg.result(),
                                      iter_L2_avg.result(),
                                      iter_L3_avg.result()])
            if iter % 50 == 0:
                if verbose:
                    print("iteration {:03d}: Loss: {:.3f}"
                          " L1: {:.3f} L2: {:.3f} L3: {:.3f}".format(iter,
                                                                     iter_loss_avg.result(),
                                                                     iter_L1_avg.result(),
                                                                     iter_L2_avg.result(),
                                                                     iter_L3_avg.result()))

    def eval(self, S_val, t_val):
        fitted_option_val = self.dgm(t_val, S_val)
        return fitted_option_val
