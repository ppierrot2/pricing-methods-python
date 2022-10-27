from pricing_python.diffusion import BlackScholesProcessMulti
from .dgm import DGMNet
import tensorflow as tf
import numpy as np
from .hessian_fd import fd_hessian


def payoff_call_mean(S, K):
    I = tf.reduce_mean(S, axis=1)
    G = tf.maximum(tf.subtract(I, K * tf.ones_like(I)), 0)
    return G


def payoff_put_mean(S, K):
    I = tf.reduce_mean(S, axis=1)
    G = tf.maximum(tf.subtract(K * tf.ones_like(I), I), 0)
    return G


def payoff_worstof_put_basket(S, K):
    I = tf.reduce_max(S, axis=1)
    G = tf.maximum(tf.subtract(K * tf.ones_like(I), I), 0)
    return G


class EurBasketDGM:

    t_min = 0 + 1e-10  # time lower bound
    S_min = 0.0 + 1e-10  # spot price lower bound
    S_multiplier = 1.5  # multiplier for oversampling : draw S from [S_min, S_max * S_multiplier]

    def __init__(self, model: BlackScholesProcessMulti, K,
                 maturity, riskfree, dgm_net: DGMNet,
                 payoff_fn=payoff_call_mean, S_max=None,
                 n_sim_interior=1000, n_sim_terminal=100):
        self.model = model
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.payoff_fn = payoff_fn
        self.dgm = dgm_net
        self.n_sim_interior = n_sim_interior
        self.n_sim_terminal = n_sim_terminal
        self.S_max = 2*K if S_max is None else S_max
        self.dimension = len(self.model.sigmas)

    def _sampler(self):

        t_interior = np.random.uniform(low=self.t_min, high=self.maturity,
                                       size=[self.n_sim_interior, 1]).astype('float32')
        S_interior = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_interior,
                                             self.dimension]).astype('float32')
        t_terminal = self.maturity * np.ones((self.n_sim_terminal, 1)).astype('float32')
        S_terminal = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_terminal,
                                             self.dimension]).astype('float32')

        return t_interior, S_interior, t_terminal, S_terminal

    def _loss(self, t_interior, S_interior, t_terminal, S_terminal):
        """ Compute total loss for training"""

        # L1: PDE
        # compute function value and derivatives at current sampled points
        V = self.dgm(t_interior, S_interior)
        V_t = tf.gradients(V, t_interior)[0]
        V_s = tf.gradients(V, S_interior)[0]
        # V_ss = tf.gradients(V_s, S_interior)[0]
        # V_ss = tf.hessians(V, S_interior)[0] # cause memory overflow
        # V_ss = tf.reduce_sum(V_ss, axis=2)
        # hack to compute the Hessian efficiently
        S_mean = tf.reduce_mean(S_interior)
        V_ss = (fd_hessian(self.dgm, S_interior, t_interior, 1.5e-6 * S_mean)
                + fd_hessian(self.dgm, S_interior, t_interior, 1.5e-7 * S_mean)
                + fd_hessian(self.dgm, S_interior, t_interior, 1.5e-8 * S_mean)) / 3
        first_order = self.riskfree * tf.reduce_sum(tf.multiply(S_interior, V_s), axis=1)
        second_order = tf.multiply(self.model.covariances.astype('float32'), V_ss)
        second_order = tf.map_fn(fn=lambda i: 0.5 * tf.tensordot(S_interior[i],
                                                                 tf.linalg.matvec(second_order[i],
                                                                                  S_interior[i]), axes=1),
                                 elems=tf.range(tf.shape(S_interior)[0]),
                                 dtype=tf.float32)
        diff_V = V_t + second_order + first_order - self.riskfree * V
        L1 = tf.reduce_mean(tf.square(diff_V))

        # L3: initial/terminal condition
        target_payoff = self.payoff_fn(S_terminal, self.K)
        fitted_payoff = self.dgm(t_terminal, S_terminal)

        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L3

    def solve(self, sampling_iter=100, steps_per_sample=10,
              learning_rate=0.001, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        train_losses = []

        @tf.function
        def train_one_iter(t_interior, S_interior, t_terminal, S_terminal):
            with tf.GradientTape() as tape:
                L1, L3 = self._loss(t_interior, S_interior, t_terminal, S_terminal)
                loss = L1 + L3
            gradients = tape.gradient(loss, self.dgm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.dgm.trainable_variables))
            return loss, L1, L3

        iter_loss_avg = tf.keras.metrics.Mean()
        iter_L1_avg = tf.keras.metrics.Mean()
        iter_L3_avg = tf.keras.metrics.Mean()

        for iter in range(sampling_iter):

            t_int, S_int, t_term, S_term = self._sampler()
            for _ in range(steps_per_sample):
                loss, L1, L3 = train_one_iter(t_int, S_int, t_term, S_term)
                iter_loss_avg.update_state(loss)
                iter_L1_avg.update_state(L1)
                iter_L3_avg.update_state(L3)

            train_losses.append([iter_loss_avg.result(), iter_L1_avg.result(), iter_L3_avg.result()])
            if verbose:
                print("iteration {:03d}: Loss: {:.3f} L1: {}  L3: {}".format(iter,
                                                                             iter_loss_avg.result(),
                                                                             iter_L1_avg.result(),
                                                                             iter_L3_avg.result()))

    def eval(self, S_val, t_val):
        fitted_option_val = self.dgm(t_val, S_val)
        return fitted_option_val
