import sys, os
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR+"/..")
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
                 call=True, S_max=None, n_sim_interior=1000,
                 n_sim_terminal=100):
        self.model = model
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.call = call
        self.dgm = dgm_net
        self.n_sim_interior = n_sim_interior
        self.n_sim_terminal = n_sim_terminal
        self.S_max = 2*K if S_max is None else S_max

    def _sampler(self):

        t_interior = np.random.uniform(low=self.t_min, high=self.maturity, size=[self.n_sim_interior, 1])
        S_interior = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_interior, 1])
        t_terminal = self.maturity * np.ones((self.n_sim_terminal, 1))
        S_terminal = np.random.uniform(low=self.S_min, high=self.S_max * self.S_multiplier,
                                       size=[self.n_sim_terminal, 1])

        return t_interior, S_interior, t_terminal, S_terminal

    def _loss(self, t_interior, S_interior, t_terminal, S_terminal):
        """ Compute total loss for training"""

        # L1: PDE
        # compute function value and derivatives at current sampled points
        V = self.dgm(t_interior, S_interior)
        V_t = tf.gradients(V, t_interior)[0]
        V_s = tf.gradients(V, S_interior)[0]
        V_ss = tf.gradients(V_s, S_interior)[0]
        # V_ss = tf.hessians(V, S_interior)[0] # cause memory overflow

        diff_V = V_t + 0.5 * self.model.sigma**2 * S_interior**2 * V_ss +\
                 self.riskfree * S_interior * V_s - self.riskfree*V
        L1 = tf.reduce_mean(tf.square(diff_V))

        # L3: initial/terminal condition
        if self.call:
            target_payoff = tf.maximum(tf.subtract(S_terminal, self.K * tf.ones_like(S_terminal)), 0)
        else:
            target_payoff = tf.maximum(tf.subtract(self.K * tf.ones_like(S_terminal), S_terminal), 0)

        fitted_payoff = self.dgm(t_terminal, S_terminal)

        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L3

    def solve(self, sampling_iter=100, steps_per_sample=10,
              learning_rate=0.001, verbose=True, **opt_params):

        t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        S_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        t_terminal_tnsr = tf.placeholder(tf.float32, [None, 1])
        S_terminal_tnsr = tf.placeholder(tf.float32, [None, 1])

        L1_tnsr, L3_tnsr = self._loss(t_interior_tnsr, S_interior_tnsr, t_terminal_tnsr, S_terminal_tnsr)
        loss_tnsr = L1_tnsr + L3_tnsr
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, **opt_params).minimize(loss_tnsr)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # train
            for i in range(sampling_iter):
                t_interior, S_interior, t_terminal, S_terminal = self._sampler()

                for _ in range(steps_per_sample):
                    loss, L1, L3, _ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                               feed_dict={t_interior_tnsr: t_interior,
                                                          S_interior_tnsr: S_interior,
                                                          t_terminal_tnsr: t_terminal,
                                                          S_terminal_tnsr: S_terminal})
                if verbose:
                    print("iteration {:03d}: Loss: {:.3f} L1: {} L3: {}".format(i, loss, L1, L3))
            saver.save(sess, DIR + "/models/" + "curr_model.ckpt")

    def eval(self, S_val, t_val):
        S_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        self.V = self.dgm(t_interior_tnsr, S_interior_tnsr)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, DIR + '/models/curr_model.ckpt')
            fitted_option_val = sess.run(self.V, feed_dict={S_interior_tnsr: S_val,
                                                            t_interior_tnsr: t_val})
        return fitted_option_val
