import numpy as np
from pricing_python.diffusion import BlackScholesProcessMulti
import tensorflow as tf
from tensorflow.keras.models import clone_model
from .networks import FullyConnectedMLP
DELTA_CLIP = 50.0
tf.config.run_functions_eagerly(True)  # for debugging


def payoff_call_mean(S, K):
    I = tf.reduce_mean(S, axis=1, keepdims=True)
    G = tf.maximum(tf.subtract(I, K * tf.ones_like(I)), 0)
    return G


class EurBasketBSDE:
    """Implementation of deep BSDE using multiple neural network approach"""
    def __init__(self, model: BlackScholesProcessMulti, init_vals, K,
                 maturity, riskfree, network_y, network_z,
                 payoff_fn=payoff_call_mean,
                 n_path=1000, n_step=100):
        self.model = model
        self.init_vals = init_vals
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.payoff_fn = payoff_fn
        self.network_y = network_y
        self.network_z = network_z
        self.n_path = n_path
        self.n_step = n_step
        self.delta_t = self.maturity / self.n_step
        self.dimension = self.model.dim
        self.networks = [self.network_y] + [clone_model(self.network_z) for _ in range(1, n_step)]
        self.sigmas = np.array(self.model.sigmas).astype('float32')

    def _sampler(self):
        X, _ = self.model.simulate(n_path=self.n_path, n_step=self.n_step, init_val=self.init_vals,
                                   riskfree=self.riskfree, maturity=self.maturity)
        DWs = self.model.wiener_diffs
        return X.astype('float32'), DWs.astype('float32')

    def _one_step_forward(self, prev, inputs):
        Y_prev, Z_prev, step = prev
        X, DWs = inputs
        Z = self.networks[int(step.numpy())](X)
        Y = (1 + self.riskfree) * self.delta_t * Y_prev + \
                tf.reduce_sum(tf.multiply(self.sigmas,
                                          tf.multiply(X, tf.multiply(Z_prev, DWs))),
                              axis=1, keepdims=True)
        return Y, Z, step + 1

    def _loss(self, inputs):
        X, DWs = inputs
        # Y_0 = self.networks[0](X[:, 0, :])
        # Z_0 = self.networks[1](X[:, 0, :])
        # Y_1 = (1 + self.riskfree) * self.delta_t * Y_0 + \
        #       tf.reduce_sum(tf.multiply(self.sigmas,
        #                                 tf.multiply(X[:, 0, :], tf.multiply(Z_0, DWs[:, 0, :]))),
        #                     axis=1, keepdims=True)
        #
        # X = tf.transpose(X[:, 1:, :], perm=[1, 0, 2])
        # DWs = tf.transpose(DWs[:, 1:, :], perm=[1, 0, 2])
        # elems = tf.stack([X, DWs], axis=1)
        # Y_all, Z_all, _ = tf.scan(self._one_step_forward, elems, initializer=(Y_1, Z_0, 1.))
        # terminal_cond = self.payoff_fn(X[:, self.n_step - 1, :], self.K)
        # loss = tf.reduce_mean(tf.square(Y_all[-1] - terminal_cond))

        Y = self.networks[0](X[:, 0, :])
        for i in range(self.n_step):
            Z = self.networks[i](X[:, i, :])
            Y = (1 + self.riskfree) * self.delta_t * Y + \
                  tf.reduce_sum(tf.multiply(self.sigmas,
                                            tf.multiply(X[:, i, :], tf.multiply(Z, DWs[:, i, :]))),
                                axis=1, keepdims=True)
        terminal_cond = self.payoff_fn(X[:, self.n_step - 1, :], self.K)
        loss = tf.reduce_mean(tf.square(Y - terminal_cond))
        return loss

    def solve(self, epochs=10, batch_size=100, learning_rate=0.001, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        self.train_losses = []
        if batch_size is None:
            batch_size = self.n_path

        X, DWs = self._sampler()
        train_data = tf.data.Dataset.from_tensor_slices((X.astype('float32'), DWs.astype('float32')))
        train_data = train_data.batch(batch_size)

        trainable_variables = []
        for i in range(self.n_step):
            trainable_variables.extend(self.networks[i].trainable_variables)

        @tf.function
        def train_one_iter(inputs):
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss(inputs)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            del tape
            return loss

        for ep in range(epochs):
            ep_loss_avg = tf.keras.metrics.Mean()
            for _, train_batch in enumerate(train_data):
                loss = train_one_iter(train_batch)
                ep_loss_avg.update_state(loss)
            self.train_losses.append(ep_loss_avg.result())
            if verbose:
                print("iteration {:03d}: Loss: {:.3f} Y_0 {}: ".format(ep,
                                                                       ep_loss_avg.result(),
                                                                       self.networks[0](
                                                                           np.array(self.init_vals).reshape(1, -1))))


class EurBasketBSDEConnectedNet:
    """Implementation of deep BSDE using a single neural network"""
    def __init__(self, model: BlackScholesProcessMulti, init_vals, K,
                 maturity, riskfree, y_init_range, payoff_fn=payoff_call_mean,
                 n_path=1000, n_step=100, hidden_shape=(50, 50,),
                 activation=None, use_bias=False):
        self.model = model
        self.init_vals = init_vals
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.sigmas = np.array(self.model.sigmas).astype('float32')
        self.payoff_fn = payoff_fn
        self.n_path = n_path
        self.n_step = n_step
        self.delta_t = self.maturity / self.n_step
        self.dimension = self.model.dim
        self.hidden_shape = hidden_shape
        self.network = FullyConnectedMLP(self.n_step, self.delta_t, self.sigmas,
                                         self.riskfree, z_dim=self.dimension,
                                         y_init_inf=y_init_range[0], y_init_sup=y_init_range[1],
                                         hidden_shape=hidden_shape, activation=activation,
                                         use_bias=use_bias)

    def _sampler(self):
        X, _ = self.model.simulate(n_path=self.n_path, n_step=self.n_step, init_val=self.init_vals,
                                   riskfree=self.riskfree, maturity=self.maturity)
        DWs = self.model.wiener_diffs
        return X, DWs

    def _loss(self, inputs, training):
        X, _ = inputs
        Y = self.network(inputs, training)
        terminal_cond = self.payoff_fn(X[:, self.n_step - 1, :], self.K)
        loss = tf.reduce_mean(tf.square(Y - terminal_cond))
        return loss

    def solve(self, epochs=10, batch_size=100, learning_rate=0.1, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        self.train_losses = []
        if batch_size is None:
            batch_size = self.n_path

        X, DWs = self._sampler()
        self.X = X
        self.DWs = DWs
        train_data = tf.data.Dataset.from_tensor_slices((X.astype('float32'), DWs.astype('float32')))
        train_data = train_data.batch(batch_size)

        @tf.function
        def train_one_iter(inputs):
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss(inputs, training=True)
            grads = tape.gradient(loss, self.network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
            del tape
            return loss

        for ep in range(epochs):
            ep_loss_avg = tf.keras.metrics.Mean()
            for step, train_batch in enumerate(train_data):
                loss = train_one_iter(train_batch)
                ep_loss_avg.update_state(loss)
            self.train_losses.append([ep_loss_avg.result(), self.network.y_init.numpy()[0]])
            if verbose:
                print("iteration {:03d}: Loss: {:.3f} Y0 : {}".format(ep, ep_loss_avg.result(),
                                                                      self.network.y_init.numpy()[0]))

    def eval(self):
        """eval model using new simulated SDE paths"""
        data = self._sampler()
        X, _, history = self.network.predict(data)
        return X, history


class EurBasketBSDEConnectedNetV2:
    """Implementation of deep BSDE using a single neural network. Improving the loss function as
    in Ref https://github.com/frankhan91/DeepBSDE"""
    def __init__(self, model: BlackScholesProcessMulti, init_vals, K,
                 maturity, riskfree, y_init_range, payoff_fn=payoff_call_mean,
                 n_step=100, hidden_shape=(50,50,),
                 activation=None, use_bias=False):
        self.model = model
        self.init_vals = init_vals
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.sigmas = np.array(self.model.sigmas).astype('float32')
        self.payoff_fn = payoff_fn
        self.n_step = n_step
        self.delta_t = self.maturity / self.n_step
        self.dimension = self.model.dim
        self.hidden_shape = hidden_shape
        self.y_init_range = y_init_range
        self.network = FullyConnectedMLP(self.n_step, self.delta_t, self.sigmas,
                                         self.riskfree, z_dim=self.dimension,
                                         y_init_inf=y_init_range[0], y_init_sup=y_init_range[1],
                                         hidden_shape=hidden_shape, activation=activation,
                                         use_bias=use_bias)

    def _sampler(self, n_path):
        X, _ = self.model.simulate(n_path=n_path, n_step=self.n_step, init_val=self.init_vals,
                                   riskfree=self.riskfree, maturity=self.maturity)
        DWs = self.model.wiener_diffs
        return X.astype('float32'), DWs.astype('float32')

    def _loss(self, inputs, training):
        X, _ = inputs
        Y = self.network(inputs, training)
        terminal_cond = self.payoff_fn(X[:, - 1, :], self.K)
        # loss = tf.reduce_mean(tf.square(Y - terminal_cond))
        delta = Y - terminal_cond
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss += 1000 * (tf.maximum(self.network.y_init[0] - self.y_init_range[1], 0) + tf.maximum(
            self.y_init_range[0] - self.network.y_init[0], 0))
        return loss

    def solve(self, sampling_iter=10, batch_size=100, learning_rate=0.1, verbose=True, **opt_params):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)
        self.train_losses = []

        @tf.function
        def train_one_iter(inputs):
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss(inputs, training=True)
            grads = tape.gradient(loss, self.network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
            del tape
            return loss

        loss_avg = tf.keras.metrics.Mean()
        for iter in range(sampling_iter):
            batch_data = self._sampler(batch_size)
            loss = train_one_iter(batch_data)
            if iter % 50 == 0:
                loss_avg.update_state(loss)
                self.train_losses.append([loss_avg.result(), self.network.y_init.numpy()[0]])
                if verbose:
                    print("iteration {:03d}: Avg Loss: {:.3f} Y0: {}".format(iter,
                                                                             loss_avg.result(),
                                                                             self.network.y_init.numpy()[0]))

    def eval(self, n_path=1000):
        """eval model using new simulated SDE paths"""
        data = self._sampler(n_path=n_path)
        X, _, history = self.network.predict(data)
        return X, history


class EurBasketDBDP:
    """Deep backward dynamic programming scheme, Ref : Deep backward schemes for
        high-dimensional nonlinear PDEs """
    def __init__(self, model: BlackScholesProcessMulti, init_vals, K,
                 maturity, riskfree, network_y, network_z,
                 payoff_fn=payoff_call_mean,
                 n_path=1000, n_step=100):
        self.model = model
        self.init_vals = init_vals
        self.K = K
        self.maturity = maturity
        self.riskfree = riskfree
        self.payoff_fn = payoff_fn
        self.n_path = n_path
        self.n_step = n_step
        self.delta_t = self.maturity / self.n_step
        self.dimension = self.model.dim
        self.networks_y = [clone_model(network_y) for _ in range(n_step)]
        self.networks_z = [clone_model(network_z) for _ in range(n_step)]
        self.sigmas = np.array(self.model.sigmas).astype('float32')

    def _sampler(self):
        X, _ = self.model.simulate(n_path=self.n_path, n_step=self.n_step, init_val=self.init_vals,
                                   riskfree=self.riskfree, maturity=self.maturity)
        DWs = self.model.wiener_diffs
        return X, DWs

    def _loss(self, inputs, step):
        X, X_sc, DWs = inputs
        if step == self.n_step - 1:
            Y = self.payoff_fn(X[:, step+1, :], self.K)
        else:
            Y = self.networks_y[step+1](X_sc[:, step+1, :])
        Z = self.networks_z[step](X_sc[:, step, :])
        Y_new = self.networks_y[step](X_sc[:, step, :])
        delta = Y - (1 + self.riskfree) * self.delta_t * Y_new - \
                tf.reduce_sum(tf.multiply(self.sigmas,
                                          tf.multiply(X[:, step, :],
                                                      tf.multiply(Z, DWs[:, step, :]))),
                              axis=1, keepdims=True)
        loss = tf.reduce_mean(tf.square(delta))
        return loss

    def _weight_initializer(self, step):
        """initialize current step network with trained weights from previous step network"""
        for i, l in enumerate(self.networks_y[step+1].layers[1:]):
            self.networks_y[step].layers[i+1].set_weights(l.get_weights())
        for i, l in enumerate(self.networks_z[step+1].layers[1:]):
            self.networks_z[step].layers[i+1].set_weights(l.get_weights())

    def _solve_step(self, data, step, epochs, learning_rate, verbose, **opt_params):

        if step < self.n_step - 1:
            self.networks_y[step + 1].trainable = False
            self.networks_z[step + 1].trainable = False
            self._weight_initializer(step)

        train_losses = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **opt_params)

        @tf.function
        def train_one_iter(inputs, step):
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss(inputs, step=step)
            grads = tape.gradient(loss, self.networks_y[step].trainable_variables +
                                  self.networks_z[step].trainable_variables)
            optimizer.apply_gradients(zip(grads, self.networks_y[step].trainable_variables +
                                          self.networks_z[step].trainable_variables))
            del tape
            return loss

        for ep in range(epochs):
            ep_loss_avg = tf.keras.metrics.Mean()
            for _, train_batch in enumerate(data):
                loss = train_one_iter(train_batch, step)
                ep_loss_avg.update_state(loss)
            train_losses.append(ep_loss_avg.result())
            if verbose:
                print("iteration {:03d}: Loss: {:.3f}".format(ep, ep_loss_avg.result()))

        return train_losses

    def solve(self, epochs=10, batch_size=100, learning_rate=0.001, scale_data=False,
              verbose=True, **opt_params):

        self.train_losses = []
        if batch_size is None:
            batch_size = self.n_path

        X, DWs = self._sampler()
        X_sc = X.copy()
        if scale_data:
            X_sc[:, 1:, :] = (X_sc[:, 1:, :] - np.mean(X_sc[:, 1:, :], axis=0)) / np.std(X_sc[:, 1:, :], axis=0)
            X_sc[:, 0, :] = 0.

        train_data = tf.data.Dataset.from_tensor_slices((X.astype('float32'),
                                                         X_sc.astype('float32'),
                                                         DWs.astype('float32')))
        train_data = train_data.batch(batch_size)

        for step in range(self.n_step-1, -1, -1):
            print('start training step ', step)
            step_losses = self._solve_step(train_data, step, epochs, learning_rate, verbose, **opt_params)
            self.train_losses.append(step_losses)
        return X, X_sc, DWs
