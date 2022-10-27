import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization


def mlp(input_dim, output_dim,
        hidden_shape=(100,), activation='relu',
        use_bias=True, batch_normalization=True):
    input = Input(shape=(input_dim,))
    if batch_normalization:
        x = BatchNormalization()(input)
    else:
        x = input
    for i in range(len(hidden_shape)):
        x = Dense(hidden_shape[i],
                  activation=activation,
                  use_bias=use_bias)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
    output = Dense(output_dim, activation='linear')(x)
    return Model(input, output)


class FullyConnectedMLP(Model):
    """Connected MLP networks for Eur Basket BSDE.
    Adapted from Ref : github.com/frankhan91/DeepBSDE"""
    def __init__(self, n_step, delta_t, sigmas, riskfree, z_dim, y_init_inf,
                 y_init_sup, hidden_shape, activation=None, use_bias=False):
        super(FullyConnectedMLP, self).__init__()
        self.n_step = n_step
        self.z_dim = z_dim
        self.delta_t = delta_t
        self.sigmas = sigmas
        self.riskfree = riskfree
        self.y_init = tf.Variable(np.random.uniform(low=y_init_inf,
                                                    high=y_init_sup,
                                                    size=[1]),
                                  dtype='float32'
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.z_dim]),
                                  dtype='float32'
                                  )

        self.subnet = [
            MLPSubNet(output_dim=self.z_dim,
                      hidden_shape=hidden_shape,
                      activation=activation,
                      use_bias=use_bias)
            for _ in range(self.n_step-1)
        ]

    def call(self, inputs, training):
        X, DWs = inputs
        all_one_vec = tf.ones(shape=[tf.shape(DWs)[0], 1], dtype='float32')
        Y = all_one_vec * self.y_init
        Z = tf.matmul(all_one_vec, self.z_init)

        for i in range(0, self.n_step-1):
            Y = Y + self.riskfree * Y * self.delta_t + \
                tf.reduce_sum(tf.multiply(self.sigmas,
                                          tf.multiply(X[:, i, :], tf.multiply(Z, DWs[:, i, :]))),
                              axis=1, keepdims=True)

            Z = self.subnet[i](X[:, i+1, :], training)
        # last step
        Y = Y + self.riskfree * Y * self.delta_t + \
            tf.reduce_sum(tf.multiply(self.sigmas,
                                      tf.multiply(X[:, -2, :], tf.multiply(Z, DWs[:, -1, :]))),
                          axis=1, keepdims=True)
        return Y

    def predict_step(self, data):

        X, DWs = data[0]
        history = tf.TensorArray(dtype='float32', size=self.n_step)
        all_one_vec = tf.ones(shape=[tf.shape(DWs)[0], 1], dtype='float32')
        Y = all_one_vec * self.y_init
        Z = tf.matmul(all_one_vec, self.z_init)

        history = history.write(0, Y)

        for i in range(0, self.n_step-1):
            Y = Y + self.riskfree * Y * self.delta_t + \
                tf.reduce_sum(tf.multiply(self.sigmas,
                                          tf.multiply(X[:, i, :],
                                                      tf.multiply(Z, DWs[:, i, :]))),
                              axis=1, keepdims=True)
            history = history.write(i + 1, Y)
            Z = self.subnet[i](X[:, i+1, :], training=False)

        Y = Y + self.riskfree * Y * self.delta_t + \
            tf.reduce_sum(tf.multiply(self.sigmas,
                                      tf.multiply(X[:, -2, :], tf.multiply(Z, DWs[:, -1, :]))),
                          axis=1, keepdims=True)
        history = history.write(self.n_step - 1, Y)
        history = tf.transpose(history.stack(), perm=[1, 2, 0])
        return X, DWs, history


class MLPSubNet(Model):
    def __init__(self, output_dim, hidden_shape, activation=None, use_bias=False):
        super(MLPSubNet, self).__init__()
        self.hidden_shape = hidden_shape
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.dense_layers = [
            Dense(self.hidden_shape[i], use_bias=self.use_bias, activation=activation)
            for i in range(len(hidden_shape))
        ]
        self.batch_normalizations = [
            BatchNormalization(
                momentum=0.99,
            ) for _ in range(len(hidden_shape) + 2)
        ]
        self.last_layer = Dense(self.output_dim, use_bias=self.use_bias, activation='linear')

    def call(self, x, training):
        for i in range(len(self.hidden_shape)):
            x = self.batch_normalizations[i](x, training)
            x = self.dense_layers[i](x)
        x = self.batch_normalizations[-2](x, training)
        x = self.last_layer(x)
        x = self.batch_normalizations[-1](x, training)
        return x
