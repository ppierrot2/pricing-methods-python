"""Ref : https://github.com/adolfocorreia/DGM
        https://github.com/alialaradi/DeepGalerkinMethod
        https://github.com/ZewenShen/hdp"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


def _get_activation(activ_name):
    if activ_name == "tanh":
        return tf.nn.tanh
    elif activ_name == "relu":
        return tf.nn.relu
    elif activ_name == "sigmoid":
        return tf.nn.sigmoid
    else:
        return tf.identity


class LSTMLayer(Layer):

    def __init__(self, output_dim, input_dim, activation_1="tanh",
                 activation_2="tanh", initializer='glorot_uniform'):
        """
        Custom LSTM layer for DGM Net

        Parameters
        ----------
        output_dim : int
            dimensionality of input data
        input_dim : int
            number of outputs for LSTM layers
        activation_1 : str
            activation function for Z, G, R. {'tanh', 'relu', 'sigmoid'}
        activation_2 : str
            activation function for H. {'tanh', 'relu', 'sigmoid'}
        """
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_1 = _get_activation(activation_1)
        self.activation_2 = _get_activation(activation_2)

        self.Uz = self.add_weight("Uz", shape=[self.input_dim, self.output_dim],
                                  initializer=initializer)
        self.Ug = self.add_weight("Ug", shape=[self.input_dim, self.output_dim],
                                  initializer=initializer)
        self.Ur = self.add_weight("Ur", shape=[self.input_dim, self.output_dim],
                                  initializer=initializer)
        self.Uh = self.add_weight("Uh", shape=[self.input_dim, self.output_dim],
                                  initializer=initializer)

        self.Wz = self.add_weight("Wz", shape=[self.output_dim, self.output_dim],
                                  initializer=initializer)
        self.Wg = self.add_weight("Wg", shape=[self.output_dim, self.output_dim],
                                  initializer=initializer)
        self.Wr = self.add_weight("Wr", shape=[self.output_dim, self.output_dim],
                                  initializer=initializer)
        self.Wh = self.add_weight("Wh", shape=[self.output_dim, self.output_dim],
                                  initializer=initializer)

        # bias
        self.bz = self.add_weight("bz", shape=[1, self.output_dim])
        self.bg = self.add_weight("bg", shape=[1, self.output_dim])
        self.br = self.add_weight("br", shape=[1, self.output_dim])
        self.bh = self.add_weight("bh", shape=[1, self.output_dim])

    # main function to be called
    def call(self, S, X):
        """
        Compute output of a LSTMLayer for a given inputs S,X .

        Parameters
        ----------
            S: output of previous layer
            X: data input

        Returns
        -------
        S_new:
        """

        # compute components of LSTM layer output
        Z = self.activation_1(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.activation_1(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.activation_1(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.activation_2(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)),
                                   self.bh))

        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))

        return S_new


class DenseLayer(Layer):

    def __init__(self, output_dim, input_dim, activation="tanh", initializer='glorot_uniform'):
        """
        Dense layer for DGM Net

        Parameters
        ----------
        output_dim : int
            dimensionality of input data
        input_dim : int
            number of outputs for LSTM layers
        activation : str
            activation function of LSTM nodes. {'tanh', 'relu', 'sigmoid'}
        """
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = _get_activation(activation)

        self.W = self.add_weight("W", shape=[self.input_dim, self.output_dim],
                                 initializer=initializer)
        self.b = self.add_weight("b", shape=[1, self.output_dim],
                                 initializer=initializer)

    def call(self, inputs):
        out = tf.add(tf.matmul(inputs, self.W), self.b)
        return self.activation(out)


class DGMNet(Model):

    def __init__(self, hidden_units, n_layers, input_dim,
                 lstm_activation_1="tanh", lstm_activation_2="tanh",
                 last_activation="linear"):
        """
        Parameters
        ----------
            hidden_units: int
                dimension of layers output in the network
            n_layers: int
               number of intermediate LSTM layers
            input_dim: int
                spatial dimension of input data (EXCLUDES time dimension)
            lstm_activation_1: str
                activation 1 of LSTM layers. Default "tanh"
            lstm_activation_2: str
                activation 2 of LSTM layers. Default "tanh"
        """
        super(DGMNet, self).__init__()
        self.initial_layer = DenseLayer(hidden_units, input_dim + 1, activation="tanh")
        self.n_layers = n_layers
        self.LSTMLayers = []
        for _ in range(self.n_layers):
            self.LSTMLayers.append(LSTMLayer(hidden_units, input_dim + 1,
                                             activation_1=lstm_activation_1,
                                             activation_2=lstm_activation_2))
        self.final_layer = DenseLayer(1, hidden_units, activation=last_activation)

    def call(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs
        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''

        # define input vector as time-space pairs
        X = tf.concat([t, x], 1)
        S = self.initial_layer.call(X)
        for i in range(self.n_layers):
            S = self.LSTMLayers[i].call(S, X)
        result = self.final_layer.call(S)
        return result