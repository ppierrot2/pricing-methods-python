import numpy as np


def payoff_call_basket(S_T, K, weights):
    """
    payoff call basket option
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = np.dot(np.transpose(weights), S_T)
    G = np.maximum(I_T - K, 0)
    return G


def payoff_put_basket(a, S_T, K):
    """
    payoff put basket option
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = np.dot(a, S_T)
    G = np.maximum(K - I_T, 0)
    return G


def payoff_worstof_put_basket(S_T, K):
    """
    payoff put basket of kind max(K - max(S1,..,Sd))
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = S_T.max(axis=1)
    G = np.maximum(K - I_T, 0)
    return G