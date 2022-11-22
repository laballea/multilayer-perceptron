import numpy as np
from utils.utils_ml import not_zero
import sys


def relu(inputs):
    """
    ReLU Activation Function
    """
    return np.maximum(0, inputs)

def softmax(inputs):
    """
    Softmax Activation Function
    """
    exp_scores = not_zero(np.exp(inputs))
    tmp = not_zero(np.sum(exp_scores, axis=1, keepdims=True))
    probs = exp_scores / tmp
    return probs

act_funct = {
    "relu":relu,
    "softmax":softmax
}

def relu_derivative(dA, Z):
    """
    ReLU Derivative Function
    """
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_derivative(dA, Z):
    """
    ReLU Derivative Function
    """
    return dA

der_funct = {
    "relu":relu_derivative,
    "softmax":softmax_derivative
}
