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
    exp_scores = not_zero(np.exp(np.clip(inputs, -709.78, 709.78)))
    tmp = not_zero(np.sum(exp_scores, axis=1, keepdims=True))
    probs = exp_scores / tmp
    return probs

def sigmoid(inputs):
    """
    Sigmoid Activation Function
    """
    return 1 / (1 + np.exp(-inputs))

def tanh(inputs):
    """
    Sigmoid Activation Function
    """
    return (2 / (1 + np.exp(-2 * inputs))) - 1

act_funct = {
    "relu":relu,
    "softmax":softmax,
    "sigmoid":sigmoid,
    "tanh":tanh,
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
    Softmax Derivative Function
    """
    return dA

def sigmoid_derivative(dA, Z):
    """
    Sigmoid Derivative Function
    """
    s = sigmoid(Z)
    dZ = dA * (1 - s) * s
    return dZ

def tanh_derivative(dA, Z):
    """
    tanh Derivative Function
    """
    s = tanh(Z)
    dZ = dA * (1 - s**2)
    return dZ

der_funct = {
    "relu":relu_derivative,
    "softmax":softmax_derivative,
    "sigmoid":sigmoid_derivative,
    "tanh":tanh_derivative,
}
