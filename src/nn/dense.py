import numpy as np
import math
from numpy.random import rand


from .activation import act_funct, der_funct
from .regularizer import Regularizer

class DenseLayer:
    def __init__(self, neurons, act_name="relu", regularizer: Regularizer=Regularizer()):
        self.neurons = neurons
        self.act_name = act_name
        self.reg = regularizer
    
    def forward(self, inputs, weights, bias, act_name):
        """
        Single Layer Forward Propagation
        """
        Z_curr = np.dot(inputs, weights.T) + bias
        A_curr = act_funct[act_name](inputs=Z_curr)
        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, act_name):
        """
        Single Layer Backward Propagation
        """
        dZ = der_funct[act_name](dA_curr, Z_curr)
        dW = np.dot(A_prev.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, W_curr)
        return dA, dW, db #+ self.reg.regularize(W_curr)