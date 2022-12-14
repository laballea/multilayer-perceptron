import numpy as np
from copy import deepcopy


from .activation import act_funct, der_funct
from .regularizer import Regularizer
from .optimizers import BasicOptim


class DenseLayer:
    def __init__(self, neurons, act_name="relu", regularizer: Regularizer=Regularizer()):
        self.neurons = neurons
        self.act_name = act_name
        self.reg = regularizer

        self.W = None
        self.b = None

        self.mA = None
        self.mI = None

        self.dW = None
        self.db = None

        self.optimizer = None
    
    def compile(self, input_dim, optimizer):
        self.input_dim = input_dim
        self.activation = act_funct[self.act_name]
        self.activation_der = der_funct[self.act_name]
        self.W = np.random.uniform(low=-1, high=1, size=(self.neurons, input_dim))
        self.b = np.zeros((1, self.neurons))
        self.optimizer: BasicOptim = deepcopy(optimizer)

    def forward(self, inputs, save):
        """
        Single Layer Forward Propagation
        """
        Z_curr = np.dot(inputs, self.W.T) + self.b
        A_curr = self.activation(inputs=Z_curr)
        if save:
            self.mI, self.mZ = inputs, Z_curr
        return A_curr

    def backward(self, dA_curr):
        """
        Single Layer Backward Propagation
        """
        dZ = self.activation_der(dA_curr, self.mZ)
        dW = np.dot(self.mI.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W)

        self.dW, self.db = dW, db #+ self.reg.regularize(W_curr)
        return dA
    
    def update(self, total_it:int, lr: float):
        self.W, self.b = self.optimizer._update_wb(total_it, self.W, self.dW.T, self.b, self.db, lr)