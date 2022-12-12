import numpy as np

class Regularizer():
    def __init__(self, lambda_: float = 0):
        self.lambda_ = lambda_

    def regularize(self, weights):
        return 0

class l2(Regularizer):
    def __init__(self, lambda_: float = 0):
        self.lambda_ = lambda_
    
    def regularize(self, weights: np.ndarray):
        m = len(weights)
        return (self.lambda_ / 2 * m) * np.sum(weights**2)

class l1(Regularizer):
    def __init__(self, lambda_: float = 0):
        self.lambda_ = lambda_
    
    def regularize(self, weights: np.ndarray):
        m = len(weights)
        return (self.lambda_ / 2 * m) * np.sum(np.abs(weights))