import numpy as np


class BasicOptim():
    def __init__(self):
        self.name = "basic"
        pass
    
    def _update_wb(self, t: int, w: np.ndarray, dw: np.ndarray, b: np.ndarray, db: np.ndarray, lr: float):
        w = w - lr * dw
        b = b - lr * db
        return w, b


class AdamOptim(BasicOptim):
    #https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    #https://www.youtube.com/watch?v=JXQT_vxqwIs
    def __init__(self, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8):
        self.name = "adam"
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def _update_wb(self, t: int, w: np.ndarray, dw: np.ndarray, b: np.ndarray, db: np.ndarray, lr: float):
        t += 1
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1-self.beta2) * (dw**2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1-self.beta2) * (db**2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1**t)
        m_db_corr = self.m_db / (1 - self.beta1**t)
        v_dw_corr = self.v_dw / (1 - self.beta2**t)
        v_db_corr = self.v_db / (1 - self.beta2**t)

        ## update weights and biases
        w = w - lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.eps))
        b = b - lr * (m_db_corr / (np.sqrt(v_db_corr) + self.eps))
        return w, b


optimizer_dict = {
    "basic":BasicOptim,
    "adam":AdamOptim
}