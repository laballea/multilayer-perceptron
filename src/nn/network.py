import numpy as np
from tqdm import tqdm
import sys

from utils.utils_ml import not_zero
from nn.dense import DenseLayer

class AdamOptim():
    #https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    #https://www.youtube.com/watch?v=JXQT_vxqwIs
    def __init__(self, beta1: float=0.9, beta2: float=0.999, eps: float=1e-15):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
    
    def _update_wb(self, w: np.ndarray, b:np.ndarray, dw: np.ndarray, db: np.ndarray, lr: float):
        self.t += 1
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
        m_db_corr = self.m_db / (1 - self.beta1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)
        v_db_corr = self.v_db / (1 - self.beta2 ** self.t)

        w = w - lr * (m_dw_corr / (np.sqrt(v_dw_corr + self.eps)))
        b = b - lr * (m_db_corr / (np.sqrt(v_db_corr + self.eps)))
        return w, b

class Network:
    def __init__(self, name=None):
        self.supported_opt = ["basic", "adam"]

        self.layerSize = []
        self.network = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons
        self.params = [] ## W, b
        self.memory = [] ## Z, A
        self.gradients = [] ## dW, db
        self.eps = 1e-15
        self.loss = []  # store loss
        self.accuracy = []  # store accuracy
        self.total_it = 0
        self.name = name
        self.opt = "basic"

        
    def add(self, layer: DenseLayer):
        """
        Add layers to the network
        """
        self.network.append(layer)
        self.layerSize.append(layer.neurons)


    def _forwardprop(self, data, test=False):
        """
        Performs one full forward pass through network
        """
        A_curr = data  # current activation result

        # iterate over layers Weight and bias
        for i in range(len(self.params)):
            A_prev = A_curr  # save activation result
            # calculate forward for specific layer
            A_curr, Z_curr = self.network[i].forward(inputs=A_prev,
                                                     weights=self.params[i]['W'],
                                                     bias=self.params[i]['b'],
                                                     act_name=self.architecture[i]['activation']
                                                    )
            # save data for backwardprop
            if (not test):
                self.memory.append({'inputs':A_prev, 'Z':Z_curr})
        return A_curr

    def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        num_samples = len(actual)

        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_prev = dscores
        for idx, layer in reversed(list(enumerate(self.network))):
            dA_curr = dA_prev
            A_prev = self.memory[idx]['inputs']
            Z_curr = self.memory[idx]['Z']
            W_curr = self.params[idx]['W']
            act_name = self.architecture[idx]['activation']
            dA_prev, dW_curr, db_curr = layer.backward(dA_curr, W_curr, Z_curr, A_prev, act_name)
            self.gradients.append({'dW':dW_curr, 'db':db_curr})

    def basic_update_wb(self, w: np.ndarray, b:np.ndarray, dw: np.ndarray, db: np.ndarray, lr: float):
        w = w - lr * dw
        b = b - lr * db
        return w, b

    def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        for idx, layer in enumerate(self.network):
            dw, db = list(reversed(self.gradients))[idx]['dW'].T, list(reversed(self.gradients))[idx]['db']
            w, b = self.params[idx]['W'], self.params[idx]['b']
            new_w, new_b = self.params[idx]['opt'](w, b, dw, db, lr)
            self.params[idx]['W'] = new_w
            self.params[idx]['b'] = new_b

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        # for each sample get the index of the maximum value and compare it to the actual set
        # then compute the mean of this False/True array
        return float(np.mean(np.argmax(predicted, axis=1)==actual))

    def _calculate_loss(self, predicted, actual):
        """
        Calculate cross-entropy loss after each iteration
        """
        # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        samples = len(actual)
        correct_logprobs = -np.log(not_zero(predicted[range(samples), actual]))
        data_loss = np.sum(correct_logprobs) / samples
        return float(data_loss)

    def train(self, train, test, epochs, lr=0.01):
        """
        Train the model using SGD
        """
        X_train, Y_train = train[0], train[1]
        X_test, Y_test = test[0], test[1]
        for i in tqdm(range(epochs), disable=True):
            yhat_train = self._forwardprop(X_train)  # calculate prediction
            self._backprop(predicted=yhat_train, actual=Y_train)

            yhat_test = self._forwardprop(X_test, test=True)  # calculate prediction for test
            self.accuracy.append(self._get_accuracy(predicted=yhat_test, actual=Y_test))  # get accuracy
            self.loss.append(self._calculate_loss(predicted=yhat_test, actual=Y_test))  # get loss

            self._update(lr=lr)
            self.total_it += 1

    def _init_architect(self, data):
        """
        Initialize model architecture
        """
        shape = data.shape[1]
        for idx, layer in enumerate(self.network):
            self.architecture.append({'input_dim':shape, 
                                    'output_dim':self.network[idx].neurons,
                                    'activation':layer.act_name})
            shape = self.network[idx].neurons
        return self
    
    def reset(self):
        """
        Reset model weight and bias, gradients, memory etc..
        """

        np.random.seed(99)
        self.memory = []
        self.gradients = []
        self.loss = []
        self.params = []
        self.accuracy = []
        self.total_it = 0
        for i in range(len(self.architecture)):
            self.params.append({
                'W':np.random.uniform(low=-1, high=1,
                size=(self.architecture[i]['output_dim'],
                        self.architecture[i]['input_dim'])),
                'b':np.zeros((1, self.architecture[i]['output_dim'])),
                'opt':self.compile_opt()
            })
    
    def reset_weights(self):
        for i in range(len(self.architecture)):
            self.params[i] = {
                'W':np.random.uniform(low=-1, high=1, 
                size=(self.architecture[i]['output_dim'],
                        self.architecture[i]['input_dim'])),
                'b':np.zeros((1, self.architecture[i]['output_dim'])),
                'opt':self.compile_opt()
            }

    def compile(self, data, params, optimizer="basic"):
        """
        Initialize model parameters depending of input shape and architecture
        """
        self.layerSize.insert(0, data.shape[1])
        self._init_architect(data)
        if optimizer in self.supported_opt:
            self.opt = optimizer
        np.random.seed(99)
        if (len(params) == 0):
            for i in range(len(self.architecture)):
                self.params.append({
                    'W':np.random.uniform(low=-1, high=1, 
                    size=(self.architecture[i]['output_dim'],
                            self.architecture[i]['input_dim'])),
                    'b':np.zeros((1, self.architecture[i]['output_dim'])),
                    'opt':self.compile_opt()
                })
        else:
            for param in params:
                self.params.append({
                    'W':param["W"],
                    'b':param["b"],
                    'opt':self.compile_opt()
                })
            # self.params = params
        return self

    def compile_opt(self):
        if self.opt == "basic":
            return self.basic_update_wb
        elif self.opt == "adam":
            return AdamOptim()._update_wb
