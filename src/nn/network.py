import numpy as np
from tqdm import tqdm
import time
from copy import deepcopy

from utils.utils_ml import not_zero
from nn.dense import DenseLayer
from nn.optimizers import BasicOptim


class Network:
    def __init__(self, name=None):
        self.layerSize = []
        self.network = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons

        self.params = [] ## W, optimizer

        self.memory = [] ## Z, A
        self.gradients = [] ## dW


        self.loss = []  # store loss
        self.loss_tr = []  # store loss
        self.accuracy = []  # store accuracy
        self.accuracy_tr = []  # store accuracy

        self.total_it = 0
        self.name = name
        self.opt = BasicOptim()
        self.eps = 1e-15

        
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

    def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        for idx, layer in enumerate(self.network):
            dw = list(reversed(self.gradients))[idx]['dW'].T
            db = list(reversed(self.gradients))[idx]['db']
            w = self.params[idx]['W']
            b = self.params[idx]['b']
            new_w, new_b = self.params[idx]['opt'](self.total_it, w, dw, b, db, lr)
            self.params[idx]['W'] = new_w.astype(np.float32)
            self.params[idx]['b'] = new_b.astype(np.float32)

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        # for each sample get the index of the maximum value and compare it to the actual set
        # then compute the mean of this False/True array
        return float(np.mean(np.argmax(predicted, axis=1)==actual))

    def _calculate_loss(self, predicted: np.ndarray, actual: np.ndarray):
        """
        Calculate cross-entropy loss after each iteration
        """
        # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        samples = len(actual)
        correct_logprobs = -np.log(not_zero(predicted[range(samples), actual.astype(int)]))
        data_loss = np.sum(correct_logprobs) / samples
        return float(data_loss)

    def predict(self, X):
        y_hat = self._forwardprop(X, test=True)
        return y_hat

    def train(self, train, test, epochs, lr=0.01):
        """
        Train the model using SGD
        """
        X_train, Y_train = train[0], train[1]
        X_test, Y_test = test[0], test[1]
        for i in tqdm(range(epochs), leave=False):
            yhat_train = self._forwardprop(X_train)  # calculate prediction
            self.accuracy_tr.append(self._get_accuracy(predicted=yhat_train, actual=Y_train))  # get accuracy
            self.loss_tr.append(self._calculate_loss(predicted=yhat_train, actual=Y_train))  # get loss
            self._backprop(predicted=yhat_train, actual=Y_train)

            yhat_test = self._forwardprop(X_test, test=True)  # calculate prediction for test
            self.accuracy.append(self._get_accuracy(predicted=yhat_test, actual=Y_test))  # get accuracy
            self.loss.append(self._calculate_loss(predicted=yhat_test, actual=Y_test))  # get loss

            self._update(lr=lr)
            self.total_it += 1
            time.sleep(0.03)

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
    
    def reset_weights(self):
        self.params = []
        for i in range(len(self.architecture)):
            self.params.append({
                'W':np.random.uniform(low=-1, high=1, 
                        size=(self.architecture[i]['output_dim'], 
                              self.architecture[i]['input_dim'])),
                'b':np.zeros((1, self.architecture[i]['output_dim'])),
                'opt':deepcopy(self.opt)._update_wb
            })

    def compile(self, data, params, optimizer:BasicOptim=None):
        """
        Initialize model parameters depending of input shape and architecture
        """
        self.layerSize.insert(0, data.shape[1])
        self._init_architect(data)
        if isinstance(optimizer, BasicOptim):
            self.opt = optimizer
        if (len(params) == 0):
            self.reset_weights()
        else:
            for param in params:
                self.params.append({
                    'W':param["W"],
                    'b':param["b"],
                    'opt':deepcopy(self.opt)._update_wb
                })
        return self
