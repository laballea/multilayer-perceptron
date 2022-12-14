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
        self.network: list[DenseLayer] = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons

        self.loss = []  # store loss
        self.loss_tr = []  # store loss
        self.accuracy = []  # store accuracy
        self.accuracy_tr = []  # store accuracy

        self.total_it = 0
        self.name = name
        self.opt = BasicOptim()
        self.eps = 1e-15

    def _forwardprop(self, data, save=True):
        """
        Performs one full forward pass through network
        """
        A_curr = data  # current activation result

        # iterate over layers Weight and bias
        for layer in self.network:
            # calculate forward propagation for specific layer
            # save the ouput in A_curr and transfer it to the next layer
            A_curr = layer.forward(A_curr, save)
        return A_curr

    def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        num_samples = len(actual)

        # calculate loss derivative of our algorithm
        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_curr = dscores
        for layer in reversed(self.network):
            # calculate backward propagation for specific layer
            dA_curr = layer.backward(dA_curr)

    def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        for layer in self.network:
            # update layer Weights and bias
            layer.update(self.total_it, lr)

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
        y_hat = self._forwardprop(X, save=False)
        return y_hat

    def train(self, train, test, epochs, lr=0.01):
        """
        Train the model using GD
        """
        X_train, Y_train = train[0], train[1]
        X_test, Y_test = test[0], test[1]
        for i in tqdm(range(epochs), leave=False):
            yhat_train = self._forwardprop(X_train)  # prediction on train set
            yhat_test = self._forwardprop(X_test, save=False)  # prediction on test set

            #get accuracy and crossentropy on respectively train and test set, and save them
            self.accuracy_tr.append(self._get_accuracy(predicted=yhat_train, actual=Y_train))
            self.loss_tr.append(self._calculate_loss(predicted=yhat_train, actual=Y_train))
            self.accuracy.append(self._get_accuracy(predicted=yhat_test, actual=Y_test))
            self.loss.append(self._calculate_loss(predicted=yhat_test, actual=Y_test))
            
            self._backprop(predicted=yhat_train, actual=Y_train)
            self._update(lr=lr)
            self.total_it += 1
            time.sleep(0.03)

    def _init_architect(self, data):
        """
        Initialize model architecture
        """
        input_shape = data.shape[1]
        for idx, layer in enumerate(self.network):
            self.architecture.append({'input_dim':input_shape, 
                                    'output_dim':self.network[idx].neurons,
                                    'activation':layer.act_name})
            layer.compile(input_dim=input_shape, optimizer=self.opt)
            input_shape = self.network[idx].neurons
        return self

    def compile(self, data, params, optimizer:BasicOptim=None):
        """
        Initialize model parameters depending of input shape and architecture
        """
        self.layerSize.insert(0, data.shape[1])

        if isinstance(optimizer, BasicOptim):
            self.opt = optimizer
        self._init_architect(data)
        if params is not None:
            self.set_params(params)

        return self

    def add(self, layer: DenseLayer):
        """
        Add layers to the network
        """
        self.network.append(layer)
        self.layerSize.append(layer.neurons)

    def get_params(self):
        """
        Returns all weight and bias of layer contained in network formatted in a list
        """
        params = []
        for layer in self.network:
            params.append({"W":layer.W.astype(float).tolist(), "b": layer.b.astype(float).tolist()})
        return params

    def set_params(self, params):
        """
        Set all weight and bias of layer contained in network
        """
        for param, layer in zip(params, self.network):
            layer.W = np.array(param["W"])
            layer.b = np.array(param["b"])

    def update_yaml(self, previous_yaml: dict):
        """
        Given a dict, update accuracy, loss, total iteration and weights and bias
        """
        previous_yaml["accuracy_hist"] += self.accuracy
        previous_yaml["loss_hist"] += self.loss
        previous_yaml["total_it"] += int(self.total_it)
        previous_yaml["params"] = self.get_params()
