import numpy as np
from activations import relu, relu_backward, sigmoid, sigmoid_backward, activations_ft


nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)  # consistant randomization depending of the seed
    number_of_layers = len(nn_architecture)  # store the number of layers
    params_values = {}  #  store weight and bias for each

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
    return params_values

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_name="relu"):
    try:
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        activation_func = activations_ft[activation_name]
        return activation_func(Z_curr), Z_curr
    except Exception as inst:
        raise Exception(inst)

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activation_name = layer["activation"]
        W_curr = params_values['W' + str(layer_idx)]
        b_curr = params_values['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_name)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
    return A_curr, memory
print(init_layers(nn_architecture))