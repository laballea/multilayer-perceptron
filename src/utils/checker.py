def check_positive(values, labels=None):
    if not isinstance(values, list):
        raise TypeError(f"{values} must be a list.")
    if (labels is None):
        labels= ["Argument" for _ in range(len(values))]
    for value, label in zip(values, labels):
        if (value < 0):
            raise ValueError(f"{label} must be positive.")

def check_diff(value0, value1, labels=["Argument0", "Argument1"]):
    if value0 < value1:
        raise ValueError(f"{labels[0]} must be greater than {labels[1]}.")

def check_type(values, type_, labels=None):
    if not isinstance(values, list):
        raise TypeError(f"{values} must be a list.")
    if (labels is None):
        labels= ["Argument" for _ in range(len(values))]
    for value, label in zip(values, labels):
        if not isinstance(value, type_):
            raise TypeError(f"{label} must be of type {type}.")

def arg_checker(learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons):
    check_type([learning_rate], float, ["l"])
    check_type([max_iter, max_layers, max_neurons, min_layers, min_neurons],
                int,
                ["r", "maxL", "maxN", "minL", "minN"])
    check_positive([learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons],
                    ["l", "r", "maxL", "maxN", "minL", "minN"])
    check_diff(max_neurons, min_neurons, ["maxN", "minN"])
    check_diff(max_layers, min_layers, ["maxL", "minL"])