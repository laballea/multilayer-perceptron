from nn.network import Network
from nn.dense import DenseLayer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from utils.utils_ml import *
import getopt
import sys
from utils.checker import arg_checker
import itertools
from tqdm import tqdm
import time
import yaml
from nn.visualize import Visualize
from matplotlib import pyplot as plt
import matplotlib


def get_data(path):
    data = pd.read_csv(path, index_col=0)
    X = data.iloc[:, 1:].copy().to_numpy()
    Y = data.iloc[:, 0].copy()
    Y_n = LabelEncoder().fit_transform(Y).reshape(-1, 1)
    return np.array(X), np.array(Y), np.array(Y_n),

def get_combs(minN, maxN, minL, maxL):
    """
    get all possible structure combination depending of min and max layer, and min and max Neuron per layer
    (maxN - minN)**(maxL - minL) possibilities
    """
    pow = range(minN, maxN)
    combs = []
    for num_layer in range(minL, maxL + 1):
        combs += list(itertools.product(list(itertools.product(pow)), repeat=num_layer))
    return combs


def reset(minN, maxN, minL, maxL, output, act_output):
    """
    reset the yml file, all data is lost
    """
    models = {}

    combs = get_combs(minN, maxN, minL, maxL)
    models["data"] = {
        "best_model":None,
        "number_of_models":len(combs),
    }
    models["models"] = {}
    for comb in combs:
        comb = [value[0] for value in comb]
        comb.append(output)
        model_name = '_'.join(str(x) for x in comb)
        models["models"][model_name] = {
            "structure":list(comb),
            "name":model_name,
            "act_output":act_output,
            "accuracy_hist": [],
            "accuracy": -1,
            "loss_hist": [],
            "params":[],
            "total_it": 0
        }
    with open("models.yml", 'w') as outfile:
        yaml.dump(models, outfile, default_flow_style=None)
    return models


def compile_model(yml_model, X):
    """
    create the Network class depending on the structure
    """
    model = Network(name=yml_model["name"])
    for neuron_nb in yml_model["structure"][:-1]:
        model.add(DenseLayer(neuron_nb, act_name="relu"))
    model.add(DenseLayer(yml_model["structure"][-1], act_name=yml_model["act_output"]))
    params = []
    for el in yml_model["params"]:
        params.append({"W":np.array(el["W"]), "b":np.array(el["b"])})
    model.compile(X, params)
    return model


def compile_models(yml_file, X):
    """
    create all model
    """
    models = []
    for key, model in yml_file["models"].items():
        models.append(compile_model(model, X))
    return models


def train(yml_file, models, X, Y, max_iter, lr=0.1):
    for model in tqdm(models, leave=False):
        yml_model = yml_file["models"][model.name]
        for k_folds in cross_validation(X, Y, 10):
            x_train, y_train, x_test, y_test = k_folds
            x_train = normalize(x_train)
            x_test = normalize(x_test)
            model.reset_weights()
            model.train(train=[x_train, y_train.astype(int)], test=[x_test, y_test.astype(int)], epochs=max_iter, lr=lr)
            time.sleep(1)
        yml_model["accuracy_hist"] += model.accuracy
        yml_model["loss_hist"] += model.loss
        yml_model["total_it"] += int(model.total_it)
        params = []
        for el in model.params:
            params.append({"W":el["W"].tolist(), "b":el["b"].tolist()})
        yml_model["params"] = params
        time.sleep(0.5)


def find_best(yml_file, max_iter):
    accuracy_list = []
    for model in yml_file["models"].values():
        size = len(model["accuracy_hist"])
        accuracy_list.append(sum(np.array(model["accuracy_hist"][max_iter:size:max_iter])))
        # np.array([model["accuracy_hist"][:len(model["accuracy_hist"]):len(model["accuracy_hist"])/10] if len(model["accuracy_hist"]) > 0 else 0 )
    name_list = np.array([model["name"] for model in yml_file["models"].values()])
    best = name_list[np.argmax(accuracy_list, axis=0)]
    yml_file["data"]["best_model"] = str(best)


def display(yml_file, X):
    yml_model = yml_file["models"][yml_file["data"]["best_model"]]
    model = compile_model(yml_model, X)
    model.accuracy = yml_model["accuracy_hist"]
    model.loss = yml_model["loss_hist"]
    model.total_it = len(model.accuracy)
    Visualize().evol_2(range(model.total_it), model.loss, label=["iteration", "loss"], title=f"Model loss of {model.name}")
    Visualize().evol_2(range(model.total_it), model.accuracy, label=["accuracy", "loss"], title=f"Model accuracy {model.name}")
    Visualize().draw_nn(model)
    for eval in ["accuracy_hist", "loss_hist"]:
        quantil = np.quantile([model[eval].pop() for model in yml_file["models"].values()], (0.25, 0.5, 0.75))
        cmap = matplotlib.cm.get_cmap('Spectral')
        plt.figure()
        for idx, yml_model in enumerate(yml_file["models"].values()):
            alpha = 1
            if (yml_model[eval].pop() < quantil[0]):
                alpha = 0.1
            elif (yml_model[eval].pop() < quantil[1]):
                alpha = 0.3
            elif (yml_model[eval].pop() < quantil[2]):
                alpha = 0.5
            plt.plot(range(len(yml_model[eval])), yml_model[eval], c=cmap(idx%50/50), label=yml_model["name"], alpha=alpha)
            if ((idx % 50 == 0 and idx != 0) or idx == len(yml_file["models"].values()) - 1):
                plt.legend()
                plt.ylabel(eval)
                plt.xlabel("total iteration")
                if (idx != len(yml_file["models"].values()) - 1):
                    plt.figure()
    Visualize().plot()

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "l:r:", ["maxL=", "maxN=", "minL=", "minN=", "reset", "train", "display"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    X, Y, Y_n = get_data("../ressources/data.csv")
    output = 2
    learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons = 0.01, 100, 2, int(len(X.T)/2), 2, 5
    with open("models.yml", "r") as stream:
        try:
            yml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for opt, arg in opts:
        if (opt == "-l"):
            learning_rate = float(arg)
        if (opt == "-r"):
            max_iter = int(arg)
        if (opt in ['--maxL']):
            max_layers = int(arg)
        if (opt in ['--maxN']):
            max_neurons = int(arg)
        if (opt in ['--minL']):
            min_layers = int(arg)
        if (opt in ['--minN']):
            min_neurons = int(arg)
    arg_checker(learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons)
    for opt, arg in opts:
        if opt == '--reset':
            reset(min_neurons, max_neurons, min_layers, max_layers, output, "softmax")
        if opt == '--train':
            models = compile_models(yml_file, X)
            train(yml_file, models, X, Y_n, max_iter, learning_rate)
            find_best(yml_file, max_iter)
            with open("models.yml", 'w') as outfile:
                yaml.dump(yml_file, outfile, default_flow_style=None)
        if opt == '--display':
            display(yml_file, X)

if __name__ == "__main__":
    main(sys.argv[1:])