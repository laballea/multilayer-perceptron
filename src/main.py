
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import getopt
import sys
from matplotlib import pyplot as plt
import matplotlib
import itertools
from tqdm import tqdm
import time
import yaml
import seaborn as sns


from utils.checker import arg_checker
from nn.visualize import Visualize
from utils.utils_ml import data_spliter, normalize
from nn.network import Network
from nn.dense import DenseLayer
from nn.regularizer import *
from utils.metrics import f1_score_, recall_score_, precision_score_


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

def clean(yml_file):
    yml_file["data"]["best_model"] = None
    for name in yml_file["models"]:
        yml_file["models"][name]["accuracy_hist"] = []
        yml_file["models"][name]["loss_hist"] = []
        yml_file["models"][name]["params"] = []
        yml_file["models"][name]["total_it"] = 0
    with open("models.yml", 'w') as outfile:
        yaml.dump(yml_file, outfile, default_flow_style=None)

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


def compile_model(yml_model, X, optimizer="basic"):
    """
    create the Network class depending on the structure
    """
    model = Network(name=yml_model["name"])
    for neuron_nb in yml_model["structure"][:-1]:
        model.add(DenseLayer(neuron_nb, act_name="tanh"))
    model.add(DenseLayer(yml_model["structure"][-1], act_name=yml_model["act_output"]))
    params = []
    for el in yml_model["params"]:
        params.append({"W":np.array(el["W"]), "b":np.array(el["b"])})
    model.compile(X, params, optimizer=optimizer)
    return model


def compile_models(yml_file, X, opt="basic"):
    """
    create all model
    """
    models = []
    for key, model in yml_file["models"].items():
        models.append(compile_model(model, X, opt))
    return models


def train(yml_file, models, X, Y, max_iter, lr=0.01):
    X_train, Y_train, X_test, Y_test = data_spliter(X, Y, 0.75)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    X = normalize(X)
    Y = Y.reshape(len(Y),)
    for model in tqdm(models, leave=True):
        yml_model = yml_file["models"][model.name]
        model.train(train=[X, Y.astype(int)], test=[X, Y.astype(int)], epochs=max_iter, lr=lr)
        
        time.sleep(0.5)
        yml_model["accuracy_hist"] += model.accuracy
        yml_model["loss_hist"] += model.loss
        yml_model["total_it"] += int(model.total_it)
        params = []
        for el in model.params:
            params.append({"W":el["W"].astype(float).tolist(), "b":el["b"].astype(float).tolist()})
        yml_model["params"] = params


def predict(yml_file, X, Y):
    model_yml = yml_file["models"][yml_file["data"]["best_model"]]
    # model_yml = yml_file["models"]["15_8_2"]
    model = compile_model(model_yml, X)
    X_ = normalize(X)
    Y = Y.reshape(len(Y),)
    Y_hat = model.predict(X_)
    print(f"accuracy = {model._get_accuracy(Y_hat, Y)} | loss = {model._calculate_loss(Y_hat, Y)}")
    print(f"f1_score = {f1_score_((np.argmax(Y_hat, axis=1)).reshape(-1, 1), Y.reshape(-1, 1))}")
    print(f"recall_score = {recall_score_((np.argmax(Y_hat, axis=1)).reshape(-1, 1), Y.reshape(-1, 1))}")
    print(f"precision_score = {precision_score_((np.argmax(Y_hat, axis=1)).reshape(-1, 1), Y.reshape(-1, 1))}")


def find_best(yml_file, max_iter):
    accuracy_list = []
    for model in yml_file["models"].values():
        size = len(model["accuracy_hist"])
        accuracy_list.append(np.array(model["accuracy_hist"][-1]))
    name_list = np.array([model["name"] for model in yml_file["models"].values()])
    best = name_list[np.argmax(accuracy_list, axis=0)]
    print(best)
    yml_file["data"]["best_model"] = str(best)


def display(yml_file, X):
    yml_model = yml_file["models"][yml_file["data"]["best_model"]]
    model = compile_model(yml_model, X)
    model.accuracy = yml_model["accuracy_hist"]
    model.loss = yml_model["loss_hist"]
    model.total_it = len(model.accuracy)
    Visualize().evol_2(range(model.total_it), model.loss, label=["iteration", "loss"], title=f"Model loss of {model.name}")
    Visualize().evol_2(range(model.total_it), model.accuracy, label=["iteration", "accuracy"], title=f"Model accuracy {model.name}")
    Visualize().draw_nn(model)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    for idx, eval in enumerate(["accuracy_hist", "loss_hist"]):
        quantil = np.quantile([model[eval].pop() for model in yml_file["models"].values()], (0.25, 0.5, 0.75))
        cmap = matplotlib.cm.get_cmap('Spectral')
        # plt.figure()
        for idx_model, yml_model in enumerate(yml_file["models"].values()):
            axs[idx].plot(range(len(yml_model[eval])), yml_model[eval], c=cmap(idx_model%5/5), label=yml_model["name"])
        axs[idx].legend()
        axs[idx].set_ylabel(eval)
        axs[idx].set_xlabel("total iteration")
    Visualize().plot()


def test_opt(yml_file, X, Y, max_iter, lr):
    X_train, Y_train, X_test, Y_test = data_spliter(X, Y, 0.75)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    data = {
        "it":range(max_iter)
    }
    for model_name in tqdm(yml_file["models"], leave=True):
        for opt in ["basic", "adam"]:
            model = compile_model(yml_file["models"][model_name], X, optimizer=opt)
            model.train(train=[X_train, Y_train.astype(int)], test=[X_test, Y_test.astype(int)], epochs=max_iter, lr=lr)
            data[f"accuracy_{opt}"] = model.accuracy
            data[f"accuracy_tr_{opt}"] = model.accuracy_tr
            data[f"loss_{opt}"] = model.loss
            data[f"loss_tr_{opt}"] = model.loss_tr
    
        plt.figure()
        last = data[f"accuracy_adam"][-1]
        plt.title(f"model_name accuracy_adam {last}")
        sns.lineplot(x='it', y='value', hue='variable', data=pd.melt(pd.DataFrame(data), ['it']))
        plt.ylabel("accuracy")
    plt.show()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:l:r:o:", ["file=", "maxL=", "maxN=", "minL=", "minN=", "predict", "reset", "clean", "train", "display", "opt"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    output = 2
    learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons = 0.01, 100, 1, 1, 1, 1
    optimizer = "basic"
    file = "../ressources/data.csv"

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
        if opt in ['-o']:
            optimizer = str(arg)

    X, Y, Y_n = get_data(file)
    np.random.seed(1)
    arg_checker(learning_rate, max_iter, max_layers, max_neurons, min_layers, min_neurons)

    for opt, arg in opts:
        if opt in ['--clean']:
            clean(yml_file)
        if opt in ['--reset']:
            reset(min_neurons, max_neurons, min_layers, max_layers, output, "softmax")
        if opt in ['--train']:
            models = compile_models(yml_file, X, opt=optimizer)
            train(yml_file, models, X, Y_n, max_iter, learning_rate)
            find_best(yml_file, max_iter)
            with open("models.yml", 'w') as outfile:
                yaml.dump(yml_file, outfile, default_flow_style=None)
        if opt in ['--display']:
            display(yml_file, X)
        if opt in ['--opt']:
            test_opt(yml_file, X, Y_n, max_iter, learning_rate)
        if opt in ['--predict']:
            predict(yml_file, X, Y_n)

if __name__ == "__main__":
    main(sys.argv[1:])