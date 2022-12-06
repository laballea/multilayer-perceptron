import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D  # for legend handle
import seaborn as sns
import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Features:
    def __init__(self, label, data):
        self.label = label
        self.data = data


def scatterplot(data: pd.DataFrame):
    data = data.sample(100)
    sns.pairplot(data=data.iloc[:, 0:30], hue=0)

def plot_camembert(data):
    figure = plt.figure()
    label = {'Malignient':'tab:red', 'Benign':'tab:blue'}
    colors = {'red':'M', 'blue':'B'}
    y = Features("Type", data.iloc[:, 1])
    pie = [data.iloc[:, 0].value_counts()["M"], data.iloc[:, 0].value_counts()["B"]]
    plt.pie(pie, labels=label, autopct='%1.2f%%',
            colors=colors, shadow=True)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in label.items()]
    plt.legend(title='Type', handles=handles, bbox_to_anchor=(0.8, 1), loc='upper left')


def visualize():
    label = {'Malignient':'tab:red', 'Benign':'tab:blue'}
    colors = {'M':'red', 'B':'blue'}
    data = pd.read_csv("../ressources/data.csv", index_col=0, names=range(0, 31))
    plot_camembert(data)
    scatterplot(data)
    plt.show()

visualize()