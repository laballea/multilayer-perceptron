import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D  # for legend handle


class Features:
    def __init__(self, label, data):
        self.label = label
        self.data = data


def plot_features_3d(axis, feat0, feat1, y):
    # fig = plt.figure()
    cmap = matplotlib.colors.ListedColormap(['red', 'green'])
    colors = {'M':'tab:red', 'B':'tab:blue'}
    label = {'Malignient':'tab:red', 'Benign':'tab:blue'}
    axis.set_title("{} x {}".format(feat0.label, feat1.label))
    axis.scatter(feat0.data, feat1.data, c=y.data.map(colors))


def plot_camembert(data):
    figure = plt.figure()
    label = {'Malignient':'tab:red', 'Benign':'tab:blue'}
    colors = {'red':'M', 'blue':'B'}
    y = Features("Type", data.iloc[:, 1])
    pie = [data.iloc[:, 1].value_counts()["M"], data.iloc[:, 1].value_counts()["B"]]
    plt.pie(pie, labels=label, autopct='%1.2f%%',
            colors=colors, shadow=True)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in label.items()]
    plt.legend(title='Type', handles=handles, bbox_to_anchor=(0.8, 1), loc='upper left')


def visualize(data):
    label = {'Malignient':'tab:red', 'Benign':'tab:blue'}
    colors = {'M':'red', 'B':'blue'}

    y = Features("Type", data.iloc[:, 1])
    figure, axis = plt.subplots(5, 6)
    n = 2
    for i in range(0, 5):
        for j in range(0, 6):
            plot_features_3d(axis[i, j], Features("θ" + str(n), data.iloc[:, n]), Features("index", range(0, len(data))), y)
            n += 1
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in label.items()]
    plt.legend(title='Type', handles=handles, bbox_to_anchor=(0, 0), loc='upper left')
    plot_camembert(data)
    plt.show()
