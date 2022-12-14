import matplotlib.pyplot as plt
import numpy as np


from nn.nn_visu import DrawNN
from nn.network import Network


class Visualize():
    def plot(self):
        plt.show()

    def evol_2(self, feat0, feat1, color="blue", label=["X", "Y"], title=None):
        plt.figure()
        plt.title(title)
        plt.plot(feat0, feat1, c=color)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    
    def draw_nn(self, model: Network):
        network = DrawNN(model.architecture, model.get_params())
        network.draw()
