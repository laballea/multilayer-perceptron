from matplotlib import pyplot
from math import cos, sin, atan
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

class Neuron():
    def __init__(self, x, y, act_name, w):
        self.x = x
        self.y = y
        self.act_name = act_name
        self.w = w

    def draw(self, neuron_radius, layer_type):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        if (layer_type != 0 and self.act_name is not None):
            text = TextPath((0, 0), self.act_name, size=neuron_radius)
            text = TextPath((self.x - (2 * neuron_radius) - text.get_extents().width, self.y), self.act_name, size=neuron_radius)
            pyplot.gca().add_patch(PathPatch(text, color="black", linewidth=0.5))

class Layer():
    def __init__(self, network, number_of_neurons, max_neurons, params,act_name):
        self.vertical_distance_between_layers = 12
        self.horizontal_distance_between_neurons = 4
        self.neuron_radius = 0.5
        self.act_name = act_name
        self.max_neurons = max_neurons
        self.params = params
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in zip(range(number_of_neurons)):
            w = None
            if (self.params is not None):
                w = self.params['W'][iteration]
            neuron = Neuron(x, self.y, self.act_name, w)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.max_neurons - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, idx):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap_neg = cm.get_cmap("Reds")
        cmap_pos = cm.get_cmap("Greens")
        cmap = cmap_neg if neuron1.w[idx]< 0 else cmap_pos
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
                             color=m.to_rgba(neuron1.w[idx]))
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius, layerType)
            if self.previous_layer:
                for idx, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, idx)
        # write Text
        x_text = self.max_neurons * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, max_neurons, layer_params):
        self.max_neurons = max_neurons
        self.layers = []
        self.layers_params = layer_params
        self.layertype = 0

    def add_layer(self, layer, params):
        layer = Layer(self, layer['input_dim'], self.max_neurons, params=params, act_name=layer['activation'])
        self.layers.append(layer)

    def draw(self):
        pyplot.figure(figsize=(10, 10), dpi=60)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)

class DrawNN():
    def __init__(self, neural_network, layer_params):
        last = neural_network[-1]
        output = {'input_dim': last['output_dim'], 'output_dim': 0, 'activation': last['activation']}
        last['activation'] = neural_network[-2]['activation']
        neural_network.append(output)
        self.neural_network = neural_network
        self.layer_params = layer_params
        self.layer_params.insert(0, None)

    def draw(self):
        widest_layer = max(item['input_dim'] for item in self.neural_network)
        network = NeuralNetwork(widest_layer, self.layer_params)
        for layer, params in zip(self.neural_network, self.layer_params):
            network.add_layer(layer, params)
        network.draw()
