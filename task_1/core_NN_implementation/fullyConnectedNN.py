import numpy as np

from task_1.ReLuLayer.ReLu.ReLu import ReLuLayer
from task_1.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer
from task_1.dropout.dropout import Dropout
from task_1.softmaxLayer.softmax.softmax import SoftmaxLayer


class fullyConnectedNN:
    def __init__(self, input_size, output_size, hidden_layers, learning_rate = 0.001, dropout_rate = 0.05, regularization_rate = 0.01):
        """
        Parameters:

        input_size (int): Number of input features
        output_size (int): Number of output neurons
        hidden_layers (list): List of hidden layer sizes
        learning_rate (float): Learning rate
        dropout_rate (float): Dropout rate
        regularization_rate (float): L2 regularization rate
        """

        # Init params
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.regularization_rate = regularization_rate

        # Init dropout
        self.dropout = Dropout(rate = dropout_rate)

        #Init activation layers
        self.softmax_layer = SoftmaxLayer()
        self.relu_layer = ReLuLayer()
        self.sigmoid_layer = SigmoidLayer()

        # Init weights and biases
        self.weights = []
        self.biases = []
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range (len(self.layer_sizes)-1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass

    def calculate_loss(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass




