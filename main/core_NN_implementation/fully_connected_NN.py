import numpy as np

from main.ReLuLayer.ReLu.ReLu import ReLuLayer
from main.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer
from main.dropout import dropout
from main.softmaxLayer.softmax.softmax import SoftmaxLayer


class fully_connected_NN:
    def __init__(self, input_size, output_size, learning_rate, hidden_layers, dropout_rate, regularization_rate):
        """
        Parameters:

        input_size
        output_size
        learning_rate
        hidden_layers
        dropout_rate
        regularization_rate (L2)
        """

        # Init params
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.regularization_rate = regularization_rate

        # Init dropout
        self.dropout = dropout.Dropout(rate = dropout_rate)

        #Init activation layers
        self.softmax_layer = SoftmaxLayer()
        self.relu_layer = ReLuLayer()
        self.sigmoid_layer = SigmoidLayer()

        # Init weights and biases
        self.weights = []
        self.biases = []

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




