import numpy as np

from task_1.ReLuLayer.ReLu.ReLu import ReLuLayer
from task_1.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer
from task_1.dropout.dropout import Dropout
from task_1.softmaxLayer import SoftmaxLayer


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

        # Structure
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        # Init weights and biases
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
                        for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]


    def forward_propagation(self, X, training = True):

        inputs = X
        # Stores outputs to be used for backward pass
        self.layer_outputs = []

        for i in range(len(self.weights) - 1):
            # z = XW + b
            z = np.dot(inputs, self.weights[i]) + self.biases[i]

            # ReLU activation
            a = self.relu_layer.forward(z)

            # Use dropout when training
            if training:
                self.dropout.forward(a, training=training)
                inputs = self.dropout.output
            else:
                inputs = a

            self.layer_outputs.append(inputs)

        # Softmax
        z_final = np.dot(inputs, self.weights[-1]) + self.biases[-1]
        predictions = self.softmax_layer.forward(z_final)

        # Softmax output
        self.layer_outputs.append(predictions)
        return predictions

    def backward_propagation(self, X, y):
        m = X.shape[0]  # Number of samples
        dz = self.layer_outputs[-1] - y  # Error gradient at output layer

        # Loop through the layers in reverse order
        for i in reversed(range(len(self.weights))):
            # Gradient with respect to weights
            if i > 0:
                dw = np.dot(self.layer_outputs[i - 1].T, dz) / m
            else:
                dw = np.dot(X.T, dz) / m

            # Gradient with respect to biases
            db = np.sum(dz, axis=0, keepdims=True) / m

            # L2 Regularization for weights
            dw += self.regularization_rate * self.weights[i]

            # Update weights and biases using gradient descent
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

            # Propagate the error gradient to the previous layer
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i - 1], function = 'relu')


    def activation_derivative(self, activation_output, function='relu'):
        # ReLU: 1 = positive input, else 0
        if function == 'relu':
            return np.where(activation_output > 0, 1, 0)

        # Sigmoid: sig_out * (1 = sig_out)
        elif function == 'sigmoid':
            return activation_output * (1 - activation_output)

        else:
            raise ValueError(f"Invalid activation function. {function}")

    def calculate_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Use epsilon to avoid log(0)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m

        #L2
        l2_loss = 0.5 * self.regularization_rate * sum(np.sum(w ** 2) for w in self.weights)
        return loss + l2_loss

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward_propagation(X, training=True)

            # Loss
            loss = self.calculate_loss(y, y_pred)

            # Backward pass
            self.backward_propagation(X, y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        probabilities = self.forward_propagation(X, training=False)
        return np.argmax(probabilities, axis=1)




