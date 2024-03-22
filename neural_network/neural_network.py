import numpy as np
import logging
from scipy.special import expit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    A class representing a neural network.

    Parameters:
    - alpha (float): The learning rate of the neural network.
    - iterations (int): The number of iterations for training the neural network.
    - aspect_ratio (tuple): The aspect ratio of the input data.

    Attributes:
    - alpha (float): The learning rate of the neural network.
    - iterations (int): The number of iterations for training the neural network.
    - aspect_ratio (tuple): The aspect ratio of the input data.
    - hidden_layer_size (int): The size of the hidden layer in the neural network.
    """

    def __init__(self, alpha, hidden_layers, iterations, aspect_ratio):
        logger.info("Initializing Neural Network")
        self.alpha = alpha
        self.iterations = iterations
        self.aspect_ratio = aspect_ratio
        self.hidden_layer_size = hidden_layers

    def init_params(self):
        """
        Initializes the parameters of the neural network.

        Returns:
        - W1: numpy array of shape (hidden_layer_size, aspect_ratio[0] * aspect_ratio[1]), representing the weights of the first layer.
        - b1: numpy array of shape (hidden_layer_size, 1), representing the biases of the first layer.
        - W2: numpy array of shape (1, hidden_layer_size), representing the weights of the second layer.
        - b2: numpy array of shape (1, 1), representing the bias of the second layer.
        """
        W1 = np.random.randn(self.hidden_layer_size, self.aspect_ratio[0] * self.aspect_ratio[1])
        b1 = np.zeros((self.hidden_layer_size, 1))
        W2 = np.random.randn(1, self.hidden_layer_size)
        b2 = np.zeros((1, 1))
        return W1, b1, W2, b2

    def ReLU(self, Z):
        """
        Applies the Rectified Linear Unit (ReLU) activation function element-wise to the input array Z.

        Returns:
        - numpy.ndarray: Output array after applying ReLU activation function.
        """
        return np.maximum(Z, 0)

    def sigmoid(self, Z):
        """
        Applies the sigmoid activation function to the input Z.

        Returns:
        - numpy.ndarray: The output of the sigmoid function.
        """
        return expit(Z)

    def forward_prop(self, W1, b1, W2, b2, X):
        """
        Performs forward propagation in the neural network.

        Parameters:
        - W1 (numpy.ndarray): Weight matrix of the first layer.
        - b1 (numpy.ndarray): Bias vector of the first layer.
        - W2 (numpy.ndarray): Weight matrix of the second layer.
        - b2 (numpy.ndarray): Bias vector of the second layer.
        - X (numpy.ndarray): Input data.

        Returns:
        - Z1 (numpy.ndarray): Output of the first linear unit.
        - A1 (numpy.ndarray): Activation of the first layer.
        - Z2 (numpy.ndarray): Output of the second linear unit.
        - A2 (numpy.ndarray): Activation of the second layer.
        """
        Z1 = np.dot(W1, X) + b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2

    def deriv_ReLU(self, Z):
        """
        Compute the derivative of the ReLU activation function.

        Returns:
        - numpy.ndarray: Derivative of the ReLU function.
        """
        return Z > 0

    def back_prop(self, Z1, A1, A2, W2, X, Y):
        """
        Performs backpropagation to compute the gradients of the neural network's parameters.

        Parameters:
        - Z1: The output of the first linear layer (before applying the activation function)
        - A1: The output of the first activation function (ReLU) applied to Z1
        - A2: The output of the second activation function (sigmoid) applied to the final linear layer's output
        - W2: The weight matrix of the second linear layer
        - X: The input data
        - Y: The true labels

        Returns:
        - dW1: The gradient of the weight matrix W1
        - db1: The gradient of the bias vector b1
        - dW2: The gradient of the weight matrix W2
        - db2: The gradient of the bias vector b2
        """
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = np.dot(W2.T, dZ2) * self.deriv_ReLU(Z1)

        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        """
        Updates the parameters of the neural network using gradient descent.

        Parameters:
        - W1 (ndarray): The weights of the first layer.
        - b1 (ndarray): The biases of the first layer.
        - W2 (ndarray): The weights of the second layer.
        - b2 (ndarray): The biases of the second layer.
        - dW1 (ndarray): The gradients of the weights of the first layer.
        - db1 (ndarray): The gradients of the biases of the first layer.
        - dW2 (ndarray): The gradients of the weights of the second layer.
        - db2 (ndarray): The gradients of the biases of the second layer.

        Returns:
        - tuple: A tuple containing the updated weights and biases of the network.
        """
        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1
        W2 = W2 - self.alpha * dW2
        b2 = b2 - self.alpha * db2
        return W1, b1, W2, b2

    def get_prediction(self, A2):
        """
        Returns the predicted class labels based on the activation values in A2.

        Parameters:
        - A2 (numpy.ndarray): The activation values of the output layer.

        Returns:
        - numpy.ndarray: An array of predicted class labels, where 1 represents a positive prediction and 0 represents a negative prediction.
        """
        return (A2 > 0.5).astype(int)

    def get_accuracy(self, predictions, Y):
        """
        Calculates the accuracy of the predictions.

        Parameters:
        - predictions (numpy.ndarray): The predicted values.
        - Y (numpy.ndarray): The true values.

        Returns:
        - float: The accuracy of the predictions.
        """
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y):
        """
        Performs gradient descent optimization to train the neural network.

        Args:
        - X (numpy.ndarray): Input data of shape (num_features, num_examples).
        - Y (numpy.ndarray): True labels of shape (1, num_examples).

        Returns:
        - tuple: Updated weights and biases after training (W1, b1, W2, b2).
        """
        W1, b1, W2, b2 = self.init_params()
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, A2, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 10 == 0:
                accuracy = self.get_accuracy(self.get_prediction(A2), Y)
                logger.info(f"Iteration: {i}/{self.iterations}. Accuracy: {accuracy}")
        return W1, b1, W2, b2
