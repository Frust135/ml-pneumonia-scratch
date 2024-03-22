import numpy as np
import logging
from scipy.special import expit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNetwork:
    def __init__(self, alpha, iterations, aspect_ratio):
        logger.info("Initializing Neural Network")
        self.alpha = alpha
        self.iterations = iterations
        self.aspect_ratio = aspect_ratio
        self.hidden_layer_size = 200

    def init_params(self):
        W1 = np.random.randn(self.hidden_layer_size, self.aspect_ratio[0] * self.aspect_ratio[1])
        b1 = np.zeros((self.hidden_layer_size, 1))
        W2 = np.random.randn(1, self.hidden_layer_size)
        b2 = np.zeros((1, 1))
        return W1, b1, W2, b2

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def sigmoid(self, Z):
        return expit(Z)

    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = np.dot(W1, X) + b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2

    def deriv_ReLU(self, Z):
        return Z > 0

    def back_prop(self, Z1, A1, A2, W2, X, Y):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = np.dot(W2.T, dZ2) * self.deriv_ReLU(Z1)

        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1
        W2 = W2 - self.alpha * dW2
        b2 = b2 - self.alpha * db2
        return W1, b1, W2, b2

    def get_prediction(self, A2):
        return (A2 > 0.5).astype(int)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y):
        W1, b1, W2, b2 = self.init_params()
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, A2, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 10 == 0:
                accuracy = self.get_accuracy(self.get_prediction(A2), Y)
                logger.info(f"Iteration: {i}/{self.iterations}. Accuracy: {accuracy}")
        return W1, b1, W2, b2
