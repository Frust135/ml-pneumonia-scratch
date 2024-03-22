from neural_network.neural_network import NeuralNetwork
from neural_network.create_data import CreateData

if __name__ == "__main__":
    """
    This script trains a neural network model to classify pneumonia images.
    
    Steps:
    1. Create the dataset using the CreateData class.
    2. Initialize a NeuralNetwork object with specified parameters.
    3. Perform gradient descent to train the neural network using the training data.
    """
    dataset = CreateData()
    neural_network = NeuralNetwork(alpha=0.01, iterations=dataset.x_train.shape[1], aspect_ratio=dataset.aspect_ratio)
    neural_network.gradient_descent(dataset.x_train, dataset.y_train)
