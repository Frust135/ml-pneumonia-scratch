from neural_network.neural_network import NeuralNetwork
from neural_network.create_data import CreateData

if __name__ == "__main__":
    dataset = CreateData()
    neural_network = NeuralNetwork(alpha=0.01, iterations=dataset.x_train.shape[1], aspect_ratio=dataset.aspect_ratio)
    neural_network.gradient_descent(dataset.x_train, dataset.y_train)
