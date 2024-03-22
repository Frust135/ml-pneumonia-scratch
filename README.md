# Neural Network from Scratch for Pneumonia Detection

This repository contains a simple implementation of a neural network built from scratch using Numpy. The neural network is trained to classify pneumonia images into two categories: normal and pneumonia.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    ```
2. Navigate to the project directory:
    ```bash
    cd tu_repositorio
    ```
3. Install the required dependencies (it is recommended to use a virtual environment):

    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script to train the neural network:

    ```bash
    python main.py
    ```

## Dataset
The dataset consists of X-ray images of lungs with and without pneumonia. The images are categorized into two classes: **normal** and **pneumonia**. 

You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Structure
The project is organized as follows:

- **neural_network**: Contains the implementation of the neural network.
    - **neural_network.py**: Defines the NeuralNetwork class with methods for training and inference.
    - **create_data.py**: Provides functionality for loading and formatting the dataset.
- **main.py**: Script to train the neural network using the dataset.
## Dependencies
- Python 3
- Numpy
- OpenCV
- Matplotlib
- Scipy (Only for the sigmoid equation) *TODO: Do the sigmoid equation only with Numpy*

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
Santiago (santiagosaav.99@gmail.com)

Feel free to reach out with any questions or feedback!