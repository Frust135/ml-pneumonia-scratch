import cv2
import numpy as np
from os import path, listdir
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreateData:
    """
    A class for creating and formatting data for a neural network model.

    Attributes:
        aspect_ratio (tuple): The desired aspect ratio for resizing images.
        dataset_train_normal (ndarray): Array of images from the "test/NORMAL" folder.
        dataset_train_pneumonia (ndarray): Array of images from the "test/PNEUMONIA" folder.
        dataset_test_normal (ndarray): Array of images from the "train/NORMAL" folder.
        dataset_test_pneumonia (ndarray): Array of images from the "train/PNEUMONIA" folder.
        x_train (ndarray): Transposed array of training images.
        y_train (ndarray): Array of training labels.
        x_test (ndarray): Transposed array of testing images.
        y_test (ndarray): Array of testing labels.
    """

    def __init__(self):
        """
        Initializes the CreateData object.
        """
        logger.info("Initializing Data Creation")
        self.aspect_ratio = (1024, 1024)
        self.dataset_train_normal = self.convert_image_to_array("test/NORMAL")
        self.dataset_train_pneumonia = self.convert_image_to_array("test/PNEUMONIA")
        self.dataset_test_normal = self.convert_image_to_array("train/NORMAL")
        self.dataset_test_pneumonia = self.convert_image_to_array("train/PNEUMONIA")
        self.x_train, self.y_train, self.x_test, self.y_test = self.format_data()

    def convert_image_to_array(self, folder_path):
        """
        Converts images in a given folder to a numpy array.

        Args:
            folder_path (str): The path to the folder containing the images.

        Returns:
            numpy.ndarray: A numpy array containing the flattened images.

        Raises:
            None

        """
        folder_path = path.join("dataset", folder_path)
        jpeg_files = [f for f in listdir(folder_path) if f.endswith(".jpeg")]
        images = []
        for jpeg_file in jpeg_files:
            image = plt.imread(folder_path + "/" + jpeg_file)
            resized_image = cv2.resize(image, self.aspect_ratio, interpolation=cv2.INTER_LINEAR)
            if resized_image.shape != self.aspect_ratio:
                logger.error(f"Image {jpeg_file} has the wrong aspect ratio: {resized_image.shape}.")
                continue
            flattened_image = resized_image.reshape(-1)
            images.append(flattened_image)
        return np.array(images)

    def show_image(self, image):
        """
        Display the given image in a window.
        Use it mainly for debugging purposes.
        
        Parameters:
            - image: The image to be displayed.
        """
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def format_data(self):
        """
        Formats the data by concatenating the training and testing datasets,
        shuffling the data, and returning the formatted data.

        Returns:
            x_train (numpy.ndarray): The formatted training data.
            y_train (numpy.ndarray): The labels for the training data.
            x_test (numpy.ndarray): The formatted testing data.
            y_test (numpy.ndarray): The labels for the testing data.
        """
        logger.info("Formatting data")
        
        normal = np.zeros(len(self.dataset_train_normal))
        pneumonia = np.ones(len(self.dataset_train_pneumonia))
        x_train = np.concatenate((self.dataset_train_normal, self.dataset_train_pneumonia))
        y_train = np.concatenate((normal, pneumonia)).astype(int)

        # Shuffle
        indices_train = np.random.permutation(len(x_train))
        x_train = x_train[indices_train]
        y_train = y_train[indices_train]
        
        normal = np.zeros(len(self.dataset_test_normal))
        pneumonia = np.ones(len(self.dataset_test_pneumonia))
        x_test = np.concatenate((self.dataset_test_normal, self.dataset_test_pneumonia))
        y_test = np.concatenate((normal, pneumonia)).astype(int)

        # Shuffle
        indices_test = np.random.permutation(len(x_test))
        x_test = x_test[indices_test]
        y_test = y_test[indices_test]

        return x_train.T, y_train, x_test.T, y_test
