import cv2
import numpy as np
from os import path, listdir
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreateData:
    def __init__(self):
        logger.info("Initializing Data Creation")
        self.aspect_ratio = (1024, 1024)
        self.dataset_train_normal = self.convert_image_to_array("test/NORMAL")
        self.dataset_train_pneumonia = self.convert_image_to_array("test/PNEUMONIA")
        self.dataset_test_normal = self.convert_image_to_array("train/NORMAL")
        self.dataset_test_pneumonia = self.convert_image_to_array("train/PNEUMONIA")
        self.x_train, self.y_train, self.x_test, self.y_test = self.format_data()

    def convert_image_to_array(self, folder_path):
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
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def format_data(self):
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
