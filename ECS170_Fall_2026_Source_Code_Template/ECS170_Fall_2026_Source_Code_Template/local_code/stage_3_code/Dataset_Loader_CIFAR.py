import pickle
import os
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader_CIFAR(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading CIFAR-10 dataset...")
        # Join folder path and file name ('CIFAR')
        file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        X_train, y_train = [], []
        X_test, y_test = [], []

        for instance in data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])

        for instance in data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        return {
            'X_train': np.array(X_train),
            'y_train': np.array(y_train),
            'X_test': np.array(X_test),
            'y_test': np.array(y_test)
        }