'''
Dataset loader for 3 Data Types
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle #came from the data provided
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'ORL', 'MNIST', or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _to_arrays(self, instances):
        X = np.array([inst['image'] for inst in instances], dtype=np.float32) / 255.0
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        name = self.dataset_source_file_name
        if name == 'MNIST':
            X = X[:, None, :, :]
        elif name == 'ORL':
            X = X[:, None, :, :, 0]
            y = y - 1
        elif name == 'CIFAR':
            X = X.transpose(0, 3, 1, 2)
        return X, y

    def load(self):
        print(f'loading {self.dataset_source_file_name}...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = self._to_arrays(data['train'])
        X_test, y_test = self._to_arrays(data['test'])
        print(f'train {X_train.shape}, test {X_test.shape}')
        return {'train': {'X': X_train, 'y': y_train},
                'test':  {'X': X_test,  'y': y_test}}