'''
Dataset loader for the Stage 3 image datasets (ORL, MNIST, CIFAR)
'''

from local_code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader_ORL(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        # 1. open the pickle file and load the saved dictionary {'train': [...], 'test': [...]}
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        result = {}
        for split in ['train', 'test']:
            X = []
            y = []
            # 2. go through every instance and collect its image and label
            for instance in data[split]:
                X.append(instance['image'])
                y.append(instance['label'])

            # turn the lists into numpy arrays and scale pixels from 0-255 to 0-1
            X = np.array(X, dtype=np.float32) / 255.0
            y = np.array(y, dtype=np.int64)

            # 3. fix the image shape for PyTorch: (N, channels, height, width)
            if self.dataset_source_file_name == 'ORL':
                X = X[:, :, :, 0]               # using the red channel since other 2 channels are similar
                X = X[:, np.newaxis, :, :]      
            elif X.ndim == 3:                   
                X = X[:, np.newaxis, :, :]
            else:                              
                X = X.transpose(0, 3, 1, 2)

            # 4. ORL labels go 1-40, but CrossEntropyLoss needs 0-39
            if self.dataset_source_file_name == 'ORL':
                y = y - 1

            print(split, 'X shape:', X.shape, 'y range:', y.min(), 'to', y.max())
            result[split] = {'X': X, 'y': y}

        return result