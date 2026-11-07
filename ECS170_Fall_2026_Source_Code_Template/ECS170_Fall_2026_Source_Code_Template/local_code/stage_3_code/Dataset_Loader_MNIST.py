'''
Loader for the MNIST pickle file
'''

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
import torch


class Dataset_Loader_MNIST(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading MNIST data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        result = {}
        for split in ['train', 'test']:
            images = [instance['image'] for instance in data[split]]
            labels = [instance['label'] for instance in data[split]]

            X = torch.FloatTensor(np.array(images))   # (N, 28, 28)
            X = X.unsqueeze(1) / 255.0                # (N, 1, 28, 28), scaled to 0-1
            y = torch.LongTensor(labels)              # (N,)
            result[split] = {'X': X, 'y': y}

        print('loaded', len(result['train']['y']), 'train and', len(result['test']['y']), 'test images.')
        return result