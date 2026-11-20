'''
Dataset loader for CSV Data
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import csv

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data from CSV...')
        X = []
        y = []
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Convert strings to integers
                values = [int(val) for val in row]
                # First column is label (y), remaining are features (X)
                y.append(values[0])
                X.append(values[1:])
        print(f'Loaded {len(X)} data points.')
        return {'X': X, 'y': y}