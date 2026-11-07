'''
CNN method class for MNIST
'''

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # block 1: (N, 1, 28, 28) -> (N, 8, 26, 26) -> pool -> (N, 8, 13, 13)
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        # flatten: (N, 8, 13, 13) -> (N, 1352)
        self.flatten = nn.Flatten()

        # output layer: (N, 1352) -> (N, 10)
        self.fc_layer = nn.Linear(1352, 10)

    def forward(self, x):
        h = self.pool_1(self.relu_1(self.conv_1(x)))
        h = self.flatten(h)
        return self.fc_layer(h)   # raw scores, no softmax

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        batch_size = 128
        n = X.shape[0]
        self.loss_history = []   # one loss value per epoch, for the plot

        for epoch in range(self.max_epoch):
            perm = torch.randperm(n)          # shuffle the order each epoch
            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]  # indices for this batch
                y_pred = self.forward(X[idx])
                train_loss = loss_function(y_pred, y[idx])

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)

            avg_loss = epoch_loss / n
            self.loss_history.append(avg_loss)
            print('Epoch:', epoch, 'Loss:', avg_loss)

    def test(self, X):
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 1000):
                out = self.forward(X[i:i + 1000])
                preds.append(out.max(1)[1])
        return torch.cat(preds)

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}