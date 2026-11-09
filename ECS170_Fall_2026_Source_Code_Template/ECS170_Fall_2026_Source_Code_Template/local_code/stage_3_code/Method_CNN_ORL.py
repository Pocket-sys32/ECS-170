'''
CNN model for the ORL face dataset (40 people)
'''

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn


class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # convolution block 1: 1 gray channel in, 16 pattern maps out
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # convolution block 2: 16 maps in, 32 maps out
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        # classifier (like the Stage 2 MLP): 32 maps of 28 x 23 -> 128 -> 40 people
        self.flatten = nn.Flatten()
        self.fc_layer_1 = nn.Linear(32 * 28 * 23, 128)
        self.relu_3 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))   # (N, 16, 56, 46)
        x = self.pool_2(self.relu_2(self.conv_2(x)))   # (N, 32, 28, 23)
        x = self.flatten(x)                            # (N, 20608)
        x = self.relu_3(self.fc_layer_1(x))            # (N, 128)
        y_pred = self.fc_layer_2(x)                    # (N, 40) raw scores, no softmax
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # convert the data to tensors once, before the loop
        X_tensor = torch.FloatTensor(X)
        y_true = torch.LongTensor(y)
        self.loss_history = []

        # ORL has only 360 images, so we use the whole training set as one batch
        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)
            self.loss_history.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(X))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']} 