import torch
import torch.nn as nn
from local_code.base_class.method import method
import numpy as np

class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch = 20
    #learning_rate = 5e-4
    learning_rate = 1e-3
    optimizer_choice = 'Adam'

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # 2 blocks (main)

        # 1st round 3x3 (original):  (3, 32, 32) -> (32, 16, 16)
        # 3x3 out_channels = 16: (3, 32, 32) -> (16, 16, 16)
        # 5x5 (highest accuracy): (3, 32,32) ->  (32, 14, 14)
        # Other ablation: learning rate of 5e-4 instead of 1e-3

        #self.conv_layer_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        #self.conv_layer_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv_layer_1 = nn.Conv2d(3, 32, kernel_size=5)
        self.norm_1 = nn.BatchNorm2d(32)
        #self.norm_1 = nn.BatchNorm2d(16) # abltn
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)


        # 2nd block 3x3 (original): (32, 16, 16) -> (64, 8, 8)
        # 3x3 out_channels = 32: (16, 16, 16) -> (32, 8, 8)
        # 5x5: (32, 14, 14) -> Output: (64, 5, 5)

        #self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=5)
        #self.conv_layer_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm_2 = nn.BatchNorm2d(64)
        #self.norm_2 = nn.BatchNorm2d(32) # abltn
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: 2 conv 3x3: 64, 8, 8 -> 4096
        # 1 conv 3x3: 32, 16, 16 -> 8192
        # 2 conv 5x5: 64, 5, 5 -> 1600
        # 2 conv 3x3: 32, 8, 8 -> 2048
        self.flatten = nn.Flatten()

        #self.fc_layer_1 = nn.Linear(64 * 8 * 8, 128) # 2 conv 3x3
        #self.fc_layer_1 = nn.Linear(32 * 16 * 16, 128) # 1 conv
        self.fc_layer_1 = nn.Linear(64 * 5 * 5, 128)  # 2 conv 5x5
        #self.fc_layer_1 = nn.Linear(32 * 8 * 8, 128) # 2 conv 3x3

        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc_layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convert numpy arrays to torch tensors if necessary
        # Otherwise might get AttributeError: 'numpy.ndarray' object has no attribute 'dim'. Did you mean: 'ndim'?
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Permute (Batch, H, W, C) -> (Batch, C, H, W) for PyTorch Conv2d
        if x.dim() == 4 and x.shape[-1] == 3:
            # permute() makes the tensor non-contiguous in memory
            x = x.permute(0, 3, 1, 2).contiguous()

        x = x.float() / 255.0  # Scale RGB pixel values from 0-255 to 0-1

        h = self.pool_1(self.relu_1(self.norm_1(self.conv_layer_1(x))))
        h = self.pool_2(self.relu_2(self.norm_2(self.conv_layer_2(h))))
        h = self.flatten(h)
        h = self.dropout(self.relu_fc(self.fc_layer_1(h)))
        return self.fc_layer_2(h)

    def train(self, X, y):
        # Convert numpy arrays to torch tensors if necessary
        # Otherwise might get AttributeError: 'numpy.ndarray' object has no attribute 'dim'. Did you mean: 'ndim'?
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)


        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss()

        batch_size = 64
        n = X.shape[0]
        self.loss_history = []

        for epoch in range(self.max_epoch):
            perm = torch.randperm(n)
            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
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
        # Convert numpy arrays to torch tensors if necessary
        # Otherwise might get AttributeError: 'numpy.ndarray' object has no attribute 'dim'. Did you mean: 'ndim'?
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        predicted = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 1000):
                out = self.forward(X[i:i + 1000])
                predicted.append(out.max(1)[1])
        return torch.cat(predicted)

    def run(self):
        print("Method running...")
        print("--Start training")
        self.train(self.data['train']['X'], self.data['train']['y'])
        print("--start testing...")
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}