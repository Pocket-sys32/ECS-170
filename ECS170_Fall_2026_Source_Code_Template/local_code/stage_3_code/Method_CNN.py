import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


from local_code.base_class.method import method


class _CNNBase(method, nn.Module):
    input_channels = 1
    num_classes = 10

    def __init__(self, mName=None, mDescription=None, max_epoch=15,
                 learning_rate=1e-3, batch_size=128, validation_fraction=0.1,
                 learning_curve_path=None, device=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.learning_curve_path = learning_curve_path
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.num_classes),
        )
        self.to(self.device)
        self.history = {'train_loss': [], 'validation_loss': [], 'validation_accuracy': []}

    def forward(self, x):
        return self.classifier(self.features(x))

    def _evaluate(self, loader, loss_function):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self(images)
                total_loss += loss_function(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    def _save_learning_curves(self):
        if not self.learning_curve_path:
            return
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(self.history['train_loss']) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['validation_loss'], label='Validation')
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss')
        axes[0].legend()
        axes[1].plot(epochs, self.history['validation_accuracy'], label='Validation')
        axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Validation accuracy')
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.learning_curve_path)
        plt.close(figure)

    def run(self):
        train_x = np.asarray(self.data['train']['X'], dtype=np.float32)
        train_y = np.asarray(self.data['train']['y'], dtype=np.int64)
        test_x = np.asarray(self.data['test']['X'], dtype=np.float32)
        test_y = np.asarray(self.data['test']['y'], dtype=np.int64)

        if train_x.ndim != 4 or train_x.shape[1] != self.input_channels:
            raise ValueError(f'Expected training images shaped (N, {self.input_channels}, H, W), got {train_x.shape}')
        if test_x.ndim != 4 or test_x.shape[1] != self.input_channels:
            raise ValueError(f'Expected test images shaped (N, {self.input_channels}, H, W), got {test_x.shape}')
        if train_y.min() < 0 or train_y.max() >= self.num_classes:
            raise ValueError(f'Training labels must be in [0, {self.num_classes - 1}]')

        from sklearn.model_selection import train_test_split
        train_indices, validation_indices = train_test_split(
            np.arange(len(train_y)), test_size=self.validation_fraction,
            random_state=42, stratify=train_y)

        train_tensor = TensorDataset(torch.from_numpy(train_x[train_indices]),
                                     torch.from_numpy(train_y[train_indices]))
        validation_tensor = TensorDataset(torch.from_numpy(train_x[validation_indices]),
                                          torch.from_numpy(train_y[validation_indices]))
        test_tensor = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_tensor, batch_size=self.batch_size)
        test_loader = DataLoader(test_tensor, batch_size=self.batch_size)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.max_epoch):
            self.train(True)
            running_loss = 0.0
            seen = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

            validation_loss, validation_accuracy = self._evaluate(validation_loader, loss_function)
            self.history['train_loss'].append(running_loss / seen)
            self.history['validation_loss'].append(validation_loss)
            self.history['validation_accuracy'].append(validation_accuracy)
            print(f'Epoch {epoch + 1}/{self.max_epoch}: '
                  f'train_loss={self.history["train_loss"][-1]:.4f}, '
                  f'val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.4f}')

        self._save_learning_curves()
        _, test_accuracy = self._evaluate(test_loader, loss_function)
        self.eval()
        predictions = []
        with torch.no_grad():
            for images, _ in test_loader:
                predictions.extend(self(images.to(self.device)).argmax(dim=1).cpu().tolist())
        print(f'Test accuracy: {test_accuracy:.4f}')
        return {'pred_y': np.asarray(predictions), 'true_y': test_y,
                'history': self.history, 'test_accuracy': test_accuracy}


class Method_CNN_MNIST(_CNNBase):
    input_channels = 1
    num_classes = 10


class Method_CNN_ORL(_CNNBase):
    input_channels = 1
    num_classes = 40


class Method_CNN_CIFAR(_CNNBase):
    input_channels = 3
    num_classes = 10
