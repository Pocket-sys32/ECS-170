'''
Concrete MethodRNN class for a specific learning of text
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from local_code.base_class.method import method


class Method_RNN(method, nn.Module):
    def __init__(
        self,
        mName=None,
        mDescription=None,
        vocab_size=30000,
        embedding_dim=50,
        hidden_size=128,
        max_epoch=10,
        learning_rate=1e-3,
        batch_size=64,
        learning_curve_path=None,
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_curve_path = learning_curve_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 2)
        self.to(self.device)

        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

    def forward(self, token_ids):
        # token_ids has shape (batch_size, sequence_length).
        mask = token_ids.ne(0)
        lengths = mask.sum(dim=1).clamp(min=1)
        embedded = self.embedding(token_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.rnn(packed)

        # Shape: (batch_size, sequence_length, hidden_size).
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=token_ids.size(1),
        )

        # Sum hidden states at real tokens, then divide by the token count.
        summed = (output * mask.unsqueeze(-1)).sum(dim=1)
        pooled = summed / lengths.unsqueeze(-1)

        return self.classifier(self.dropout(pooled))

    def _evaluate(self, loader, loss_function):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for token_ids, labels in loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)
                logits = self(token_ids)

                total_loss += loss_function(logits, labels).item() * len(labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def _load_glove(self, path):
        word_to_id = self.data["word_to_id"]
        weights = self.embedding.weight
        matched = 0

        with open(path, encoding="utf-8") as glove_file, torch.no_grad():
            for line in glove_file:
                parts = line.split()
                if len(parts) != self.embedding.embedding_dim + 1:
                    continue

                word_id = word_to_id.get(parts[0])
                if word_id is None or word_id < 2:
                    continue

                vector = weights.new_tensor([float(value) for value in parts[1:]])
                weights[word_id].copy_(vector)
                matched += 1

        print(f"GloVe matched {matched:,} of {len(word_to_id) - 2:,} vocabulary words.")


    def run(self):
        glove_path = getattr(self, "glove_path", None)
        if glove_path is not None:
            self._load_glove(glove_path)
        train_x = np.asarray(self.data["train"]["X"], dtype=np.int64)
        train_y = np.asarray(self.data["train"]["y"], dtype=np.int64)
        test_x = np.asarray(self.data["test"]["X"], dtype=np.int64)
        test_y = np.asarray(self.data["test"]["y"], dtype=np.int64)

        train_indices, validation_indices = train_test_split(
            np.arange(len(train_y)),
            test_size=0.1,
            random_state=42,
            stratify=train_y,
        )

        train_set = TensorDataset(
            torch.from_numpy(train_x[train_indices]),
            torch.from_numpy(train_y[train_indices]),
        )
        validation_set = TensorDataset(
            torch.from_numpy(train_x[validation_indices]),
            torch.from_numpy(train_y[validation_indices]),
        )
        test_set = TensorDataset(
            torch.from_numpy(test_x),
            torch.from_numpy(test_y),
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        validation_loader = DataLoader(
            validation_set, batch_size=self.batch_size
        )
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )

        for epoch in range(self.max_epoch):
            self.train()
            total_loss = 0.0
            total = 0

            for token_ids, labels in train_loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self(token_ids)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total += len(labels)

            validation_loss, validation_accuracy = self._evaluate(
                validation_loader, loss_function
            )
            self.history["train_loss"].append(total_loss / total)
            self.history["validation_loss"].append(validation_loss)
            self.history["validation_accuracy"].append(validation_accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch}: "
                f"train_loss={total_loss / total:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"val_accuracy={validation_accuracy:.4f}"
            )

        test_loss, test_accuracy = self._evaluate(test_loader, loss_function)
        print(f"Test accuracy: {test_accuracy:.4f}")

        if self.learning_curve_path:
            import matplotlib.pyplot as plt

            epochs = range(1, self.max_epoch + 1)
            figure, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(epochs, self.history["train_loss"], label="Train")
            axes[0].plot(epochs, self.history["validation_loss"], label="Validation")
            axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
            axes[0].legend()
            axes[1].plot(
                epochs, self.history["validation_accuracy"], label="Validation"
            )
            axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation accuracy")
            axes[1].legend()
            figure.tight_layout()
            figure.savefig(self.learning_curve_path)
            plt.close(figure)

        predictions = []
        self.eval()
        with torch.no_grad():
            for token_ids, _ in test_loader:
                logits = self(token_ids.to(self.device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())

        return {
            "pred_y": np.asarray(predictions),
            "true_y": test_y,
            "history": self.history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }