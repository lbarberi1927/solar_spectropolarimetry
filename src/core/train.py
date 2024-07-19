# Training function
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.NN_config import hparams, DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

root = get_project_root()

def train(train_loader):
    model = MLP()
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.TRAIN.LEARNING_RATE)

    for epoch in range(hparams.TRAIN.EPOCHS):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{hparams.TRAIN.EPOCHS}], Loss: {epoch_loss:.4f}')


def load_data(profile):
    x = np.loadtxt(os.path.join(root, DATA_FOLDER, profile, "x_train.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(root, DATA_FOLDER, profile, "y_train.csv"), delimiter=",")

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)

    return x, y


def main():
    x, y = load_data("I")
    train_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=hparams.TRAIN.BATCH_SIZE,
        shuffle=True,
    )
    train(train_loader)


if __name__ == '__main__':
    main()
