# Training function
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.NN_config import hparams, SAVE_FOLDER
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root, verify_path_exists

root = get_project_root()
np.random.seed(hparams.SEED)
torch.manual_seed(hparams.SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, profile):
    print(f"Training for profile {profile}", flush=True)
    model = MLP()
    if hparams.TRAIN.PRE_TRAINED:
        model_state_dict = torch.load(
            os.path.join(root, SAVE_FOLDER, profile, hparams.TRAIN.EXISTING_NAME)
        )
        model.load_state_dict(model_state_dict)

    model = model.to(torch.device(device))
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
        print(f"Epoch [{epoch + 1}/{hparams.TRAIN.EPOCHS}], Loss: {epoch_loss:.8f}")

    if hparams.TRAIN.SAVE_MODEL:
        if device == "cuda":
            model = model.cpu()

        save_path = os.path.join(root, SAVE_FOLDER, profile, hparams.TRAIN.NAME)
        verify_path_exists(Path(save_path).parent)
        torch.save(model.state_dict(), save_path)


def load_data(profile):
    y = np.loadtxt(
        os.path.join(root, DATA_FOLDER, profile, "y_train.csv"), delimiter=","
    )
    y = torch.from_numpy(y).to(torch.float32)

    return y


def main():
    print(
        f"preparing training with params: \n"
        f"Learning Rate: {hparams.TRAIN.LEARNING_RATE} \n"
        f"NN Hidden Dimension: {hparams.MODEL.HIDDEN_DIM} \n"
        f"Pre-Trained Model: {str(hparams.TRAIN.PRE_TRAINED)} \n"
        f"Saving model: {str(hparams.TRAIN.SAVE_MODEL)} to {hparams.TRAIN.NAME} \n"
        f"Device: {device} \n",
        flush=True,
    )
    x = np.loadtxt(
        os.path.join(root, DATA_FOLDER, "x_train.csv"), delimiter=","
    )
    x = torch.from_numpy(x).to(torch.float32)
    for profile in ["I", "Q", "U", "V"]:
        y = load_data(profile)
        x = x.to(torch.device(device))
        y = y.to(torch.device(device))
        train_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=hparams.TRAIN.BATCH_SIZE,
            shuffle=True,
        )
        train(train_loader, profile)


if __name__ == "__main__":
    main()
