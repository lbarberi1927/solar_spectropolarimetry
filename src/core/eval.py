import os

import numpy as np
import torch
from skfda.preprocessing.dim_reduction import FPCA
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from configs.NN_config import hparams
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

root = get_project_root()


def evaluate(model, test_loader):
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")


def load_data(profile):
    x = np.loadtxt(
        os.path.join(root, DATA_FOLDER, profile, "x_test.csv"), delimiter=","
    )
    y = np.loadtxt(
        os.path.join(root, DATA_FOLDER, profile, "y_test.csv"), delimiter=","
    )

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)

    return x, y


def plot(model, x, y, profile):
    index = np.random.randint(0, len(x))
    x = x[index].unsqueeze(0)
    y = y[index].numpy()

    fpca = FPCA(n_components=hparams.FPCA.N_COMPONENTS)
    fpca_dictionary = np.load(
        os.path.join(root, DATA_FOLDER, profile, f"{profile}_decomposition.npy"),
        allow_pickle="TRUE",
    ).item()

    fpca.components_ = fpca_dictionary["components_"]
    fpca.mean_ = fpca_dictionary["mean_"]
    fpca.singular_values_ = fpca_dictionary["singular_values"]

    y = fpca.inverse_transform(y)

    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy()
        y_pred = fpca.inverse_transform(y_pred)

        # Plot both FDataGrid objects
        fig, ax = plt.subplots()
        y.plot(axes=ax, color="blue", label="Observation")
        y_pred.plot(axes=ax, color="red", label="Prediction")

        ax.set_title(f"{profile}")
        ax.legend()
        plt.show()


def main():
    for profile in ["I", "Q", "U", "V"]:
        print(f"evaluating profile {profile}")
        x, y = load_data(profile)
        test_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=hparams.EVAL.BATCH_SIZE,
            shuffle=False,
        )
        model_state_dict = torch.load(
            os.path.join(root, "logs", profile, hparams.EVAL.NAME)
        )
        model = MLP()
        model.load_state_dict(model_state_dict)
        evaluate(model, test_loader)

        if hparams.EVAL.PLOT:
            plot(model, x, y, profile)


if __name__ == "__main__":
    main()
