import os

import numpy as np
import skfda
import torch
from skfda.misc.scoring import mean_squared_error
from skfda.preprocessing.dim_reduction import FPCA
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from configs.NN_config import hparams, SAVE_FOLDER
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

root = get_project_root()


def load_labels(profile, obs="", extension="csv"):
    y = np.loadtxt(
        os.path.join(root, DATA_FOLDER, profile, f"y_{obs}test.{extension}"), delimiter=","
    )
    y = torch.from_numpy(y).to(torch.float32)

    return y


def evaluate(model, test_loader, obs_loader, fpca):
    model.eval()
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        MSE_loss = 0.0
        f_MSE_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            MSE_loss += loss.item()

        avg_MSE_loss = MSE_loss / len(test_loader)
        print(f"Test MSE Loss: {avg_MSE_loss:.8f}")

        if hparams.EVAL.EXTENDED_EVAL:
            for inputs, y_obs in obs_loader:
                outputs = model(inputs)
                rec_outputs = fpca.inverse_transform(outputs.numpy())

                original_outputs = skfda.FDataGrid(data_matrix=y_obs, grid_points=rec_outputs.grid_points)
                f_MSE_loss += mean_squared_error(rec_outputs, original_outputs)

        avg_fMSE_loss = f_MSE_loss / len(test_loader)
        print(f"Test L2 Loss: {avg_fMSE_loss:.8f}")


def plot(model, x, y, y_obs, profile, index, fpca):
    x = x[index].unsqueeze(0)
    y = y[index].numpy()
    y_obs = y_obs[index].numpy()

    y = fpca.inverse_transform(y)
    y_obs = skfda.FDataGrid(data_matrix=y_obs, grid_points=y.grid_points)

    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy()
        y_pred = fpca.inverse_transform(y_pred)

        # Plot both FDataGrid objects
        fig, ax = plt.subplots()
        y_obs.plot(axes=ax, color="black", label="Observation")
        y.plot(axes=ax, color="blue", linestyle="dashed", label="Reconstruction")
        y_pred.plot(axes=ax, color="red", linestyle="dashed", label="Prediction")

        ax.set_title(f"Sample Observation, Reconstruction, and Prediction: {profile}")
        ax.legend()
        ax.set_xlim(-1.5, 1.5)
        #ax.set_ylim(-1, 1)
        ax.set_xlabel('Δλ')
        ax.set_ylabel(profile)
        plt.show()


def main():
    index = 8677 #np.random.randint(0, 10000)
    x = np.loadtxt(
        os.path.join(root, DATA_FOLDER, "x_test.csv"), delimiter=","
    )
    x = torch.from_numpy(x).to(torch.float32)
    for profile in ["I", "Q", "U", "V"]:
        print(f"evaluating profile {profile}")
        y = load_labels(profile)

        test_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=hparams.EVAL.BATCH_SIZE,
            shuffle=False,
        )
        model_state_dict = torch.load(
            os.path.join(root, SAVE_FOLDER, profile, hparams.EVAL.NAME)
        )
        model = MLP()
        model.load_state_dict(model_state_dict)

        if hparams.EVAL.EXTENDED_EVAL:
            y_obs = load_labels(profile, "obs_")

            fpca = FPCA(n_components=hparams.FPCA.N_COMPONENTS)
            fpca_dictionary = np.load(
                os.path.join(root, DATA_FOLDER, profile, f"{profile}_decomposition.npy"),
                allow_pickle="TRUE",
            ).item()

            fpca.components_ = fpca_dictionary["components_"]
            fpca.mean_ = fpca_dictionary["mean_"]
            fpca.singular_values_ = fpca_dictionary["singular_values"]

            plot(model, x, y, y_obs, profile, index, fpca)
            obs_loader = DataLoader(
                TensorDataset(x, y_obs),
                batch_size=hparams.EVAL.BATCH_SIZE,
                shuffle=False,
            )
        else:
            fpca = None
            obs_loader = None

        evaluate(model, test_loader, obs_loader, fpca)


if __name__ == "__main__":
    main()
